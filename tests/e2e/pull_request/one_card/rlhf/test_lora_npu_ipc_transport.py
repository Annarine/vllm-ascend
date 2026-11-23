# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Cross-process NPU IPC transport and owned LoRA staging, without an LLM.

Run directly when a model-level pytest conftest is incompatible with the
installed upstream runtime:

    python tests/e2e/pull_request/one_card/rlhf/test_lora_npu_ipc_transport.py

This exercises real NPU IPC exports, imports and LoRAWeightUpdateSession.consume.
It deliberately does not call prepare/commit, install an adapter or validate
model outputs. The manager dependency fails immediately if installation is
attempted. No vLLM modules or NPU hardware APIs are mocked.
"""

import argparse
import os
import traceback
from functools import cached_property
from multiprocessing.connection import Connection

import torch
import torch.multiprocessing as mp
import torch_npu  # noqa: F401
from vllm.config import DeviceConfig, VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource
from vllm.lora.weight_update import LoRAWeightUpdateRequest, LoRAWeightUpdateSession

from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
    NPUIPCTrainerInitInfo,
    NPUIPCTrainerWeightTransferEngine,
    NPUIPCWeightTransferEngine,
)

CONTROL_TIMEOUT_SECONDS = 120
SHUTDOWN_TIMEOUT_SECONDS = 10


class _NoAdapterInstallation:
    """Unused manager dependency: this test covers only receiving and copying."""

    def prepare_tensor_adapter(self, request):
        raise AssertionError("This transport test must not prepare an adapter")

    def install_prepared_adapter(self, model):
        raise AssertionError("This transport test must not install an adapter")


class _TensorSource(WeightSource):
    def __init__(self, tensors):
        self.tensors = tensors

    @cached_property
    def metadata(self):
        return [ParamMeta(name, tensor.dtype, tuple(tensor.shape)) for name, tensor in self.tensors.items()]

    def iter_full(self):
        yield from self.tensors.items()


class _PipeControlClient:
    """A real inter-process control connection for the trainer's transport API."""

    def __init__(self, connection: Connection):
        self.connection = connection
        self.verification = None

    def receive(self):
        if not self.connection.poll(CONTROL_TIMEOUT_SECONDS):
            raise TimeoutError("NPU IPC consumer did not acknowledge its operation")
        reply = self.connection.recv()
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def request(self, operation, payload=None):
        self.connection.send((operation, payload))
        return self.receive()

    def init_weight_transfer_engine(self, init_info):
        self.request("init", init_info)

    def start_lora_weight_update(self, request):
        self.request("start", request)

    def update_lora_weights(self, update_info):
        self.request("receive", update_info)

    def finish_lora_weight_update(self, update_id=None):
        # This endpoint acknowledges transport/staging only. It intentionally
        # does not implement the model-level prepare/commit lifecycle.
        self.verification = self.request("verify_staging", update_id)

    def abort_lora_weight_update(self, update_id=None):
        self.request("abort", update_id)


def _consume_in_child(connection: Connection, device_index: int):
    try:
        device = torch.device("npu", device_index)
        torch.npu.set_device(device)
        transfer_config = WeightTransferConfig(backend="npu_ipc")
        config = VllmConfig(device_config=DeviceConfig(device=device), weight_transfer_config=transfer_config)
        model = torch.nn.Linear(4, 4, bias=False, device=device)
        original_base = model.weight.detach().clone()
        receiver = NPUIPCWeightTransferEngine(transfer_config, config, device, model)
        session = None
        imported = {}
        connection.send({"ready": True, "pid": os.getpid()})
        while True:
            operation, payload = connection.recv()
            if operation == "shutdown":
                receiver.shutdown()
                connection.send({"stopped": True})
                break
            if operation == "init":
                receiver.init_transfer_engine(receiver.parse_init_info(payload))
                assert receiver.supports_lora_weight_updates
            elif operation == "start":
                assert session is None
                spec = LoRAWeightUpdateRequest(**payload)
                session = LoRAWeightUpdateSession(_NoAdapterInstallation(), spec, device)
            elif operation == "receive":
                assert session is not None
                update = dict(payload)
                assert update.pop("update_id") == session.request.update_id
                imported = dict(receiver.receive_tensors(receiver.parse_update_info(update)))
                session.consume(list(imported.items()))
                torch.npu.synchronize()
            elif operation == "verify_staging":
                assert session is not None
                assert payload == session.request.update_id
                assert set(imported) == set(session.request.names)
                assert set(session._tensors) == set(imported)
                version = session.request.version
                expected_values = {}
                for name, tensor in imported.items():
                    offset = (version - 1) * 128 + (32 if name.endswith(".lora_B.weight") else 0)
                    expected = torch.arange(8, dtype=torch.float32, device=device).reshape(tensor.shape) + offset
                    expected_values[name] = expected
                    owned = session._tensors[name]
                    assert owned.data_ptr() != tensor.data_ptr()
                    torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
                    torch.testing.assert_close(owned, expected, rtol=0, atol=0)
                b_name = next(name for name in imported if name.endswith(".lora_B.weight"))
                session._tensors[b_name].add_(13)
                torch.npu.synchronize()
                for name, tensor in imported.items():
                    torch.testing.assert_close(tensor, expected_values[name], rtol=0, atol=0)
                torch.testing.assert_close(model.weight, original_base, rtol=0, atol=0)
                count = len(imported)
                session.abort()
                session = None
                imported.clear()
                expected_values.clear()
                del tensor, owned, expected
                torch.npu.synchronize()
                connection.send(
                    {
                        "transport_complete": True,
                        "owned_copy_isolated": True,
                        "tensor_count": count,
                        "version": version,
                        "pid": os.getpid(),
                    }
                )
                continue
            elif operation == "abort":
                assert session is not None
                assert payload is None or payload == session.request.update_id
                session.abort()
                session = None
                imported.clear()
                torch.npu.synchronize()
            else:
                raise ValueError(f"Unknown transport operation: {operation}")
            connection.send({"ok": True})
    except BaseException:
        connection.send({"error": traceback.format_exc()})
        raise
    finally:
        connection.close()


def test_npu_lora_ipc_transport_copies_owned_storage(device_index: int = 0):
    """Native NPU IPC crosses a process boundary and staging never aliases it."""
    assert hasattr(NPUIPCTrainerWeightTransferEngine, "send_lora_weights"), "Install the upstream LoRA IPC API build"
    device = torch.device("npu", device_index)
    torch.npu.set_device(device)
    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe()
    process = context.Process(target=_consume_in_child, args=(child_connection, device_index))
    process.start()
    child_connection.close()
    client = _PipeControlClient(parent_connection)
    try:
        ready = client.receive()
        assert ready["ready"]
        assert ready["pid"] != os.getpid()
        prefix = "base_model.model.model.layers.0.self_attn.q_proj"
        tensors = {
            f"{prefix}.lora_A.weight": torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4),
            f"{prefix}.lora_B.weight": torch.arange(8, dtype=torch.float32, device=device).reshape(4, 2) + 32,
        }
        source = _TensorSource(tensors)
        transfer = NPUIPCTrainerWeightTransferEngine.trainer_init(
            NPUIPCTrainerInitInfo(rank=0, packed=False), client=client, source=source
        )
        for version in (1, 2):
            if version == 2:
                source.tensors = {name: tensor + 128 for name, tensor in tensors.items()}
            original = {name: tensor.clone() for name, tensor in source.tensors.items()}
            transfer.send_lora_weights(
                {
                    "update_id": f"transport-policy-{version}",
                    "lora_name": "policy",
                    "lora_int_id": 1,
                    "version": version,
                    "expected_version": version - 1 if version > 1 else None,
                    "peft_config": {"r": 2, "lora_alpha": 4, "target_modules": ["q_proj"]},
                    "names": [meta.name for meta in source.metadata],
                    "dtype_names": ["float32", "float32"],
                    "shapes": [[2, 4], [4, 2]],
                }
            )
            assert client.verification == {
                "transport_complete": True,
                "owned_copy_isolated": True,
                "tensor_count": 2,
                "version": version,
                "pid": ready["pid"],
            }
            for name, tensor in source.tensors.items():
                torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
        client.request("shutdown")
        process.join(SHUTDOWN_TIMEOUT_SECONDS)
        assert process.exitcode == 0
        print("Two NPU IPC snapshots and owned LoRA staging passed; model prepare/commit was not executed.")
    finally:
        if process.is_alive():
            process.terminate()
        process.join(SHUTDOWN_TIMEOUT_SECONDS)
        parent_connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int, default=0)
    arguments = parser.parse_args()
    test_npu_lora_ipc_transport_copies_owned_storage(arguments.device_index)
