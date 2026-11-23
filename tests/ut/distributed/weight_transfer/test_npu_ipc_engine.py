#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Regression tests for the NPU IPC weight transfer engine.

These cover the bugs that broke ``examples/rl/rlhf_http_npu_ipc.py``:

1. ``NPUIPCWeightTransferEngine.__init__`` did not accept the ``model``
   argument that ``WeightTransferEngineFactory.create_engine`` passes,
   raising ``TypeError: __init__() takes 3 positional arguments but 4
   were given`` at engine construction.
2. ``receive_weights`` / ``packed_npu_ipc_consumer`` unpacked the stored
   IPC handle as ``func, args`` even though the producer stored only the
   ``reduce_tensor`` *args*, raising ``ValueError: too many values to
   unpack (expected 2)``. Aligned with upstream vLLM's CUDA IPC engine:
   the producer stores args only and the consumer rebuilds with the
   well-known ``rebuild_npu_tensor``.
3. The NPU IPC worker skipped the layerwise reload START/FINISH lifecycle,
   so runtime-formatted weights were not restored before loading and graph-
   visible storage was not preserved after loading.
4. The packed receive path decoded tensors but never passed them to
   ``model.load_weights``.
"""

import inspect
import os
import sys
import types
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from functools import cached_property
from importlib import import_module
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource, WeightTransferEngine
from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerWeightTransferEngine

from vllm_ascend.distributed.weight_transfer import npu_ipc_engine
from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
    NPUIPCTrainerInitInfo,
    NPUIPCTrainerWeightTransferEngine,
    NPUIPCWeightTransferEngine,
)
from vllm_ascend.utils import vllm_version_is

_MODULE = "vllm_ascend.distributed.weight_transfer.npu_ipc_engine"


def _patch_rebuild_npu_tensor(rebuild_func):
    """Install a fake ``torch_npu.multiprocessing.reductions`` module.

    The engine imports ``rebuild_npu_tensor`` lazily from ``torch_npu``,
    which is only a stub on CPU CI runners, so provide a fake submodule.
    """
    fake_mod = types.ModuleType("torch_npu.multiprocessing.reductions")
    fake_mod.rebuild_npu_tensor = rebuild_func  # type: ignore[attr-defined]
    return patch.dict(
        sys.modules,
        {
            "torch_npu.multiprocessing": types.ModuleType("torch_npu.multiprocessing"),
            "torch_npu.multiprocessing.reductions": fake_mod,
        },
    )


def _patch_reload_module(*, initialize=None, finalize=None):
    """Provide the lazy-imported reload helpers without importing vLLM models."""
    fake_mod = types.ModuleType("vllm.model_executor.model_loader.reload")
    fake_mod.initialize_layerwise_reload = initialize or MagicMock()  # type: ignore[attr-defined]
    fake_mod.finalize_layerwise_reload = finalize or MagicMock()  # type: ignore[attr-defined]
    fake_mod.start_reload = fake_mod.initialize_layerwise_reload  # type: ignore[attr-defined]
    fake_mod.finish_reload = fake_mod.finalize_layerwise_reload  # type: ignore[attr-defined]
    return patch.dict(
        sys.modules,
        {"vllm.model_executor.model_loader.reload": fake_mod},
    )


def test_init_accepts_model_argument():
    """Bug 1: __init__ must accept the optional ``model`` argument."""
    params = inspect.signature(NPUIPCWeightTransferEngine.__init__).parameters
    assert "model" in params


def test_init_passes_model_to_super():
    captured: dict = {}

    def fake_init_v1(self, config, vllm_config, device, model):
        captured["args"] = (config, vllm_config, device, model)

    with patch.object(WeightTransferEngine, "__init__", fake_init_v1):
        device = torch.device("npu:0")
        NPUIPCWeightTransferEngine("config", "vllm_config", device, "model")

    assert captured["args"] == ("config", "vllm_config", device, "model")


def test_unpacked_send_stores_reduce_tensor_args_only():
    """Bug 2 (producer): the handle stores only the ``reduce_tensor`` args.

    This matches upstream vLLM's CUDA IPC engine, which drops the rebuild
    func and relies on the consumer using the well-known rebuild function.
    """
    rebuild_args = (None, None, None, None, None, None, 999, None)

    def rebuild(*args):
        return torch.zeros(3)

    fake_reduce = MagicMock(return_value=(rebuild, rebuild_args))

    with (
        _patch_rebuild_npu_tensor(rebuild),
        patch("torch.multiprocessing.reductions.reduce_tensor", fake_reduce),
        patch("vllm.distributed.weight_transfer.ipc_engine.reduce_tensor", fake_reduce),
    ):
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
            NPUIPCTrainerWeightTransferEngine,
        )

        engine = object.__new__(NPUIPCTrainerWeightTransferEngine)
        engine.client = MagicMock()
        engine.is_sender = True
        engine.npu_uuid = "node-0"
        engine.gpu_uuid = engine.npu_uuid
        source = MagicMock()
        source.__iter__.return_value = iter([("model.weight", torch.zeros(3))])
        source.iter_full.return_value = iter([("model.weight", torch.zeros(3))])
        with patch.object(engine, "_all_gather_and_merge_handles", side_effect=lambda handles: handles):
            engine._send_unpacked(source)

        stored = engine.client.update_weights.call_args.args[0]["ipc_handles"][0]["node-0"]

        # Only the args tuple is stored, not a (func, args) pair.
        assert stored == rebuild_args


def test_receive_weights_rebuilds_with_rebuild_npu_tensor():
    """Bug 2 (consumer): receive_weights rebuilds via ``rebuild_npu_tensor``.

    Verifies the args-only handle is consumed without unpacking errors and
    that the receiver's device index is written into the rebuild args.
    """
    npu_uuid = "node-0"
    device_index = 0

    rebuilt_weight = torch.tensor([1.0, 2.0, 3.0])
    seen = {}

    def fake_rebuild(*args):
        seen["args"] = args
        return rebuilt_weight

    # Sender stores 999 at index 6; the receiver must overwrite it.
    rebuild_args = (None, None, None, None, None, None, 999, None)

    kwargs = dict(
        names=["model.weight"],
        dtype_names=["float32"],
        shapes=[[3]],
        ipc_handles=[{npu_uuid: rebuild_args}],
    )

    update_info = NPUIPCWeightTransferEngine.update_info_cls(**kwargs)

    engine = object.__new__(NPUIPCWeightTransferEngine)
    received: dict[str, list[tuple[str, torch.Tensor]]] = {}
    engine.model = MagicMock()
    engine.device = MagicMock(index=device_index)
    engine.packed = False
    engine.model.load_weights.side_effect = lambda weights: received.update(weights=weights)

    with (
        _patch_rebuild_npu_tensor(fake_rebuild),
        patch(f"{_MODULE}.npu_generate_uuid", return_value=npu_uuid) as mock_uuid,
    ):
        engine.receive_weights(update_info)

    mock_uuid.assert_called_once_with(device_index)
    engine.model.load_weights.assert_called_once()
    assert received["weights"][0][0] == "model.weight"
    assert torch.equal(received["weights"][0][1], rebuilt_weight)
    # Index 6 (device index) overwritten with the receiver's device.
    assert seen["args"][6] == device_index


def test_start_weight_update():
    config = WeightTransferConfig(backend="npu_ipc")
    vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"), weight_transfer_config=config)
    engine = NPUIPCWeightTransferEngine(config, vllm_config, torch.device("cpu"), torch.nn.Linear(2, 2))
    mock_init = MagicMock()

    with _patch_reload_module(initialize=mock_init):
        engine.start_weight_update()

    mock_init.assert_called_once_with(engine.model)


def test_finish_weight_update():
    engine = object.__new__(NPUIPCWeightTransferEngine)
    engine.model = MagicMock()
    engine.model_config = MagicMock()
    engine.device = torch.device("cpu")
    engine._packed_importer = MagicMock()
    mock_finalize = MagicMock()

    with _patch_reload_module(finalize=mock_finalize):
        engine.finish_weight_update()

    if vllm_version_is("0.33.0"):
        mock_finalize.assert_called_once_with(engine.model, engine.model_config)
    else:
        mock_finalize.assert_called_once_with(engine.model, engine.model_config, engine.device)
    engine._packed_importer.close.assert_called_once_with()


def test_receive_packed_weights_loads_model():
    packed_weights = [("model.weight", torch.tensor([1.0, 2.0, 3.0]))]
    update_info = MagicMock(
        tensor_sizes=[12],
        ipc_handles={"node-0": ("packed-handle",)},
        names=["model.weight"],
        shapes=[[3]],
        dtype_names=["float32"],
    )

    engine = object.__new__(NPUIPCWeightTransferEngine)
    engine.model = MagicMock()
    engine.device = MagicMock(index=0)
    engine.packed = True
    engine._packed_importer = MagicMock()

    with (
        patch(f"{_MODULE}.npu_generate_uuid", return_value="node-0"),
        patch(
            f"{_MODULE}.packed_npu_ipc_consumer",
            return_value=packed_weights,
        ) as mock_consumer,
    ):
        engine.receive_weights(update_info)

    mock_consumer.assert_called_once_with(
        ipc_handle=update_info.ipc_handles,
        physical_npu_id="node-0",
        names=update_info.names,
        shapes=update_info.shapes,
        dtype_names=update_info.dtype_names,
        tensor_sizes=update_info.tensor_sizes,
        device_index=0,
        importer=engine._packed_importer,
    )
    engine.model.load_weights.assert_called_once_with(packed_weights)


@pytest.mark.parametrize("packed", [False, True])
def test_trainer_init_uses_assigned_npu_without_cuda(packed):
    client = MagicMock()
    source = [("w", torch.ones(1))]
    accelerator = types.SimpleNamespace(current_device_index=lambda: 2)
    with (
        patch.object(torch, "accelerator", accelerator, create=True),
        patch(f"{_MODULE}.npu_generate_uuid", return_value="node-7") as uuid,
        patch.object(torch.cuda, "get_device_properties", side_effect=AssertionError("CUDA used")),
    ):
        engine = NPUIPCTrainerWeightTransferEngine(
            client=client,
            source=source,
            packed=packed,
            packed_buffer_size_bytes=16,
        )
    assert engine.source is source
    assert engine.device_index == 2
    assert engine.gpu_uuid == engine.npu_uuid == "node-7"
    assert engine.packed is packed
    assert engine.packed_buffer_size_bytes == 16
    if hasattr(IPCTrainerWeightTransferEngine, "send_lora_weights"):
        assert engine._pending_lora_exports is None
        assert engine.device == torch.device("npu", 2)
    uuid.assert_called_once_with(2)


def test_receive_tensors_imports_npu_storage_without_loading_base_weights():
    """The adapter receive path uses NPU reconstruction without applying a model reload."""

    class WeightModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((2, 2), -1.0))

        def load_weights(self, weights):
            with torch.no_grad():
                for name, tensor in weights:
                    getattr(self, name).copy_(tensor)

    config = WeightTransferConfig(backend="npu_ipc")
    vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"), weight_transfer_config=config)
    model = WeightModel()
    engine = NPUIPCWeightTransferEngine(config, vllm_config, torch.device("cpu:0"), model)
    tensor = torch.arange(4.0).reshape(2, 2)
    original = model.weight.detach().clone()
    info = engine.parse_update_info(
        dict(
            names=["weight"],
            dtype_names=["float32"],
            shapes=[[2, 2]],
            ipc_handles=[{"node-0": (None,) * 6 + (99, tensor)}],
        )
    )

    def rebuild(*args):
        assert args[6] == 0
        return args[-1].view_as(args[-1])

    with (
        _patch_rebuild_npu_tensor(rebuild),
        patch(f"{_MODULE}.npu_generate_uuid", return_value="node-0"),
        patch.object(torch.cuda, "get_device_properties", side_effect=AssertionError("CUDA used")),
        patch(
            "vllm.distributed.weight_transfer.ipc_engine.rebuild_cuda_tensor", side_effect=AssertionError("CUDA used")
        ),
    ):
        [(name, received)] = engine.receive_tensors(info)
        assert name == "weight"
        assert received.data_ptr() == tensor.data_ptr()
        torch.testing.assert_close(received, tensor, rtol=0, atol=0)
        torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
        engine.receive_weights(info)
    torch.testing.assert_close(model.weight, tensor, rtol=0, atol=0)
    supported = hasattr(IPCTrainerWeightTransferEngine, "send_lora_weights")
    assert engine.supports_lora_weight_updates is supported
    engine.init_transfer_engine(engine.init_info_cls(packed=True))
    assert not engine.supports_lora_weight_updates


class _LoRAOnlySource(WeightSource):
    def __init__(self, tensors):
        self.tensors = tensors

    @cached_property
    def metadata(self):
        return [ParamMeta(name, tensor.dtype, tuple(tensor.shape)) for name, tensor in self.tensors.items()]

    def iter_full(self):
        yield from self.tensors.items()


class _CPUStorageNPUTrainer(NPUIPCTrainerWeightTransferEngine):
    """Use CPU tensors while exercising the actual NPU export and lifecycle hooks."""

    def _initialize_device(self):
        self.device_index = 0
        self.device = torch.device("cpu")
        self.npu_uuid = "node-0"
        self.gpu_uuid = self.npu_uuid


class _LoRAControlClient:
    """Record the external control service while exposing transferred values."""

    def __init__(self):
        self.calls = []
        self.payload = None

    def init_weight_transfer_engine(self, init_info):
        assert init_info == {"packed": False}
        self.calls.append(("init", None))

    def start_lora_weight_update(self, request):
        self.calls.append(("start", request["update_id"]))

    def update_lora_weights(self, update_info):
        self.payload = update_info
        self.calls.append(("update", update_info["update_id"]))

    def finish_lora_weight_update(self, update_id=None):
        self.calls.append(("finish", update_id))

    def abort_lora_weight_update(self, update_id=None):
        self.calls.append(("abort", update_id))


def test_inherited_lora_send_uses_dynamic_npu_reducer_and_retains_owner():
    """Shared LoRA sends preserve ownership and cloned snapshots through NPU hooks."""
    if not hasattr(IPCTrainerWeightTransferEngine, "send_lora_weights"):
        # Older upstream releases keep their base-model transfer APIs; the new
        # LoRA API is available only after its upstream extension is installed.
        assert not hasattr(NPUIPCTrainerWeightTransferEngine, "send_lora_weights")
        return

    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    tensors = {
        f"{prefix}.lora_A.weight": torch.arange(8.0).reshape(2, 4),
        f"{prefix}.lora_B.weight": torch.arange(8.0).reshape(4, 2),
    }
    originals = {name: tensor.clone() for name, tensor in tensors.items()}
    exported_pointers = []

    def rebuild(*args):
        return args[-1]

    def reduce(tensor):
        exported_pointers.append(tensor.data_ptr())
        return rebuild, (None,) * 6 + (0, tensor)

    client = _LoRAControlClient()
    with (
        _patch_rebuild_npu_tensor(rebuild),
        patch("torch.multiprocessing.reductions.reduce_tensor", side_effect=reduce),
        patch(
            "vllm.distributed.weight_transfer.ipc_engine.reduce_tensor",
            side_effect=AssertionError("stale CUDA reducer used"),
        ),
        patch.object(torch.cuda, "get_device_properties", side_effect=AssertionError("CUDA used")),
        patch.object(torch.cuda, "ipc_collect", side_effect=AssertionError("CUDA used")),
        patch.object(torch, "npu", types.SimpleNamespace(synchronize=lambda: None), create=True),
    ):
        engine = _CPUStorageNPUTrainer.trainer_init(
            NPUIPCTrainerInitInfo(rank=0, packed=False), client=client, source=_LoRAOnlySource(tensors)
        )
        engine.send_lora_weights(
            {
                "update_id": "policy-1",
                "lora_name": "policy",
                "lora_int_id": 1,
                "version": 1,
                "expected_version": None,
                "peft_config": {"r": 2, "lora_alpha": 4, "target_modules": ["q_proj"]},
                "names": list(tensors),
                "dtype_names": ["float32", "float32"],
                "shapes": [[2, 4], [4, 2]],
            }
        )
    assert client.calls == [("init", None), ("start", "policy-1"), ("update", "policy-1"), ("finish", "policy-1")]
    assert len(exported_pointers) == len(tensors)
    assert all(pointer != tensor.data_ptr() for pointer, tensor in zip(exported_pointers, tensors.values()))
    assert engine._pending_lora_exports is None
    assert client.payload is not None
    assert len(client.payload["ipc_handles"]) == len(tensors)
    for name, handle in zip(tensors, client.payload["ipc_handles"]):
        torch.testing.assert_close(handle["node-0"][-1], originals[name], rtol=0, atol=0)
        torch.testing.assert_close(tensors[name], originals[name], rtol=0, atol=0)


@pytest.mark.parametrize("is_sender", [False, True])
def test_packed_sender_waits_before_reusing_each_chunk(is_sender):
    engine = object.__new__(NPUIPCTrainerWeightTransferEngine)
    engine.client = MagicMock()
    engine.is_sender = is_sender
    engine.packed = True
    engine.npu_uuid = "node-0"
    engine.packed_buffer_size_bytes = 16
    engine.source = MagicMock()
    events: list[str] = []

    def chunks(**kwargs):
        assert kwargs["npu_uuid"] == "node-0"
        assert kwargs["buffer_size_bytes"] == 16
        for i in range(2):
            yield dict(
                names=[f"w{i}"], dtype_names=["float32"], shapes=[[4]], tensor_sizes=[16], ipc_handle={"node-0": (i,)}
            )
            assert events[-1] == "sync"
            events.append("reuse")

    engine.client.update_weights.side_effect = lambda payload: events.append("update")
    with (
        patch.object(engine, "_all_gather_and_merge_handles", side_effect=lambda handles: handles),
        patch.object(engine, "_post_send_sync", side_effect=lambda: events.append("sync")),
        patch(f"{_MODULE}.packed_npu_ipc_producer", side_effect=chunks),
    ):
        engine._send_packed(engine.source)
    expected_chunk = ["update", "sync", "reuse"] if is_sender else ["sync", "reuse"]
    assert events == expected_chunk * 2
    assert engine.client.update_weights.call_count == (2 if is_sender else 0)


@pytest.mark.parametrize("is_sender", [False, True])
def test_multi_rank_handles_are_merged_and_synchronized_on_npu(is_sender):
    engine = object.__new__(NPUIPCTrainerWeightTransferEngine)
    engine.is_sender = is_sender
    local = [{"node-0": (0,)}, {"node-0": (1,)}]
    remote = [{"node-1": (2,)}, {"node-1": (3,)}]
    events = []

    def gather(output, handles):
        assert handles is local
        output[:] = [local, remote]
        events.append("gather")

    with (
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "get_world_size", return_value=2),
        patch.object(torch.distributed, "all_gather_object", side_effect=gather),
        patch.object(torch.distributed, "barrier", side_effect=lambda: events.append("barrier")),
        patch.object(torch, "npu", types.SimpleNamespace(synchronize=lambda: events.append("npu")), create=True),
        patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CUDA used")),
    ):
        result = engine._all_gather_and_merge_handles(local)
    assert events == ["gather", "barrier", "npu"]
    assert result == ([{**local[i], **remote[i]} for i in range(2)] if is_sender else [{}, {}])


def test_packed_receive_rejects_missing_sizes():
    engine = object.__new__(NPUIPCWeightTransferEngine)
    engine.device = types.SimpleNamespace(index=1)
    engine.packed = True
    engine.model = MagicMock()
    with (
        patch(f"{_MODULE}.npu_generate_uuid", return_value="node-1"),
        pytest.raises(ValueError, match="tensor_sizes.*required"),
    ):
        engine.receive_weights(types.SimpleNamespace(tensor_sizes=None))
    engine.model.load_weights.assert_not_called()


def _legacy_mtp_validation_fixture():
    """Supply the removed release API when testing its engine branch on main."""
    enabled = ContextVar("test_mtp_completeness_check_enabled", default=True)

    @contextmanager
    def disable_mtp_completeness_check():
        token = enabled.set(False)
        try:
            yield
        finally:
            enabled.reset(token)

    return types.SimpleNamespace(
        is_mtp_completeness_check_enabled=enabled.get,
        disable_mtp_completeness_check=disable_mtp_completeness_check,
    )


def test_receive_scopes_mtp_completeness_check_to_loading():
    # This test targets the release-only engine branch on both CPU lanes.
    # The main branch (no legacy import) is covered by Nov10 contract tests.
    mtp_validation = (
        import_module("vllm.model_executor.model_loader.mtp_validation")
        if vllm_version_is("0.33.0")
        else _legacy_mtp_validation_fixture()
    )
    is_mtp_completeness_check_enabled = mtp_validation.is_mtp_completeness_check_enabled

    engine = object.__new__(NPUIPCWeightTransferEngine)
    engine.device = types.SimpleNamespace(index=1)
    engine.packed = False
    engine.model = MagicMock()
    info = engine.parse_update_info(
        dict(names=["w"], dtype_names=["float32"], shapes=[[1]], ipc_handles=[{"node-1": (None,) * 6 + (9,)}])
    )

    def load(weights):
        assert not is_mtp_completeness_check_enabled()
        raise RuntimeError("load failed")

    engine.model.load_weights.side_effect = load
    with (
        patch(f"{_MODULE}.vllm_version_is", return_value=True),
        patch(f"{_MODULE}.import_module", return_value=mtp_validation),
        _patch_rebuild_npu_tensor(lambda *args: torch.ones(1)),
        patch(f"{_MODULE}.npu_generate_uuid", return_value="node-1"),
        pytest.raises(RuntimeError, match="load failed"),
    ):
        engine.receive_weights(info)
    assert is_mtp_completeness_check_enabled()


def test_uuid_uses_assigned_device_with_visibility_mapping():
    npu_ipc_engine.npu_generate_uuid.cache_clear()
    try:
        with (
            patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "3,7"}),
            patch(f"{_MODULE}.get_ip", return_value="host"),
            patch.object(torch, "accelerator", types.SimpleNamespace(current_device_index=lambda: 0), create=True),
        ):
            assert npu_ipc_engine.npu_generate_uuid(1) == "host-7"
    finally:
        npu_ipc_engine.npu_generate_uuid.cache_clear()


@pytest.mark.parametrize("cleanup", ["finish_weight_update", "shutdown"])
def test_packed_export_imported_once_with_fresh_payload_and_release(cleanup):
    """Multiple chunks share an import, but load fresh independent weights."""

    def fake_base_init(self, config, vllm_config, device, model):
        self.device, self.model = device, model
        self.model_config = None

    model = MagicMock()
    with patch.object(WeightTransferEngine, "__init__", fake_base_init):
        engine = NPUIPCWeightTransferEngine(None, None, torch.device("cpu:0"), model)
    engine.packed = True
    staging = torch.tensor([1.0, 2.0])
    imports: list[weakref.ReferenceType[torch.Tensor]] = []
    loaded: list[torch.Tensor] = []

    def rebuild(*args):
        assert args[6] == 0
        assert args[-1] == f"export-{len(imports) + 1}"
        mapping = staging.view(torch.uint8)
        imports.append(weakref.ref(mapping))
        return mapping

    model.load_weights.side_effect = lambda weights: loaded.append(weights[0][1])
    info = engine.update_info_cls(
        names=["w"],
        dtype_names=["float32"],
        shapes=[[2]],
        tensor_sizes=[8],
        ipc_handles={"node-0": (None,) * 6 + (999, "export-1")},
    )
    with (
        _patch_rebuild_npu_tensor(rebuild),
        _patch_reload_module(),
        patch(f"{_MODULE}.npu_generate_uuid", return_value="node-0"),
    ):
        engine.receive_weights(info)
        staging.copy_(torch.tensor([3.0, 4.0]))
        engine.receive_weights(info)
        assert len(imports) == 1
        assert imports[0]() is not None
        assert torch.equal(loaded[0], torch.tensor([1.0, 2.0]))
        assert torch.equal(loaded[1], torch.tensor([3.0, 4.0]))
        getattr(engine, cleanup)()
        assert imports[0]() is None
        # Cleanup is idempotent; a later update must open a new mapping.
        getattr(engine, cleanup)()
        next_info = replace(info, ipc_handles={"node-0": (None,) * 6 + (999, "export-2")})
        engine.receive_weights(next_info)
        assert len(imports) == 2
        engine.shutdown()
        assert imports[1]() is None


def test_packed_export_replacement_and_failed_rebuild_release():
    from vllm_ascend.distributed.weight_transfer.packed_tensor import NPUPackedBufferImporter

    importer = NPUPackedBufferImporter()
    staging = torch.zeros(8, dtype=torch.uint8)
    refs = []

    def rebuild(*args):
        if args[-1] == "bad-export":
            raise RuntimeError("cannot import")
        mapping = staging.view(-1)
        refs.append(weakref.ref(mapping))
        return mapping

    with _patch_rebuild_npu_tensor(rebuild):
        importer.rebuild(["same-storage", "refcount-slot-1"])
        importer.rebuild(["same-storage", "refcount-slot-1"])
        assert len(refs) == 1
        importer.rebuild(["same-storage", "refcount-slot-2"])
        assert refs[0]() is None
        assert refs[1]() is not None
        with pytest.raises(RuntimeError, match="cannot import"):
            importer.rebuild(["same-storage", "bad-export"])
        assert refs[1]() is not None
        importer.close()
        assert refs[1]() is None
