# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU IPC-based weight transfer engine using Ascend IPC for communication."""

import os
import socket
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from vllm.distributed.weight_transfer.base import (
    TrainerWeightTransferEngine,
)
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerInitInfo,
    IPCTrainerWeightTransferEngine,
    IPCWeightTransferEngine,
    IPCWeightTransferInitInfo,
    IPCWeightTransferUpdateInfo,
)

from vllm_ascend.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    NPUPackedBufferImporter,
    packed_npu_ipc_consumer,
    packed_npu_ipc_producer,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.weight_transfer import WeightTransferConfig
    from vllm.distributed.weight_transfer.base import (
        VLLMWeightSyncClient,
        WeightSource,
    )


def vllm_version_is(target_version: str) -> bool:
    """Defer release-only compatibility checks until model weights are loaded."""
    from vllm_ascend.utils import vllm_version_is as check_version

    return check_version(target_version)


@dataclass
class NPUIPCWeightTransferInitInfo(IPCWeightTransferInitInfo):
    """Worker-side NPU IPC wire parameters, shared with upstream IPC."""


@lru_cache(maxsize=1)
def get_ip() -> str:
    try:
        # try to get ip from network interface
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:  # noqa: BLE001
        # fallback to get ip from hostname
        return socket.gethostbyname(socket.gethostname())


@lru_cache(maxsize=1)
def npu_generate_uuid(logical_device: int | None = None) -> str:
    """Generate a unique identifier for the current process's physical NPU chip.

    Returns ``{host_ip}-{physical_chip_id}`` where ``host_ip`` is the local
    machine's IP address and ``physical_chip_id`` is derived from the current
    logical device index mapped through ``ASCEND_RT_VISIBLE_DEVICES``. The
    logical index is read from the current device when it is not provided.

    On Ascend NPU, ``torch.accelerator.current_device_index()`` returns the
    *logical* device index. When ``ASCEND_RT_VISIBLE_DEVICES`` is set, it
    maps logical indices to physical chip IDs (e.g., ``ASCEND_RT_VISIBLE_DEVICES=2,3``
    means logical device 0 → physical chip 2, logical device 1 → physical chip 3).
    If the env var is not set, the logical index is used directly as the
    physical chip ID (identity mapping).

    The result is cached because it is constant for the lifetime of the
    process. Both the trainer and inference worker processes co-located
    on the same physical NPU chip will produce the same UUID, which is
    required for NPU IPC handle matching.
    """
    if logical_device is None:
        logical_device = torch.accelerator.current_device_index()
    visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", None)
    if visible_devices:
        physical_device = int(visible_devices.split(",")[logical_device].strip())
    else:
        physical_device = logical_device
    return f"{get_ip()}-{physical_device}"


@dataclass
class NPUIPCTrainerInitInfo(IPCTrainerInitInfo):
    """NPU IPC trainer init info — overrides the backend key only.

    ``IPCTrainerInitInfo`` already provides ``packed`` and
    ``packed_buffer_size_bytes``; this subclass only rebinds the
    factory ``backend`` from ``"ipc"`` to ``"npu_ipc"``.
    """

    backend: ClassVar[str] = "npu_ipc"


@dataclass
class NPUIPCWeightTransferUpdateInfo(IPCWeightTransferUpdateInfo):
    """Upstream IPC metadata plus the existing NPU packed-buffer byte sizes."""

    tensor_sizes: list[int] | None = None


class NPUIPCWeightTransferEngine(IPCWeightTransferEngine):
    """
    Weight transfer engine using NPU IPC for communication between
    trainer and workers.

    This implementation uses Ascend NPU IPC to transfer weights from the
    trainer (rank 0) to all inference workers. IPC handles are used to
    share memory between processes on the same node.

    Requires ``torch_npu`` to be imported (which patches
    ``torch.multiprocessing.reductions.reduce_tensor`` to support
    NPU tensors via ``_share_npu_()`` / ``rebuild_npu_tensor``).
    """

    init_info_cls = NPUIPCWeightTransferInitInfo
    update_info_cls = NPUIPCWeightTransferUpdateInfo

    def __init__(
        self,
        config: "WeightTransferConfig",
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        if not hasattr(IPCTrainerWeightTransferEngine, "_collect_ipc"):
            self._packed_importer = NPUPackedBufferImporter()

    if not hasattr(IPCTrainerWeightTransferEngine, "_collect_ipc"):

        @property
        def supports_lora_weight_updates(self) -> bool:
            """Advertise adapter transport only with the upstream LoRA IPC hooks."""
            return (
                not self.packed
                and hasattr(IPCWeightTransferEngine, "_rebuild_ipc_tensor")
                and hasattr(IPCTrainerWeightTransferEngine, "send_lora_weights")
                and hasattr(IPCTrainerWeightTransferEngine, "_initialize_device")
            )

    def _get_device_uuid(self) -> str:
        return npu_generate_uuid(self.device.index)

    def _rebuild_ipc_tensor(self, args: tuple) -> torch.Tensor:
        # Import only on NPU workers, after torch_npu registers its backend.
        from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

        local_args = list(args)
        if len(local_args) <= 6:
            raise ValueError("NPU IPC rebuild arguments do not contain a device index")
        local_args[6] = self.device.index
        return rebuild_npu_tensor(*local_args)

    if not hasattr(IPCTrainerWeightTransferEngine, "_collect_ipc"):

        def start_weight_update(self) -> None:
            """Start the base-model reload without creating a CUDA packed consumer."""
            if vllm_version_is("0.33.0"):
                from vllm.model_executor.model_loader.reload import initialize_layerwise_reload

                initialize_layerwise_reload(self.model)
            else:
                from vllm.model_executor.model_loader.reload import start_reload

                start_reload(self.model)
            self._packed_importer.close()

        def finish_weight_update(self) -> None:
            """Finalize the base model and release this export's NPU IPC mapping."""
            try:
                if vllm_version_is("0.33.0"):
                    from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

                    finalize_layerwise_reload(self.model, self.model_config)
                else:
                    from vllm.model_executor.model_loader.reload import finish_reload

                    finish_reload(self.model, self.model_config, self.device)
            finally:
                self._packed_importer.close()

        def shutdown(self) -> None:
            self._packed_importer.close()

    @staticmethod
    def trainer_send_weights(*args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "The static NPU IPC trainer path has been replaced by "
            "NPUIPCTrainerWeightTransferEngine. Build it via "
            "WeightTransferTrainerFactory.trainer_init("
            "NPUIPCTrainerInitInfo(...), client=..., "
            "source=...) and drive it with send_weights()."
        )

    if not hasattr(IPCTrainerWeightTransferEngine, "_collect_ipc"):

        def receive_weights(self, update_info: IPCWeightTransferUpdateInfo) -> None:
            """Receive weights from the trainer via NPU IPC handles.

            Args:
                update_info: NPU IPC update info containing parameter names,
                    dtypes, shapes, and IPC handles.
            """
            weights = self.receive_tensors(update_info)

            mtp_check = nullcontext()
            if vllm_version_is("0.33.0"):
                # This module is release-only; avoid importing it on newer main.
                mtp_validation = import_module("vllm.model_executor.model_loader.mtp_validation")
                mtp_check = mtp_validation.disable_mtp_completeness_check()

            with mtp_check:
                self.model.load_weights(weights)

        def receive_tensors(self, update_info: IPCWeightTransferUpdateInfo) -> list[tuple[str, torch.Tensor]]:
            """Import NPU tensors without applying them to either model or adapter."""
            if not self.packed and hasattr(IPCWeightTransferEngine, "_rebuild_ipc_tensor"):
                return super().receive_tensors(update_info)

            device_index = self.device.index
            physical_npu_id = self._get_device_uuid()

            if self.packed:
                if update_info.tensor_sizes is None:
                    raise ValueError("`tensor_sizes` is required when packed=True")
                assert isinstance(update_info.ipc_handles, dict)
                return packed_npu_ipc_consumer(
                    ipc_handle=update_info.ipc_handles,
                    physical_npu_id=physical_npu_id,
                    names=update_info.names,
                    shapes=update_info.shapes,
                    dtype_names=update_info.dtype_names,
                    tensor_sizes=update_info.tensor_sizes,
                    device_index=device_index,
                    importer=self._packed_importer,
                )
            else:
                assert isinstance(update_info.ipc_handles, list)
                weights = []
                for name, ipc_handle in zip(
                    update_info.names,
                    update_info.ipc_handles,
                ):
                    if physical_npu_id not in ipc_handle:
                        raise ValueError(
                            f"IPC handle not found for NPU UUID {physical_npu_id}. "
                            f"Available UUIDs: {list(ipc_handle.keys())}. "
                            f"This may indicate that the trainer and worker are "
                            f"not co-located on the same physical NPU (node)."
                        )

                    weight = self._rebuild_ipc_tensor(ipc_handle[physical_npu_id])
                    weights.append((name, weight))
                return weights


class NPUIPCTrainerWeightTransferEngine(IPCTrainerWeightTransferEngine):
    """Trainer-side NPU IPC weight transfer engine.

    Inherits the handshake, packed/unpacked sends and payload construction
    when upstream provides device hooks. Older releases retain their NPU
    packed compatibility path.
    """

    init_info_cls = NPUIPCTrainerInitInfo
    update_info_cls = NPUIPCWeightTransferUpdateInfo

    def __init__(
        self,
        *,
        client: "VLLMWeightSyncClient",
        source: "WeightSource",
        is_sender: bool = True,
        packed: bool = False,
        packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    ) -> None:
        if hasattr(IPCTrainerWeightTransferEngine, "_initialize_device"):
            super().__init__(
                client=client,
                source=source,
                is_sender=is_sender,
                packed=packed,
                packed_buffer_size_bytes=packed_buffer_size_bytes,
            )
        else:
            # Released vLLM queries CUDA directly in its IPC constructor and
            # has no LoRA IPC API; keep its existing base-weight path working.
            TrainerWeightTransferEngine.__init__(
                self,
                client=client,
                source=source,
                is_sender=is_sender,
            )
            self.packed = packed
            self.packed_buffer_size_bytes = packed_buffer_size_bytes
            self._initialize_device()

    def _initialize_device(self) -> None:
        self.device_index = torch.accelerator.current_device_index()
        self.device = torch.device("npu", self.device_index)
        self.npu_uuid = npu_generate_uuid(self.device_index)
        # Upstream's unpacked sender treats gpu_uuid as an opaque wire key.
        self.gpu_uuid = self.npu_uuid

    def _export_ipc_tensor(self, tensor: torch.Tensor) -> tuple:
        from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

        # torch_npu replaces this module attribute when it registers its IPC
        # reducer. Reading it here avoids a stale CUDA-only imported alias.
        rebuild, args = torch.multiprocessing.reductions.reduce_tensor(tensor)
        if rebuild is not rebuild_npu_tensor:
            raise ValueError("The NPU IPC reducer did not produce an NPU tensor handle")
        return args

    def _synchronize_device(self) -> None:
        torch.npu.synchronize()

    if not hasattr(IPCTrainerWeightTransferEngine, "_collect_ipc"):

        def _all_gather_and_merge_handles(
            self,
            handles: list[dict[str, tuple]],
        ) -> list[dict[str, tuple]]:
            """All-gather and merge IPC handle dicts across ranks in one call.

            Each rank contributes a list of ``{npu_uuid: ipc_args}`` dicts.
            A single all_gather_object collects every rank's full list, then
            the sender merges per-index so each dict maps every NPU UUID to
            its args. No-op (returns handles unchanged) when no distributed
            group exists.
            """
            if hasattr(IPCTrainerWeightTransferEngine, "_synchronize_device"):
                return super()._all_gather_and_merge_handles(handles)
            if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
                return handles

            world_size = torch.distributed.get_world_size()
            gathered: list[list[dict[str, tuple]] | None] = [None] * world_size
            torch.distributed.all_gather_object(gathered, handles)
            torch.distributed.barrier()
            torch.npu.synchronize()

            if self.is_sender:
                merged: list[dict[str, tuple]] = []
                for param_idx in range(len(handles)):
                    m: dict[str, tuple] = {}
                    for rank_handles in gathered:
                        if rank_handles is not None:
                            m.update(rank_handles[param_idx])
                    merged.append(m)
                return merged
            return [{} for _ in handles]

        @staticmethod
        def _post_send_sync() -> None:
            """Synchronize each send and keep all trainer ranks in step."""
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                torch.distributed.barrier()
            torch.npu.synchronize()

        def _send_packed(self, source: "WeightSource") -> None:
            """Send weights in bounded-memory chunks (packed mode)."""
            # Both supported versions drive the WeightSource iter_full protocol.
            iterator = source.iter_full()
            for chunk in packed_npu_ipc_producer(
                iterator=iterator,
                npu_uuid=self.npu_uuid,
                post_iter_func=lambda item: item[1],
                buffer_size_bytes=self.packed_buffer_size_bytes,
            ):
                ipc_handle = self._all_gather_and_merge_handles([chunk["ipc_handle"]])[0]
                self._do_send(
                    names=chunk["names"],
                    dtype_names=chunk["dtype_names"],
                    shapes=chunk["shapes"],
                    ipc_handles=ipc_handle,
                    tensor_sizes=chunk["tensor_sizes"],
                )
                # Per-chunk barrier: the producer reuses a single IPC buffer
                # across chunks. Without syncing every rank here, non-sender
                # ranks race ahead and overwrite their buffer while their
                # colocated worker is still reading the current chunk, silently
                # corrupting the transfer.
                self._post_send_sync()

        def _do_send(
            self,
            names: list[str],
            dtype_names: list[str],
            shapes: list[list[int]],
            ipc_handles: list[dict[str, tuple]] | dict[str, tuple],
            tensor_sizes: list[int] | None = None,
        ) -> None:
            """Keep the NPU packed wire format while reusing upstream unpacked sends."""
            if tensor_sizes is None:
                super()._do_send(
                    names=names,
                    dtype_names=dtype_names,
                    shapes=shapes,
                    ipc_handles=ipc_handles,
                )
            elif self.is_sender:
                self.client.update_weights(
                    asdict(
                        NPUIPCWeightTransferUpdateInfo(
                            names=names,
                            dtype_names=dtype_names,
                            shapes=shapes,
                            ipc_handles=ipc_handles,
                            tensor_sizes=tensor_sizes,
                        )
                    )
                )
