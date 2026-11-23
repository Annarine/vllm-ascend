# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Train a PEFT LoRA step and update a colocated Ascend rollout policy.

Requires an upstream vLLM build with the LoRA IPC APIs and device hooks.
Run on one NPU with enough HBM for both the trainer and inference model:

    ASCEND_RT_VISIBLE_DEVICES=0 python examples/rl/rlhf_lora_npu_ipc.py

The worker is a separate process on the same physical NPU. Only complete
LoRA A/B tensors are transferred; the base model remains unchanged.
"""

import argparse
import os
from functools import cached_property
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    ParamMeta,
    WeightSource,
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.lora.request import LoRARequest

from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
    NPUIPCTrainerInitInfo,
    NPUIPCTrainerWeightTransferEngine,
)


class AdapterSource(WeightSource):
    """Export the selected PEFT adapter as complete, canonically named tensors."""

    def __init__(self, model):
        self.model = model

    @cached_property
    def metadata(self):
        return [ParamMeta(name, tensor.dtype, tuple(tensor.shape)) for name, tensor in self.iter_full()]

    def iter_full(self):
        yield from sorted(get_peft_model_state_dict(self.model).items())


class OfflineClient:
    """Adapt the trainer control plane to an idle offline LLM."""

    def __init__(self, llm):
        self.llm = llm

    def init_weight_transfer_engine(self, init_info):
        self.llm.init_weight_transfer_engine(WeightTransferInitRequest(init_info))

    def start_weight_update(self):
        self.llm.start_weight_update()

    def update_weights(self, update_info):
        self.llm.update_weights(WeightTransferUpdateRequest(update_info))

    def finish_weight_update(self, weight_version=None):
        self.llm.finish_weight_update(weight_version)

    def start_lora_weight_update(self, request):
        self.llm.start_lora_weight_update(request)

    def update_lora_weights(self, update_info):
        self.llm.update_lora_weights(WeightTransferUpdateRequest(update_info))

    def finish_lora_weight_update(self, update_id=None):
        self.llm.finish_lora_weight_update(update_id)

    def abort_lora_weight_update(self, update_id=None):
        self.llm.abort_lora_weight_update(update_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--local-files-only", action="store_true", help="Require --model to name a local model directory"
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    args = parser.parse_args()
    model_path = Path(args.model).expanduser()
    local_model = model_path.is_dir()
    if args.local_files_only and not local_model:
        raise ValueError(f"Local model directory does not exist: {args.model}")
    model_name = str(model_path.resolve()) if local_model else args.model
    local_files_only = args.local_files_only or local_model
    if not hasattr(LLM, "start_lora_weight_update") or not hasattr(
        NPUIPCTrainerWeightTransferEngine, "send_lora_weights"
    ):
        raise RuntimeError(
            "NPU LoRA IPC requires an upstream vLLM build containing the LoRA update APIs and IPC device hooks. "
            "Install the upstream LoRA IPC development branch before running this example."
        )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        dtype="bfloat16",
        enable_lora=True,
        max_loras=1,
        max_cpu_loras=1,
        max_lora_rank=8,
        max_model_len=256,
        enforce_eager=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        weight_transfer_config=WeightTransferConfig(backend="npu_ipc"),
        additional_config={"rl_config": {"enabled": True}},
    )
    config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    trainer = get_peft_model(
        AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, local_files_only=local_files_only).to(
            device
        ),
        config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    optimizer = torch.optim.AdamW((p for p in trainer.parameters() if p.requires_grad), lr=1e-3)
    source = AdapterSource(trainer)
    transfer = NPUIPCTrainerWeightTransferEngine.trainer_init(
        NPUIPCTrainerInitInfo(rank=0, packed=False), client=OfflineClient(llm), source=source
    )
    peft_config = config.to_dict()
    peft_config["target_modules"] = sorted(peft_config["target_modules"])
    policy = LoRARequest("policy", 1, "ipc://policy")
    prompt = "The capital of France is"

    for version in (1, 2):
        if version == 2:
            trainer.train()
            batch = tokenizer(prompt + " Paris.", return_tensors="pt").to(device)
            optimizer.zero_grad()
            loss = trainer(**batch, labels=batch["input_ids"]).loss
            loss.backward()
            optimizer.step()
            print(f"Training loss: {loss.item():.4f}")
        # The source remains unchanged while the sender freezes this revision.
        torch.npu.synchronize()
        transfer.send_lora_weights(
            {
                "update_id": f"policy-{version}",
                "lora_name": "policy",
                "lora_int_id": 1,
                "version": version,
                "expected_version": version - 1 if version > 1 else None,
                "peft_config": peft_config,
                "names": [meta.name for meta in source.metadata],
                "dtype_names": [str(meta.dtype).removeprefix("torch.") for meta in source.metadata],
                "shapes": [list(meta.shape) for meta in source.metadata],
            }
        )
        result = llm.generate(prompt, SamplingParams(temperature=0, max_tokens=16), lora_request=policy)[0]
        print(f"Policy version {version}: {result.outputs[0].text!r}")


if __name__ == "__main__":
    main()
