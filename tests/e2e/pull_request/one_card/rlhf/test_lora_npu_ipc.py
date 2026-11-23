# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Real NPU adapter transactions compared with the same checkpoint weights.

The standalone entry point runs the same assertions without importing pytest's
model-level conftest. To use a prepared local model and retain the evidence:

    python tests/e2e/pull_request/one_card/rlhf/test_lora_npu_ipc.py \\
        --model /home/z00850655/serving/lora-npu-ipc-20261123/models/Qwen3-0.6B \\
        --local-files-only --work-dir /tmp/lora-npu-ipc-e2e
"""

import argparse
import json
import os
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401
from safetensors.torch import save_file
from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.weight_transfer.base import (
    ParamMeta,
    WeightSource,
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.inputs import TokensPrompt
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
from vllm.lora.request import LoRARequest

from vllm_ascend.ascend_config import clear_ascend_config
from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
    NPUIPCTrainerInitInfo,
    NPUIPCTrainerWeightTransferEngine,
)

MODEL_NAME = "Qwen/Qwen3-0.6B"
RANK = 8
PROMPT = "A training system samples responses, evaluates rewards, and updates its policy. " * 4
TRACE_TOKENS = [444, 2210, 13]
GREEDY_MAX_TOKENS = 8
DEFAULT_ADAPTER_MODULES = ("q_proj", "v_proj")
SHORT_MAX_MODEL_LEN = 256
PREFIX_CACHE_MAX_MODEL_LEN = 4608
PREFIX_CACHE_TAIL_TOKENS = 64


class AdapterSource(WeightSource):
    def __init__(self, tensors):
        self.tensors = tensors

    @cached_property
    def metadata(self):
        return [ParamMeta(name, tensor.dtype, tuple(tensor.shape)) for name, tensor in self.tensors.items()]

    def iter_full(self):
        yield from self.tensors.items()


class OfflineClient:
    def __init__(self, llm):
        self.llm = llm

    def init_weight_transfer_engine(self, init_info):
        self.llm.init_weight_transfer_engine(WeightTransferInitRequest(init_info))

    def start_lora_weight_update(self, request):
        self.llm.start_lora_weight_update(request)

    def update_lora_weights(self, update_info):
        self.llm.update_lora_weights(WeightTransferUpdateRequest(update_info))

    def finish_lora_weight_update(self, update_id=None):
        self.llm.finish_lora_weight_update(update_id)

    def abort_lora_weight_update(self, update_id=None):
        self.llm.abort_lora_weight_update(update_id)


def _registered_adapter_modules(worker, adapter_modules, rank):
    """Return geometry from real worker LoRA wrappers, without exporting weights."""
    assert worker.vllm_config.parallel_config.tensor_parallel_size == 1
    manager = worker.model_runner.lora_manager._adapter_manager
    metadata = []
    matched = set()
    for wrapper_name, wrapper in sorted(manager.modules.items()):
        module_names = manager.packed_modules.get(wrapper_name, [wrapper_name])
        for index, module_name in enumerate(module_names):
            aliases = [module_name, *manager._get_text_lora_names(module_name)]
            targets = [
                target
                for target in adapter_modules
                if target in aliases or ("." not in target and module_name.rsplit(".", 1)[-1] == target)
            ]
            if not targets:
                continue
            assert len(targets) == 1, f"Overlapping adapter targets for {module_name}: {targets}"
            target = targets[0]
            assert isinstance(wrapper, BaseLinearLayerWithLoRA), f"Not a dense LoRA linear: {wrapper_name}"
            assert len(module_names) == len(wrapper.lora_a_stacked) == len(wrapper.lora_b_stacked)
            slot_a = wrapper.lora_a_stacked[index]
            slot_b = wrapper.lora_b_stacked[index]
            assert slot_a.shape[-2] >= rank and slot_b.shape[-1] >= rank
            parent = manager.model.get_submodule(wrapper_name.rpartition(".")[0])
            if "Glm5Next" in type(manager.model).__name__:
                assert len(adapter_modules) == 1 and "." in target, "Select one exact GLM KDA projection module"
                assert ".layers.0." in module_name and module_name.endswith(".o_proj"), (
                    f"GLM acceptance targets the layer-0 KDA output projection: {module_name}"
                )
                assert type(parent).__name__ == "Glm5NextLinearAttention", (
                    f"GLM acceptance targets KDA, got {type(parent).__name__}: {wrapper_name}"
                )
            peft_name = target if "." in target else module_name
            metadata.append(
                {
                    "module_name": module_name,
                    "wrapper_name": wrapper_name,
                    "wrapper_type": type(wrapper).__name__,
                    "parent_type": type(parent).__name__,
                    "peft_name": peft_name,
                    "a_shape": [rank, slot_a.shape[-1]],
                    "b_shape": [slot_b.shape[-2], rank],
                }
            )
            matched.add(target)
    assert matched == set(adapter_modules), (
        f"Adapter targets were not registered: {sorted(set(adapter_modules) - matched)}; "
        f"registered wrappers: {sorted(manager.modules)}"
    )
    assert metadata, "No registered dense LoRA modules selected"
    return metadata


def _prefix_cache_metadata(worker):
    """Read the cache matching granularity resolved by the real engine core."""
    config = worker.model_runner.kv_cache_config
    assert config.cache_hit_alignment_tokens is not None and config.cache_hit_alignment_tokens > 0
    assert config.hash_block_size is not None and config.hash_block_size > 0
    return {
        "cache_hit_alignment_tokens": config.cache_hit_alignment_tokens,
        "hash_block_size": config.hash_block_size,
        "group_block_sizes": [group.kv_cache_spec.block_size for group in config.kv_cache_groups],
    }


@contextmanager
def _spawned_llm(**kwargs):
    """Use the real LLM and the same cleanup contract as VllmRunner."""
    # The isolated test sends its trusted inspection functions to the worker.
    with patch.dict(
        os.environ,
        {
            "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        },
    ):
        llm = LLM(**kwargs)
        try:
            yield llm
        finally:
            try:
                llm.llm_engine.engine_core.shutdown()
            finally:
                clear_ascend_config()
                cleanup_dist_env_and_memory()


@pytest.mark.e2e_model(MODEL_NAME)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="lora,runtime_lora,weight_transfer,prefix_caching,logprobs",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="eager",
)
def test_npu_lora_ipc_matches_checkpoint_and_preserves_previous_revision(tmp_path):
    """Two IPC revisions match disk adapters; failed prepare keeps the old policy."""
    run_lora_ipc_e2e(MODEL_NAME, tmp_path)


def run_lora_ipc_e2e(
    model_name: str,
    work_dir: Path,
    *,
    local_files_only: bool = False,
    gpu_memory_utilization: float = 0.35,
    device_index: int = 0,
    adapter_modules: tuple[str, ...] | list[str] = DEFAULT_ADAPTER_MODULES,
    additional_config: dict | None = None,
    attention_config: dict | None = None,
    text_only: bool = False,
    verify_prefix_cache: bool = False,
):
    """Run the complete, real LLM assertions for pytest or a standalone caller."""
    assert hasattr(LLM, "start_lora_weight_update"), "Install the upstream LoRA IPC API build for this test"
    if device_index != 0:
        raise ValueError(
            "Single-worker IPC uses logical NPU 0; select the physical chip with ASCEND_RT_VISIBLE_DEVICES"
        )
    model_path = Path(model_name).expanduser()
    local_model = model_path.is_dir()
    if local_files_only and not local_model:
        raise ValueError(f"Local model directory does not exist: {model_name}")
    if local_model:
        model_name = str(model_path.resolve())
    adapter_modules = tuple(part.strip() for target in adapter_modules for part in target.split(","))
    if (
        not adapter_modules
        or any(not target for target in adapter_modules)
        or len(set(adapter_modules)) != len(adapter_modules)
    ):
        raise ValueError("Adapter modules must be nonempty, unique names")
    target_suffixes = sorted({target.rsplit(".", 1)[-1] for target in adapter_modules})
    if additional_config is not None and not isinstance(additional_config, dict):
        raise ValueError("Additional config must be a JSON object")
    additional_config = dict(additional_config or {})
    rl_config = additional_config.get("rl_config", {})
    if not isinstance(rl_config, dict):
        raise ValueError("Additional config rl_config must be a JSON object")
    additional_config["rl_config"] = {**rl_config, "enabled": True}
    if attention_config is not None and not isinstance(attention_config, dict):
        raise ValueError("Attention config must be a JSON object")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "result.json").write_text(
        json.dumps({"status": "running", "model": model_name}, indent=2), encoding="utf-8"
    )
    device = torch.device("npu", device_index)
    torch.npu.set_device(device)
    peft_config = {
        "r": RANK,
        "lora_alpha": 16,
        "target_modules": target_suffixes,
        "bias": "none",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
    }
    policy = LoRARequest("policy", 1, "ipc://policy")
    params = SamplingParams(
        temperature=0, max_tokens=len(TRACE_TOKENS), trace_decode_token_ids=TRACE_TOKENS, logprobs=1
    )
    greedy_params = SamplingParams(temperature=0, max_tokens=GREEDY_MAX_TOKENS)
    llm_options = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if text_only else {}
    if attention_config is not None:
        llm_options["attention_config"] = attention_config
    max_model_len = PREFIX_CACHE_MAX_MODEL_LEN if verify_prefix_cache else SHORT_MAX_MODEL_LEN

    with _spawned_llm(
        model=model_name,
        tokenizer=model_name,
        dtype="bfloat16",
        enable_lora=True,
        max_loras=3,
        max_cpu_loras=3,
        max_lora_rank=RANK,
        lora_target_modules=target_suffixes,
        max_model_len=max_model_len,
        max_num_seqs=1 if verify_prefix_cache else 4,
        max_num_batched_tokens=256,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=True,
        enable_trace_replay=True,
        weight_transfer_config=WeightTransferConfig(backend="npu_ipc"),
        additional_config=additional_config,
        **llm_options,
    ) as llm:
        metadata_by_worker = llm.collective_rpc(_registered_adapter_modules, args=(adapter_modules, RANK))
        assert len(metadata_by_worker) == 1, "This acceptance test requires a single inference worker"
        adapter_metadata = metadata_by_worker[0]
        (work_dir / "adapter_modules.json").write_text(json.dumps(adapter_metadata, indent=2), encoding="utf-8")
        print(f"Registered adapter modules: {json.dumps(adapter_metadata)}", flush=True)
        cache_metadata = None
        prompt = PROMPT
        if verify_prefix_cache:
            cache_metadata_by_worker = llm.collective_rpc(_prefix_cache_metadata)
            assert len(cache_metadata_by_worker) == 1
            cache_metadata = cache_metadata_by_worker[0]
            prompt_length = cache_metadata["cache_hit_alignment_tokens"] + PREFIX_CACHE_TAIL_TOKENS
            assert prompt_length + len(TRACE_TOKENS) <= max_model_len, (
                f"Resolved cache alignment does not fit the acceptance context: {cache_metadata}"
            )
            seed_tokens = llm.get_tokenizer().encode(PROMPT, add_special_tokens=False)
            assert seed_tokens, "The real model tokenizer produced an empty prompt"
            repeats = (prompt_length + len(seed_tokens) - 1) // len(seed_tokens)
            prompt_tokens = (seed_tokens * repeats)[:prompt_length]
            assert len(prompt_tokens) == prompt_length
            prompt = TokensPrompt(prompt_token_ids=prompt_tokens)
            print(f"Prefix cache verification: {json.dumps(cache_metadata)}, prompt_tokens={prompt_length}", flush=True)
        base_traces = []
        for label, base_prompt in (("short", PROMPT), ("acceptance", prompt)):
            completion = llm.generate(base_prompt, params, use_tqdm=False)[0].outputs[0]
            assert completion.logprobs is not None
            base_scores = [step[token].logprob for token, step in zip(TRACE_TOKENS, completion.logprobs)]
            base_traces.append({"prompt": label, "logprobs": base_scores})
            (work_dir / "base_traces.json").write_text(json.dumps(base_traces, indent=2), encoding="utf-8")
            print(f"Base model trace: {json.dumps(base_traces[-1])}", flush=True)
        assert all(bool(torch.isfinite(torch.tensor(trace["logprobs"])).all()) for trace in base_traces), (
            f"The base model produced nonfinite logprobs before LoRA updates: {base_traces}"
        )
        generator = torch.Generator(device="cpu").manual_seed(2026)
        checkpoint = {}
        for module in adapter_metadata:
            prefix = f"base_model.model.{module['peft_name']}"
            checkpoint[f"{prefix}.lora_A.weight"] = torch.randn(*module["a_shape"], generator=generator) * 0.05
            checkpoint[f"{prefix}.lora_B.weight"] = torch.randn(*module["b_shape"], generator=generator) * 0.1
        checkpoint = dict(sorted(checkpoint.items()))
        checksums = llm.collective_rpc("compute_weight_checksums")
        source = AdapterSource({name: tensor.to(device) for name, tensor in checkpoint.items()})
        transfer = NPUIPCTrainerWeightTransferEngine.trainer_init(
            NPUIPCTrainerInitInfo(rank=0, packed=False), client=OfflineClient(llm), source=source
        )

        cache_observations = []

        def scores(request, label, cache_path):
            request_output = llm.generate(prompt, params, lora_request=request, use_tqdm=False)[0]
            assert request_output.prompt_token_ids is not None
            observation = {
                "label": label,
                "cache_path": cache_path,
                "lora_int_id": request.lora_int_id,
                "prompt_length": len(request_output.prompt_token_ids),
                "num_cached_tokens": request_output.num_cached_tokens,
                "num_cache_creation_tokens": request_output.num_cache_creation_tokens,
            }
            cache_observations.append(observation)
            (work_dir / "cache_observations.json").write_text(
                json.dumps(cache_observations, indent=2), encoding="utf-8"
            )
            print(f"Request cache observation: {json.dumps(observation)}", flush=True)
            if verify_prefix_cache:
                assert observation["prompt_length"] == prompt_length
                assert isinstance(request_output.num_cached_tokens, int), "Missing per-request prefix cache statistics"
                if cache_path == "cold":
                    assert request_output.num_cached_tokens == 0, f"Unexpected stale prefix hit: {observation}"
                else:
                    assert request_output.num_cached_tokens > 0, f"Repeated prompt did not hit the cache: {observation}"
            output = request_output.outputs[0]
            assert list(output.token_ids) == TRACE_TOKENS
            assert output.logprobs is not None
            assert len(output.logprobs) == len(TRACE_TOKENS)
            result = torch.tensor(
                [step[token].logprob for token, step in zip(TRACE_TOKENS, output.logprobs)], device="cpu"
            )
            observation["logprobs"] = result.tolist()
            (work_dir / "cache_observations.json").write_text(
                json.dumps(cache_observations, indent=2), encoding="utf-8"
            )
            assert bool(torch.isfinite(result).all()), f"Nonfinite LoRA inference scores: {observation}"
            return result

        references = []
        greedy_generations = []
        for version in (1, 2):
            if version == 2:
                checkpoint = {
                    name: tensor * (-2 if name.endswith("lora_B.weight") else 1) for name, tensor in checkpoint.items()
                }
                source.tensors = {name: tensor.to(device) for name, tensor in checkpoint.items()}
            request = {
                "update_id": f"policy-{version}",
                "lora_name": "policy",
                "lora_int_id": 1,
                "version": version,
                "expected_version": version - 1 if version > 1 else None,
                "peft_config": peft_config,
                "names": list(checkpoint),
                "dtype_names": ["float32"] * len(checkpoint),
                "shapes": [list(tensor.shape) for tensor in checkpoint.values()],
            }
            if version == 2:
                stale_request = {**request, "update_id": "stale-start", "expected_version": None}
                with pytest.raises(Exception, match="expected LoRA version"):
                    llm.start_lora_weight_update(stale_request)
                # A rejected start cleared caches; repopulate the old policy so
                # the successful version-2 update must invalidate a real hit.
                torch.testing.assert_close(
                    scores(policy, "policy-1-after-rejected-start-cold", "cold"),
                    references[-1]["cold"],
                    rtol=1e-5,
                    atol=1e-5,
                )
                torch.testing.assert_close(
                    scores(policy, "policy-1-after-rejected-start-warm", "warm"),
                    references[-1]["warm"],
                    rtol=1e-5,
                    atol=1e-5,
                )
            transfer.send_lora_weights(request)
            for name, tensor in source.tensors.items():
                torch.testing.assert_close(tensor.cpu(), checkpoint[name], rtol=0, atol=0)
            path = work_dir / f"adapter-{version}"
            path.mkdir(exist_ok=True)
            (path / "adapter_config.json").write_text(json.dumps(peft_config), encoding="utf-8")
            save_file(checkpoint, path / "adapter_model.safetensors")
            reference_request = LoRARequest(f"reference-{version}", version + 1, str(path))
            cold_reference = scores(reference_request, f"reference-{version}-cold", "cold")
            warm_reference = scores(reference_request, f"reference-{version}-warm", "warm")
            references.append({"cold": cold_reference, "warm": warm_reference})
            torch.testing.assert_close(
                scores(policy, f"policy-{version}-cold", "cold"), cold_reference, rtol=1e-5, atol=1e-5
            )
            # Compare repeated prompts on the same cached prefix path for both identities.
            torch.testing.assert_close(
                scores(policy, f"policy-{version}-warm", "warm"), warm_reference, rtol=1e-5, atol=1e-5
            )
            print(
                f"LoRA revision {version} matched its checkpoint: "
                f"cold_logprobs={cold_reference.tolist()}, warm_logprobs={warm_reference.tolist()}",
                flush=True,
            )
            greedy_comparison = {"version": version}
            for label, adapter in (("reference", reference_request), ("policy", policy)):
                greedy_result = llm.generate(prompt, greedy_params, lora_request=adapter, use_tqdm=False)[0]
                completion = greedy_result.outputs[0]
                greedy_comparison[label] = {
                    "token_ids": list(completion.token_ids),
                    "text": completion.text,
                    "num_cached_tokens": greedy_result.num_cached_tokens,
                }
            greedy_generations.append(greedy_comparison)
            (work_dir / "greedy_generations.json").write_text(
                json.dumps(greedy_generations, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            assert greedy_comparison["reference"]["token_ids"], "Greedy reference generated no tokens"
            assert greedy_comparison["policy"]["token_ids"] == greedy_comparison["reference"]["token_ids"], (
                f"Greedy token IDs differ from the checkpoint: {greedy_comparison}"
            )
            print(f"LoRA greedy generation: {json.dumps(greedy_comparison, ensure_ascii=False)}", flush=True)

        assert not torch.allclose(references[0]["cold"], references[1]["cold"], rtol=1e-5, atol=1e-5)
        request.update(update_id="policy-3", version=3, expected_version=2)
        llm.start_lora_weight_update(request)
        with pytest.raises(Exception, match="update_id"):
            llm.abort_lora_weight_update("stale-update")
        with pytest.raises(Exception, match="Incomplete LoRA update"):
            llm.finish_lora_weight_update("policy-3")
        llm.abort_lora_weight_update("policy-3")
        # Starting the incomplete transaction already cleared the old prefix cache.
        torch.testing.assert_close(
            scores(policy, "policy-2-after-abort-cold", "cold"), references[-1]["cold"], rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            scores(policy, "policy-2-after-abort-warm", "warm"), references[-1]["warm"], rtol=1e-5, atol=1e-5
        )
        assert llm.collective_rpc("compute_weight_checksums") == checksums
    report = {
        "status": "passed",
        "model": model_name,
        "adapter_modules": adapter_metadata,
        "additional_config": additional_config,
        "attention_config": attention_config,
        "text_only": text_only,
        "revisions": 2,
        "trace_token_ids": TRACE_TOKENS,
        "base_traces": base_traces,
        "greedy_max_tokens": GREEDY_MAX_TOKENS,
        "greedy_generations": greedy_generations,
        "reference_logprobs": [
            {path: scores.tolist() for path, scores in reference.items()} for reference in references
        ],
        "source_weights_unchanged": True,
        "base_weights_unchanged": True,
        "prefix_cache_enabled": True,
        "verify_prefix_cache": verify_prefix_cache,
        "cache_metadata": cache_metadata,
        "cache_observations": cache_observations,
        "stale_abort_rejected": True,
        "rejected_start_recovered": True,
        "failed_prepare_preserved_revision": 2,
    }
    (work_dir / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME, help="Model repository or an existing local model directory")
    parser.add_argument("--local-files-only", action="store_true", help="Require --model to name a local directory")
    parser.add_argument("--work-dir", type=Path, help="Keep generated adapter checkpoints and result.json here")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument(
        "--text-only", action="store_true", help="Set existing image/video limits to zero for text-only inference"
    )
    parser.add_argument(
        "--verify-prefix-cache",
        action="store_true",
        help="Use a longer token-ID prompt and require cold misses and warm prefix-cache hits",
    )
    parser.add_argument(
        "--additional-config",
        type=json.loads,
        help="Existing vLLM Ascend additional config as JSON; rl_config.enabled is always true",
    )
    parser.add_argument("--attention-config", type=json.loads, help="Existing upstream attention config as JSON")
    parser.add_argument(
        "--adapter-modules",
        nargs="+",
        default=DEFAULT_ADAPTER_MODULES,
        help="LoRA module suffixes or exact text-model paths; GLM requires one exact layer-0 KDA o_proj path",
    )
    parser.add_argument(
        "--device-index", type=int, default=0, help="Logical NPU index; the single-worker test requires 0"
    )
    args = parser.parse_args()

    def run(directory):
        return run_lora_ipc_e2e(
            args.model,
            Path(directory),
            local_files_only=args.local_files_only,
            gpu_memory_utilization=args.gpu_memory_utilization,
            device_index=args.device_index,
            adapter_modules=args.adapter_modules,
            additional_config=args.additional_config,
            attention_config=args.attention_config,
            text_only=args.text_only,
            verify_prefix_cache=args.verify_prefix_cache,
        )

    if args.work_dir is not None:
        report = run(args.work_dir)
    else:
        with TemporaryDirectory(prefix="lora-npu-ipc-e2e-") as directory:
            report = run(directory)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
