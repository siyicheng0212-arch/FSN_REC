"""Measure deployed CUDA inference resources from a trusted CVM best checkpoint.

This profiles synthetic normalized RGB tensors at the checkpoint's declared
shape. It reads no medical videos or labels and computes no recognition score.
Decoder/cache loading, preprocessing, host transfers and training are outside
the timed region. An explicit new output path is required; no result is replaced.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping

import torch
from torch import nn

from .models import ClinicalHierarchyModel, build_model, prediction_probabilities
from .train import ROOT, model_arguments, selected_frame_count, sha256_file, source_digest


class DeploymentWrapper(nn.Module):
    """Keep only modules required by the selected learned inference path.

    Flat/capacity/aux-flat deployment needs the trained flat head alone. Soft
    hierarchy needs group and non-singleton fine heads. Hard hierarchy uses
    the same vectorized fine heads for every sample; this does not claim the
    lower cost of dynamically executing only the selected conditional head.
    No labels, clip order, source metadata or oracle route enter this wrapper.
    """

    def __init__(self, model: ClinicalHierarchyModel, decoder: str | None = None):
        super().__init__()
        self.mode = model.mode
        self.taxonomy = model.taxonomy
        self.trained_heads = model.trained_heads
        self.backbone = model.backbone
        if self.mode == "hierarchy":
            self.decoder = decoder or "soft"
            if self.decoder not in ("soft", "hard") or not {"group", "conditional"}.issubset(self.trained_heads):
                raise ValueError("hierarchy deployment requires trained group/fine heads and hard/soft decoding")
            self.group_head = model.group_head
            self.conditional_heads = model.conditional_heads
        else:
            self.decoder = decoder or "flat"
            if self.decoder != "flat" or "flat" not in self.trained_heads:
                raise ValueError("flat/capacity/aux-flat deployment requires its trained flat head")
            self.flat_head = model.flat_head

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        features = self.backbone(video)
        if self.mode != "hierarchy":
            return self.flat_head(features).float().softmax(dim=1)
        conditional = [head(features) if len(group) > 1 else features.new_zeros(features.shape[0], 1)
                       for group, head in zip(self.taxonomy.groups, self.conditional_heads)]
        outputs = {"group_logits": self.group_head(features), "conditional_logits": conditional,
                   "trained_heads": self.trained_heads}
        return prediction_probabilities(outputs, "hierarchy", self.taxonomy, decoding=self.decoder)


def parameter_counts(model: nn.Module, deployment: nn.Module) -> dict[str, int]:
    """Machine counts, with training-state flags restored before calling."""
    return {
        "stored_model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "deployment_active_parameters": sum(parameter.numel() for parameter in deployment.parameters()),
        "training_trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "deployment_buffer_elements": sum(buffer.numel() for buffer in deployment.buffers()),
        "deployment_parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in deployment.parameters()),
        "deployment_buffer_bytes": sum(buffer.numel() * buffer.element_size() for buffer in deployment.buffers()),
    }


def declared_input_shape(config: Mapping[str, Any], batch_size: int) -> tuple[int, ...]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    preprocessing = config.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise ValueError("checkpoint is missing its declared preprocessing")
    frames = selected_frame_count(config["backbone"], preprocessing["frames"])
    crop = preprocessing["crop_size"]
    if not isinstance(crop, (list, tuple)) or len(crop) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in crop):
        raise ValueError("checkpoint crop_size must contain two positive integers")
    return batch_size, 3, frames, crop[0], crop[1]


def synthetic_normalized_input(config: Mapping[str, Any], batch_size: int, device: torch.device,
                               seed: int = 42) -> torch.Tensor:
    """Deterministic uniform RGB -> declared normalization, prepared outside timing."""
    shape = declared_input_shape(config, batch_size)
    preprocessing = config["preprocessing"]
    mean, std = preprocessing["mean"], preprocessing["std"]
    if len(mean) != 3 or len(std) != 3 or not all(math.isfinite(value) for value in (*mean, *std)) or any(value <= 0 for value in std):
        raise ValueError("normalization requires three finite means and positive standard deviations")
    generator = torch.Generator(device=device).manual_seed(seed)
    rgb = torch.rand(shape, generator=generator, device=device, dtype=torch.float32)
    mean_tensor = rgb.new_tensor(mean).view(1, 3, 1, 1, 1)
    std_tensor = rgb.new_tensor(std).view(1, 3, 1, 1, 1)
    return ((rgb - mean_tensor) / std_tensor).contiguous()


def _autocast(device: torch.device, amp: str):
    if amp == "none":
        return nullcontext()
    if amp not in ("bfloat16", "float16"):
        raise ValueError("AMP must be none, bfloat16 or float16")
    dtype = torch.bfloat16 if amp == "bfloat16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _quantile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = fraction * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def profile_cuda_batch(deployment: nn.Module, inputs: torch.Tensor, *, amp: str = "none",
                       warmup: int = 20, iterations: int = 100) -> dict[str, Any]:
    if inputs.device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("resource timing requires actual CUDA; CPU is not a timing fallback")
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations positive")
    device = inputs.device
    deployment.eval()
    with torch.cuda.device(device), torch.inference_mode():
        with _autocast(device, amp):
            checked = deployment(inputs)
        if checked.shape != (inputs.shape[0], 7) or not torch.isfinite(checked).all().item():
            raise RuntimeError("deployed path did not return finite seven-class probabilities")
        del checked
        for _ in range(warmup):
            with _autocast(device, amp):
                deployment(inputs)
        torch.cuda.synchronize(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        # CUDA events measure the actual device workload, not a CPU stopwatch.
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        wall_started = time.perf_counter()
        for start, end in zip(starts, ends):
            start.record()
            with _autocast(device, amp):
                output = deployment(inputs)
            end.record()
            del output
        ends[-1].synchronize()
        host_wall_seconds = time.perf_counter() - wall_started
        latencies = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
        span_ms = float(starts[0].elapsed_time(ends[-1]))
        if not all(math.isfinite(value) and value > 0 for value in latencies) or span_ms <= 0:
            raise RuntimeError("invalid CUDA event timings")
        return {
            "batch_size": inputs.shape[0], "input_shape": list(inputs.shape),
            "input_dtype": str(inputs.dtype), "amp": amp, "warmup_iterations": warmup,
            "measured_iterations": iterations, "latency_ms": {
                "mean": statistics.mean(latencies), "median": statistics.median(latencies),
                "p90": _quantile(latencies, .9), "p95": _quantile(latencies, .95),
                "min": min(latencies), "max": max(latencies), "samples": latencies},
            "cuda_measured_span_ms": span_ms,
            "throughput_clips_per_second": inputs.shape[0] * iterations * 1000. / span_ms,
            "host_launch_and_synchronize_seconds": host_wall_seconds,
            "host_observed_throughput_clips_per_second": inputs.shape[0] * iterations / host_wall_seconds,
            "memory_bytes": {
                "baseline_allocated_model_and_input": baseline_allocated,
                "baseline_reserved_after_warmup": baseline_reserved,
                "peak_allocated": torch.cuda.max_memory_allocated(device),
                "peak_reserved": torch.cuda.max_memory_reserved(device),
                "incremental_peak_allocated_over_baseline": torch.cuda.max_memory_allocated(device) - baseline_allocated},
        }


def _git_state() -> dict[str, Any]:
    def run(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": run("rev-parse", "HEAD"), "working_tree_status": run("status", "--porcelain")}


def load_deployment_checkpoint(path: Path, decoder: str | None = None):
    """Accept only our trusted selected finetune checkpoint; strict source/state."""
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint_sha256 = sha256_file(path)
    # Our own checkpoints include optimizer/RNG metadata; never use untrusted
    # externally supplied pickle files here. Pretrained files use strict adapters.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if sha256_file(path) != checkpoint_sha256:
        raise RuntimeError("checkpoint changed while loading")
    if not isinstance(checkpoint, Mapping) or "config" not in checkpoint or "model" not in checkpoint:
        raise ValueError("profile requires a trusted CVM selected best checkpoint")
    config = dict(checkpoint["config"])
    if config.get("code_hash") != source_digest():
        raise RuntimeError("runnable source differs from checkpoint training; freeze the same implementation before profiling")
    expected_stage = "frozen_features" if config.get("backbone_training", "finetune") == "frozen" else "finetune"
    if (checkpoint.get("stage") != expected_stage or checkpoint.get("interrupted") or
            checkpoint.get("epoch") != checkpoint.get("best_epoch") or
            not isinstance(checkpoint.get("selected_val_macro_f1"), (int, float)) or
            not math.isfinite(checkpoint["selected_val_macro_f1"])):
        raise ValueError("checkpoint is not a complete validation-selected post-warmup best checkpoint")
    model = build_model(**model_arguments(config, pretrained=False))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.configure_trainable(backbone_trainable=config.get("backbone_training", "finetune") != "frozen")
    if model.load_report.get("sha256") and model.load_report["sha256"] != config.get("checkpoint_sha256"):
        raise RuntimeError("external pretrained checkpoint changed since training")
    selected_decoder = decoder or config["main_decoder"]
    deployment = DeploymentWrapper(model, selected_decoder)
    counts = parameter_counts(model, deployment)
    if config.get("parameters_trainable_finetune") != counts["training_trainable_parameters"]:
        raise RuntimeError("actual training trainable parameter count differs from saved configuration")
    metadata = {"checkpoint_sha256": checkpoint_sha256, "checkpoint_epoch": checkpoint["epoch"],
                "training_git_commit": config.get("git_commit"), "training_code_hash": config["code_hash"],
                "protocol_sha256": config.get("protocol_sha256"),
                "backbone": config["backbone"], "mode": config["mode"],
                "trained_heads": list(model.trained_heads), "decoder": selected_decoder,
                "preprocessing": config["preprocessing"], "parameters": counts}
    return deployment, config, metadata


def run_profile(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"profiling output already exists: {args.output}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("actual resource profiling requires CUDA; no CPU timing or estimated result is substituted")
    deployment, config, metadata = load_deployment_checkpoint(args.checkpoint, args.decoder)
    amp = args.amp or config.get("amp", "none")
    if amp not in ("none", "float16", "bfloat16"):
        raise ValueError("invalid precision")
    with torch.cuda.device(device):
        if amp == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("requested bf16 is unsupported; precision is never silently changed")
        deployment.to(device).eval()  # frozen audit/auxiliary heads remain off-device
        report = {
            "schema": "fsn-cvm-cuda-resources-v1", "status": "running", **metadata,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "cuda": torch.version.cuda, "device": str(device),
                        "gpu_name": torch.cuda.get_device_name(device),
                        "gpu_capability": list(torch.cuda.get_device_capability(device)),
                        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
                        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                        "cudnn_deterministic": torch.backends.cudnn.deterministic,
                        **_git_state()},
            "workload": "synthetic uniform RGB normalized to the checkpoint's declared input shape",
            "classification_performance_evaluated": False,
            "timed_region": "resident GPU tensor -> backbone -> active learned heads -> seven-class probabilities",
            "excluded": ["video/cache loading", "temporal sampling", "resize/crop/normalization",
                         "host-to-device transfers", "training", "clinical accuracy measurement"],
            "hard_hierarchy_execution": "all conditional heads evaluated vectorially; no dynamic routed-head compute saving",
            "flops": {"status": "not_measured", "value": None,
                      "reason": "No validated operator-complete FLOPs estimator is invoked."},
            "batches": [],
        }
        failure = None
        try:
            for batch_size in args.batch_sizes:
                torch.cuda.empty_cache()
                inputs = synthetic_normalized_input(config, batch_size, device, config.get("seed", 42))
                report["batches"].append(profile_cuda_batch(deployment, inputs, amp=amp,
                                                            warmup=args.warmup, iterations=args.iterations))
                del inputs
            report["status"] = "completed"
        except Exception as exc:
            report["status"] = "failed"
            report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            failure = exc
        if source_digest() != config["code_hash"] or sha256_file(args.checkpoint) != metadata["checkpoint_sha256"]:
            report["status"] = "invalidated"
            report["failure"] = {"type": "ChangedSourceOrCheckpoint", "message": "Source/checkpoint changed during profiling"}
            failure = RuntimeError(report["failure"]["message"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        if failure is not None:
            raise failure


def batch_sizes(value: str) -> list[int]:
    try:
        values = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated positive integers") from exc
    if not values or any(value < 1 for value in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("batch sizes must be positive and distinct")
    return values


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--batch-sizes", type=batch_sizes, default=[1, 4])
    command.add_argument("--amp", choices=("none", "bfloat16", "float16"))
    command.add_argument("--decoder", choices=("flat", "hard", "soft"))
    command.add_argument("--warmup", type=int, default=20)
    command.add_argument("--iterations", type=int, default=100)
    return command


def main() -> None:
    command = parser()
    args = command.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        command.error("warmup must be nonnegative and iterations positive")
    run_profile(args)


if __name__ == "__main__":
    main()
