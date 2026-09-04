#!/usr/bin/env python3
"""Stage 18: causal utility and real compute cost of GLOBAL HAA and LOCAL.

The existing ``NO_RETRIEVAL`` ablation intentionally computes retrieval and
then zeros it.  That is the right causal intervention for quality, but the
wrong implementation for a cost benchmark.  This script therefore evaluates
four paths with the same frozen checkpoint and an explicit computational
bypass:

* full: GLOBAL read/write plus LOCAL;
* global_off: LOCAL only, with every GLOBAL projection/read/write skipped;
* local_off: GLOBAL only, with every LOCAL projection skipped;
* associative_off: neither associative path.

It also verifies that the bypass is numerically equivalent to the public
causal interventions before using it for timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from celllm.ablation import AblationCondition, AblationConfig, AblationTrace
from celllm.cell import piecewise_linear
from celllm600m.config import DemoConfig, build_model, parameter_report
from celllm600m.sources import SOURCES


DDP_PREFIX = "module.core."
POSITION_BINS = (
    ("0_63", 0, 64),
    ("64_127", 64, 128),
    ("128_255", 128, 256),
    ("256_511", 256, 512),
    ("512_1023", 512, 1024),
    ("1024_2047", 1024, 2048),
    ("2048_4095", 2048, 4096),
    ("4096_8190", 4096, 8191),
)


@dataclass(frozen=True)
class ComputePath:
    name: str
    global_enabled: bool
    local_enabled: bool


PATHS = (
    ComputePath("full", True, True),
    ComputePath("global_off", False, True),
    ComputePath("local_off", True, False),
    ComputePath("associative_off", False, False),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def raw_state(payload: dict) -> dict:
    state = payload["model"]
    return {
        key.removeprefix(DDP_PREFIX): value
        for key, value in state.items()
    }


def forward_block(
    model,
    tokens: torch.Tensor,
    memory,
    path: ComputePath,
    *,
    update_memory: bool = True,
):
    """Execute one native block, physically skipping disabled mechanisms."""
    core = model.core
    cell = core.cell
    if path.global_enabled and memory is None:
        memory = model.new_plastic_state(tokens.shape[0])
    embedding = core.embed(tokens)
    cell_input = cell.control_input(embedding)
    state = torch.zeros_like(embedding)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    for _ in range(core.cfg.k):
        output = piecewise_linear(state)
        global_drive = (
            cell.attention.retrieve(memory, output)
            if path.global_enabled
            else torch.zeros_like(output)
        )
        local_drive = (
            cell.local_attention(output, global_drive, mask=mask)
            if path.local_enabled
            else torch.zeros_like(output)
        )
        state = cell.dynamics.step(
            state,
            cell_input,
            extra_drive=cell.mixer(state) + global_drive + local_drive,
        )
    if path.global_enabled and update_memory:
        memory = cell.observe(memory, state, mask=mask)
    return core.readout(state), memory


@torch.inference_mode()
def logits_public_ablation(model, tokens, condition: AblationCondition):
    memory = None
    outputs = []
    ablation = AblationConfig(condition)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for chunk in tokens.split(64, dim=1):
            logits, memory = model.forward_with_state(
                chunk,
                memory,
                write_mask=torch.ones_like(chunk, dtype=torch.bool),
                ablation=ablation,
            )
            outputs.append(logits)
    return torch.cat(outputs, dim=1)


@torch.inference_mode()
def logits_compute_path(model, tokens, path: ComputePath):
    memory = None
    outputs = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for chunk in tokens.split(64, dim=1):
            logits, memory = forward_block(model, chunk, memory, path)
            outputs.append(logits)
    return torch.cat(outputs, dim=1)


def verify_bypasses(model, tokens) -> dict:
    full_public = logits_public_ablation(model, tokens, AblationCondition.NORMAL)
    full_bypass = logits_compute_path(model, tokens, PATHS[0])
    global_off = logits_compute_path(model, tokens, PATHS[1])
    result = {
        "full_public_vs_explicit_max_abs": float(
            (full_public - full_bypass).abs().max()
        )
    }
    for condition in (
        AblationCondition.NO_RETRIEVAL,
        AblationCondition.NO_WRITE,
        AblationCondition.NO_CARRY,
    ):
        logits = logits_public_ablation(model, tokens, condition)
        result[f"global_off_vs_{condition.value}_max_abs"] = float(
            (global_off - logits).abs().max()
        )
    return result


@torch.inference_mode()
def evaluate_window(model, tokens: torch.Tensor, path: ComputePath) -> dict:
    memory = None
    loss_parts = []
    correct = 0
    count = 0
    bins = {name: [0.0, 0, 0] for name, _, _ in POSITION_BINS}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, tokens.shape[1], 64):
            chunk = tokens[:, start : start + 64]
            logits, memory = forward_block(model, chunk, memory, path)
            predictions = min(chunk.shape[1], tokens.shape[1] - start - 1)
            if predictions < 1:
                continue
            scores = logits[:, :predictions].float()
            targets = tokens[:, start + 1 : start + 1 + predictions]
            losses = F.cross_entropy(
                scores.reshape(-1, scores.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            )
            hits = scores.argmax(dim=-1).eq(targets).reshape(-1)
            positions = torch.arange(
                start, start + predictions, device=tokens.device
            )
            loss_parts.append(losses.cpu())
            correct += int(hits.sum())
            count += predictions
            for name, lower, upper in POSITION_BINS:
                selected = (positions >= lower) & (positions < upper)
                if selected.any():
                    bins[name][0] += float(losses[selected].sum())
                    bins[name][1] += int(selected.sum())
                    bins[name][2] += int(hits[selected].sum())
    losses = torch.cat(loss_parts)
    return {
        "loss_nats": float(losses.mean()),
        "perplexity": math.exp(float(losses.mean())),
        "top1_accuracy": correct / count,
        "tokens": count,
        "position_bins": {
            name: {
                "loss_nats": total / n,
                "top1_accuracy": hits / n,
                "tokens": n,
            }
            for name, (total, n, hits) in bins.items()
            if n
        },
    }


def mean(values) -> float:
    return float(statistics.fmean(values))


def sample_std(values) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def bootstrap_weighted_delta(
    windows: dict[str, list[dict]],
    ablated: str,
    *,
    samples: int = 20_000,
    seed: int = 180018,
) -> dict:
    rng = np.random.default_rng(seed)
    observed = 0.0
    source_deltas = {}
    arrays = {}
    for source, rows in windows.items():
        weight = float(SOURCES[source]["weight"]) / 100.0
        values = np.asarray([
            row["full"]["loss_nats"] - row[ablated]["loss_nats"]
            for row in rows
        ])
        arrays[source] = (weight, values)
        source_deltas[source] = {
            "weight": weight,
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "values": values.tolist(),
        }
        observed += weight * float(values.mean())
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        total = 0.0
        for weight, values in arrays.values():
            total += weight * float(rng.choice(values, size=len(values)).mean())
        draws[index] = total
    return {
        "definition": "loss_full_minus_loss_ablated; positive favors ablation",
        "weighted_delta": observed,
        "bootstrap_95_ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_samples": samples,
        "sources": source_deltas,
    }


def summarize_quality(windows: dict[str, list[dict]]) -> dict:
    summary = {"paths": {}, "paired_deltas": {}}
    for path in PATHS:
        sources = {}
        weighted_loss = weighted_acc = 0.0
        for source, rows in windows.items():
            weight = float(SOURCES[source]["weight"]) / 100.0
            losses = [row[path.name]["loss_nats"] for row in rows]
            accuracies = [row[path.name]["top1_accuracy"] for row in rows]
            sources[source] = {
                "weight": weight,
                "loss_mean": mean(losses),
                "loss_sample_std": sample_std(losses),
                "top1_mean": mean(accuracies),
            }
            weighted_loss += weight * mean(losses)
            weighted_acc += weight * mean(accuracies)
        summary["paths"][path.name] = {
            "weighted_loss_nats": weighted_loss,
            "weighted_perplexity": math.exp(weighted_loss),
            "weighted_top1_accuracy": weighted_acc,
            "sources": sources,
        }
    for name in ("global_off", "local_off", "associative_off"):
        summary["paired_deltas"][name] = bootstrap_weighted_delta(windows, name)

    for path in PATHS:
        by_bin = {}
        for bin_name, _, _ in POSITION_BINS:
            per_source = []
            total_weight = 0.0
            for source, rows in windows.items():
                values = [
                    row[path.name]["position_bins"][bin_name]["loss_nats"]
                    for row in rows
                    if bin_name in row[path.name]["position_bins"]
                ]
                if values:
                    weight = float(SOURCES[source]["weight"]) / 100.0
                    per_source.append(weight * mean(values))
                    total_weight += weight
            if per_source:
                by_bin[bin_name] = sum(per_source) / total_weight
        summary["paths"][path.name]["weighted_loss_by_position"] = by_bin
    return summary


def run_quality(model, packed: Path, windows_per_source: int, tokens: int, output: Path):
    manifest = json.loads((packed / "pack-manifest.json").read_text())
    rng = np.random.default_rng(180017)
    windows: dict[str, list[dict]] = {}
    starts: dict[str, list[int]] = {}
    for source, entry in manifest["sources"].items():
        array = np.load(packed / entry["path"], mmap_mode="r")
        windows[source] = []
        starts[source] = []
        for _ in range(windows_per_source):
            start = int(rng.integers(0, array.size - tokens))
            starts[source].append(start)
            ids = np.asarray(array[start : start + tokens], dtype=np.int64).copy()
            batch = torch.from_numpy(ids).unsqueeze(0).to("cuda")
            row = {path.name: evaluate_window(model, batch, path) for path in PATHS}
            row["start"] = start
            windows[source].append(row)
        partial = {
            "protocol": {
                "windows_per_source": windows_per_source,
                "window_tokens": tokens,
                "seed": 180017,
                "starts": starts,
            },
            "windows": windows,
        }
        output.write_text(json.dumps(partial, indent=2, ensure_ascii=False) + "\n")
    return {
        "protocol": {
            "windows_per_source": windows_per_source,
            "window_tokens": tokens,
            "seed": 180017,
            "starts": starts,
        },
        "windows": windows,
        "summary": summarize_quality(windows),
    }


@torch.inference_mode()
def trace_memory(model, tokens) -> dict:
    trace = AblationTrace()
    memory = None
    per_block = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for index, chunk in enumerate(tokens.split(64, dim=1)):
            before = memory
            logits, memory = model.forward_with_state(
                chunk,
                memory,
                write_mask=torch.ones_like(chunk, dtype=torch.bool),
                trace=trace,
            )
            value_delta = (
                memory.values if before is None else memory.values - before.values
            )
            per_block.append({
                "block": index,
                "value_rms": float(memory.values.float().square().mean().sqrt()),
                "value_delta_rms": float(value_delta.float().square().mean().sqrt()),
                "strength_mean": float(memory.strength.float().mean()),
                "strength_max": float(memory.strength.float().max()),
                "logit_rms": float(logits.float().square().mean().sqrt()),
                "state_dtype": str(memory.values.dtype),
            })
    return {"aggregate": trace.to_dict(), "per_block": per_block}


def cuda_measure(function, *, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
    peak = torch.cuda.max_memory_allocated()
    return {
        "seconds": timings,
        "median_seconds": float(statistics.median(timings)),
        "mean_seconds": mean(timings),
        "sample_std_seconds": sample_std(timings),
        "resident_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "incremental_peak_bytes": max(0, peak - baseline),
    }


def sequence_inference_cost(model, tokens, path, repeats) -> dict:
    @torch.inference_mode()
    def run():
        memory = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for chunk in tokens.split(64, dim=1):
                _, memory = forward_block(model, chunk, memory, path)

    result = cuda_measure(run, warmup=1, repeats=repeats)
    result["tokens"] = tokens.numel()
    result["tokens_per_second_median"] = tokens.numel() / result["median_seconds"]
    return result


def generation_cost(model, prompt, path, generated_tokens, repeats) -> dict:
    @torch.inference_mode()
    def run():
        ids = prompt[0].tolist()
        memory = None
        pending = ids
        with torch.autocast("cuda", dtype=torch.bfloat16):
            while len(pending) > 64:
                chunk = torch.tensor([pending[:64]], device="cuda")
                _, memory = forward_block(model, chunk, memory, path)
                pending = pending[64:]
            for _ in range(generated_tokens):
                chunk = torch.tensor([pending], device="cuda")
                logits, _ = forward_block(
                    model, chunk, memory, path, update_memory=False
                )
                token = int(logits[0, -1].argmax())
                pending.append(token)
                if len(pending) > 64:
                    committed = torch.tensor([pending[:64]], device="cuda")
                    _, memory = forward_block(model, committed, memory, path)
                    pending = pending[64:]

    result = cuda_measure(run, warmup=1, repeats=repeats)
    result["prompt_tokens"] = prompt.numel()
    result["generated_tokens"] = generated_tokens
    result["tokens_per_second_median"] = generated_tokens / result["median_seconds"]
    return result


def training_cost(model, tokens, path, repeats) -> dict:
    model.train()

    def run():
        model.zero_grad(set_to_none=True)
        memory = None
        logits = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for chunk in tokens.split(64, dim=1):
                scores, memory = forward_block(model, chunk, memory, path)
                logits.append(scores)
            joined = torch.cat(logits, dim=1)
            loss = F.cross_entropy(
                joined[:, :-1].float().reshape(-1, joined.shape[-1]),
                tokens[:, 1:].reshape(-1),
            )
        loss.backward()

    result = cuda_measure(run, warmup=1, repeats=repeats)
    parameters_with_grad = [p for p in model.parameters() if p.grad is not None]
    result.update({
        "tokens": tokens.numel(),
        "tokens_per_second_median": tokens.numel() / result["median_seconds"],
        "parameters_with_grad": sum(p.numel() for p in parameters_with_grad),
        "gradient_bytes": sum(
            p.grad.numel() * p.grad.element_size() for p in parameters_with_grad
        ),
        "scope": "forward + backward; optimizer step excluded",
    })
    model.zero_grad(set_to_none=True)
    model.eval()
    torch.cuda.empty_cache()
    return result


def run_costs(model, sample, args) -> dict:
    inference_tokens = sample[:, : args.inference_tokens]
    training_tokens = sample[:, : args.training_tokens]
    prompt = sample[:, : args.prompt_tokens]
    result = {"sequence_inference": {}, "autoregressive_generation": {}, "training": {}}
    for path in PATHS:
        result["sequence_inference"][path.name] = sequence_inference_cost(
            model, inference_tokens, path, args.cost_repeats
        )
        result["autoregressive_generation"][path.name] = generation_cost(
            model, prompt, path, args.generated_tokens, args.generation_repeats
        )
        result["training"][path.name] = training_cost(
            model, training_tokens, path, args.training_repeats
        )
    for category in result.values():
        full = category["full"]["median_seconds"]
        for values in category.values():
            values["speedup_vs_full"] = full / values["median_seconds"]
            values["time_reduction_vs_full"] = 1.0 - values["median_seconds"] / full
    return result


def analytic_costs(cfg: DemoConfig) -> dict:
    d, h, k = cfg.d, cfg.h, cfg.k
    groups, leaves, radius = cfg.groups, cfg.leaves_per_group, cfg.m
    # Two FLOPs per multiply-accumulate. These are the dominant matmul/einsum
    # operations. Softmax, normalization, activations, indexing and elementwise
    # updates are deliberately reported as omitted rather than assigned a
    # misleading platform-independent FLOP count.
    global_read_per_refinement = (
        4 * d * h + 2 * groups * h + 4 * leaves * h
    )
    global_write_per_block_per_token = (
        4 * d * h
        + 2 * d
        + 2 * groups * h
        + 2 * leaves * h
        + 4 * groups * leaves * h
    )
    global_per_token = k * global_read_per_refinement + global_write_per_block_per_token
    local_per_refinement = 8 * d * h + 4 * radius * h
    local_per_token = k * local_per_refinement
    dynamics_per_refinement = 4 * (cfg.r + 1) * d
    dynamics_per_token = k * dynamics_per_refinement
    readout_per_token = 2 * d * cfg.vocab_size
    base_per_token = dynamics_per_token + readout_per_token
    values = {
        "global_major_flops_per_token": global_per_token,
        "local_major_flops_per_token": local_per_token,
        "base_major_flops_per_token": base_per_token,
        "paths": {
            "full": base_per_token + global_per_token + local_per_token,
            "global_off": base_per_token + local_per_token,
            "local_off": base_per_token + global_per_token,
            "associative_off": base_per_token,
        },
        "convention": "2 FLOPs per multiply-accumulate",
        "omits": [
            "softmax", "normalization", "activation", "indexing",
            "embedding lookup", "elementwise state and plastic updates",
        ],
    }
    full = values["paths"]["full"]
    values["path_fraction_of_full"] = {
        name: value / full for name, value in values["paths"].items()
    }
    return values


def parameter_costs(cfg: DemoConfig) -> dict:
    report = parameter_report(cfg)
    components = report["components"]
    global_parameters = (
        components["haa_projections"]
        + components["haa_write_rate"]
        + components["phylo_anchors"]
    )
    local_parameters = (
        components["local_projections"] + components["local_retrieval_scale"]
    )
    total = report["trainable_unique"]
    active = {
        "full": total,
        "global_off": total - global_parameters,
        "local_off": total - local_parameters,
        "associative_off": total - global_parameters - local_parameters,
    }
    return {
        "components": components,
        "total_unique": total,
        "global_parameters": global_parameters,
        "local_parameters": local_parameters,
        "active_parameters": active,
        "bf16_inference_weight_bytes": {name: count * 2 for name, count in active.items()},
        "bf16_training_persistent_bytes": {
            name: count * 8 for name, count in active.items()
        },
        "training_byte_formula": "BF16 weight + BF16 gradient + 2x BF16 AdamW moments = 8 bytes/active parameter; activations and DDP buckets excluded",
        "checkpoint_byte_formula": "BF16 weight + 2x BF16 AdamW moments = 6 bytes/active parameter",
    }


def generation_samples(model, tokenizer_path: Path, prompts: list[str], count: int) -> dict:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    result = {}
    for path in PATHS:
        rows = []
        for prompt in prompts:
            pending = tokenizer.encode(prompt, add_special_tokens=False).ids
            if not pending:
                continue
            memory = None
            generated = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                while len(pending) > 64:
                    chunk = torch.tensor([pending[:64]], device="cuda")
                    _, memory = forward_block(model, chunk, memory, path)
                    pending = pending[64:]
                for _ in range(count):
                    chunk = torch.tensor([pending], device="cuda")
                    logits, _ = forward_block(model, chunk, memory, path, update_memory=False)
                    token = int(logits[0, -1].argmax())
                    generated.append(token)
                    pending.append(token)
                    if len(pending) > 64:
                        committed = torch.tensor([pending[:64]], device="cuda")
                        _, memory = forward_block(model, committed, memory, path)
                        pending = pending[64:]
            counts = Counter(generated)
            bigrams = list(zip(generated, generated[1:]))
            rows.append({
                "prompt": prompt,
                "text": tokenizer.decode(generated, skip_special_tokens=True),
                "tokens": len(generated),
                "unique_token_fraction": len(counts) / max(1, len(generated)),
                "repeated_bigram_fraction": 1.0 - len(set(bigrams)) / max(1, len(bigrams)),
            })
        result[path.name] = rows
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows-per-source", type=int, default=5)
    parser.add_argument("--quality-tokens", type=int, default=2048)
    parser.add_argument("--inference-tokens", type=int, default=2048)
    parser.add_argument("--training-tokens", type=int, default=128)
    parser.add_argument("--prompt-tokens", type=int, default=96)
    parser.add_argument("--generated-tokens", type=int, default=128)
    parser.add_argument("--cost-repeats", type=int, default=7)
    parser.add_argument("--generation-repeats", type=int, default=3)
    parser.add_argument("--training-repeats", type=int, default=5)
    parser.add_argument("--sample-generation-tokens", type=int, default=96)
    args = parser.parse_args()
    if args.quality_tokens < 2048:
        raise ValueError("quality-tokens must be at least 2048 for all position bins")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cfg = DemoConfig()
    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if payload.get("config_sha256") != cfg.fingerprint():
        raise RuntimeError("checkpoint fingerprint does not match DemoConfig")
    metadata = {
        key: payload.get(key)
        for key in (
            "architecture", "next_step", "tokens_seen", "world_size",
            "micro_batch_size", "config_sha256",
        )
    }
    optimizer_bytes = Counter()
    for state in payload["optimizer"]["state"].values():
        for value in state.values():
            if torch.is_tensor(value):
                optimizer_bytes[str(value.dtype)] += value.numel() * value.element_size()
    model = build_model()
    model.load_state_dict(raw_state(payload), strict=True)
    del payload
    model = model.to("cuda", dtype=torch.bfloat16).eval()

    manifest = json.loads((args.packed / "pack-manifest.json").read_text())
    sample_entry = manifest["sources"]["fineweb_edu"]
    array = np.load(args.packed / sample_entry["path"], mmap_mode="r")
    needed = max(args.quality_tokens, args.inference_tokens, args.training_tokens, 2048)
    ids = np.asarray(array[1_800_018 : 1_800_018 + needed], dtype=np.int64).copy()
    sample = torch.from_numpy(ids).unsqueeze(0).to("cuda")

    report = {
        "status": "running",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": vars(args) | {"paths": [asdict(path) for path in PATHS]},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "hostname": platform.node(),
        },
        "checkpoint": metadata | {
            "path": str(args.checkpoint.resolve()),
            "size_bytes": args.checkpoint.stat().st_size,
            "sha256": sha256(args.checkpoint),
            "optimizer_tensor_bytes_by_dtype": dict(optimizer_bytes),
        },
        "parameter_cost": parameter_costs(cfg),
        "analytic_major_flops": analytic_costs(cfg),
    }
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")

    report["bypass_verification"] = verify_bypasses(model, sample[:, :256])
    if any(value != 0.0 for value in report["bypass_verification"].values()):
        raise RuntimeError(f"bypass equivalence failed: {report['bypass_verification']}")
    report["memory_trace"] = trace_memory(model, sample[:, :2048])
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")

    quality_partial = args.output.with_suffix(".quality-partial.json")
    report["quality"] = run_quality(
        model,
        args.packed,
        args.windows_per_source,
        args.quality_tokens,
        quality_partial,
    )
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")

    report["cost"] = run_costs(model, sample, args)
    report["generation_samples"] = generation_samples(
        model,
        args.tokenizer,
        [
            "Once upon a time, in a small village",
            "Artificial intelligence is a field of computer science that",
            "A água é essencial para a vida porque",
        ],
        args.sample_generation_tokens,
    )
    report["status"] = "complete"
    report["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "quality": report["quality"]["summary"],
        "cost": report["cost"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
