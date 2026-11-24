"""Measure one checkpoint at one input length in a fresh process."""

import argparse
import json
import resource
import statistics
import time
from pathlib import Path

import torch

from .experiment import environment
from .model import load_model, parameters
from .protocol import predict, sha256, synchronize


def measure(function, repeats, device):
    function()
    synchronize(device)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        synchronize(device)
        elapsed.append(time.perf_counter() - start)
    return {
        "median_ms": statistics.median(elapsed) * 1000,
        "samples_ms": [seconds * 1000 for seconds in elapsed],
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device)
            if torch.device(device).type == "cuda"
            else None
        ),
    }


@torch.inference_mode()
def benchmark(checkpoint, length, repeats, decode_tokens, device, threads, mode=None):
    torch.set_num_threads(threads)
    model, arm, vocab = load_model(checkpoint, device)
    mode = mode or arm
    if (arm == "transformer") != (mode == "transformer"):
        raise ValueError("modo incompatível com a arquitetura do checkpoint")
    random = torch.Generator().manual_seed(180017)
    ids = torch.randint(2, vocab, (1, length), generator=random).to(device)

    prefill = measure(
        lambda: predict(model, ids, mode, last_only=True), repeats, device
    )
    prefill["input_tokens_per_second"] = length * 1000 / prefill["median_ms"]
    prefill["process_max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    def decode():
        prefix = ids
        for _ in range(decode_tokens):
            logits = predict(model, prefix, mode, last_only=True)[:, -1]
            token = logits.argmax(dim=-1, keepdim=True)
            prefix = torch.cat((prefix, token), dim=1)

    decoding = measure(decode, repeats, device)
    decoding["output_tokens_per_second"] = decode_tokens * 1000 / decoding["median_ms"]
    decoding["process_max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "trained_with": arm,
        "inference_mode": mode,
        "parameters": parameters(model),
        "checkpoint_sha256": sha256(checkpoint),
        "environment": environment(device),
        "length": length,
        "batch_size": 1,
        "decode_tokens": decode_tokens,
        "prefill": prefill,
        "greedy_from_full_prefix": decoding,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("carry", "reset", "lamp-off", "transformer"))
    args = parser.parse_args()
    if min(args.length, args.repeats, args.decode_tokens, args.threads) < 1:
        parser.error("comprimento, repetições, tokens e threads precisam ser positivos")
    result = benchmark(
        args.checkpoint,
        args.length,
        args.repeats,
        args.decode_tokens,
        args.device,
        args.threads,
        args.mode,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
