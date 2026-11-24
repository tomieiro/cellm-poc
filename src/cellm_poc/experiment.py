"""Run the matched carry/reset experiment and preserve its inputs and outputs."""

import argparse
import json
import os
import platform
import random
import statistics
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import tokenizers
import torch

from .model import Shape, build_model, load_model, parameters, transformer_width
from .protocol import (
    boundary,
    evaluate,
    evaluate_recall,
    generate,
    prepare,
    recall_inputs,
    sha256,
    train_arm,
)


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def environment(device):
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "celnn": version("celnn"),
        "tokenizers": tokenizers.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": device,
        "gpu": (
            torch.cuda.get_device_name(device)
            if torch.device(device).type == "cuda"
            else None
        ),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "threads": torch.get_num_threads(),
    }


def smoke_documents(output):
    data = output / "fixture"
    data.mkdir()
    paths = [data / f"{name}.jsonl" for name in ("train", "validation", "test")]
    phrases = [
        "The garden has yellow roses and green leaves. The river carries cool water.",
        "The library keeps a record of the stars. A small fox watches the morning rain.",
        "A quiet teacher looks out of the window. The school opens after the storm.",
    ]
    for split, path in enumerate(paths):
        with path.open("w", encoding="utf-8") as stream:
            for row in range(6 if split == 0 else 2):
                text = f"Story {split}-{row}. " + " ".join(
                    phrases[(split + row + i) % len(phrases)] for i in range(12)
                )
                stream.write(json.dumps({"text": text}) + "\n")
    return paths


def validate(config, shape):
    budgets = config["budgets"]
    seeds = config["seeds"]
    if not budgets or budgets != sorted(set(budgets)) or budgets[0] <= 0:
        raise ValueError("os orçamentos devem ser positivos, únicos e crescentes")
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("as sementes devem ser únicas e não negativas")
    if min(asdict(shape).values()) < 1:
        raise ValueError("os parâmetros da arquitetura devem ser positivos")
    if config["context"] < 2 * shape.block or config["context"] % shape.block:
        raise ValueError("o contexto deve conter pelo menos dois blocos completos")
    if not 0 < config["overlap"] < shape.block:
        raise ValueError("overlap deve ser menor que o bloco e maior que zero")
    if (
        min(
            config["batch_size"],
            config["recall_trials"],
            config["generation_tokens"],
            config["boundary_windows"],
        )
        < 1
        or config["vocab_size"] < 258
    ):
        raise ValueError("tamanho de lote, amostras ou vocabulário inválido")
    if (
        config["learning_rate"] <= 0
        or not config["prompts"]
        or not all(prompt.strip() for prompt in config["prompts"])
        or not config["recall_distances"]
    ):
        raise ValueError("taxa de aprendizado, prompts ou distâncias inválidos")
    if not config["smoke"] and (config["context"], shape.block, config["overlap"]) != (
        2048,
        64,
        32,
    ):
        raise ValueError(
            "o protocolo de continuidade requer contexto 2048, bloco 64 e overlap 32"
        )


def recall_summary(rows):
    groups = {}
    for distance in sorted({row["distance"] for row in rows}):
        group = [row for row in rows if row["distance"] == distance]
        count = len(group)
        groups[distance] = {
            "pairs": count,
            "accuracy": sum(
                (row["target"] == row["top1"])
                + (row["alternative_target"] == row["alternative_top1"])
                for row in group
            )
            / (2 * count),
            "target_probability": statistics.mean(
                (row["target_probability"] + row["alternative_probability"]) / 2
                for row in group
            ),
            "probability_delta": statistics.mean(
                (
                    row["target_probability"]
                    - row["target_probability_after_swap"]
                    + row["alternative_probability"]
                    - row["alternative_probability_before_swap"]
                )
                / 2
                for row in group
            ),
        }
    return groups


def evaluate_arm(model, arm, test, cases, tokenizer, config, boundary_indices, device):
    modes = ("carry", "reset", "lamp-off") if arm == "carry" else (arm,)
    result = {}
    for mode in modes:
        natural = evaluate(model, mode, test, config["batch_size"], device)
        recalled = evaluate_recall(model, mode, cases, device)
        completions = []
        for prompt in config["prompts"]:
            ids = tokenizer.encode(prompt).ids
            generated = generate(model, mode, ids, config["generation_tokens"], device)
            completions.append(
                {
                    "prompt": prompt,
                    "generated_ids": generated,
                    "completion": tokenizer.decode(generated),
                }
            )
        result[mode] = {
            "natural": natural,
            "recall_by_distance": recall_summary(recalled),
            "recall_pairs": recalled,
            "generation": completions,
        }
    if arm == "carry":
        rows = [
            dict(row, window=index)
            for index in boundary_indices
            for row in boundary(model, test[index], config["overlap"], device)
        ]
        targets = sum(row["targets"] for row in rows)
        result["overlap32" if config["overlap"] == 32 else "overlap"] = {
            "segments": len(rows),
            "targets": targets,
            "improved_segments": sum(
                row["overlap_nll"] < row["reset_nll"] for row in rows
            ),
            "reset_nll": sum(row["reset_nll"] * row["targets"] for row in rows)
            / targets,
            "overlap_nll": sum(row["overlap_nll"] * row["targets"] for row in rows)
            / targets,
            "rows": rows,
        }
    return result


def run(config_file, paths, output, device, threads):
    config = json.loads(config_file.read_text())
    shape = Shape(**config["shape"])
    validate(config, shape)
    if config["smoke"] != (not paths):
        raise ValueError("informe train, validation e test para o protocolo completo")
    if threads < 1:
        raise ValueError("o número de threads precisa ser positivo")
    output.mkdir(parents=True)
    if config["smoke"]:
        paths = smoke_documents(output)
    torch.set_num_threads(threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    tokenizer, (train, validation, test) = prepare(
        paths, config["vocab_size"], config["context"]
    )
    tokenizer.save(str(output / "tokenizer.json"))
    vocabulary = tokenizer.get_vocab_size()
    control_width = transformer_width(vocabulary, shape)
    splits = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "windows": len(rows),
            "targets": sum(len(window) - 1 for window in rows),
        }
        for name, path, rows in zip(
            ("train", "validation", "test"), paths, (train, validation, test)
        )
    }
    manifest = {
        "config": config,
        "splits": splits,
        "vocabulary": vocabulary,
        "tokenizer_sha256": sha256(output / "tokenizer.json"),
        "environment": environment(device),
        "source_sha256": {
            module.name: sha256(module)
            for module in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    write_json(output / "manifest.json", manifest | {"status": "running"})

    available = [idx for idx, row in enumerate(test) if len(row) > shape.block + 1]
    if not available:
        raise ValueError(
            "o teste precisa ter uma janela que atravesse uma fronteira de bloco"
        )
    rng = random.Random(config["evaluation_seed"])
    boundary_indices = rng.sample(
        available, min(config["boundary_windows"], len(available))
    )
    arms = ("carry", "reset", "transformer")
    for seed in config["seeds"]:
        records = {}
        for arm in arms:
            torch.manual_seed(seed)
            model = build_model(arm, vocabulary, shape, control_width)
            trained = train_arm(
                model,
                arm,
                train,
                validation,
                config["budgets"],
                config["batch_size"],
                seed,
                device,
                config["learning_rate"],
            )
            records[arm] = {
                "seed": seed,
                "trained_with": arm,
                "parameters": parameters(model),
                **trained,
            }
            torch.save(
                {
                    "arm": arm,
                    "shape": asdict(shape),
                    "vocabulary": vocabulary,
                    "transformer_width": control_width,
                    "seed": seed,
                    "weights": model.state_dict(),
                },
                output / f"{seed}-{arm}.pt",
            )
            write_json(output / f"{seed}-{arm}.json", records[arm])
            print(f"seed={seed} arm={arm} treino concluído", flush=True)
            del model
        if (
            records["carry"]["initial_weights_sha256"]
            != records["reset"]["initial_weights_sha256"]
            or len({record["batch_sha256"] for record in records.values()}) != 1
        ):
            raise RuntimeError(f"pareamento violado na semente {seed}")

    pairs = recall_inputs(
        tokenizer,
        train,
        test,
        config["recall_distances"],
        config["recall_trials"],
        config["context"],
        config["evaluation_seed"],
    )
    deltas = []
    for seed in config["seeds"]:
        records = {}
        for arm in arms:
            model, _, _ = load_model(output / f"{seed}-{arm}.pt", device)
            path = output / f"{seed}-{arm}.json"
            result = json.loads(path.read_text())
            result["test"] = evaluate_arm(
                model, arm, test, pairs, tokenizer, config, boundary_indices, device
            )
            write_json(path, result)
            records[arm] = result
            print(
                f"seed={seed} arm={arm} nll={result['test'][arm]['natural']['nll']:.5f}",
                flush=True,
            )
            del model

        carry = records["carry"]["test"]
        nll = carry["carry"]["natural"]["nll"]
        deltas.append(
            {
                "seed": seed,
                "carry_minus_trained_reset_nll": nll
                - records["reset"]["test"]["reset"]["natural"]["nll"],
                "carry_minus_inference_reset_nll": nll
                - carry["reset"]["natural"]["nll"],
                "lamp_enabled_minus_disabled_nll": nll
                - carry["lamp-off"]["natural"]["nll"],
                "carry_minus_transformer_nll": nll
                - records["transformer"]["test"]["transformer"]["natural"]["nll"],
            }
        )
    paired = {
        name: statistics.mean(elt[name] for elt in deltas)
        for name in deltas[0]
        if name != "seed"
    }
    write_json(
        output / "summary.json",
        {"deltas_by_seed": deltas, "mean_paired_deltas": paired},
    )
    write_json(output / "manifest.json", manifest | {"status": "complete"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--data", nargs=3, type=Path, metavar=("TRAIN", "VALIDATION", "TEST")
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", default=2, type=int)
    args = parser.parse_args()
    try:
        run(args.config, args.data, args.output, args.device, args.threads)
    except (ValueError, OSError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
