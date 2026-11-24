"""Recompute archived language modeling metrics from experimental records."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean


LAMP_ARTIFACT = "lamp/stage19-haa738-within-checkpoint-ablation.json"
SOURCE_WEIGHTS = dict(
    fineweb_edu=0.30,
    tulu3=0.20,
    cosmopedia=0.15,
    smol_smoltalk=0.13,
    fineweb2_pt=0.12,
    pes2o=0.05,
    wikipedia_en=0.05,
)


def summarize_lamp(windows, weights):
    if set(windows) != set(weights) or not math.isclose(sum(weights.values()), 1):
        raise ValueError("fontes e pesos da ablação de LAMP inconsistentes")
    rows = [row for source in windows.values() for row in source]
    if any(
        row["full"]["tokens"] != 2047 or row["local_off"]["tokens"] != 2047
        for row in rows
    ):
        raise ValueError("a ablação deve usar os mesmos 2.047 alvos por janela")
    return {
        "windows": len(rows),
        "targets_per_arm": 2047 * len(rows),
        **{
            f"{arm}_nll": sum(
                weights[source] * mean(row[arm]["loss_nats"] for row in group)
                for source, group in windows.items()
            )
            for arm in ("full", "local_off")
        },
    }


def text8_notebook(path):
    notebook = json.loads(path.read_text())
    by_rung = {}
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            for line in "".join(output.get("text", [])).splitlines():
                if line.startswith('{"rung":'):
                    row = json.loads(line)
                    by_rung[row["rung"]] = row
    if set(by_rung) != set("ABCDEFGH"):
        raise ValueError("faltam resultados da escada Text8 no notebook executado")
    return by_rung


def summarize_d4(records):
    held_out = {
        (row["held_out"]["tokens"], row["held_out"]["utf8_bytes"])
        for row in records.values()
    }
    if len(held_out) != 1 or set(records) != {"cellm", "wide", "deep"}:
        raise ValueError("os braços D4 devem usar o mesmo conjunto de avaliação")
    eval_tokens, eval_bytes = next(iter(held_out))
    result = {}
    for arm, row in records.items():
        nll = row["held_out"]["nll"]
        bpc = nll * eval_tokens / (math.log(2) * eval_bytes)
        if row["target_tokens"] < 50_000_000 or not math.isclose(
            bpc, row["held_out"]["bpc"]
        ):
            raise ValueError(f"orçamento ou BPC incompatível no registro D4 de {arm}")
        result[arm] = {
            "trained_tokens": row["target_tokens"],
            "eval_tokens": eval_tokens,
            "eval_utf8_bytes": eval_bytes,
            "nll": nll,
            "bpc": bpc,
        }
    return result


def d4_report(directory):
    cell = [
        json.loads(line)
        for line in (directory / "limpo-linguagem.jsonl").read_text().splitlines()
        if line.strip()
    ]
    records = {"cellm": cell[-1]}
    for arm in ("wide", "deep"):
        records[arm] = json.loads((directory / f"transformer-{arm}.json").read_text())[
            "history"
        ][-1]
    return summarize_d4(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artefacts", nargs="?", type=Path, default=Path("artefacts"))
    args = parser.parse_args()
    lamp = json.loads((args.artefacts / LAMP_ARTIFACT).read_text())
    result = {
        "text8_reported": text8_notebook(
            args.artefacts / "text8/experiment-0-gpu.executed.ipynb"
        ),
        "tinystories_d4": d4_report(args.artefacts / "tinystories-d4"),
        "lamp": summarize_lamp(lamp["quality"]["windows"], SOURCE_WEIGHTS),
    }
    if result["lamp"]["windows"] != 35:
        raise ValueError("a ablação histórica de LAMP requer 35 janelas")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
