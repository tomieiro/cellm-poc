"""Build and size the CellLM-D20 model from the validated BANK + LOCAL core.

This module never defines a new architecture. It imports the same
``CellLMChatModel`` the B0.x line validated and only chooses its shape.

A structural fact that governs everything here: **the architecture has no depth
knob.** One shared cell is applied ``k`` times, so capacity scales through width
(``d``), vocabulary and memory width only.

This is a property, not a defect. The Chua-Yang core holds on the order of ten
thousand parameters and is reused at every one of the ``k`` evolution steps, so
its share of *storage* is a poor proxy for its share of *computation*: CellLM
scales compute by reusing parameters over time rather than by stacking
independent layers. The search therefore never tries to inflate the core -- that
would change the architecture. It only avoids spending the budget on the symbol
table when an equally sized alternative spends less.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from celllm.chat_model import CellLMChatModel
from celllm.config import (
    LocalAssociativeConfig,
    ModelConfig,
    StateMatchedBankConfig,
)

# Frozen by the B0.x line. These are not search knobs.
DYNAMICS_STEPS = 15
CELNN_RADIUS = 1
LOCAL_RADIUS = 2
SPATIAL = "diagonal"
MIXER = "dense"
HEBB_RATE = 0.1
MIN_RETENTION = 0.99
RETRIEVAL_SCALE = 0.1
LOCAL_RETRIEVAL_SCALE = 0.1


@dataclass(frozen=True)
class D20Shape:
    """The searchable part of the configuration."""

    dimensions: int
    cells: int
    vocab_size: int
    memory_size: int

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            n=self.cells,
            d=self.dimensions,
            r=CELNN_RADIUS,
            k=DYNAMICS_STEPS,
            vocab_size=self.vocab_size,
            spatial=SPATIAL,
            mixer=MIXER,
        )

    def bank_config(self) -> StateMatchedBankConfig:
        return StateMatchedBankConfig(
            slots=self.cells,
            key_size=self.memory_size,
            value_size=self.memory_size,
            learning_rate=HEBB_RATE,
            min_retention=MIN_RETENTION,
            retrieval_scale=RETRIEVAL_SCALE,
            chunk_size=self.cells,
        )

    def local_config(self) -> LocalAssociativeConfig:
        return LocalAssociativeConfig(
            radius=LOCAL_RADIUS,
            key_size=self.memory_size,
            value_size=self.memory_size,
            gated=False,
            retrieval_scale=LOCAL_RETRIEVAL_SCALE,
        )


def build_model(shape: D20Shape) -> CellLMChatModel:
    """Instantiate the validated BANK + LOCAL-UNGATED-R2 core at this shape."""
    return CellLMChatModel(
        shape.model_config(),
        bank=shape.bank_config(),
        local=shape.local_config(),
    )


def plastic_state_floats(model: CellLMChatModel) -> int:
    """Floats of associative state carried per sequence.

    This is not a parameter count and the search would miss it otherwise, but it
    is what decides the achievable batch size: the BANK state is
    ``slots x key_size x value_size`` and is held in FP32 by protocol, so it
    grows with the square of ``memory_size``. A shape that wins on parameter
    composition can still be the wrong choice if it throttles the A100.
    """
    state = model.new_plastic_state(1)

    def measure(item) -> int:
        import torch

        if torch.is_tensor(item):
            return item.numel()
        if isinstance(item, (tuple, list)):
            return sum(measure(entry) for entry in item)
        if hasattr(item, "__dict__"):
            return sum(measure(value) for value in vars(item).values())
        return 0

    return measure(state)


def count_parameters(model: CellLMChatModel) -> dict:
    """Report parameters by role, not just the total.

    The embedding share matters: a configuration that hits 20M by making the
    embedding table enormous has not scaled the mechanism under test.
    """
    groups = {
        "embedding": 0,
        "head": 0,
        "chua_yang_core": 0,
        "projection": 0,
        "bank_memory": 0,
        "local_memory": 0,
        "other": 0,
    }
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        if "embed" in name:
            groups["embedding"] += count
        elif "readout" in name:
            groups["head"] += count
        elif "cell.dynamics" in name:
            groups["chua_yang_core"] += count
        elif "cell.mixer" in name:
            groups["projection"] += count
        elif "local_attention" in name:
            groups["local_memory"] += count
        elif "attention" in name:
            groups["bank_memory"] += count
        else:
            groups["other"] += count

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    embedding_like = groups["embedding"] + groups["head"]
    memory = groups["bank_memory"] + groups["local_memory"]
    return {
        "total": total,
        "trainable": trainable,
        "by_component": groups,
        "share_by_component": {
            name: value / total for name, value in groups.items()
        },
        "embedding_parameters": embedding_like,
        "embedding_fraction": embedding_like / total,
        "non_embedding_parameters": total - embedding_like,
        "non_embedding_fraction": (total - embedding_like) / total,
        "chua_yang_parameters": groups["chua_yang_core"],
        # Storage share only. The core is reused at every one of the k evolution
        # steps, so this number says nothing about its share of computation.
        "chua_yang_fraction": groups["chua_yang_core"] / total,
        "chua_yang_applications_per_forward": DYNAMICS_STEPS,
        "memory_parameters": memory,
        "memory_fraction": memory / total,
        "projection_parameters": groups["projection"],
        "projection_fraction": groups["projection"] / total,
        "head_parameters": groups["head"],
    }


def search_architecture(
    *,
    target: int = 20_000_000,
    minimum: int = 19_000_000,
    maximum: int = 21_000_000,
    dimensions: tuple[int, ...] = (
        1536, 1792, 2048, 2304, 2560, 2816, 3072,
    ),
    vocab_sizes: tuple[int, ...] = (4096, 6144, 8192),
    memory_sizes: tuple[int, ...] = (
        64, 96, 128, 192, 256, 384, 512, 768,
    ),
    cells: int = 16,
    max_plastic_state_mb: float = 8.0,
    output: str | Path | None = None,
) -> dict:
    """Enumerate candidate shapes and select lexicographically.

    Selection order:

    1. the total must lie inside ``[minimum, maximum]``;
    2. the per-sequence plastic state must fit ``max_plastic_state_mb``;
    3. among those, minimise ``embedding_fraction``;
    4. ties break by distance to ``target``.

    The plastic-state constraint exists because ``memory_size`` buys
    non-embedding parameters cheaply while growing the carried state
    quadratically. Without it the search happily picks a shape that looks best
    on composition and then throttles the batch size on the A100.

    Distance to the target alone would happily buy the last few hundred
    thousand parameters with vocabulary, which spends the budget on the symbol
    table instead of the model. If the admissible band is empty the search falls
    back to nearest-to-target and says so, rather than silently returning an
    out-of-band shape.
    """
    candidates = []
    for dimension in dimensions:
        for vocab in vocab_sizes:
            for memory in memory_sizes:
                shape = D20Shape(dimension, cells, vocab, memory)
                model = build_model(shape)
                report = count_parameters(model)
                state_floats = plastic_state_floats(model)
                state_mb = state_floats * 4 / 1e6
                total = report["total"]
                candidates.append(
                    {
                        "dimensions": dimension,
                        "cells": cells,
                        "vocab_size": vocab,
                        "memory_size": memory,
                        "total_parameters": total,
                        "embedding_parameters": report["embedding_parameters"],
                        "embedding_fraction": report["embedding_fraction"],
                        "non_embedding_parameters": report[
                            "non_embedding_parameters"
                        ],
                        "chua_yang_parameters": report["chua_yang_parameters"],
                        "chua_yang_fraction": report["chua_yang_fraction"],
                        "memory_parameters": report["memory_parameters"],
                        "projection_parameters": report[
                            "projection_parameters"
                        ],
                        "head_parameters": report["head_parameters"],
                        "distance_to_target": abs(total - target),
                        "plastic_state_floats_per_sequence": state_floats,
                        "plastic_state_mb_per_sequence_fp32": state_mb,
                        "in_band": minimum <= total <= maximum,
                        "state_fits": state_mb <= max_plastic_state_mb,
                    }
                )
    admissible = [
        item
        for item in candidates
        if item["in_band"] and item["state_fits"]
    ]
    if admissible:
        admissible.sort(
            key=lambda item: (
                item["embedding_fraction"],
                item["distance_to_target"],
            )
        )
        best = admissible[0]
        rule = (
            "in-band and state-fitting, minimum embedding_fraction"
        )
    else:
        candidates.sort(key=lambda item: item["distance_to_target"])
        best = candidates[0]
        rule = (
            "FALLBACK: no candidate is both in-band and state-fitting; "
            "nearest to target"
        )
    candidates.sort(
        key=lambda item: (
            not (item["in_band"] and item["state_fits"]),
            item["embedding_fraction"],
        )
    )
    result = {
        "target_parameters": target,
        "band": [minimum, maximum],
        "max_plastic_state_mb": max_plastic_state_mb,
        "selection_rule": rule,
        "admissible_candidates": len(admissible),
        "frozen": {
            "dynamics_steps": DYNAMICS_STEPS,
            "celnn_radius": CELNN_RADIUS,
            "local_radius": LOCAL_RADIUS,
            "spatial": SPATIAL,
            "mixer": MIXER,
            "attention": "bank + local-ungated-r2",
        },
        "note": (
            "The architecture has no depth knob: one shared cell is applied "
            "dynamics_steps times, so capacity scales through width, "
            "vocabulary and memory width only. The Chua-Yang core is small in "
            "storage precisely because it is reused at every evolution step; "
            "its parameter share is not its compute share, and the search "
            "never inflates it."
        ),
        "selected": best,
        "candidates": candidates,
    }
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    return result


def shape_from_search(result: dict) -> D20Shape:
    selected = result["selected"]
    return D20Shape(
        dimensions=selected["dimensions"],
        cells=selected["cells"],
        vocab_size=selected["vocab_size"],
        memory_size=selected["memory_size"],
    )
