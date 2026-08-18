"""CellLM-D20 showcase: a scaled run of the validated BANK + LOCAL core.

This package never redefines the architecture. It imports `celllm` and only
adds what a scaled showcase needs: sizing, data acquisition and gating, a
first-token-balanced curriculum, held-out behavioural evaluation, segment
training with language replay, multi-objective selection, and reporting.

The D20 is a demonstration branch. Its numbers must not be mixed with the
causal conclusions of the A1/B0.x line.
"""

__all__ = [
    "architecture",
    "chat",
    "curriculum",
    "datasets",
    "evaluation",
    "provenance",
    "report",
    "selection",
    "smoke",
    "training",
    "tokenizer",
]
