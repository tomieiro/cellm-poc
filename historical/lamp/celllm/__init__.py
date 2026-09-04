"""CellLM — Cellular Language Model."""

from typing import Any

from celllm.ablation import (
    NORMAL_ABLATION,
    AblationCondition,
    AblationConfig,
    AblationTrace,
)
from celllm.associative_field import LocalAssociativeMessagePassing
from celllm.config import (
    LocalAssociativeConfig,
    ModelConfig,
    PhyloConfig,
    TrainConfig,
)
from celllm.controls import GatedConvLM
from celllm.hebbian_attention import HebbianAttention
from celllm.language_data import (
    EncodedLanguageWindow,
    LanguageDocument,
    LanguageDocumentDataset,
    collate_language_windows,
    encode_document_windows,
    load_language_jsonl,
)
from celllm.model import CelNNLanguageModel, PhyloCelNNLanguageModel
from celllm.phylo_bank import PhyloBank, PhyloState
from celllm.provenance import sha256_file


def __getattr__(name: str) -> Any:
    """Load ladder exports lazily so ``python -m celllm.ladder`` stays clean."""
    if name in {"RUNGS", "build_rung"}:
        from celllm import ladder

        return getattr(ladder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "NORMAL_ABLATION",
    "RUNGS",
    "AblationCondition",
    "AblationConfig",
    "AblationTrace",
    "CelNNLanguageModel",
    "EncodedLanguageWindow",
    "GatedConvLM",
    "HebbianAttention",
    "LanguageDocument",
    "LanguageDocumentDataset",
    "LocalAssociativeConfig",
    "LocalAssociativeMessagePassing",
    "ModelConfig",
    "PhyloBank",
    "PhyloCelNNLanguageModel",
    "PhyloConfig",
    "PhyloState",
    "TrainConfig",
    "build_rung",
    "collate_language_windows",
    "encode_document_windows",
    "load_language_jsonl",
    "sha256_file",
]
