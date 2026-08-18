"""CellLM — Cellular Language Model."""

from typing import Any

from celllm.ablation import (
    NORMAL_ABLATION,
    AblationCondition,
    AblationConfig,
    AblationTrace,
)
from celllm.attention import (
    BankWriteAction,
    BankWriteCandidates,
    CausalFieldPropagation,
    ChuaYangHebbianFieldAttention,
    DeltaHebbianAttention,
    InterBlockFieldCarry,
    LocalAssociativeMessagePassing,
    StateMatchedGlobalBankAttention,
)
from celllm.chat_stage8 import (
    BankBlockDiagnostics,
    BankTrajectoryMetrics,
    CandidateUtility,
    DialogueUtilityResult,
    diagnose_bank_block,
    evaluate_bank_write_utilities,
)
from celllm.config import (
    CYHFAConfig,
    HebbianAttentionConfig,
    LocalAssociativeConfig,
    ModelConfig,
    PlasticityConfig,
    StateMatchedBankConfig,
    TrainConfig,
)
from celllm.controls import GatedConvLM
from celllm.language_data import (
    EncodedLanguageWindow,
    LanguageDocument,
    LanguageDocumentDataset,
    collate_language_windows,
    encode_document_windows,
    load_language_jsonl,
)
from celllm.model import (
    CYHFACelNNLanguageModel,
    CelNNLanguageModel,
    HebbianAttentionCelNNLanguageModel,
    PlasticCelNNLanguageModel,
    StateMatchedBankCelNNLanguageModel,
)
from celllm.plastic_checkpoint import (
    load_plastic_checkpoint,
    save_plastic_checkpoint,
)
from celllm.provenance import sha256_file


def __getattr__(name: str) -> Any:
    """Load ladder exports lazily so ``python -m celllm.ladder`` stays clean."""
    if name in {"RUNGS", "build_rung"}:
        from celllm import ladder

        return getattr(ladder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "RUNGS",
    "NORMAL_ABLATION",
    "AblationCondition",
    "AblationConfig",
    "AblationTrace",
    "BankBlockDiagnostics",
    "BankTrajectoryMetrics",
    "BankWriteAction",
    "BankWriteCandidates",
    "CandidateUtility",
    "CYHFAConfig",
    "CYHFACelNNLanguageModel",
    "CausalFieldPropagation",
    "CelNNLanguageModel",
    "DeltaHebbianAttention",
    "DialogueUtilityResult",
    "GatedConvLM",
    "HebbianAttentionCelNNLanguageModel",
    "HebbianAttentionConfig",
    "ChuaYangHebbianFieldAttention",
    "InterBlockFieldCarry",
    "EncodedLanguageWindow",
    "LanguageDocument",
    "LanguageDocumentDataset",
    "LocalAssociativeConfig",
    "LocalAssociativeMessagePassing",
    "ModelConfig",
    "PlasticCelNNLanguageModel",
    "PlasticityConfig",
    "StateMatchedBankCelNNLanguageModel",
    "StateMatchedBankConfig",
    "StateMatchedGlobalBankAttention",
    "TrainConfig",
    "build_rung",
    "collate_language_windows",
    "diagnose_bank_block",
    "evaluate_bank_write_utilities",
    "encode_document_windows",
    "load_plastic_checkpoint",
    "load_language_jsonl",
    "save_plastic_checkpoint",
    "sha256_file",
]
