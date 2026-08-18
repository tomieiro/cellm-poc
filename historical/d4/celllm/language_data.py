"""Document-bounded language-model examples for CellLM pretraining."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from celllm.chat_tokenizer import ChatTokenizer


@dataclass(frozen=True)
class LanguageDocument:
    """One immutable source document with a stable external identifier."""

    document_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("language document id must not be empty")
        if not self.text.strip():
            raise ValueError("language document text must not be empty")


@dataclass(frozen=True)
class EncodedLanguageWindow:
    """One independent window that never crosses a document boundary."""

    token_ids: torch.Tensor
    target_mask: torch.Tensor
    length: int
    document_id: str
    window_index: int


def encode_document_windows(
    document: LanguageDocument,
    tokenizer: ChatTokenizer,
    *,
    max_length: int,
) -> list[EncodedLanguageWindow]:
    """Split a document into independent BOS-prefixed next-token windows."""
    if max_length < 2:
        raise ValueError("language window length must be at least two")
    payload = [*tokenizer.encode(document.text), tokenizer.eos_id]
    capacity = max_length - 1
    windows = []
    for index, start in enumerate(range(0, len(payload), capacity)):
        content = payload[start : start + capacity]
        ids = [tokenizer.special_id("<bos>"), *content]
        length = len(ids)
        padding = max_length - length
        mask = [False, *([True] * len(content)), *([False] * padding)]
        ids.extend([tokenizer.pad_id] * padding)
        windows.append(
            EncodedLanguageWindow(
                token_ids=torch.tensor(ids, dtype=torch.long),
                target_mask=torch.tensor(mask, dtype=torch.bool),
                length=length,
                document_id=document.document_id,
                window_index=index,
            )
        )
    return windows


class LanguageDocumentDataset(Dataset):
    """Pre-encoded windows with no packing across source documents."""

    def __init__(
        self,
        documents: list[LanguageDocument],
        tokenizer: ChatTokenizer,
        *,
        max_length: int,
    ) -> None:
        seen = set()
        repeated = []
        for document in documents:
            if document.document_id in seen:
                repeated.append(document.document_id)
            seen.add(document.document_id)
        if repeated:
            raise ValueError(f"duplicate language document id: {repeated[0]}")
        self.utf8_bytes = sum(
            len(document.text.encode("utf-8")) for document in documents
        )
        self.examples = [
            window
            for document in documents
            for window in encode_document_windows(
                document, tokenizer, max_length=max_length
            )
        ]
        if not self.examples:
            raise ValueError("language dataset must contain documents")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedLanguageWindow:
        return self.examples[index]

    @property
    def target_tokens(self) -> int:
        return sum(int(item.target_mask.sum()) for item in self.examples)


def collate_language_windows(
    examples: list[EncodedLanguageWindow],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trim batch padding while preserving next-token target alignment."""
    length = max(item.length for item in examples)
    return (
        torch.stack([item.token_ids[:length] for item in examples]),
        torch.stack([item.target_mask[:length] for item in examples]),
    )


def load_language_jsonl(path: str | Path) -> list[LanguageDocument]:
    """Load rows shaped as ``{"id": ..., "text": ...}``."""
    documents = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                documents.append(
                    LanguageDocument(str(record["id"]), str(record["text"]))
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid language document at line {line_number}: {error}"
                ) from error
    if not documents:
        raise ValueError("language JSONL must contain at least one document")
    return documents
