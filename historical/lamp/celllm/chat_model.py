"""Assistant-targeted training on the CellLM-v3 HAA-Phylo core."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from celllm.ablation import (
    NORMAL_ABLATION,
    AblationConfig,
    AblationTrace,
)
from celllm.config import LocalAssociativeConfig, ModelConfig, PhyloConfig
from celllm.model import PhyloCelNNLanguageModel


class CellLMChatModel(nn.Module):
    """A small chat LM with explicit memory between causal context blocks."""

    def __init__(
        self,
        model: ModelConfig,
        *,
        phylo: PhyloConfig | None = None,
        local: LocalAssociativeConfig | None = None,
        global_memory: bool = True,
    ) -> None:
        super().__init__()
        self.core = PhyloCelNNLanguageModel(
            model, phylo, local=local, global_memory=global_memory
        )
        self.phylo_config = self.core.phylo_config
        self.local_config = self.core.local_config

    @property
    def cfg(self) -> ModelConfig:
        return self.core.cfg

    @property
    def uses_associative_state(self) -> bool:
        return True

    @property
    def chunk_size(self) -> int:
        return self.phylo_config.chunk_size

    @property
    def local_retrieval_scale(self) -> torch.Tensor | None:
        if self.local_config is None:
            return None
        return self.core.cell.local_attention.retrieval_scale

    def new_plastic_state(self, batch_size: int):
        return self.core.new_memory_state(batch_size)

    def forward_with_state(self, *args, **kwargs):
        if "update_plasticity" in kwargs:
            kwargs["update_memory"] = kwargs.pop("update_plasticity")
        return self.core.forward_with_state(*args, **kwargs)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.core(token_ids)

    def sequence_logits(
        self,
        token_ids: torch.Tensor,
        *,
        ablation: AblationConfig = NORMAL_ABLATION,
        trace: AblationTrace | None = None,
    ) -> torch.Tensor:
        """Run native chunks while preserving explicit associative state."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        memory = self.new_plastic_state(token_ids.shape[0])
        logits = []
        token_chunks = token_ids.split(self.chunk_size, dim=1)
        mask_chunks = token_ids.ne(0).split(self.chunk_size, dim=1)
        for chunk, write_mask in zip(token_chunks, mask_chunks):
            kwargs = {
                "write_mask": write_mask,
                "ablation": ablation,
                "trace": trace,
            }
            chunk_logits, memory = self.forward_with_state(
                chunk, memory, **kwargs
            )
            logits.append(chunk_logits)
        return torch.cat(logits, dim=1)

    def loss_statistics(
        self,
        token_ids: torch.Tensor,
        assistant_mask: torch.Tensor,
        *,
        ablation: AblationConfig = NORMAL_ABLATION,
        trace: AblationTrace | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return assistant NLL sums and token counts for each dialogue."""
        if token_ids.shape != assistant_mask.shape:
            raise ValueError("token_ids and assistant_mask must have same shape")
        predictions = self.sequence_logits(
            token_ids, ablation=ablation, trace=trace
        )[:, :-1]
        targets = token_ids[:, 1:]
        mask = assistant_mask[:, 1:]
        if not torch.any(mask):
            raise ValueError("batch has no assistant targets")
        losses = F.cross_entropy(
            predictions.reshape(-1, self.cfg.vocab_size),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        return (losses * mask).sum(dim=1), mask.sum(dim=1)

    def loss(
        self,
        token_ids: torch.Tensor,
        assistant_mask: torch.Tensor,
        *,
        ablation: AblationConfig = NORMAL_ABLATION,
        trace: AblationTrace | None = None,
    ) -> torch.Tensor:
        """Cross-entropy only where the next token belongs to the assistant."""
        loss_sums, token_counts = self.loss_statistics(
            token_ids,
            assistant_mask,
            ablation=ablation,
            trace=trace,
        )
        return loss_sums.sum() / token_counts.sum()
