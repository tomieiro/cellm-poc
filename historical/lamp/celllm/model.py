"""Character language model driven by causal CelNN dynamics."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from celllm.ablation import (
    NORMAL_ABLATION,
    AblationConfig,
    AblationTrace,
)
from celllm.cell import (
    CelNNCell,
    PhyloCelNNCell,
)
from celllm.config import (
    LocalAssociativeConfig,
    ModelConfig,
    PhyloConfig,
)


class CelNNLanguageModel(nn.Module):
    """Embed characters, evolve a cellular field, and decode once."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d)
        self.cell = CelNNCell(cfg, causal=True)
        self.readout = nn.Linear(cfg.d, cfg.vocab_size, bias=True)
        self.readout.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return next-character logits for each sequence position."""
        embedding = self.embed(tokens)
        cell_input = self.cell.control_input(embedding)
        state = torch.zeros_like(embedding)
        for _ in range(self.cfg.k):
            state = self.cell.step(state, cell_input)
        return self.readout(state)

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return mean next-character cross-entropy in nats."""
        logits = self(tokens)
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, self.cfg.vocab_size),
            tokens[:, 1:].reshape(-1),
        )


class PhyloCelNNLanguageModel(nn.Module):
    """CellLM-v3 with global HAA-Phylo and optional LOCAL association."""

    def __init__(
        self,
        cfg: ModelConfig,
        phylo: PhyloConfig | None = None,
        local: LocalAssociativeConfig | None = None,
        global_memory: bool = True,
    ) -> None:
        super().__init__()
        # Respect the requested mixer. This line used to force
        # replace(cfg, mixer="dense") unconditionally, silently discarding
        # the caller's choice, which made the channel-mixing axis dead code
        # in every associative architecture and put a d^2 matmul inside every
        # one of the k steps -- half of the measured FLOPs -- in direct
        # violation of the no-dense-local-comm invariant and of
        # Architecture.md, which prescribes low-rank or grouped mixing as the
        # default. Behaviour for mixer="dense" is unchanged.
        self.cfg = cfg
        self.phylo_config = phylo or PhyloConfig(chunk_size=self.cfg.n)
        self.local_config = local
        self.embed = nn.Embedding(self.cfg.vocab_size, self.cfg.d)
        self.global_memory = global_memory
        self.cell = PhyloCelNNCell(
            self.cfg, self.phylo_config, local=local, causal=True,
            global_memory=global_memory,
        )
        self.readout = nn.Linear(
            self.cfg.d, self.cfg.vocab_size, bias=True
        )
        self.readout.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def new_memory_state(self, batch_size: int):
        return self.cell.new_memory_state(batch_size)

    def forward_with_state(
        self,
        tokens: torch.Tensor,
        memory_state=None,
        *,
        update_memory: bool = True,
        write_mask: torch.Tensor | None = None,
        ablation: AblationConfig = NORMAL_ABLATION,
        trace: AblationTrace | None = None,
    ):
        if memory_state is None:
            memory_state = self.new_memory_state(tokens.shape[0])
        if not ablation.carry_enabled:
            memory_state = self.new_memory_state(tokens.shape[0])
        if trace is not None and memory_state is not None:
            trace.record("block_input_keys", memory_state.keys)
            trace.record("block_input_values", memory_state.values)
            trace.record("block_input_strength", memory_state.strength)
        embedding = self.embed(tokens)
        cell_input = self.cell.control_input(embedding)
        state = torch.zeros_like(embedding)
        for _ in range(self.cfg.k):
            state = self.cell.step_with_memory(
                state,
                cell_input,
                memory_state,
                retrieval_enabled=ablation.retrieval_enabled,
                local_mask=write_mask,
                trace=trace,
            )
        next_memory = (
            self.cell.observe(
                memory_state,
                state,
                mask=write_mask,
                write_enabled=ablation.write_enabled,
                trace=trace,
            )
            if update_memory
            else memory_state
        )
        return self.readout(state), next_memory

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_state(tokens)
        return logits

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        chunks = tokens.split(self.phylo_config.chunk_size, dim=1)
        memory = self.new_memory_state(tokens.shape[0])
        logits = []
        for chunk in chunks:
            chunk_logits, memory = self.forward_with_state(chunk, memory)
            logits.append(chunk_logits)
        joined = torch.cat(logits, dim=1)
        return F.cross_entropy(
            joined[:, :-1].reshape(-1, self.cfg.vocab_size),
            tokens[:, 1:].reshape(-1),
        )
