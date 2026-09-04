"""CellLM adapter around the canonical libPyCelNN differentiable dynamics."""

from __future__ import annotations

import torch
from celnn import DifferentiableCellularNetwork
from celnn import piecewise_linear as _piecewise_linear
from torch import nn

from celllm.ablation import AblationTrace
from celllm.associative_field import LocalAssociativeMessagePassing
from celllm.config import (
    LocalAssociativeConfig,
    ModelConfig,
    PhyloConfig,
)
from celllm.hebbian_attention import HebbianAttention
from celllm.mixers import build_mixer


def piecewise_linear(x: torch.Tensor) -> torch.Tensor:
    """Expose the libPyCelNN canonical saturation for CellLM callers."""
    return _piecewise_linear(x)


class CelNNCell(nn.Module):
    """Compose libPyCelNN dynamics with caller-owned channel mixing."""

    def __init__(self, cfg: ModelConfig, causal: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.dynamics = DifferentiableCellularNetwork(
            radius=cfg.r,
            channels=cfg.d,
            activation="piecewise_linear",
            boundary="constant",
            dt=cfg.eta,
            steps=1,
            method="euler",
            causal=causal,
            shared_channels=cfg.spatial == "scalar",
        )
        with torch.no_grad():
            self.dynamics.feedback.normal_(std=0.1)
            self.dynamics.control.normal_(std=0.1)
        self.mixer = build_mixer(cfg.mixer, cfg.d)

    def control_input(self, embedding: torch.Tensor) -> torch.Tensor:
        """Bound learned embeddings to the classical CelNN input range."""
        return torch.tanh(embedding) if self.cfg.bound_drive else embedding

    def step(self, x: torch.Tensor, cell_input: torch.Tensor) -> torch.Tensor:
        """Advance through libPyCelNN with an external channel-mixing drive."""
        return self.dynamics.step(
            x,
            cell_input,
            extra_drive=self.mixer(x),
        )


class PhyloCelNNCell(CelNNCell):
    """CellNN dynamics with global HAA-Phylo and validated LOCAL messages."""

    def __init__(
        self,
        cfg: ModelConfig,
        phylo: PhyloConfig,
        local: LocalAssociativeConfig | None = None,
        causal: bool = True,
        global_memory: bool = True,
    ) -> None:
        super().__init__(cfg, causal=causal)
        # global_memory=False omits the HAA-Phylo projections entirely rather
        # than zeroing their output. Stage 18 measured the bypass path, which
        # answers "can this checkpoint be served without them"; omitting them
        # at construction is what a from-scratch baseline needs, and it is the
        # only way the parameters and their optimiser state actually go away.
        self.attention = None if not global_memory else HebbianAttention(
            cfg.d,
            phylo.associative_size,
            groups=phylo.groups,
            leaves_per_group=phylo.leaves_per_group,
            read_temperature=phylo.read_temperature,
            write_temperature=phylo.write_temperature,
            key_rate=phylo.key_rate,
            value_rate=phylo.value_rate,
            retention=phylo.retention,
        )
        self.local_attention = (
            None
            if local is None
            else LocalAssociativeMessagePassing(cfg.d, local)
        )

    def new_memory_state(self, batch_size: int):
        if self.attention is None:
            return None
        return self.attention.new_state(batch_size)

    def step_with_memory(
        self,
        x: torch.Tensor,
        cell_input: torch.Tensor,
        memory_state,
        *,
        retrieval_enabled: bool = True,
        local_mask: torch.Tensor | None = None,
        trace: AblationTrace | None = None,
    ) -> torch.Tensor:
        next_state, _ = self.refine_with_memory(
            x,
            cell_input,
            memory_state,
            retrieval_enabled=retrieval_enabled,
            local_mask=local_mask,
            trace=trace,
        )
        return next_state

    def refine_with_memory(
        self,
        x: torch.Tensor,
        cell_input: torch.Tensor,
        memory_state,
        *,
        retrieval_enabled: bool = True,
        local_mask: torch.Tensor | None = None,
        trace: AblationTrace | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance once and expose the retrieval drive for diagnostics."""
        output = piecewise_linear(x)
        memory_drive = (
            None
            if self.attention is None
            else self.attention.retrieve(memory_state, output)
        )
        if memory_drive is not None and not retrieval_enabled:
            memory_drive = torch.zeros_like(memory_drive)
        if trace is not None and memory_drive is not None:
            trace.record("retrieval_drive", memory_drive)
        local_drive = (
            None
            if self.local_attention is None
            else self.local_attention(output, memory_drive, mask=local_mask)
        )
        if trace is not None and local_drive is not None:
            trace.record("local_retrieval_drive", local_drive)
        # Sum only the terms that exist. Adding a zero tensor is arithmetically
        # free but not memory-free: at d=24320 each one is 3 MB, allocated k
        # times per block.
        extra = self.mixer(x)
        if memory_drive is not None:
            extra = extra + memory_drive
        if local_drive is not None:
            extra = extra + local_drive
        next_state = self.dynamics.step(x, cell_input, extra_drive=extra)
        # The drive is returned for diagnostics and the only caller discards
        # it. Materialising a zero tensor here just to keep the type uniform
        # cost 1.25 GiB of activations across k steps and 32 blocks at
        # d=24320 -- more than the parameters the arm was meant to save.
        return next_state, memory_drive

    def observe(
        self,
        memory_state,
        activity: torch.Tensor,
        mask=None,
        *,
        write_enabled: bool = True,
        trace: AblationTrace | None = None,
    ):
        if self.attention is None:
            updated = memory_state          # stays None; nothing to carry
        else:
            updated = self.attention.observe(
                memory_state, piecewise_linear(activity), mask=mask
            ) if write_enabled else memory_state
        if trace is not None and memory_state is not None:
            trace.record_delta(
                "write_keys_delta", memory_state.keys, updated.keys
            )
            trace.record_delta(
                "write_values_delta", memory_state.values, updated.values
            )
        return updated
