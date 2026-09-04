"""Stateless causal LOCAL associative path for CellLM-v3."""

from __future__ import annotations

import torch
from torch import nn

from celllm.config import LocalAssociativeConfig


class LocalAssociativeMessagePassing(nn.Module):
    """UNGATED normalized associative messages from preceding cells."""

    def __init__(self, dimensions: int, config: LocalAssociativeConfig) -> None:
        super().__init__()
        self.dimensions = int(dimensions)
        self.config = config
        self.query = nn.Linear(dimensions, config.key_size, bias=False)
        self.key = nn.Linear(dimensions, config.key_size, bias=False)
        self.value = nn.Linear(dimensions, config.value_size, bias=False)
        self.output = nn.Linear(config.value_size, dimensions, bias=False)
        scale = torch.tensor([float(config.retrieval_scale)])
        if config.learnable_retrieval_scale:
            self.retrieval_scale = nn.Parameter(scale)
        else:
            self.register_buffer("retrieval_scale", scale)

    @staticmethod
    def _features(projected: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.elu(projected.float()) + 1.0

    def forward(
        self,
        activity: torch.Tensor,
        global_drive: torch.Tensor | None,
        *,
        mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if activity.ndim != 3:
            raise ValueError("activity must have shape (batch, cells, features)")
        # global_drive is only shape-checked; LOCAL never reads its content.
        # None is therefore exact, not an approximation, and it lets the no-HAA
        # arm skip allocating a zero tensor of (batch, cells, d) on every one of
        # the k refinement steps of every block.
        if global_drive is not None and global_drive.shape != activity.shape:
            raise ValueError("global drive must match activity")
        if mask is not None and mask.shape != activity.shape[:2]:
            raise ValueError("local mask must match batch and cells")
        batch, cells, _ = activity.shape
        query = self._features(self.query(activity))
        keys = self._features(self.key(activity))
        values = torch.tanh(self.value(activity)).float()
        numerator = activity.new_zeros(
            (batch, cells, self.config.value_size), dtype=torch.float32
        )
        denominator = activity.new_zeros((batch, cells), dtype=torch.float32)
        edge_weights = activity.new_zeros(
            (batch, cells, self.config.radius), dtype=torch.float32
        )
        positions = torch.arange(cells, device=activity.device)
        for offset in range(1, self.config.radius + 1):
            source_positions = (positions - offset).clamp_min(0)
            source_keys = keys.index_select(1, source_positions)
            source_values = values.index_select(1, source_positions)
            valid = positions.ge(offset)[None].expand(batch, -1)
            if mask is not None:
                valid = (
                    valid
                    & mask.bool()
                    & mask.index_select(1, source_positions).bool()
                )
            kernel = (query * source_keys).sum(dim=-1) / self.config.key_size
            weight = kernel.float() * valid.to(torch.float32)
            numerator = numerator + weight[..., None] * source_values
            denominator = denominator + weight
            edge_weights[..., offset - 1] = weight
        retrieved = numerator / (denominator[..., None] + self.config.epsilon)
        drive = self.retrieval_scale * self.output(retrieved.to(activity.dtype))
        if mask is not None:
            drive = drive * mask[..., None].to(drive.dtype)
        if return_weights:
            normalized = edge_weights / (
                denominator[..., None] + self.config.epsilon
            )
            return drive, normalized
        return drive
