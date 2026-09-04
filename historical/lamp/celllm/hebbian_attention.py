"""Projected Hierarchical Hebbian Associative Attention."""

from __future__ import annotations

import torch
from torch import nn

from celllm.phylo_bank import PhyloBank, PhyloState


class HebbianAttention(nn.Module):
    """Learn projections around the recurrent, fixed-capacity PhyloBank."""

    def __init__(self, input_size: int, associative_size: int, **bank_options) -> None:
        super().__init__()
        self.query = nn.Linear(input_size, associative_size, bias=False)
        self.key = nn.Linear(input_size, associative_size, bias=False)
        self.value = nn.Linear(input_size, associative_size, bias=False)
        self.output = nn.Linear(associative_size, input_size, bias=False)
        self.write_rate = nn.Linear(input_size, 1)
        nn.init.zeros_(self.write_rate.weight)
        nn.init.constant_(self.write_rate.bias, -1.0)
        self.phylo_bank = PhyloBank(associative_size, **bank_options)

    def new_state(self, batch_size: int, device=None, dtype=None) -> PhyloState:
        return self.phylo_bank.new_state(batch_size, device=device, dtype=dtype)

    def associations(self, activity: torch.Tensor):
        return (
            self.key(activity),
            torch.tanh(self.value(activity)),
            torch.sigmoid(self.write_rate(activity)),
        )

    def read(self, state: PhyloState, query: torch.Tensor) -> torch.Tensor:
        return self.phylo_bank.read(state, query)

    def write(
        self,
        state: PhyloState,
        key: torch.Tensor,
        value: torch.Tensor,
        rate: torch.Tensor,
    ) -> PhyloState:
        return self.phylo_bank.write(state, key, value, rate)

    def retrieve(self, state: PhyloState, activity: torch.Tensor) -> torch.Tensor:
        return self.output(self.read(state, self.query(activity)))

    def observe(self, state: PhyloState, activity: torch.Tensor, mask=None) -> PhyloState:
        key, value, rate = self.associations(activity)
        if mask is not None:
            rate = rate * mask[..., None].to(rate.dtype)
        return self.write(state, key, value, rate)

    def forward(
        self,
        state: PhyloState,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rate: torch.Tensor,
        update_memory: bool = True,
    ):
        retrieved = self.read(state, query)
        if update_memory:
            state = self.write(state, key, value, rate)
        return retrieved, state
