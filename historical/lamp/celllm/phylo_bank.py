"""Fixed two-level plastic associative store used by HAA-Phylo."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class PhyloState:
    """The complete recurrent HAA-Phylo state."""

    keys: torch.Tensor
    values: torch.Tensor
    strength: torch.Tensor

    def reset(self) -> PhyloState:
        """Return an independent zero-history state with the same anchors."""
        return PhyloState(
            self.keys,
            torch.zeros_like(self.values),
            torch.ones_like(self.strength),
        )


class PhyloBank(nn.Module):
    """Dense G x L hierarchical memory with synchronous Hebbian writes."""

    def __init__(
        self,
        dimensions: int,
        *,
        groups: int = 8,
        leaves_per_group: int = 8,
        read_temperature: float = 1.0,
        write_temperature: float = 1.0,
        key_rate: float = 0.1,
        value_rate: float = 0.1,
        retention: float = 0.99,
        initial_strength: float = 1.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if groups < 1 or leaves_per_group < 1 or dimensions < 1:
            raise ValueError("PhyloBank dimensions must be positive")
        if read_temperature <= 0 or write_temperature <= 0:
            raise ValueError("routing temperatures must be positive")
        if key_rate < 0 or value_rate < 0:
            raise ValueError("plastic rates must be non-negative")
        if not 0 < retention <= 1:
            raise ValueError("retention must be in (0, 1]")
        if initial_strength <= 0 or epsilon <= 0:
            raise ValueError("initial strength and epsilon must be positive")
        self.dimensions = dimensions
        self.groups = groups
        self.leaves_per_group = leaves_per_group
        self.read_temperature = read_temperature
        self.write_temperature = write_temperature
        self.key_rate = key_rate
        self.value_rate = value_rate
        self.retention = retention
        self.initial_strength = initial_strength
        self.epsilon = epsilon
        self.anchors = nn.Parameter(torch.empty(groups, leaves_per_group, dimensions))
        nn.init.normal_(self.anchors, std=dimensions**-0.5)

    @property
    def capacity(self) -> int:
        return self.groups * self.leaves_per_group

    def new_state(
        self, batch_size: int, device=None, dtype=None
    ) -> PhyloState:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        reference = self.anchors
        device = reference.device if device is None else device
        dtype = reference.dtype if dtype is None else dtype
        keys = F.normalize(reference.to(device=device, dtype=dtype), dim=-1)
        keys = keys.unsqueeze(0).expand(batch_size, -1, -1, -1)
        values = torch.zeros_like(keys)
        strength = torch.full(
            (batch_size, self.groups, self.leaves_per_group),
            self.initial_strength,
            device=device,
            dtype=dtype,
        )
        return PhyloState(keys, values, strength)

    def group_prototypes(self, state: PhyloState) -> torch.Tensor:
        weighted = (state.strength[..., None] * state.keys).sum(dim=2)
        total = state.strength.sum(dim=2, keepdim=True)
        return F.normalize(weighted / (total + self.epsilon), dim=-1)

    def route_groups(self, state: PhyloState, vectors: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(vectors, dim=-1, eps=self.epsilon)
        scores = torch.einsum("bnh,bgh->bng", normalized, self.group_prototypes(state))
        return scores.argmax(dim=-1)

    @staticmethod
    def _gather_groups(tensor: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        batch = torch.arange(tensor.shape[0], device=tensor.device)[:, None]
        return tensor[batch, groups]

    def read(self, state: PhyloState, query: torch.Tensor) -> torch.Tensor:
        if query.ndim != 3 or query.shape[-1] != self.dimensions:
            raise ValueError("query must have shape [B, n, h]")
        normalized = F.normalize(query, dim=-1, eps=self.epsilon)
        groups = self.route_groups(state, normalized)
        keys = self._gather_groups(state.keys, groups)
        values = self._gather_groups(state.values, groups)
        scores = torch.einsum("bnh,bnlh->bnl", normalized, keys)
        weights = torch.softmax(scores / self.read_temperature, dim=-1)
        return torch.einsum("bnl,bnlh->bnh", weights, values)

    def write(
        self,
        state: PhyloState,
        key: torch.Tensor,
        value: torch.Tensor,
        rate: torch.Tensor,
    ) -> PhyloState:
        """Commit all candidates together against the same pre-write state."""
        if key.ndim != 3 or key.shape != value.shape:
            raise ValueError("key and value must share shape [B, n, h]")
        if key.shape[-1] != self.dimensions:
            raise ValueError("key width does not match the PhyloBank")
        if rate.shape == key.shape[:2] + (1,):
            rate = rate.squeeze(-1)
        if rate.shape != key.shape[:2]:
            raise ValueError("rate must have shape [B, n] or [B, n, 1]")

        normalized = F.normalize(key, dim=-1, eps=self.epsilon)
        groups = self.route_groups(state, normalized)
        selected_keys = self._gather_groups(state.keys, groups)
        scores = torch.einsum("bnh,bnlh->bnl", normalized, selected_keys)
        leaf_weights = torch.softmax(scores / self.write_temperature, dim=-1)
        assignment = F.one_hot(groups, self.groups).to(key.dtype)
        mass_by_group = rate.clamp(0, 1)[..., None, None] * assignment[..., :, None]
        mass_by_group = mass_by_group * leaf_weights[..., None, :]

        mass = mass_by_group.sum(dim=1)
        key_sum = torch.einsum("bngl,bnh->bglh", mass_by_group, normalized)
        value_sum = torch.einsum("bngl,bnh->bglh", mass_by_group, value)
        denominator = mass[..., None] + self.epsilon
        mean_key = key_sum / denominator
        mean_value = value_sum / denominator

        key_alpha = 1.0 - torch.exp(-self.key_rate * mass)
        value_alpha = 1.0 - torch.exp(-self.value_rate * mass)
        next_keys = F.normalize(
            (1.0 - key_alpha[..., None]) * state.keys
            + key_alpha[..., None] * mean_key,
            dim=-1,
            eps=self.epsilon,
        )
        next_values = state.values + value_alpha[..., None] * (
            mean_value - state.values
        )
        next_strength = self.retention * state.strength + mass
        return PhyloState(next_keys, next_values, next_strength)

    def diagnostics(self, state: PhyloState, vectors: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return collapse-sensitive routing and geometry measurements."""
        normalized = F.normalize(vectors, dim=-1, eps=self.epsilon)
        groups = self.route_groups(state, normalized)
        group_usage = F.one_hot(groups, self.groups).float().mean(dim=(0, 1))
        selected = self._gather_groups(state.keys, groups)
        leaf_scores = torch.einsum("bnh,bnlh->bnl", normalized, selected)
        leaf_weights = torch.softmax(leaf_scores / self.read_temperature, dim=-1)
        group_assignment = F.one_hot(groups, self.groups).float()
        slot_usage = torch.einsum(
            "bng,bnl->gl", group_assignment, leaf_weights
        ) / float(groups.numel())
        entropy = lambda p: -(p * p.clamp_min(self.epsilon).log()).sum()
        keys = state.keys
        within = torch.einsum("bglh,bgmh->bglm", keys, keys)
        prototypes = self.group_prototypes(state)
        cross = torch.einsum("bgh,bjh->bgj", prototypes, prototypes)
        return {
            "group_usage": group_usage,
            "leaf_usage": slot_usage,
            "group_routing_entropy": entropy(group_usage),
            "leaf_routing_entropy": entropy(slot_usage),
            "strength_min": state.strength.min(),
            "strength_mean": state.strength.mean(),
            "strength_max": state.strength.max(),
            "within_group_key_similarity": within.mean(),
            "cross_group_key_similarity": cross.mean(),
        }
