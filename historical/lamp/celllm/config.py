"""Configuration objects for Experiment 0."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Shape and dynamics of a CellLM capacity-gate model.

    The receptive-field constraint ``k * r + 1 >= n`` keeps locality out of
    Experiment 0: with a causal neighbourhood, the reach after ``k`` Euler
    steps is ``k * r + 1``. A failure can therefore be attributed to capacity
    rather than distance.
    """

    n: int = 64
    d: int = 128
    r: int = 2
    k: int = 32
    eta: float = 0.5
    vocab_size: int = 27
    spatial: str = "diagonal"
    mixer: str = "none"
    bound_drive: bool = True

    def __post_init__(self) -> None:
        if self.k * self.r + 1 < self.n:
            minimum_k = -(-(self.n - 1) // self.r)
            raise ValueError(
                f"receptive field k*r+1={self.k * self.r + 1} does not cover "
                f"n={self.n}; raise k to at least {minimum_k}"
            )
        if self.spatial not in {"scalar", "diagonal"}:
            raise ValueError(f"unknown spatial template {self.spatial!r}")

    @property
    def offsets(self) -> tuple[int, ...]:
        """Causal neighbourhood offsets, from ``-r`` through zero."""
        return tuple(range(-self.r, 1))

    @property
    def n_offsets(self) -> int:
        """Number of offsets in the causal neighbourhood."""
        return self.r + 1


@dataclass(frozen=True)
class TrainConfig:
    """Training defaults for a capacity-ladder run."""

    steps: int = 20_000
    batch_size: int = 128
    lr: float = 3e-3
    warmup: int = 500
    grad_clip: float = 1.0
    seed: int = 42
    eval_every: int = 1_000
    eval_batches: int = 50


@dataclass(frozen=True)
class PhyloConfig:
    """The deliberately fixed two-level HAA-Phylo topology."""

    groups: int = 8
    leaves_per_group: int = 8
    associative_size: int = 32
    read_temperature: float = 1.0
    write_temperature: float = 1.0
    key_rate: float = 0.1
    value_rate: float = 0.1
    retention: float = 0.99
    chunk_size: int = 64

    def __post_init__(self) -> None:
        # 8 x 8 is the frozen showcase topology and stays the default. The
        # capacity study needs other shapes, and phylo_bank contracts groups,
        # leaves and n as independent indices, so nothing else assumes 8 x 8.
        if self.groups < 1 or self.leaves_per_group < 1:
            raise ValueError("PhyloBank topology must be positive")
        if self.associative_size < 1:
            raise ValueError("associative_size must be positive")
        if self.read_temperature <= 0 or self.write_temperature <= 0:
            raise ValueError("routing temperatures must be positive")
        if self.key_rate < 0 or self.value_rate < 0:
            raise ValueError("plastic rates must be non-negative")
        if not 0 < self.retention <= 1:
            raise ValueError("retention must be in (0, 1]")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")


@dataclass(frozen=True)
class LocalAssociativeConfig:
    """Stateless causal associative messages inside one native block."""

    radius: int = 2
    key_size: int = 32
    value_size: int = 32
    retrieval_scale: float = 0.1
    learnable_retrieval_scale: bool = True
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.radius < 1:
            raise ValueError("local associative radius must be positive")
        if self.key_size < 1 or self.value_size < 1:
            raise ValueError("local key and value sizes must be positive")
        if self.retrieval_scale < 0:
            raise ValueError("local retrieval scale must be non-negative")
        if self.epsilon <= 0:
            raise ValueError("local epsilon must be positive")
