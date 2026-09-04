"""CellLM-v3-HAA-Phylo-600M-Chat: the frozen architecture for the chat run.

One axis moves from the ctx2048 study: vocab_size 32768 -> 16384. The core is
untouched -- 120,777,730 parameters against the 600M run's 120,794,114, a
0.01% difference -- so this is a pure reallocation of lexical capacity, not a
smaller model in the sense that matters for reasoning.

Three measurements decided it, and only the third is the real reason:

  * bytes/token on the materialised mixture: 4.495 (32k) -> 4.144 (16k), so
    the same text costs 8.5% more tokens. Portuguese pays 13.0%.
  * throughput with the core held fixed: 16k buys about 8% more tok/s, since
    k=16 refinement dominates and the LM head is a small share. Those two
    cancel: the same corpus costs the same wall clock either way.
  * optimiser state: 239M fewer parameters frees roughly 2.4 GiB per rank
    (bf16 weights plus fp32 Adam m and v), which buys micro-batch.

The 32k vocabulary was NOT wasted -- 31,268 of 32,768 tokens clear 1e-6
frequency on the corpus, and only 21.9M parameters sit in the rare tail. The
case for 16k is memory, not waste. Numbers above come from the 32k tokenizer;
bytes/token must be remeasured once the 16k tokenizer is trained.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DemoConfig:
    name: str = "CellLM-v3-HAA-Phylo-600M-Chat"
    vocab_size: int = 16_384
    d: int = 24_320
    h: int = 1_024
    n: int = 64
    k: int = 16
    r: int = 4
    m: int = 2
    dt: float = 0.5
    mixer: str = "none"
    groups: int = 32
    leaves_per_group: int = 32
    read_temperature: float = 1.0
    write_temperature: float = 1.0
    key_rate: float = 0.1
    value_rate: float = 0.1
    retention: float = 0.99
    read_before_write: bool = True
    exact_carry: bool = True
    tied_embedding_head: bool = True
    plastic_dtype: str = "bfloat16"
    model_dtype: str = "bfloat16"
    seed: int = 2026
    sequence_length: int = 64
    # 2048 measured at 5.72 GiB on an 8 GiB card with block checkpointing.
    context_tokens: int = 2_048
    # Both are preflight outputs, not architecture. run_preflight.py reports
    # the largest micro_batch that fits; changing them does not change the LR
    # trajectory, because the schedule is driven by tokens_seen (schedule.py).
    #: False omits the HAA-Phylo projections at construction -- the arm that
    #: fills the empty cell of stage 18's square. It is architecture, so it is
    #: frozen and inside the fingerprint: a checkpoint from one value must
    #: never load into the other.
    global_memory: bool = True
    micro_batch_size: int = 4
    world_size: int = 4
    walltime_hours: float = 12.0

    #: Sizing knobs the preflight is allowed to set. They change how the work
    #: is packed onto GPUs, never what is computed: the LR trajectory follows
    #: tokens_seen / target_tokens, so a different micro-batch or world size
    #: reproduces the same run.
    TUNABLE = ("micro_batch_size", "world_size", "walltime_hours")

    def validate(self) -> None:
        # Each arm is its own frozen architecture, so compare against the
        # canonical config FOR THIS ARM. global_memory is not a tunable: a
        # checkpoint from one arm must never load into the other, which is why
        # it sits outside TUNABLE and inside the fingerprint below.
        expected = asdict(DemoConfig(global_memory=self.global_memory))
        changed = {
            key: (expected[key], value)
            for key, value in asdict(self).items()
            if value != expected[key] and key not in self.TUNABLE
        }
        if changed:
            raise ValueError(f"frozen architecture changed: {changed}")
        if self.micro_batch_size < 1 or self.world_size < 1:
            raise ValueError("micro_batch_size and world_size must be positive")
        # Capacity is deliberately decoupled from n here; that is the study.
        if self.groups < 1 or self.leaves_per_group < 1:
            raise ValueError("HAA-Phylo capacity must be positive")
        if self.context_tokens % self.sequence_length:
            raise ValueError("context_tokens must be a multiple of sequence_length")
        if self.context_tokens < 2 * self.sequence_length:
            raise ValueError("exact carry needs at least two blocks")
        if self.k * self.r + 1 < self.n:
            raise ValueError("causal cone does not cover all lattice cells")
        total = parameter_report(self)["trainable_unique"]
        lo, hi = (580_000_000, 620_000_000) if self.global_memory else (480_000_000, 520_000_000)
        if not lo <= total <= hi:
            raise ValueError(
                f"parameter gate failed: {total:,} outside [{lo:,}; {hi:,}] "
                f"for global_memory={self.global_memory}"
            )

    def fingerprint(self) -> str:
        """Hash the architecture only, so preflight sizing never moves it."""
        fields = {k: v for k, v in asdict(self).items() if k not in self.TUNABLE}
        # The with-HAA architecture predates this field. Adding a flag that
        # names the existing behaviour must not change that architecture's
        # identity, or every checkpoint already trained -- job 12038 included --
        # would stop matching its own config and refuse to resume. So the flag
        # enters the hash only when it selects the new arm.
        if fields.pop("global_memory", True):
            pass
        else:
            fields["global_memory"] = False
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def parameter_report(cfg: DemoConfig) -> dict[str, object]:
    """Exact unique-parameter count for the current v3 model."""
    d, h, vocab = cfg.d, cfg.h, cfg.vocab_size
    components = {
        "embedding_and_tied_lm_head": vocab * d,
        "lm_head_bias": vocab,
        "chua_yang_core": (2 * (cfg.r + 1) + 1) * d,
        "haa_projections": 4 * d * h if cfg.global_memory else 0,
        "haa_write_rate": (d + 1) if cfg.global_memory else 0,
        "phylo_anchors": (cfg.groups * cfg.leaves_per_group * h
                          if cfg.global_memory else 0),
        "local_projections": 4 * d * h,
        "local_retrieval_scale": 1,
    }
    total = sum(components.values())
    return {
        "trainable_unique": total,
        "registered_unique": total,
        "frozen_dormant_local_gate": 0,
        "shared_weight_counted_once": True,
        "in_600m_gate": 580_000_000 <= total <= 620_000_000,
        "components": components,
    }


def active_config() -> "DemoConfig":
    """The arm this process trains, selected by CELLLM_GLOBAL_MEMORY.

    Architecture is frozen per arm, so the choice cannot live in a CLI flag
    that a resubmission might forget: it is read once, echoed, and lands in the
    fingerprint, so a checkpoint from one arm cannot load into the other.
    """
    import os

    raw = os.environ.get("CELLLM_GLOBAL_MEMORY", "1").strip().lower()
    if raw not in {"0", "1", "true", "false"}:
        raise SystemExit(
            f"CELLLM_GLOBAL_MEMORY must be 0/1/true/false, got {raw!r}"
        )
    import dataclasses

    return dataclasses.replace(
        DemoConfig(), global_memory=raw in {"1", "true"}
    )


def architecture_name(cfg: "DemoConfig") -> str:
    """Stable checkpoint label for each physically distinct arm."""
    return (
        "celllm-v3-haa-phylo-600m"
        if cfg.global_memory
        else "celllm-v3-local-only-498m"
    )


def build_model(cfg: "DemoConfig | None" = None):
    """Build v3 from the source tree copied into the submission bundle.

    Takes an explicit config so the no-HAA arm can be constructed without
    monkeypatching this module -- which silently broke validate(), since that
    needs the real DemoConfig to know what the arm should look like.
    """
    from celllm.chat_model import CellLMChatModel
    from celllm.config import LocalAssociativeConfig, ModelConfig, PhyloConfig

    cfg = cfg or active_config()
    cfg.validate()
    # Construct the historical full arm first even for the no-HAA baseline.
    # The original run consumed RNG while initialising HAA *before* LOCAL. If
    # we simply skip those modules, the same seed gives LOCAL different random
    # weights and a one-seed comparison confounds architecture with a different
    # initialisation. Dropping the module after initialisation preserves every
    # shared tensor and the post-build RNG state bit-for-bit while still
    # returning a physical 498M model with no HAA parameter or optimiser state.
    model = CellLMChatModel(
        ModelConfig(
            n=cfg.n, d=cfg.d, r=cfg.r, k=cfg.k, eta=cfg.dt,
            vocab_size=cfg.vocab_size, spatial="diagonal", mixer=cfg.mixer,
        ),
        phylo=PhyloConfig(
            groups=cfg.groups, leaves_per_group=cfg.leaves_per_group,
            associative_size=cfg.h, read_temperature=cfg.read_temperature,
            write_temperature=cfg.write_temperature, key_rate=cfg.key_rate,
            value_rate=cfg.value_rate, retention=cfg.retention,
            chunk_size=cfg.n,
        ),
        local=LocalAssociativeConfig(
            radius=cfg.m, key_size=cfg.h, value_size=cfg.h,
            retrieval_scale=0.1, learnable_retrieval_scale=True,
        ),
        global_memory=True,
    )
    if not cfg.global_memory:
        model.core.cell.attention = None
        model.core.global_memory = False
    return model


def assert_runtime(model) -> None:
    """Fail before distributed training if runtime differs from the protocol."""
    cfg = active_config()
    cfg.validate()
    core = model.core
    # The arm actually built must be the arm selected. Getting this wrong
    # trains the other architecture for twelve hours and the loss curve looks
    # perfectly healthy while doing it.
    has_haa = core.cell.attention is not None
    if has_haa != cfg.global_memory:
        raise AssertionError(
            f"CELLLM_GLOBAL_MEMORY selected global_memory={cfg.global_memory} "
            f"but the constructed model {'has' if has_haa else 'lacks'} HAA"
        )
    assert core.cfg.mixer == "none"
    assert sum(p.numel() for p in core.cell.mixer.parameters()) == 0
    assert core.cfg.eta == cfg.dt and core.cfg.k == cfg.k and core.cfg.r == cfg.r
    assert core.phylo_config.groups == cfg.groups
    assert core.phylo_config.leaves_per_group == cfg.leaves_per_group
    assert core.phylo_config.associative_size == cfg.h
    assert core.phylo_config.chunk_size == cfg.n
    assert core.local_config.radius == cfg.m
    assert core.local_config.key_size == cfg.h
    assert core.local_config.value_size == cfg.h
    assert core.readout.weight is core.embed.weight
    actual = sum(p.numel() for p in model.parameters())
    expected = parameter_report(cfg)["trainable_unique"]
    assert actual == expected, f"runtime parameters {actual:,} != {expected:,}"
