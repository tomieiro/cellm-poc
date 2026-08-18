"""A parameter-matched decoder-only Transformer, as a control for CellLM.

The comparison is only meaningful if everything except the architecture is held
fixed, so this reuses the D20 tokenizer, corpus, splits, budget and evaluation
verbatim. What it deliberately does *not* match is compute:

* CellLM applies its shared cell ``k = 15`` times per token, so at equal
  parameter count it performs far more work per token;
* CellLM sees 16 tokens directly and reaches further only through its
  associative state, while this Transformer attends over the whole 256-token
  window.

Both differences favour a different side, and both are reported rather than
argued away. A parameter-matched result is a statement about parameters, not
about compute or context.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    dimensions: int
    layers: int
    heads: int
    context: int = 256
    # 0 keeps the full causal mask. A positive value limits each position to
    # that many tokens including itself, which is the control for CellLM's
    # 16-cell direct window: it isolates *selecting* from *seeing far*.
    attention_window: int = 0
    ffn_multiplier: int = 4
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.dimensions % self.heads:
            raise ValueError("dimensions must divide evenly among heads")


class Block(nn.Module):
    """Pre-norm attention + MLP, the standard decoder block."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(cfg.dimensions)
        self.attention = nn.MultiheadAttention(
            cfg.dimensions, cfg.heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(cfg.dimensions)
        hidden = cfg.ffn_multiplier * cfg.dimensions
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dimensions, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.dimensions),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm_attention(x)
        attended, _ = self.attention(
            normed, normed, normed, attn_mask=mask, need_weights=False
        )
        x = x + attended
        return x + self.mlp(self.norm_mlp(x))


class BaselineTransformer(nn.Module):
    """Decoder-only Transformer exposing the same surface as CellLMChatModel.

    ``sequence_logits``, ``loss`` and ``loss_statistics`` mirror the CellLM API
    so the existing trainer and evaluators work against either model without a
    branch.
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dimensions)
        self.positions = nn.Embedding(cfg.context, cfg.dimensions)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.dimensions)
        self.readout = nn.Linear(cfg.dimensions, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            # CellLM ties its readout to the embedding table, so the control
            # does too; untied would hand the Transformer a second vocabulary
            # matrix and break the parameter match.
            self.readout.weight = self.embed.weight
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    @property
    def chunk_size(self) -> int:
        return self.cfg.context

    def sequence_logits(self, token_ids: torch.Tensor, **_) -> torch.Tensor:
        batch, length = token_ids.shape
        if length > self.cfg.context:
            raise ValueError(
                f"sequence of {length} exceeds the {self.cfg.context}-token "
                "context; the CellLM comparison uses 256"
            )
        positions = torch.arange(length, device=token_ids.device)
        x = self.embed(token_ids) + self.positions(positions)[None]
        index = torch.arange(length, device=token_ids.device)
        delta = index[:, None] - index[None, :]
        visible = delta >= 0
        if self.cfg.attention_window:
            visible = visible & (delta < self.cfg.attention_window)
        mask = torch.zeros(length, length, device=token_ids.device)
        mask = mask.masked_fill(~visible, float("-inf"))
        for block in self.blocks:
            x = block(x, mask)
        return self.readout(self.norm(x))

    def loss_statistics(
        self, token_ids: torch.Tensor, target_mask: torch.Tensor, **_
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predictions = self.sequence_logits(token_ids)[:, :-1]
        targets = token_ids[:, 1:]
        mask = target_mask[:, 1:]
        if not torch.any(mask):
            raise ValueError("batch has no targets")
        losses = F.cross_entropy(
            predictions.reshape(-1, self.cfg.vocab_size),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        return (losses * mask).sum(dim=1), mask.sum(dim=1)

    def loss(
        self, token_ids: torch.Tensor, target_mask: torch.Tensor, **_
    ) -> torch.Tensor:
        sums, counts = self.loss_statistics(token_ids, target_mask)
        return sums.sum() / counts.sum()

    def new_plastic_state(self, batch: int):
        """No recurrent state; present so shared code paths do not branch."""
        return None


def count_parameters(model: BaselineTransformer) -> dict:
    groups = {"embedding": 0, "position": 0, "attention": 0, "mlp": 0, "norm": 0}
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        if "embed" in name:
            groups["embedding"] += count
        elif "positions" in name:
            groups["position"] += count
        elif "attention" in name:
            groups["attention"] += count
        elif "mlp" in name:
            groups["mlp"] += count
        else:
            groups["norm"] += count
    total = sum(p.numel() for p in model.parameters())
    embedding_like = groups["embedding"] + groups["position"]
    return {
        "total": total,
        "by_component": groups,
        "embedding_parameters": embedding_like,
        "embedding_fraction": embedding_like / total,
        "non_embedding_parameters": total - embedding_like,
    }


def match_parameters(
    target: int,
    vocab_size: int,
    *,
    dimensions: tuple[int, ...] = (128, 192, 256, 320, 384, 448, 512),
    layers: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10, 12),
    heads_options: tuple[int, ...] = (4, 8),
    context: int = 256,
    tolerance: float = 0.02,
) -> dict:
    """Find the shape closest to ``target`` parameters.

    Depth is a free knob here, unlike CellLM, which has none. That asymmetry is
    part of what the comparison is about, so the search reports the chosen depth
    rather than pinning it.
    """
    candidates = []
    for dimension in dimensions:
        for layer_count in layers:
            for heads in heads_options:
                if dimension % heads:
                    continue
                cfg = TransformerConfig(
                    vocab_size=vocab_size, dimensions=dimension,
                    layers=layer_count, heads=heads, context=context,
                )
                report = count_parameters(BaselineTransformer(cfg))
                candidates.append(
                    {
                        "dimensions": dimension,
                        "layers": layer_count,
                        "heads": heads,
                        "total_parameters": report["total"],
                        "embedding_fraction": report["embedding_fraction"],
                        "non_embedding_parameters": report[
                            "non_embedding_parameters"
                        ],
                        "distance": abs(report["total"] - target),
                        "relative_error": abs(report["total"] - target) / target,
                    }
                )
    candidates.sort(key=lambda item: item["distance"])
    best = candidates[0]
    return {
        "target_parameters": target,
        "tolerance": tolerance,
        "within_tolerance": best["relative_error"] <= tolerance,
        "selected": best,
        "candidates": candidates[:15],
    }


def build_matched(target: int, vocab_size: int, **kwargs) -> BaselineTransformer:
    result = match_parameters(target, vocab_size, **kwargs)
    selected = result["selected"]
    return BaselineTransformer(
        TransformerConfig(
            vocab_size=vocab_size,
            dimensions=selected["dimensions"],
            layers=selected["layers"],
            heads=selected["heads"],
            context=kwargs.get("context", 256),
        )
    )


def forward_flops_per_token(model) -> int:
    """Rough forward FLOPs per token, for reporting the compute mismatch.

    Counts matrix multiplies only. It is an estimate, not a measurement, and it
    exists so the parameter-matched claim is not mistaken for compute-matched.
    """
    cfg = model.cfg
    if isinstance(model, BaselineTransformer):
        per_layer = 4 * cfg.dimensions ** 2 + 2 * cfg.ffn_multiplier * cfg.dimensions ** 2
        return 2 * (cfg.layers * per_layer + cfg.dimensions * cfg.vocab_size)
    # CellLM: the mixer and both attention projections run once per dynamics
    # step, so its per-token cost carries a factor of k.
    dimensions = cfg.d
    memory = model.bank_config.key_size
    # The mixer cost depends on which mixer was actually built. This used to
    # hardcode dimensions**2, which reported the dense cost for every model and
    # would have hidden the whole point of a low-rank sweep.
    mixer = getattr(getattr(model, "core", model), "cell", None)
    mixer = getattr(mixer, "mixer", None)
    mixer_flops = (
        sum(p.numel() for p in mixer.parameters()) if mixer is not None
        else dimensions ** 2
    )
    per_step = mixer_flops + 8 * memory * dimensions
    return 2 * (cfg.k * per_step + dimensions * cfg.vocab_size)
