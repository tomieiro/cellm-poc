"""Shared causal cellular dynamics and the Transformer control."""

from dataclasses import dataclass

import torch
from celnn import DifferentiableCellularNetwork, piecewise_linear
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Shape:
    block: int = 64
    width: int = 128
    steps: int = 16
    radius: int = 4
    lamp_radius: int = 2
    lamp_width: int = 32
    gradient_horizon: int = 4


def previous(tensor, offset):
    """Shift along the sequence and fill the left boundary with zeros."""
    return F.pad(tensor, (0, 0, offset, 0))[:, : tensor.shape[1]]


class CellLM(nn.Module):
    def __init__(self, vocabulary, shape):
        super().__init__()
        self.shape = shape
        self.embedding = nn.Embedding(vocabulary, shape.width)
        self.dynamics = DifferentiableCellularNetwork(
            radius=shape.radius,
            channels=shape.width,
            activation="piecewise_linear",
            boundary="constant",
            dt=0.5,
            steps=shape.steps,
            method="euler",
            causal=True,
        )
        self.association = nn.Linear(shape.width, 3 * shape.lamp_width, bias=False)
        self.message = nn.Linear(shape.lamp_width, shape.width, bias=False)
        self.norm = nn.LayerNorm(shape.width)
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.dynamics.feedback, std=0.1)
        nn.init.normal_(self.dynamics.control, std=0.1)

    def local_drive(self, activity):
        match, source, messages = self.association(activity).chunk(3, dim=-1)
        match = F.elu(match) + 1
        source = F.elu(source) + 1
        messages = messages.tanh()
        positions = torch.arange(activity.shape[1], device=activity.device)
        weighted = torch.zeros_like(messages)
        total = torch.zeros_like(messages[..., :1])
        for offset in range(1, self.shape.lamp_radius + 1):
            affinity = (match * previous(source, offset)).mean(dim=-1, keepdim=True)
            strength = affinity * (positions >= offset)[None, :, None]
            weighted = weighted + strength * previous(messages, offset)
            total = total + strength
        return self.message(weighted / total.clamp_min(1e-6))

    def evolve(self, ids, field=None, *, lamp=True):
        drive = self.embedding(ids).tanh()
        state = torch.zeros_like(drive) if field is None else field[:, : ids.shape[1]]
        if not lamp:
            return self.dynamics(drive, state)
        for _ in range(self.dynamics.steps):
            message = self.local_drive(piecewise_linear(state))
            state = self.dynamics.step(state, drive, extra_drive=message)
        return state

    def readout(self, state):
        return F.linear(self.norm(state), self.embedding.weight)

    def block(self, ids, field=None, *, lamp=True):
        state = self.evolve(ids, field, lamp=lamp)
        return self.readout(state), state

    def forward(self, ids, mode, *, lamp=True, last_only=False):
        if mode not in {"carry", "reset"}:
            raise ValueError(f"modo celular inválido: {mode}")
        field = None
        logits = []
        blocks = ids.split(self.shape.block, dim=1)
        for index, tokens in enumerate(blocks, start=1):
            field = self.evolve(tokens, field, lamp=lamp)
            if not last_only:
                logits.append(self.readout(field))
            elif index == len(blocks):
                logits.append(self.readout(field[:, -1:]))
            if mode == "reset":
                field = None
            elif index % self.shape.gradient_horizon == 0:
                field = field.detach()
        return torch.cat(logits, dim=1)


class DecoderLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.projection = nn.Linear(width, width)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, states):
        batch, length, width = states.shape
        qkv = self.qkv(self.norm(states)).reshape(batch, length, 3, 4, width // 4)
        query, keys, values = qkv.permute(2, 0, 3, 1, 4).unbind()
        context = F.scaled_dot_product_attention(query, keys, values, is_causal=True)
        states = states + self.projection(
            context.transpose(1, 2).reshape(batch, length, width)
        )
        return states + self.feed_forward(states)


class Transformer(nn.Module):
    def __init__(self, vocabulary, width):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, width)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.layers = nn.Sequential(DecoderLayer(width), DecoderLayer(width))
        self.norm = nn.LayerNorm(width)

    def forward(self, ids, mode="transformer", *, last_only=False):
        states = self.embedding(ids)
        positions = torch.arange(ids.shape[1], device=ids.device, dtype=torch.float32)
        frequencies = torch.arange(
            0, states.shape[-1], 2, device=ids.device, dtype=torch.float32
        )
        angles = positions[:, None] / 10_000 ** (frequencies / states.shape[-1])
        position = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        states = self.layers(states + position.to(states.dtype))
        readout = states[:, -1:] if last_only else states
        return F.linear(self.norm(readout), self.embedding.weight)


def parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def transformer_width(vocabulary, shape):
    with torch.device("meta"):
        target = parameters(CellLM(vocabulary, shape))
        candidates = range(4, shape.width * 2 + 1, 4)
        return min(
            candidates,
            key=lambda width: abs(parameters(Transformer(vocabulary, width)) - target),
        )


def build_model(arm, vocabulary, shape, control_width):
    return (
        Transformer(vocabulary, control_width)
        if arm == "transformer"
        else CellLM(vocabulary, shape)
    )


def load_model(checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    arm, vocabulary = payload["arm"], payload["vocabulary"]
    model = build_model(
        arm, vocabulary, Shape(**payload["shape"]), payload["transformer_width"]
    )
    weights = payload["weights"]
    if arm != "transformer" and "feedback" in weights:
        # Earlier PoC checkpoints stored the present tap first.
        weights["dynamics.feedback"] = weights.pop("feedback").flip(0)
        weights["dynamics.control"] = weights.pop("control").flip(0)
        weights["dynamics.bias"] = weights.pop("bias")
    model.load_state_dict(weights)
    return model.to(device).eval(), arm, vocabulary
