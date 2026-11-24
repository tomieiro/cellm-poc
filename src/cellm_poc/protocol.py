"""Document splits, paired optimization and interventions at inference time."""

import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

PAD, EOS = 0, 1


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def documents(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, row in enumerate(stream, start=1):
            try:
                text = json.loads(row)["text"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("texto vazio")
            except (TypeError, KeyError, ValueError) as error:
                raise ValueError(
                    f"{path}:{line_number}: esperado um objeto com 'text' não vazio"
                ) from error
            yield text


def prepare(paths, vocab_size, context):
    """Fit BPE on training documents and keep the three splits independent."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<pad>", "<eos>"],
        show_progress=False,
    )
    seen = {}
    windows = []
    for split, path in enumerate(paths):
        for text in documents(path):
            key = hashlib.sha256(" ".join(text.split()).encode()).digest()
            prior = seen.setdefault(key, split)
            if prior != split:
                raise ValueError(f"documento duplicado entre partições em {path}")
        if split == 0:
            tokenizer.train_from_iterator(documents(path), trainer)
        partition = []
        for text in documents(path):
            ids = tokenizer.encode(text).ids
            if PAD in ids or EOS in ids:
                raise ValueError(f"documento em {path} contém um marcador reservado")
            ids.append(EOS)
            for start in range(0, len(ids) - 1, context):
                partition.append(ids[start : start + context + 1])
        if not partition:
            raise ValueError(f"partição sem alvos: {path}")
        windows.append(partition)
    return tokenizer, windows


def batches(windows, size, order=None):
    indices = range(len(windows)) if order is None else order
    for start in range(0, len(indices), size):
        rows = [
            torch.tensor(windows[index], dtype=torch.long)
            for index in indices[start : start + size]
        ]
        ids = pad_sequence(rows, batch_first=True, padding_value=PAD)
        yield ids[:, :-1], ids[:, 1:]


def losses(logits, targets):
    return F.cross_entropy(
        logits[targets != PAD], targets[targets != PAD], reduction="none"
    )


def predict(model, inputs, mode, *, last_only=False):
    if mode == "lamp-off":
        return model(inputs, "carry", lamp=False, last_only=last_only)
    return model(inputs, mode, last_only=last_only)


@torch.inference_mode()
def evaluate(model, mode, windows, batch_size, device):
    model.eval()
    nll, count = 0.0, 0
    for inputs, targets in batches(windows, batch_size):
        scores = losses(predict(model, inputs.to(device), mode), targets.to(device))
        nll += float(scores.sum())
        count += scores.numel()
    mean = nll / count
    return {"nll": mean, "ppl": math.exp(mean), "targets": count}


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def weight_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_arm(
    model, mode, train, validation, budgets, batch_size, seed, device, learning_rate
):
    model.to(device)
    initial = weight_digest(model)
    sampler = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    digest = hashlib.sha256()
    curve, consumed, seconds = [], 0, 0.0
    stream = iter(())
    for budget in budgets:
        model.train()
        synchronize(device)
        start = time.perf_counter()
        while consumed < budget:
            batch = next(stream, None)
            if batch is None:
                order = torch.randperm(len(train), generator=sampler).tolist()
                stream = batches(train, batch_size, order)
                batch = next(stream)
            inputs, targets = batch
            take = min(budget - consumed, int((targets != PAD).sum()))
            digest.update(f"{tuple(inputs.shape)}:{take}:".encode())
            digest.update(inputs.numpy().tobytes())
            digest.update(targets.numpy().tobytes())
            optimizer.zero_grad(set_to_none=True)
            scores = losses(model(inputs.to(device), mode), targets.to(device))
            loss = scores[:take].mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"loss não finita no braço {mode}, alvo {consumed}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            consumed += take
        synchronize(device)
        seconds += time.perf_counter() - start
        curve.append(
            {
                "trained_tokens": consumed,
                "training_seconds": seconds,
                **evaluate(model, mode, validation, batch_size, device),
            }
        )
    return {
        "learning_curve": curve,
        "initial_weights_sha256": initial,
        "final_weights_sha256": weight_digest(model),
        "batch_sha256": digest.hexdigest(),
    }


@torch.inference_mode()
def boundary(model, ids, overlap, device):
    """Score the first block-minus-overlap targets at each shared boundary."""
    block = model.shape.block
    tokens = torch.tensor(ids, device=device).unsqueeze(0)
    rows = []
    for position in range(block, len(ids) - 1, block):
        count = min(block - overlap, len(ids) - position - 1)
        targets = tokens[0, position + 1 : position + 1 + count]
        reset, _ = model.block(tokens[:, position : position + count])
        repeated, _ = model.block(tokens[:, position - overlap : position + count])
        reset_losses = F.cross_entropy(reset[0], targets, reduction="none")
        overlap_losses = F.cross_entropy(
            repeated[0, overlap:], targets, reduction="none"
        )
        rows.append(
            {
                "position": position,
                "targets": count,
                "reset_nll": float(reset_losses.mean()),
                "overlap_nll": float(overlap_losses.mean()),
            }
        )
    return rows


def recall_cases(prefix, question, filler, values, distances, trials, context, seed):
    rng = random.Random(seed)
    cases = []
    for distance in distances:
        gap = distance - len(question) - 1
        if gap < 0 or len(prefix) + distance > context or gap > len(filler):
            raise ValueError(
                f"distância de recall incompatível com o prompt ou corpus: {distance}"
            )
        for _ in range(trials):
            value, other = rng.sample(values, 2)
            start = rng.randrange(len(filler) - gap + 1)
            suffix = filler[start : start + gap] + question
            cases.append(
                (
                    distance,
                    prefix + [value] + suffix,
                    prefix + [other] + suffix,
                    value,
                    other,
                )
            )
    return cases


def recall_inputs(tokenizer, train, test, distances, trials, context, seed):
    frequent = Counter(token for row in train for token in row if token > EOS)
    values = []
    for token, _ in frequent.most_common():
        word = tokenizer.decode([token]).strip()
        if len(word) >= 2 and word.isascii() and word.isalpha():
            values.append(token)
        if len(values) == 32:
            break
    if len(values) < 2:
        raise ValueError(
            "o treino precisa ter pelo menos dois tokens lexicais para os valores do recall"
        )
    # Longest held-out document window gives a natural filler with no split crossing.
    filler = [token for token in max(test, key=len) if token > EOS]
    prefix = tokenizer.encode("The code is").ids
    question = tokenizer.encode(". What is the code? The code is").ids
    return recall_cases(
        prefix, question, filler, values, distances, trials, context, seed
    )


@torch.inference_mode()
def evaluate_recall(model, mode, cases, device):
    rows = []
    for distance, prompt, alternative, value, other in cases:
        ids = torch.tensor([prompt, alternative], device=device)
        prob = predict(model, ids, mode, last_only=True)[:, -1].softmax(dim=-1)
        winners = prob.argmax(dim=-1).tolist()
        rows.append(
            {
                "distance": distance,
                "prompt_ids": prompt,
                "target": value,
                "alternative_target": other,
                "top1": winners[0],
                "alternative_top1": winners[1],
                "target_probability": float(prob[0, value]),
                "target_probability_after_swap": float(prob[1, value]),
                "alternative_probability": float(prob[1, other]),
                "alternative_probability_before_swap": float(prob[0, other]),
            }
        )
    return rows


@torch.inference_mode()
def generate(model, mode, prompt, steps, device):
    sequence = torch.tensor([prompt], device=device)
    for _ in range(steps):
        logits = predict(model, sequence, mode, last_only=True)[:, -1]
        logits[:, PAD] = -torch.inf
        token = logits.argmax(dim=-1, keepdim=True)
        if int(token.item()) == EOS:
            break
        sequence = torch.cat((sequence, token), dim=1)
    return sequence[0, len(prompt) :].tolist()
