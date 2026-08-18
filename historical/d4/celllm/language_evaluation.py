"""Reusable likelihood, generation, and prompt-sensitivity evaluation."""

from __future__ import annotations

import math

import torch


def evaluate_language_model(model, dataset, loader, device: str) -> dict[str, float | int]:
    """Compute token-weighted NLL/PPL and byte-weighted BPC."""
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    token_count = 0
    with torch.inference_mode():
        for tokens, mask in loader:
            sums, counts = model.loss_statistics(
                tokens.to(device), mask.to(device)
            )
            loss_sum += float(sums.sum().cpu())
            token_count += int(counts.sum().cpu())
    if was_training:
        model.train()
    nll = loss_sum / token_count
    return {
        "nll": nll,
        "ppl": math.exp(nll),
        "bpc": loss_sum / (math.log(2.0) * dataset.utf8_bytes),
        "tokens": token_count,
        "utf8_bytes": dataset.utf8_bytes,
    }


def generate_language(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    *,
    temperature: float | None = None,
    top_p: float = 1.0,
    seed: int = 0,
) -> str:
    """Generate greedily or sample without consuming the training RNG."""
    if temperature is not None and temperature <= 0:
        raise ValueError("sampling temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top-p must be in (0, 1]")
    ids = [tokenizer.special_id("<bos>"), *tokenizer.encode(prompt)]
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            tokens = torch.tensor(ids, device=device)[None]
            next_logits = model.sequence_logits(tokens)[0, -1]
            if temperature is None:
                next_id = int(next_logits.argmax())
            else:
                probabilities = (next_logits / temperature).softmax(-1)
                sorted_probabilities, sorted_ids = probabilities.sort(
                    descending=True
                )
                cumulative = sorted_probabilities.cumsum(-1)
                keep = cumulative - sorted_probabilities < top_p
                filtered = sorted_probabilities * keep
                filtered = filtered / filtered.sum()
                sampled_index = torch.multinomial(
                    filtered, 1, generator=generator
                )
                next_id = int(sorted_ids[sampled_index])
            if next_id == tokenizer.eos_id:
                break
            ids.append(next_id)
    return tokenizer.decode(ids[1:])


def jensen_shannon(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return Jensen-Shannon divergence in nats with numeric non-negativity."""
    left = left.double()
    right = right.double()
    midpoint = 0.5 * (left + right)
    epsilon = torch.finfo(left.dtype).tiny
    log_midpoint = midpoint.clamp_min(epsilon).log()
    left_term = left * (left.clamp_min(epsilon).log() - log_midpoint)
    right_term = right * (right.clamp_min(epsilon).log() - log_midpoint)
    divergence = float(
        (0.5 * (left_term.sum() + right_term.sum())).cpu()
    )
    return max(0.0, divergence)


def prompt_diagnostics(
    model,
    tokenizer,
    *,
    prompts: tuple[str, ...],
    max_new_tokens: int,
    top_k: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> dict:
    """Measure contextual sensitivity and fixed greedy/sampled continuations."""
    if top_k < 1:
        raise ValueError("diagnostic top-k must be positive")
    if len(prompts) < 2:
        raise ValueError("prompt sensitivity requires at least two prompts")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    distributions = {}
    next_token_top_k = {}
    generations = {}
    with torch.inference_mode():
        for prompt_index, prompt in enumerate(prompts):
            ids = [tokenizer.special_id("<bos>"), *tokenizer.encode(prompt)]
            tokens = torch.tensor(ids, device=device)[None]
            probabilities = (
                model.sequence_logits(tokens)[0, -1].float().softmax(-1)
            )
            distributions[prompt] = probabilities
            values, indices = probabilities.topk(min(top_k, len(probabilities)))
            next_token_top_k[prompt] = [
                {
                    "token": tokenizer.decode(
                        [int(token_id)], skip_special=False
                    ),
                    "token_id": int(token_id),
                    "probability": float(probability.cpu()),
                }
                for probability, token_id in zip(values, indices)
            ]
            generations[prompt] = {
                "greedy": generate_language(
                    model, tokenizer, prompt, max_new_tokens
                ),
                "sampled": generate_language(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed + prompt_index,
                ),
            }
    pairwise_js = []
    for left_index, left_prompt in enumerate(prompts):
        for right_prompt in prompts[left_index + 1 :]:
            pairwise_js.append(
                {
                    "left": left_prompt,
                    "right": right_prompt,
                    "js_nats": jensen_shannon(
                        distributions[left_prompt], distributions[right_prompt]
                    ),
                }
            )
    if was_training:
        model.train()
    return {
        "next_token_top_k": next_token_top_k,
        "pairwise_js": pairwise_js,
        "mean_pairwise_js_nats": sum(item["js_nats"] for item in pairwise_js)
        / len(pairwise_js),
        "generations": generations,
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        },
    }
