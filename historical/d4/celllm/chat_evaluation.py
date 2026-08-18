"""Small, reproducible behavioral evaluation for conversational CellLM."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass

import torch

from celllm.chat_generation import ChatSession


@dataclass(frozen=True)
class EvaluationTurn:
    prompt: str
    expected_any: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    category: str
    turns: tuple[EvaluationTurn, ...]


SIMPLE_CHAT_CASES = (
    EvaluationCase(
        "greeting",
        "conversation",
        (EvaluationTurn("hello", ("hello", "hi")),),
    ),
    EvaluationCase(
        "identity",
        "conversation",
        (EvaluationTurn("what is your name", ("celllm",)),),
    ),
    EvaluationCase(
        "coffee",
        "everyday",
        (EvaluationTurn("do you like coffee", ("coffee",)),),
    ),
    EvaluationCase(
        "sky",
        "knowledge",
        (EvaluationTurn("what color is the sky", ("blue",)),),
    ),
    EvaluationCase(
        "arithmetic",
        "knowledge",
        (EvaluationTurn("what is two plus two", ("four",)),),
    ),
    EvaluationCase(
        "remember-name",
        "memory",
        (
            EvaluationTurn("my name is robin"),
            EvaluationTurn("what is my name", ("robin",)),
        ),
    ),
    EvaluationCase(
        "remember-color",
        "memory",
        (
            EvaluationTurn("my favorite color is purple"),
            EvaluationTurn("what is my favorite color", ("purple",)),
        ),
    ),
    EvaluationCase(
        "natural-follow-up",
        "continuity",
        (
            EvaluationTurn("hello", ("hello", "hi")),
            EvaluationTurn(
                "i am fine do you like horses", ("horse",)
            ),
            EvaluationTurn("what are you", ("celllm", "model")),
        ),
    ),
)

_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?", flags=re.UNICODE)


@dataclass(frozen=True)
class FirstWordCase:
    """One probe whose *opening word* is the thing under test."""

    prompt: str
    expected_first_word: str
    polarity: str = "neutral"


# The A1 tokenizer fragments lowercase openings ("yes" -> "y" + "es"), which
# landed a spurious "y" on nearly every B0.1 generation. Scoring the first word
# separately makes that failure measurable instead of visual, and the yes/no
# split checks the model is reading the question rather than emitting a constant.
FIRST_WORD_CASES = (
    FirstWordCase("do you like coffee", "yes", "positive"),
    FirstWordCase("do you like pizza", "yes", "positive"),
    FirstWordCase("are cats interesting", "yes", "positive"),
    FirstWordCase("do you enjoy reading", "yes", "positive"),
    FirstWordCase("is tea a good drink", "yes", "positive"),
    FirstWordCase("are you a human", "no", "negative"),
    FirstWordCase("do you sleep at night", "no", "negative"),
    FirstWordCase("can you see me", "no", "negative"),
    FirstWordCase("do you eat food", "no", "negative"),
    FirstWordCase("do you get bored", "no", "negative"),
    FirstWordCase("hello", "hello", "neutral"),
    FirstWordCase("what is your name", "my", "neutral"),
    FirstWordCase("what color is the sky", "the", "neutral"),
    FirstWordCase("tell me about cats", "cats", "neutral"),
)


def first_word(text: str) -> str:
    """The first complete decoded word, normalized; empty when there is none."""
    words = normalized_words(text)
    return words[0] if words else ""


def evaluate_first_word(
    session: ChatSession,
    cases: tuple[FirstWordCase, ...] = FIRST_WORD_CASES,
) -> dict:
    """Score whether the opening word of each answer is the expected one."""
    results = []
    for case in cases:
        session.reset()
        response = session.reply(case.prompt)
        produced = first_word(response)
        results.append(
            {
                "prompt": case.prompt,
                "polarity": case.polarity,
                "expected_first_word": case.expected_first_word,
                "produced_first_word": produced,
                "response": response,
                "correct": produced == case.expected_first_word,
                # A truncated opening is the specific B0.1 symptom: a strict,
                # non-empty prefix of the expected word, such as "y" for "yes".
                "truncated": bool(produced)
                and case.expected_first_word.startswith(produced)
                and produced != case.expected_first_word,
            }
        )

    def rate(subset: list[dict], key: str) -> float | None:
        return (
            sum(item[key] for item in subset) / len(subset) if subset else None
        )

    positive = [item for item in results if item["polarity"] == "positive"]
    negative = [item for item in results if item["polarity"] == "negative"]
    return {
        "first_word_exact": rate(results, "correct"),
        "truncated_first_word_rate": rate(results, "truncated"),
        "yes_given_positive": rate(positive, "correct"),
        "no_given_negative": rate(negative, "correct"),
        "cases": results,
    }


def normalized_words(text: str) -> tuple[str, ...]:
    """Tokenize normalized words for conservative behavioral matching."""
    return tuple(_WORD.findall(text.casefold()))


def contains_expected(response: str, expected: str) -> bool:
    """Match a complete normalized word sequence, never an arbitrary substring."""
    response_words = normalized_words(response)
    expected_words = normalized_words(expected)
    if not expected_words:
        return False
    width = len(expected_words)
    return any(
        response_words[index : index + width] == expected_words
        for index in range(len(response_words) - width + 1)
    )


def repeated_bigram_rate(text: str) -> float:
    """Return the fraction of word bigrams that repeat within one answer."""
    words = text.lower().split()
    bigrams = list(zip(words, words[1:]))
    if not bigrams:
        return 0.0
    return 1.0 - len(set(bigrams)) / len(bigrams)


def evaluate_assistant_likelihood(model, loader, device: str) -> dict:
    """Compute exact token-weighted assistant NLL over a complete split."""
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
    if token_count == 0:
        raise ValueError("evaluation split has no assistant target tokens")
    nll = loss_sum / token_count
    return {
        "nll": nll,
        "ppl": float(torch.exp(torch.tensor(nll))),
        "tokens": token_count,
    }


def evaluate_simple_chat(
    session: ChatSession,
    cases: tuple[EvaluationCase, ...] = SIMPLE_CHAT_CASES,
) -> dict:
    """Run isolated cases and report keyword recall and degeneration."""
    results = []
    for case in cases:
        session.reset()
        turns = []
        for turn in case.turns:
            response = session.reply(turn.prompt)
            passed = not turn.expected_any or any(
                contains_expected(response, keyword)
                for keyword in turn.expected_any
            )
            turns.append(
                {
                    **asdict(turn),
                    "response": response,
                    "passed": passed,
                    "repeated_bigram_rate": repeated_bigram_rate(response),
                }
            )
        scored = [turn for turn in turns if turn["expected_any"]]
        results.append(
            {
                "name": case.name,
                "category": case.category,
                "passed": all(turn["passed"] for turn in scored),
                "turns": turns,
            }
        )

    passed = sum(case["passed"] for case in results)
    scored_responses = [
        turn
        for case in results
        for turn in case["turns"]
        if turn["expected_any"]
    ]
    category_scores = {}
    for category in sorted({case.category for case in cases}):
        matching = [item for item in results if item["category"] == category]
        category_scores[category] = sum(item["passed"] for item in matching) / len(
            matching
        )
    score = passed / len(results)
    mean_repetition = sum(
        turn["repeated_bigram_rate"] for turn in scored_responses
    ) / len(scored_responses)
    normalized_responses = [
        " ".join(normalized_words(turn["response"]))
        for turn in scored_responses
    ]
    response_counts = Counter(normalized_responses)
    dominant_response, dominant_count = response_counts.most_common(1)[0]
    unique_response_rate = len(response_counts) / len(normalized_responses)
    dominant_response_rate = dominant_count / len(normalized_responses)
    accepted = (
        passed >= 6
        and category_scores.get("continuity", 0) == 1
        and category_scores.get("memory", 0) >= 0.5
        and mean_repetition <= 0.35
        and dominant_response_rate <= 0.5
    )
    return {
        "accepted": accepted,
        "acceptance_criteria": {
            "minimum_passed_cases": 6,
            "minimum_continuity_score": 1.0,
            "minimum_memory_score": 0.5,
            "maximum_mean_repeated_bigram_rate": 0.35,
            "maximum_dominant_response_rate": 0.5,
        },
        "score": score,
        "passed": passed,
        "total": len(results),
        "category_scores": category_scores,
        "mean_repeated_bigram_rate": mean_repetition,
        "unique_response_rate": unique_response_rate,
        "dominant_response_rate": dominant_response_rate,
        "dominant_response": dominant_response,
        "dominant_response_count": dominant_count,
        "cases": results,
    }
