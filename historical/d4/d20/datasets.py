"""Dataset acquisition, filtering, splitting and blocking gates for D20.

Every choice here answers a specific failure from the B0.x line:

* B0 trained on a corpus that was 95.7% one source and 47% one canned answer, so
  nothing is accepted without an audit;
* B0's splits were assigned by the same hash namespace used for subsampling, so
  an entire source landed in one split -- namespaces are separate here;
* B0.3's held-out split lost an intent to sampling noise, so splitting is
  stratified;
* gates are blocking. This is the showcase branch, but "p = 0.054, let it pass"
  is exactly how the earlier corpus reached training.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

TINYSTORIES_REPO = "roneneldan/TinyStories"
SMOLTALK_REPO = "HuggingFaceTB/smol-smoltalk"

_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?", flags=re.UNICODE)


class DataGateError(RuntimeError):
    """Raised when a corpus fails a blocking gate."""


def _hash(namespace: str, key: str, seed: int) -> str:
    return hashlib.sha256(f"{namespace}:{seed}:{key}".encode()).hexdigest()


def words(text: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(text.casefold()))


def normalize(text: str) -> str:
    return " ".join(words(text))


# --- acquisition ----------------------------------------------------------


def check_hub_access() -> None:
    """Fail fast and loudly rather than silently substituting a dataset."""
    try:
        from huggingface_hub import HfApi

        HfApi().model_info("gpt2", timeout=20)
    except Exception as error:  # noqa: BLE001 - surfaced verbatim to the user
        raise RuntimeError(
            "Hugging Face Hub is not reachable from this node. The D20 "
            "notebook downloads its datasets inside the job and will not "
            "substitute another source. Fix connectivity or pre-populate "
            f"HF_DATASETS_CACHE, then retry. Original error: {error!r}"
        ) from error


def describe_dataset(repo: str, config: str | None = None) -> dict:
    """Inspect a dataset before assuming anything about its schema."""
    from datasets import get_dataset_config_names, load_dataset

    configs = get_dataset_config_names(repo)
    chosen = config or (configs[0] if configs else None)
    probe = load_dataset(repo, chosen, split="train", streaming=True)
    first = next(iter(probe))
    return {
        "repo": repo,
        "available_configs": configs,
        "config": chosen,
        "column_names": list(first.keys()),
        "example": {
            key: (str(value)[:200] if not isinstance(value, list) else value[:2])
            for key, value in first.items()
        },
    }


def download_language(
    *,
    output: str | Path,
    limit: int = 400_000,
    config: str | None = None,
    min_chars: int = 120,
) -> dict:
    """Stream TinyStories to JSONL with `{"id", "text"}` rows."""
    from datasets import load_dataset

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stream = load_dataset(
        TINYSTORIES_REPO, config, split="train", streaming=True
    )
    written = 0
    seen: set[str] = set()
    with destination.open("w", encoding="utf-8") as handle:
        for record in stream:
            text = str(record.get("text", "")).strip()
            if len(text) < min_chars:
                continue
            identifier = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if identifier in seen:
                continue
            seen.add(identifier)
            handle.write(
                json.dumps({"id": identifier, "text": text}, ensure_ascii=False)
                + "\n"
            )
            written += 1
            if written >= limit:
                break
    if written == 0:
        raise DataGateError("language download produced no documents")
    manifest = {
        "repo": TINYSTORIES_REPO,
        "config": config,
        "split": "train",
        "requested": limit,
        "written": written,
        "min_chars": min_chars,
        "path": str(destination.resolve()),
    }
    write_download_manifest(destination, manifest)
    return manifest


def download_chat(
    *, output: str | Path, limit: int = 200_000, split: str = "train"
) -> dict:
    """Stream smol-smoltalk to JSONL, preserving `messages` and `source`."""
    from datasets import load_dataset

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stream = load_dataset(SMOLTALK_REPO, split=split, streaming=True)
    written = 0
    with destination.open("w", encoding="utf-8") as handle:
        for record in stream:
            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            handle.write(
                json.dumps(
                    {
                        "messages": [
                            {
                                "role": str(item.get("role", "")),
                                "content": str(item.get("content", "")),
                            }
                            for item in messages
                        ],
                        "source": str(record.get("source", "unknown")),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            if written >= limit:
                break
    if written == 0:
        raise DataGateError(f"chat download produced no dialogues for {split!r}")
    manifest = {
        "repo": SMOLTALK_REPO,
        "split": split,
        "requested": limit,
        "written": written,
        "path": str(destination.resolve()),
    }
    write_download_manifest(destination, manifest)
    return manifest


def write_download_manifest(path: str | Path, manifest: dict) -> Path:
    destination = Path(str(path) + ".manifest.json")
    destination.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def reusable_download(path: str | Path, *, requested: int) -> dict | None:
    """Return the cached manifest only if it satisfies the current request.

    A cache is not reusable merely because the file exists. The D20 local run
    silently trained 50M tokens over a 3,000-document smoke corpus because a
    previous smoke run had left `language_raw.jsonl` in place and the notebook
    reused it without checking how much data it held. Size is part of the
    cache key, not an afterthought.
    """
    data = Path(path)
    manifest_path = Path(str(path) + ".manifest.json")
    if not data.exists():
        return None
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("written", 0)) < requested:
        return None
    return manifest


# --- chat filtering -------------------------------------------------------


@dataclass(frozen=True)
class ChatFilter:
    """Demonstration-oriented filter for a ~20M model.

    A 20M model will not learn long-form code, advanced mathematics or tool
    calling. Filtering is by structure and length rather than by invented
    keyword lists, and every rejection reason is counted so the effect is
    inspectable rather than asserted.
    """

    max_turns: int = 6
    max_user_chars: int = 400
    # smol-smoltalk's median assistant message is ~1290 characters. A 20M model
    # is not going to learn long-form answers, but rejecting every dialogue that
    # contains one throws away most of the corpus. Dialogues are truncated at
    # the first over-long turn instead, keeping the short prefix.
    #
    # 600 was measured, not guessed. Yield and length against this limit:
    #   400 ->  6.6% kept,  90 assistant tokens/dialogue
    #   600 -> 12.3% kept, 145 assistant tokens/dialogue
    #   800 -> 18.2% kept, 205 assistant tokens/dialogue
    #  1200 -> 30.6% kept, 387 assistant tokens/dialogue
    # Beyond ~600 the answers stop being short demonstrable replies, which is
    # what this showcase is for.
    max_assistant_chars: int = 600
    min_assistant_chars: int = 8
    drop_sources: tuple[str, ...] = ()
    drop_markers: tuple[str, ...] = (
        "```",
        "\\begin{",
        "\\frac",
        "<tool_call>",
        "<function",
        "def ",
        "import ",
        "SELECT ",
    )
    reasons: dict = field(default_factory=dict)


def filter_chat(
    rows: list[dict], spec: ChatFilter = ChatFilter()
) -> tuple[list[dict], dict]:
    """Keep the demonstrable prefix of each dialogue rather than all-or-nothing.

    Two behaviours matter for yield. `system` turns are *stripped*, not treated
    as malformed ordering -- 16% of smol-smoltalk opens with one. And a dialogue
    whose third turn is a 2000-character essay still contributes its first
    exchange, so the length limit truncates instead of rejecting.
    """
    kept: list[dict] = []
    reasons: Counter = Counter()
    stripped_system = 0
    truncated = 0

    for row in rows:
        messages = row["messages"]
        before = len(messages)
        messages = [item for item in messages if item["role"] != "system"]
        if len(messages) != before:
            stripped_system += 1
        if len(messages) < 2:
            reasons["too_short_after_system_strip"] += 1
            continue
        if messages[0]["role"] != "user":
            reasons["bad_role_ordering"] += 1
            continue

        blob = " ".join(item["content"] for item in messages)
        if any(marker in blob for marker in spec.drop_markers):
            reasons["structured_content"] += 1
            continue
        if str(row.get("source", "")) in spec.drop_sources:
            reasons["dropped_source"] += 1
            continue

        accepted: list[dict] = []
        stopped = False
        for index, item in enumerate(messages[: spec.max_turns]):
            expected = "user" if index % 2 == 0 else "assistant"
            if item["role"] != expected:
                stopped = True
                break
            content = " ".join(item["content"].split())
            limit = (
                spec.max_user_chars
                if item["role"] == "user"
                else spec.max_assistant_chars
            )
            too_long = len(content) > limit
            too_short = (
                item["role"] == "assistant"
                and len(content) < spec.min_assistant_chars
            )
            if not content or too_long or too_short:
                stopped = True
                break
            accepted.append({"role": item["role"], "content": content})

        # Trim to a complete user/assistant pair.
        while accepted and accepted[-1]["role"] != "assistant":
            accepted.pop()
        if len(accepted) < 2:
            reasons["no_usable_prefix"] += 1
            continue
        if stopped or len(accepted) < len(messages):
            truncated += 1
        kept.append({"messages": accepted, "source": row.get("source", "unknown")})

    return kept, {
        "input": len(rows),
        "kept": len(kept),
        "kept_fraction": len(kept) / len(rows) if rows else 0.0,
        "system_turns_stripped": stripped_system,
        "dialogues_truncated": truncated,
        "rejection_reasons": dict(reasons.most_common()),
    }


def _percentiles(values: list[int]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p25": ordered[len(ordered) // 4],
        "median": statistics.median(ordered),
        "p75": ordered[3 * len(ordered) // 4],
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def chat_statistics(rows: list[dict]) -> dict:
    """Length and source statistics, reported before and after filtering."""
    user = [
        len(item["content"])
        for row in rows
        for item in row["messages"]
        if item["role"] == "user"
    ]
    assistant = [
        len(item["content"])
        for row in rows
        for item in row["messages"]
        if item["role"] == "assistant"
    ]
    return {
        "dialogues": len(rows),
        "turns": sum(len(row["messages"]) for row in rows),
        "turn_distribution": dict(
            Counter(len(row["messages"]) for row in rows).most_common(10)
        ),
        "sources": dict(
            Counter(str(row.get("source", "unknown")) for row in rows).most_common(15)
        ),
        "user_chars": _percentiles(user),
        "assistant_chars": _percentiles(assistant),
    }


# --- splitting ------------------------------------------------------------


def stratified_split(
    rows: list[dict],
    *,
    key: str,
    seed: int,
    validation_fraction: float = 0.05,
    test_fraction: float = 0.05,
    namespace: str = "split",
    minimum_stratum: int = 6,
    report: dict | None = None,
) -> dict[str, list[dict]]:
    """Split within each stratum so every class reaches every split.

    B0.3 lost an entire intent from its held-out set to sampling noise. Ranking
    inside the stratum removes that failure mode by construction. The namespace
    is a parameter and must differ from any namespace used for subsampling.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "unknown")), []).append(row)
    # A stratum with a handful of rows cannot reach all three splits. Dropping
    # it is the right call -- aborting the whole job over a source that
    # contributed two dialogues wastes a run, and silently folding it into
    # train would break the stratification guarantee.
    dropped = {
        name: len(items)
        for name, items in groups.items()
        if len(items) < minimum_stratum
    }
    groups = {
        name: items
        for name, items in groups.items()
        if len(items) >= minimum_stratum
    }
    if not groups:
        raise DataGateError(
            f"every stratum has fewer than {minimum_stratum} rows; "
            "nothing can be split"
        )
    if report is not None:
        report["dropped_strata"] = dropped
        report["kept_strata"] = sorted(groups)
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for stratum, items in sorted(groups.items()):
        ordered = sorted(
            items,
            key=lambda row: _hash(
                namespace,
                json.dumps(row, sort_keys=True, ensure_ascii=False),
                seed,
            ),
        )
        count = len(ordered)
        validation_count = max(1, round(count * validation_fraction))
        test_count = max(1, round(count * test_fraction))
        if count < validation_count + test_count + 1:
            raise DataGateError(
                f"stratum {stratum!r} has {count} rows, too few to split"
            )
        splits["validation"].extend(ordered[:validation_count])
        splits["test"].extend(
            ordered[validation_count : validation_count + test_count]
        )
        splits["train"].extend(ordered[validation_count + test_count :])
    return splits


def deduplicate(rows: list[dict]) -> tuple[list[dict], int]:
    seen: dict[str, dict] = {}
    for row in rows:
        key = json.dumps(row["messages"], sort_keys=True, ensure_ascii=False)
        seen.setdefault(key, row)
    return list(seen.values()), len(rows) - len(seen)


# --- gates ----------------------------------------------------------------


def first_target_words(rows: list[dict]) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        for item in row["messages"]:
            if item["role"] == "assistant":
                head = words(item["content"])
                if head:
                    counter[head[0]] += 1
                break
    return counter


def assert_data_gates(
    splits: dict[str, list[dict]],
    *,
    max_first_word_rate: float = 0.20,
    max_exact_response_rate: float = 0.05,
    require_strata: set[str] | None = None,
    stratum_key: str = "source",
) -> dict:
    """Blocking gates. Raises rather than warning."""
    report: dict = {"splits": {}, "failures": []}
    train_keys = {
        json.dumps(row["messages"], sort_keys=True, ensure_ascii=False)
        for row in splits["train"]
    }
    for name, rows in splits.items():
        if not rows:
            report["failures"].append(f"{name}: empty split")
            continue
        first = first_target_words(rows)
        top_word, top_count = first.most_common(1)[0]
        responses = Counter(
            normalize(item["content"])
            for row in rows
            for item in row["messages"]
            if item["role"] == "assistant"
        )
        top_response, top_response_count = responses.most_common(1)[0]
        strata = {str(row.get(stratum_key, "unknown")) for row in rows}
        leakage = 0
        if name != "train":
            leakage = sum(
                1
                for row in rows
                if json.dumps(row["messages"], sort_keys=True, ensure_ascii=False)
                in train_keys
            )
        entry = {
            "dialogues": len(rows),
            "dominant_first_word": top_word,
            "dominant_first_word_rate": top_count / len(rows),
            "dominant_response": top_response[:80],
            "dominant_response_rate": top_response_count / sum(responses.values()),
            "strata": sorted(strata),
            "train_leakage": leakage,
        }
        report["splits"][name] = entry
        if entry["dominant_first_word_rate"] > max_first_word_rate:
            report["failures"].append(
                f"{name}: first word {top_word!r} holds "
                f"{entry['dominant_first_word_rate']:.3f} of dialogues "
                f"(limit {max_first_word_rate})"
            )
        if entry["dominant_response_rate"] > max_exact_response_rate:
            report["failures"].append(
                f"{name}: response {top_response[:40]!r} holds "
                f"{entry['dominant_response_rate']:.3f} (limit "
                f"{max_exact_response_rate})"
            )
        if leakage:
            report["failures"].append(f"{name}: {leakage} dialogues leak from train")
        if require_strata and not require_strata <= strata:
            missing = sorted(require_strata - strata)
            report["failures"].append(f"{name}: missing strata {missing}")
    report["passed"] = not report["failures"]
    if not report["passed"]:
        raise DataGateError(
            "D20 data gates failed:\n  - " + "\n  - ".join(report["failures"])
        )
    return report


def write_jsonl(path: str | Path, rows: list[dict]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return destination


def read_jsonl(path: str | Path) -> list[dict]:
    """Read JSONL, splitting on newlines only.

    `str.splitlines()` also breaks on \v, \f, \x1c-\x1e, \x85, U+2028 and
    U+2029. TinyStories contains some of those inside its text fields, so a
    single valid JSON record was being torn into two unparseable halves.
    """
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [json.loads(line) for line in handle if line.strip()]
