"""Segment-based training with language replay, milestones and resume.

Three commitments carried over from the B0.x line:

* the budget is counted in **target tokens**, not steps or epochs, so numbers
  stay comparable across phases and against A1/B0.x;
* replay is an alternation of objectives, never TinyStories reshaped into fake
  dialogues, and its ratio is configurable per segment rather than fixed;
* a checkpoint is never selected on loss alone. Selection needs the behavioural
  and retention metrics, which B0.3 showed can move opposite to NLL.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from celllm.chat_checkpoint import load_chat_checkpoint, save_chat_checkpoint
from celllm.chat_data import ConversationDataset, collate_conversations
from celllm.language_data import (
    LanguageDocumentDataset,
    collate_language_windows,
)
from celllm.chat_evaluation import evaluate_assistant_likelihood
from celllm.chat_evaluation import evaluate_assistant_likelihood
from celllm.language_evaluation import evaluate_language_model

from d20.provenance import restore_rng_state, rng_state


@dataclass
class SegmentConfig:
    """One training segment. Changing any field starts a new segment.

    D20 is a showcase and may use a pragmatic schedule, but a schedule that
    changes silently inside a segment is untraceable. Each segment carries its
    own config and manifest.
    """

    name: str
    objective: str  # "language" | "chat"
    target_tokens: int
    learning_rate: float
    replay_ratio: float = 0.0
    batch_size: int = 32
    sequence_length: int = 256
    grad_clip: float = 1.0
    milestones: tuple[int, ...] = ()
    evaluate_every_tokens: int = 0
    checkpoint_every_steps: int = 500
    seed: int = 7
    source_checkpoint: str | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)


def replay_period(ratio: float) -> int | None:
    """Cycle length giving the requested share of language batches."""
    if not 0 <= ratio < 1:
        raise ValueError("replay ratio must lie in [0, 1)")
    if ratio == 0:
        return None
    period = round(1 / ratio)
    if period < 2:
        raise ValueError("replay ratio above 0.5 is not supported")
    return period


def is_replay_step(index: int, period: int | None) -> bool:
    """Pure function of the step index, so a resume reproduces the schedule."""
    return period is not None and index % period == period - 1


def infinite(loader):
    while True:
        yield from loader


def save_generic_checkpoint(
    path: str | Path, model, *, step: int, optimizer_state, metrics: dict
) -> None:
    """Fallback saver for models that are not CellLMChatModel.

    The CellLM saver rebuilds a model from its field/bank/local configs, which a
    baseline architecture does not have. This stores the config object itself so
    the control can be reloaded symmetrically.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "architecture": type(model).__name__,
            "step": step,
            "config": model.cfg,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer_state,
            "metrics": metrics,
        },
        temporary,
    )
    temporary.replace(destination)


def save_full_checkpoint(
    path: str | Path,
    model,
    *,
    step: int,
    optimizer,
    counters: dict,
    config: SegmentConfig,
    metrics: dict,
) -> None:
    """Checkpoint everything a preempted job needs to continue identically."""
    from celllm.chat_model import CellLMChatModel

    payload = {
        **metrics,
        "counters": counters,
        "segment_config": asdict(config),
        "rng_state": rng_state(),
    }
    if not isinstance(model, CellLMChatModel):
        save_generic_checkpoint(
            path, model, step=step,
            optimizer_state=optimizer.state_dict(), metrics=payload,
        )
        return
    save_chat_checkpoint(
        path, model, step=step,
        optimizer_state=optimizer.state_dict(), metrics=payload,
    )


def resume_state(path: str | Path, device: str):
    model, checkpoint = load_chat_checkpoint(path, device)
    metrics = checkpoint.get("metrics", {}) or {}
    restore_rng_state(metrics.get("rng_state"))
    return model, checkpoint, metrics.get("counters", {})


def load_resume_into(model, optimizer, path: str | Path, *, device: str) -> dict:
    """Restore weights, optimizer and RNG into the *existing* objects.

    `resume_state` builds a fresh model, which is wrong mid-notebook: the caller
    already holds one and everything downstream refers to it. This loads in
    place and returns the counters, so a preempted run continues on the same
    object graph and the same random stream.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    # Both savers write "model_state", "optimizer_state" and "metrics", so the
    # resume path is architecture-agnostic from here on.
    if checkpoint.get("optimizer_state") and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    metrics = checkpoint.get("metrics", {}) or {}
    restore_rng_state(metrics.get("rng_state"))
    counters = metrics.get("counters", {}) or {}
    return {
        "step": int(counters.get("step", checkpoint.get("step", 0))),
        "target_tokens": int(counters.get("target_tokens", 0)),
        "segment": metrics.get("segment"),
    }


def read_history(path: str | Path) -> list[dict]:
    """Replay `metrics.jsonl` so a resumed run keeps its earlier milestones."""
    path = Path(path)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class SegmentTrainer:
    """Run one segment against a token budget, evaluating at milestones."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        config: SegmentConfig,
        train_rows,
        valid_rows,
        replay_documents=None,
        language_eval_documents=None,
        device: str = "cuda",
        output_dir: str | Path = "runs/d20",
        evaluator=None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.evaluator = evaluator
        self.history: list[dict] = []

        if config.objective == "chat":
            self.train_data = ConversationDataset(
                train_rows, tokenizer, max_length=config.sequence_length
            )
            self.valid_data = ConversationDataset(
                valid_rows, tokenizer, max_length=config.sequence_length
            )
            collate = collate_conversations
        else:
            self.train_data = LanguageDocumentDataset(
                train_rows, tokenizer, max_length=config.sequence_length
            )
            self.valid_data = LanguageDocumentDataset(
                valid_rows, tokenizer, max_length=config.sequence_length
            )
            collate = collate_language_windows

        self.train_loader = DataLoader(
            self.train_data,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate,
            drop_last=True,
            pin_memory=device.startswith("cuda"),
        )
        self.valid_loader = DataLoader(
            self.valid_data,
            batch_size=config.batch_size,
            collate_fn=collate,
        )
        self.period = replay_period(config.replay_ratio)
        self.replay_iterator = None
        if self.period is not None:
            if replay_documents is None:
                raise ValueError("replay_ratio > 0 requires replay_documents")
            replay_data = LanguageDocumentDataset(
                replay_documents, tokenizer, max_length=config.sequence_length
            )
            self.replay_iterator = infinite(
                DataLoader(
                    replay_data,
                    batch_size=config.batch_size,
                    shuffle=True,
                    collate_fn=collate_language_windows,
                    drop_last=True,
                )
            )
        self.language_loader = None
        self.language_dataset = None
        if language_eval_documents is not None:
            self.language_dataset = LanguageDocumentDataset(
                language_eval_documents,
                tokenizer,
                max_length=config.sequence_length,
            )
            self.language_loader = DataLoader(
                self.language_dataset,
                batch_size=config.batch_size,
                collate_fn=collate_language_windows,
            )

    def evaluate(self, *, step: int, seen: int, train_loss: float | None) -> dict:
        self.model.eval()
        # The held-out metric follows the objective. A chat segment is scored on
        # assistant-only likelihood; the language evaluator needs a document
        # dataset and its byte counts, which a ConversationDataset has not got.
        if self.config.objective == "chat":
            held_out = evaluate_assistant_likelihood(
                self.model, self.valid_loader, self.device
            )
        else:
            held_out = evaluate_language_model(
                self.model, self.valid_data, self.valid_loader, self.device
            )
        language = None
        if self.language_loader is not None:
            language = evaluate_language_model(
                self.model,
                self.language_dataset,
                self.language_loader,
                self.device,
            )
        record = {
            "segment": self.config.name,
            "objective": self.config.objective,
            "step": step,
            "target_tokens": seen,
            "train_loss": train_loss,
            "held_out": held_out,
            "language_retention": language,
            "wall_clock": time.time(),
        }
        if self.evaluator is not None:
            record["behavior"] = self.evaluator(self.model, self.tokenizer)
        self.model.train()
        self.history.append(record)
        with (self.output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        return record

    def run(
        self,
        *,
        start_step: int = 0,
        start_tokens: int = 0,
        resume: bool = True,
    ) -> list[dict]:
        config = self.config
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate
        )
        latest = self.output / "latest.pt"
        final = self.output / "final.pt"
        if resume and final.exists():
            # The segment already completed. Load its end state and return the
            # recorded history instead of retraining it.
            load_resume_into(self.model, optimizer, final, device=self.device)
            self.history = read_history(self.output / "metrics.jsonl")
            print(f"[{config.name}] already complete; loaded final.pt "
                  f"({len(self.history)} recorded evaluations)")
            return self.history
        if resume and latest.exists():
            state = load_resume_into(
                self.model, optimizer, latest, device=self.device
            )
            start_step = state["step"]
            start_tokens = state["target_tokens"]
            self.history = read_history(self.output / "metrics.jsonl")
            print(f"[{config.name}] resuming at step {start_step:,}, "
                  f"{start_tokens:,} target tokens")
        self.model.train()
        iterator = infinite(self.train_loader)
        step = start_step
        seen = start_tokens
        pending = [item for item in config.milestones if item > seen]
        next_periodic = (
            seen + config.evaluate_every_tokens
            if config.evaluate_every_tokens
            else None
        )
        if seen == 0 and not self.history:
            self.evaluate(step=step, seen=seen, train_loss=None)

        # Lightweight heartbeat. Off unless D20_PROGRESS_EVERY is set, so
        # existing runs and their logs are unchanged. Deliberately one line:
        # a detached screen session is read by scrolling, not by grepping.
        progress_every = int(os.environ.get("D20_PROGRESS_EVERY", "0") or 0)
        progress_t0 = time.time()
        progress_tokens0 = seen

        autocast = torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.device.startswith("cuda"),
        )
        # Gradient accumulation. Defaults to 1, so every existing run keeps the
        # exact behaviour it had: one microbatch, one optimiser step. Raising it
        # lets a configuration that only fits at a small microbatch still take an
        # update over the same number of tokens as a wider one -- which is the
        # only honest way to compare n=64 at microbatch 1 against n=16 at 8.
        accum = max(1, int(getattr(self, "grad_accum", 1)))
        while seen < config.target_tokens:
            replay = is_replay_step(step, self.period)
            optimizer.zero_grad(set_to_none=True)
            micro_loss = None
            for _ in range(accum):
                tokens, mask = next(
                    self.replay_iterator if replay else iterator
                )
                tokens = tokens.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                with autocast:
                    # Scale so the accumulated gradient matches what a single
                    # microbatch of accum times the size would have produced.
                    loss = self.model.loss(tokens, mask) / accum
                loss.backward()
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite loss at step {step} of segment "
                        f"{config.name!r}; refusing to continue"
                    )
                micro_loss = loss.detach()
                # Replay tokens never count: segments must stay comparable at
                # equal supervision of the objective under test.
                if not replay:
                    seen += int(mask[:, 1:].sum().detach().cpu())
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), config.grad_clip
            )
            optimizer.step()
            step += 1
            # Report on the original scale, not the divided one.
            loss = micro_loss * accum

            if progress_every and step % progress_every == 0:
                share = 100.0 * seen / max(config.target_tokens, 1)
                run_s = max(time.time() - progress_t0, 1e-9)
                rate = (seen - progress_tokens0) / run_s
                left = config.target_tokens - seen
                eta = left / rate / 60.0 if rate > 0 else float("inf")
                print(
                    f"[{config.name}] step {step:>6,} | "
                    f"{seen:>10,}/{config.target_tokens:,} tok ({share:5.1f}%) | "
                    f"loss {float(loss.item()):6.4f} | {rate:6,.0f} tok/s | "
                    f"ETA {eta:5.0f} min",
                    flush=True,
                )

            reached = [item for item in pending if item <= seen]
            due = next_periodic is not None and seen >= next_periodic
            if reached or due or seen >= config.target_tokens:
                self.evaluate(
                    step=step,
                    seen=seen,
                    train_loss=None if replay else float(loss.item()),
                )
                pending = [item for item in pending if item > seen]
                if due:
                    next_periodic = seen + config.evaluate_every_tokens
            if step % config.checkpoint_every_steps == 0:
                save_full_checkpoint(
                    self.output / "latest.pt",
                    self.model,
                    step=step,
                    optimizer=optimizer,
                    counters={"target_tokens": seen, "step": step},
                    config=config,
                    metrics={"segment": config.name},
                )
        save_full_checkpoint(
            self.output / "final.pt",
            self.model,
            step=step,
            optimizer=optimizer,
            counters={"target_tokens": seen, "step": step},
            config=config,
            metrics={"segment": config.name, "history": self.history},
        )
        return self.history


def autotune_batch_size(
    model,
    tokenizer,
    rows,
    *,
    objective: str,
    device: str,
    sequence_length: int = 256,
    candidates: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256),
    safety: float = 0.85,
) -> dict:
    """Find the largest batch that runs a full step without OOM.

    A forward alone is not enough: the backward through ``k`` dynamics steps is
    what actually decides the ceiling. Each candidate therefore runs a complete
    forward, backward and optimizer step.
    """
    if objective == "chat":
        dataset = ConversationDataset(
            rows, tokenizer, max_length=sequence_length
        )
        collate = collate_conversations
    else:
        dataset = LanguageDocumentDataset(
            rows, tokenizer, max_length=sequence_length
        )
        collate = collate_language_windows

    results = []
    chosen = None
    for batch in candidates:
        if batch > len(dataset):
            break
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        loader = DataLoader(
            dataset, batch_size=batch, collate_fn=collate, drop_last=True
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-8)
        try:
            tokens, mask = next(iter(loader))
            tokens = tokens.to(device)
            mask = mask.to(device)
            start = time.time()
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.startswith("cuda"),
            ):
                loss = model.loss(tokens, mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.time() - start
            peak = (
                torch.cuda.max_memory_allocated() / 1e9
                if device.startswith("cuda")
                else 0.0
            )
            results.append(
                {
                    "batch_size": batch,
                    "ok": True,
                    "peak_memory_gb": peak,
                    "seconds_per_step": elapsed,
                    "target_tokens_per_step": int(mask[:, 1:].sum()),
                }
            )
            chosen = batch
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            # Torch raises several different types for exhaustion depending on
            # version and allocator path, so match on the message too rather
            # than on one class.
            message = str(error).lower()
            if "out of memory" not in message and not isinstance(
                error, torch.cuda.OutOfMemoryError
            ):
                raise
            results.append({"batch_size": batch, "ok": False, "reason": "OOM"})
            break
        finally:
            optimizer.zero_grad(set_to_none=True)
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    if chosen is None:
        smallest = candidates[0] if candidates else None
        raise RuntimeError(
            f"no candidate batch size completed a training step; the smallest "
            f"tried was {smallest}. The model does not fit this GPU at "
            f"sequence_length={sequence_length}. Reduce the sequence length, "
            "reduce memory_size, or use a larger device."
        )
    # Step back one candidate when the largest working batch sat close to the
    # limit: a job that OOMs at hour six is worse than one that runs slower.
    safe = chosen
    successful = [item for item in results if item.get("ok")]
    if successful and device.startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        if successful[-1]["peak_memory_gb"] > safety * total and len(successful) > 1:
            safe = successful[-2]["batch_size"]
    return {
        "candidates": results,
        "largest_working": chosen,
        "chosen": safe,
        "safety_fraction": safety,
    }
