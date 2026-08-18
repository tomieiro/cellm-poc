"""Environment, git and run provenance for the D20 showcase."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git(repository: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def environment(repository: str | Path) -> dict:
    """Capture everything needed to explain a number after the fact."""
    import torch

    repository = Path(repository)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "gpu_memory_total_gb": (
            round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
            if torch.cuda.is_available()
            else None
        ),
        "git": {
            "commit": _git(repository, "rev-parse", "HEAD"),
            "branch": _git(repository, "rev-parse", "--abbrev-ref", "HEAD"),
            # A dirty tree means the recorded commit does not fully describe
            # the code that produced the run; record the diff summary too.
            "status": _git(repository, "status", "--short"),
            "clean": _git(repository, "status", "--short") == "",
        },
        "slurm": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("SLURM_")
        },
        "environment_variables": {
            key: os.environ.get(key)
            for key in (
                "HF_HOME",
                "HF_DATASETS_CACHE",
                "TRANSFORMERS_CACHE",
                "CELL_LM_CACHE",
                "TOKENIZERS_PARALLELISM",
            )
        },
    }


def write_manifest(path: str | Path, payload: dict) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return destination


def seed_everything(seed: int) -> dict:
    """Seed every generator the run touches and report what was set."""
    import random

    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return {
        "seed": seed,
        "python_random": True,
        "numpy": True,
        "torch_cpu": True,
        "torch_cuda": torch.cuda.is_available(),
    }


def rng_state() -> dict:
    """Snapshot RNG state so a resumed run continues the same stream.

    Everything is stored as tensors and plain Python scalars. A numpy array or a
    bare tuple here would make the whole checkpoint unloadable under
    ``torch.load(weights_only=True)``, which is what `load_chat_checkpoint`
    uses -- so the training checkpoints would be unreadable by the standard
    loader and could not be served for inference.
    """
    import random

    import numpy
    import torch

    version, internal, gauss = random.getstate()
    numpy_state = numpy.random.get_state()
    return {
        "python_version": int(version),
        "python_internal": torch.tensor(list(internal), dtype=torch.int64),
        "python_gauss": float(gauss) if gauss is not None else None,
        "numpy_kind": str(numpy_state[0]),
        "numpy_keys": torch.tensor(
            numpy_state[1].astype("int64"), dtype=torch.int64
        ),
        "numpy_pos": int(numpy_state[2]),
        "numpy_has_gauss": int(numpy_state[3]),
        "numpy_cached_gauss": float(numpy_state[4]),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(state: dict) -> None:
    """Restore the generators saved by :func:`rng_state`.

    Round-tripping through ``torch.save`` loosens the types: tuples become
    lists and the CPU generator state comes back as a plain tensor rather than
    the ByteTensor ``set_rng_state`` demands, so each field is coerced back.
    """
    import random

    import numpy
    import torch

    if not state:
        return
    if "python_internal" in state:
        random.setstate((
            int(state["python_version"]),
            tuple(int(item) for item in state["python_internal"].tolist()),
            state["python_gauss"],
        ))
        numpy.random.set_state((
            state["numpy_kind"],
            numpy.asarray(state["numpy_keys"].tolist(), dtype=numpy.uint32),
            int(state["numpy_pos"]),
            int(state["numpy_has_gauss"]),
            float(state["numpy_cached_gauss"]),
        ))
    elif state.get("python") is not None:
        # Checkpoints written before the tensor-only encoding.
        version, internal, gauss = state["python"]
        random.setstate((version, tuple(internal), gauss))
        if state.get("numpy") is not None:
            numpy.random.set_state(tuple(state["numpy"]))
    cpu = state.get("torch_cpu")
    if cpu is not None:
        torch.set_rng_state(torch.as_tensor(cpu, dtype=torch.uint8).cpu())
    cuda = state.get("torch_cuda")
    if cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(item, dtype=torch.uint8).cpu() for item in cuda]
        )
