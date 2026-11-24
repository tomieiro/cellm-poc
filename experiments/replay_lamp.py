"""Re-score the archived LAMP ablation using the local weights and token windows."""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from cellm_poc.history import LAMP_ARTIFACT, SOURCE_WEIGHTS, summarize_lamp
from cellm_poc.protocol import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artefacts", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "historical/lamp"))
    from measure_stage18_haa import (
        DemoConfig,
        PATHS,
        build_model,
        evaluate_window,
        raw_state,
    )

    saved = json.loads((args.artefacts / LAMP_ARTIFACT).read_text())
    metadata = saved["checkpoint"]
    checkpoint = args.artefacts / "lamp/weights.pt"
    windows_file = args.artefacts / "lamp/windows.npz"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if (
        payload["source_sha256"] != metadata["sha256"]
        or payload["config_sha256"] != DemoConfig().fingerprint()
    ):
        raise ValueError("checkpoint incompatível com o resultado histórico")
    with torch.device("meta"):
        model = build_model()
    model.load_state_dict(raw_state(payload), strict=True, assign=True)
    del payload
    model = model.to("cuda", dtype=torch.bfloat16).eval()

    offsets = saved["quality"]["protocol"]["starts"]
    paths = {path.name: path for path in PATHS}
    windows = {}
    with np.load(windows_file, allow_pickle=False) as archive, torch.inference_mode():
        for source, starts in offsets.items():
            data = archive[source]
            if data.shape != (len(starts), 2048):
                raise ValueError(f"janelas históricas incompatíveis em {source}")
            windows[source] = []
            for offset, tokens in zip(starts, data, strict=True):
                ids = torch.tensor(tokens.astype(np.int64), device="cuda")[None]
                row = {
                    arm: evaluate_window(model, ids, paths[arm])
                    for arm in ("full", "local_off")
                }
                windows[source].append({"start": offset, **row})
            print(source, flush=True)
    result = {
        "source_checkpoint_sha256": metadata["sha256"],
        "inference_weights_sha256": sha256(checkpoint),
        "windows_sha256": sha256(windows_file),
        "torch": torch.__version__,
        "celnn": version("celnn"),
        "gpu": torch.cuda.get_device_name(),
        "windows": windows,
        "summary": summarize_lamp(windows, SOURCE_WEIGHTS),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
