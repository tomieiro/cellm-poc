"""Run the 50M-token comparison using the archived D4 architecture and trainer."""

import argparse
import json
import sys
from pathlib import Path

import torch

from cellm_poc.protocol import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artefacts", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check and not all((args.corpus, args.tokenizer, args.output)):
        parser.error("informe --corpus, --tokenizer e --output ou use --check")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "historical/d4"))
    from celllm.chat_tokenizer import ChatTokenizer
    from celllm.language_data import LanguageDocument
    from d20.architecture import D20Shape, build_model
    from d20.baseline_transformer import BaselineTransformer, TransformerConfig
    from d20.datasets import read_jsonl
    from d20.provenance import seed_everything
    from d20.training import SegmentConfig, SegmentTrainer

    if not args.check:
        checksums = dict(
            line.split()[::-1]
            for line in (
                args.artefacts / "tinystories-d4/etapa-9-d4-manifesto-sha256.txt"
            )
            .read_text()
            .splitlines()
            if line and not line.startswith("#")
        )
        expected_inputs = {
            "runs/d20_showcase/data/language_raw.jsonl": args.corpus,
            "runs/d20_showcase/tokenizer/tokenizer.json": args.tokenizer,
        }
        for name, path in expected_inputs.items():
            if sha256(path) != checksums[name]:
                raise ValueError(
                    f"SHA-256 do insumo D4 difere do registro histórico: {path}"
                )
        tokenizer = ChatTokenizer.load(args.tokenizer)
        corpus = read_jsonl(args.corpus)
        if len(corpus) != 200_000 or tokenizer.vocab_size != 2048:
            raise ValueError(
                "D4 exige o corpus ordenado de 200.000 documentos e vocabulário 2.048"
            )
        split = int(0.98 * len(corpus))
        documents = [LanguageDocument(row["id"], row["text"]) for row in corpus]
        args.output.mkdir(parents=True)
        manifest = {
            "corpus_sha256": sha256(args.corpus),
            "tokenizer_sha256": sha256(args.tokenizer),
            "torch": torch.__version__,
            "device": args.device,
            "train_documents": split,
            "validation_documents": len(corpus) - split,
        }
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    shapes = {
        "cellm": (896, 1, 3_337_865),
        "wide": (240, 4, 3_330_720),
        "deep": (144, 12, 3_340_512),
    }
    for arm, (width, layers, expected) in shapes.items():
        seed_everything(7)
        model = (
            build_model(D20Shape(width, 16, 2048, 96))
            if arm == "cellm"
            else BaselineTransformer(
                TransformerConfig(
                    vocab_size=2048,
                    context=256,
                    dimensions=width,
                    layers=layers,
                    heads=4,
                )
            )
        )
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if count != expected:
            raise ValueError(
                f"número de parâmetros incompatível em {arm}: {count} != {expected}"
            )
        print(f"{arm}: {count} parâmetros", flush=True)
        if args.check:
            continue
        config = SegmentConfig(
            name="d20_a_language" if arm == "cellm" else f"baseline_{arm}",
            objective="language",
            target_tokens=50_000_000,
            learning_rate=3e-4,
            replay_ratio=0.0,
            batch_size=8,
            sequence_length=256,
            milestones=(
                2_000_000,
                5_000_000,
                10_000_000,
                20_000_000,
                35_000_000,
                50_000_000,
            ),
            seed=7,
            checkpoint_every_steps=100 if arm == "cellm" else 200,
        )
        trainer = SegmentTrainer(
            model.to(args.device),
            tokenizer,
            config=config,
            train_rows=documents[:split],
            valid_rows=documents[split:],
            device=args.device,
            output_dir=args.output / arm,
        )
        history = trainer.run()
        (args.output / f"{arm}.json").write_text(
            json.dumps(
                {"parameters": count, "history": history}, indent=2, allow_nan=False
            )
            + "\n"
        )
        del trainer, model


if __name__ == "__main__":
    main()
