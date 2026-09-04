"""Source registry, licence provenance and the token budget derived from it.

Budgets are NOT constants here. The target is a preflight output -- measured
tok/s times available wall clock times 0.85 -- so this module stores weights
and derives token counts from whatever target the budget step produces. See
celllm600m.schedule.

`chat` sources are materialised with explicit <|system|>/<|user|>/<|assistant|>
markers rather than flattened, so the model meets turn structure during
pretraining instead of only at finetuning. They are 33% of the mixture.

aya is deliberately NOT in the pretraining mixture. Measured on its own data:
71 languages, of which Portuguese is 1.84% of the characters. At the 5% weight
it originally carried it would have delivered 0.09% of the corpus in
Portuguese and 4.75% in ~68 languages a 360M model cannot learn from a 2B
budget -- fragmenting at 2.808 bytes/token against a 3.951 mixture average.
Its 5% went to tulu3 and smol_smoltalk, holding chat at 33%. It survives in
SFT_SOURCES, filtered to Portuguese, where 8,997 human-written pairs are worth
having against smol_smoltalk's synthetic English.

`licence_review` marks a source whose terms need a decision before the corpus
is redistributed. It does not block training; it blocks shipping the corpus.
"""

from __future__ import annotations

#: Hard ceilings, measured by materialising each source to exhaustion at a 6B
#: target. Weights above these are silently unmet: the materialiser stops when
#: the dataset runs out and the mixture drifts without erroring.
#:
#:   tulu3          939,262 rows  ~706M tokens   (the whole SFT mixture)
#:   smol_smoltalk  460,341 rows  ~451M tokens   (the whole dataset)
#:
#: Together the chat sources cap at roughly 1.16B tokens. Any corpus larger
#: than ~3.5B therefore CANNOT hold 33 percent chat, no matter the weights.
#: This is a property of the data, not of the mixture: asking for more chat
#: means adding a chat source, not raising a weight.
SOURCE_TOKEN_CEILING = {"tulu3": 706_000_000, "smol_smoltalk": 451_000_000}

#: weight = share of the *token* budget, not of documents. Sums to 100.
SOURCES = {
    "fineweb_edu": {
        "dataset": "HuggingFaceTB/smollm-corpus", "config": "fineweb-edu-dedup",
        "weight": 30, "chat": False, "domain": "en_web",
        "licence": "odc-by-1.0", "licence_review": None,
    },
    "cosmopedia": {
        "dataset": "HuggingFaceTB/smollm-corpus", "config": "cosmopedia-v2",
        "weight": 15, "chat": False, "domain": "en_synthetic",
        "licence": "apache-2.0", "licence_review": None,
    },
    "fineweb2_pt": {
        "dataset": "HuggingFaceFW/fineweb-2", "config": "por_Latn",
        "weight": 12, "chat": False, "domain": "pt",
        "licence": "odc-by-1.0", "licence_review": None,
    },
    "pes2o": {
        "dataset": "allenai/peS2o", "config": "v3",
        "weight": 5, "chat": False, "domain": "stem",
        "licence": "odc-by-1.0", "licence_review": None,
    },
    "wikipedia_en": {
        "dataset": "wikimedia/wikipedia", "config": "20231101.en",
        "weight": 5, "chat": False, "domain": "encyclopaedic",
        "licence": "cc-by-sa-4.0",
        "licence_review": "Share-alike. Fine to train on; redistributing the "
                          "packed corpus would carry SA obligations. Drop this "
                          "source if the corpus itself must ship permissively.",
    },
    "tulu3": {
        "dataset": "allenai/tulu-3-sft-mixture", "config": None,
        "weight": 20, "chat": True, "domain": "instruction",
        "licence": "odc-by-1.0",
        "licence_review": "Collection is ODC-BY but subsets carry their own "
                          "terms, including non-commercial ones and model "
                          "outputs subject to third-party terms. Enumerate the "
                          "`source` field with tools/audit_tulu_subsets.py and "
                          "set TULU3_EXCLUDE before materialising.",
    },
    "smol_smoltalk": {
        "dataset": "HuggingFaceTB/smol-smoltalk", "config": None,
        "weight": 13, "chat": True, "domain": "chat",
        "licence": "apache-2.0", "licence_review": None,
    },
}

#: Tulu-3 `source` values to drop. EMPTY BY DESIGN -- populate it from the
#: audit tool against the subsets actually present, rather than from a guessed
#: list. Materialising with an empty exclude set is a deliberate choice to be
#: recorded, not a default to be inherited silently.
TULU3_EXCLUDE: set[str] = set()

#: Finetuning-only sources. aya lives here rather than in SOURCES: 100-150M
#: tokens of assistant-only loss is a scale where 8,997 human-written
#: Portuguese pairs matter, and where 30x repetition of a small set is normal
#: rather than a defect.
SFT_SOURCES = {
    "aya_pt": {
        "dataset": "CohereForAI/aya_dataset", "config": None,
        "weight": 20, "chat": True, "domain": "pt_instruction",
        "licence": "apache-2.0", "licence_review": None,
        "keep_languages": ("Portuguese",),
    },
}

SFT_WEIGHTS = {"smol_smoltalk": 45, "tulu3": 35, "aya_pt": 20}


def chat_fraction() -> float:
    total = sum(s["weight"] for s in SOURCES.values())
    return sum(s["weight"] for s in SOURCES.values() if s["chat"]) / total


def unmet_weights(target_tokens: int, margin: float = 0.05) -> dict[str, tuple[int, int]]:
    """Sources whose share of this target exceeds what the dataset contains."""
    return {
        name: (want, SOURCE_TOKEN_CEILING[name])
        for name, want in budgets(target_tokens, margin).items()
        if name in SOURCE_TOKEN_CEILING and want > SOURCE_TOKEN_CEILING[name]
    }


def budgets(target_tokens: int, margin: float = 0.05) -> dict[str, int]:
    """Tokens to materialise per source, with a small over-fetch margin.

    Packing drops the tail of any document that will not fill a context, so
    materialising exactly `target_tokens` yields slightly fewer usable ones.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if not 0.0 <= margin < 1.0:
        raise ValueError("margin must be in [0, 1)")
    total = sum(s["weight"] for s in SOURCES.values())
    scaled = target_tokens * (1.0 + margin)
    return {name: int(scaled * s["weight"] / total) for name, s in SOURCES.items()}


def needs_licence_review() -> dict[str, str]:
    return {n: s["licence_review"] for n, s in SOURCES.items() if s["licence_review"]}
