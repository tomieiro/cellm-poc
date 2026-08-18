# CellLM-D20 Showcase — report

Generated 2026-08-18T14:05:02.449556+00:00

> This is a **showcase**, not a causal experiment of the B0.x line. Its numbers must not be mixed with the B0.x conclusions.

## Architecture

```json
{
  "shape": {
    "dimensions": 896,
    "cells": 16,
    "vocab_size": 2048,
    "memory_size": 96
  },
  "parameters": {
    "total": 3337865,
    "trainable": 3337865,
    "by_component": {
      "embedding": 1835008,
      "head": 2048,
      "chua_yang_core": 4480,
      "projection": 802816,
      "bank_memory": 345861,
      "local_memory": 347652,
      "other": 0
    },
    "share_by_component": {
      "embedding": 0.5497550080665335,
      "head": 0.0006135658572171133,
      "chua_yang_core": 0.0013421753126624354,
      "projection": 0.24051781602910843,
      "bank_memory": 0.10361743210105862,
      "local_memory": 0.10415400263341987,
      "other": 0.0
    },
    "embedding_parameters": 1837056,
    "embedding_fraction": 0.5503685739237507,
    "non_embedding_parameters": 1500809,
    "non_embedding_fraction": 0.44963142607624934,
    "chua_yang_parameters": 4480,
    "chua_yang_fraction": 0.0013421753126624354,
    "chua_yang_applications_per_forward": 15,
    "memory_parameters": 693513,
    "memory_fraction": 0.20777143473447848,
    "projection_parameters": 802816,
    "projection_fraction": 0.24051781602910843,
    "head_parameters": 2048
  },
  "plastic_state_mb_per_sequence": 0.595968,
  "selection_rule": "in-band and state-fitting, minimum embedding_fraction"
}
```

## Tokenizer

```json
{
  "vocab_size": 2048,
  "special_tokens": [
    "<pad>",
    "<unk>",
    "<bos>",
    "<user>",
    "<assistant>",
    "<eos>"
  ],
  "special_token_ids": {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<user>": 3,
    "<assistant>": 4,
    "<eos>": 5
  },
  "critical_words": [
    "Yes",
    "No",
    "Hello",
    "The",
    "I",
    "A"
  ],
  "critical_failures": [],
  "fragmented_color_words": [
    "Blue",
    "Green",
    "White"
  ],
  "single_token_colors": [
    "Black",
    "Yellow",
    "Red",
    "Orange",
    "Purple",
    "Brown"
  ],
  "single_token_rate": 0.45454545454545453,
  "passed": true
}
```

## Data

```json
{
  "language": {
    "repo": "roneneldan/TinyStories",
    "split": "train",
    "requested": 200000,
    "written": 200000,
    "min_chars": 120,
    "note": "baixado manualmente apos o run contaminado"
  },
  "chat": {
    "repo": "HuggingFaceTB/smol-smoltalk",
    "split": "train",
    "requested": 200000,
    "written": 200000,
    "note": "baixado manualmente apos o run contaminado"
  },
  "filter": {
    "input": 200000,
    "kept": 26480,
    "kept_fraction": 0.1324,
    "system_turns_stripped": 30633,
    "dialogues_truncated": 8739,
    "rejection_reasons": {
      "no_usable_prefix": 134300,
      "structured_content": 39220
    }
  },
  "statistics_after": {
    "dialogues": 26436,
    "turns": 64618,
    "turn_distribution": {
      "2": 22882,
      "6": 2319,
      "4": 1235
    },
    "sources": {
      "smol-magpie-ultra-short": 8405,
      "smol-contraints": 7719,
      "openhermes-50k": 5417,
      "smollm-rewrite-30k": 1616,
      "explore-instruct-rewrite": 1278,
      "everyday-conversations": 980,
      "smol-summarize-5k": 764,
      "smol-summarize-20k": 257
    },
    "user_chars": {
      "min": 2,
      "p25": 87,
      "median": 170,
      "p75": 260,
      "p95": 363,
      "max": 400
    },
    "assistant_chars": {
      "min": 8,
      "p25": 215,
      "median": 371,
      "p75": 489,
      "p95": 579,
      "max": 600
    }
  },
  "gates": {
    "splits": {
      "train": {
        "dialogues": 24850,
        "dominant_first_word": "to",
        "dominant_first_word_rate": 0.11726358148893361,
        "dominant_response": "hello how can i help you today",
        "dominant_response_rate": 0.030427766983896994,
        "strata": [
          "everyday-conversations",
          "explore-instruct-rewrite",
          "openhermes-50k",
          "smol-contraints",
          "smol-magpie-ultra-short",
          "smol-summarize-20k",
          "smol-summarize-5k",
          "smollm-rewrite-30k"
        ],
        "train_leakage": 0
      },
      "validation": {
        "dialogues": 793,
        "dominant_first_word": "to",
        "dominant_first_word_rate": 0.11979823455233292,
        "dominant_response": "hello how can i help you today",
        "dominant_response_rate": 0.02998965873836608,
        "strata": [
          "everyday-conversations",
          "explore-instruct-rewrite",
          "openhermes-50k",
          "smol-contraints",
          "smol-magpie-ultra-short",
          "smol-summarize-20k",
          "smol-summarize-5k",
          "smollm-rewrite-30k"
        ],
        "train_leakage": 0
      },
      "test": {
        "dialogues": 793,
        "dominant_first_word": "the",
        "dominant_first_word_rate": 0.1210592686002522,
        "dominant_response": "hello how can i help you today",
        "dominant_response_rate": 0.029743589743589743,
        "strata": [
          "everyday-conversations",
          "explore-instruct-rewrite",
          "openhermes-50k",
          "smol-contraints",
          "smol-magpie-ultra-short",
          "smol-summarize-20k",
          "smol-summarize-5k",
          "smollm-rewrite-30k"
        ],
        "train_leakage": 0
      }
    },
    "failures": [],
    "passed": true
  }
}
```

## Language pretraining (D20-A)

| target tokens | validation NLL | validation PPL |
| ---: | ---: | ---: |
| 0 | 7.6240 | 2046.83 |
| 2,000,888 | 4.2667 | 71.29 |
| 5,001,636 | 3.6227 | 37.44 |
| 10,000,707 | 3.2380 | 25.48 |
| 20,000,254 | 2.9614 | 19.32 |
| 35,000,920 | 2.7011 | 14.90 |
| 50,000,801 | 2.5363 | 12.63 |

## Chat phases

| segment | assistant tokens | chat NLL | TinyStories PPL | behaviour | first word | dominant first word |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d20_b_bootstrap | 0 | 8.3077 | 12.35 | 0.021 | 0.000 | 0.335 |
| d20_c_segment_001 | 0 | 7.5227 | 13.02 | 0.262 | 0.300 | 0.131 |
| d20_c_segment_002 | 0 | 3.8473 | 16.38 | 0.064 | 0.100 | 0.426 |
| d20_b_bootstrap | 25,076 | 0.7140 | 15.42 | 0.170 | 0.222 | 0.318 |
| d20_b_bootstrap | 50,141 | 0.5524 | 15.04 | 0.284 | 0.344 | 0.165 |
| d20_b_bootstrap | 75,141 | 0.5035 | 14.80 | 0.305 | 0.322 | 0.165 |
| d20_b_bootstrap | 100,148 | 0.4879 | 14.52 | 0.234 | 0.267 | 0.119 |
| d20_b_bootstrap | 125,221 | 0.4967 | 14.23 | 0.241 | 0.278 | 0.210 |
| d20_b_bootstrap | 150,222 | 0.4764 | 13.86 | 0.284 | 0.278 | 0.176 |
| d20_b_bootstrap | 175,254 | 0.4751 | 13.65 | 0.241 | 0.256 | 0.153 |
| d20_b_bootstrap | 200,273 | 0.4760 | 13.42 | 0.262 | 0.311 | 0.170 |
| d20_c_segment_002 | 200,327 | 3.7892 | 14.98 | 0.085 | 0.122 | 0.278 |
| d20_c_segment_001 | 200,610 | 4.5367 | 16.24 | 0.149 | 0.222 | 0.261 |
| d20_b_bootstrap | 225,303 | 0.4692 | 13.50 | 0.248 | 0.267 | 0.153 |
| d20_b_bootstrap | 250,350 | 0.4494 | 13.25 | 0.284 | 0.300 | 0.159 |
| d20_b_bootstrap | 275,373 | 0.4984 | 13.04 | 0.262 | 0.278 | 0.136 |
| d20_b_bootstrap | 300,051 | 0.4862 | 13.02 | 0.262 | 0.300 | 0.131 |
| d20_c_segment_001 | 400,632 | 4.2469 | 15.96 | 0.135 | 0.222 | 0.210 |
| d20_c_segment_002 | 401,172 | 3.7305 | 14.54 | 0.092 | 0.144 | 0.381 |
| d20_c_segment_002 | 601,271 | 3.6823 | 14.49 | 0.085 | 0.122 | 0.375 |
| d20_c_segment_001 | 601,757 | 4.0540 | 16.20 | 0.050 | 0.144 | 0.176 |
| d20_c_segment_002 | 801,429 | 3.6339 | 14.39 | 0.099 | 0.144 | 0.477 |
| d20_c_segment_001 | 802,534 | 3.9238 | 16.20 | 0.085 | 0.133 | 0.210 |
| d20_c_segment_001 | 1,000,403 | 3.8473 | 16.38 | 0.064 | 0.100 | 0.426 |
| d20_c_segment_002 | 1,002,604 | 3.6007 | 14.17 | 0.099 | 0.133 | 0.426 |
| d20_c_segment_002 | 1,202,667 | 3.5567 | 14.24 | 0.092 | 0.144 | 0.381 |
| d20_c_segment_002 | 1,403,784 | 3.5197 | 14.12 | 0.099 | 0.144 | 0.455 |
| d20_c_segment_002 | 1,604,482 | 3.4906 | 14.10 | 0.085 | 0.133 | 0.682 |
| d20_c_segment_002 | 1,805,059 | 3.4561 | 14.10 | 0.099 | 0.144 | 0.540 |
| d20_c_segment_002 | 2,000,336 | 3.4329 | 14.04 | 0.085 | 0.133 | 0.591 |

## Showcase selection

```json
{
  "constraints": {
    "max_language_ppl": 18.949390620617226,
    "max_chat_nll": Infinity,
    "min_behavior_accuracy": 0.0,
    "min_first_token_accuracy": null,
    "max_dominant_first_word_rate": 0.35,
    "max_dominant_response_rate": 0.35
  },
  "evaluated": 30,
  "eligible": 19,
  "pareto_front_eligible": [
    {
      "label": "d20_b_bootstrap@0",
      "path": "runs/d20_showcase/d20_c_chat/d20_b_bootstrap/final.pt",
      "target_tokens": 0,
      "language_ppl": 12.346700531928073,
      "chat_nll": 8.307723474431288,
      "behavior_accuracy": 0.02127659574468085,
      "first_token_accuracy": 0.0,
      "dominant_first_word_rate": 0.3352272727272727,
      "dominant_response_rate": 0.017045454545454544
    },
    {
      "label": "d20_b_bootstrap@75141",
      "path": "runs/d20_showcase/d20_c_chat/d20_b_bootstrap/final.pt",
      "target_tokens": 75141,
      "language_ppl": 14.803841332228563,
      "chat_nll": 0.5034808784678735,
      "behavior_accuracy": 0.3049645390070922,
      "first_token_accuracy": 0.32222222222222224,
      "dominant_first_word_rate": 0.16477272727272727,
      "dominant_response_rate": 0.056818181818181816
    },
    {
      "label": "d20_b_bootstrap@250350",
      "path": "runs/d20_showcase/d20_c_chat/d20_b_bootstrap/final.pt",
      "target_tokens": 250350,
      "language_ppl": 13.253008020233327,
      "chat_nll": 0.4494036901157533,
      "behavior_accuracy": 0.28368794326241137,
      "first_token_accuracy": 0.3,
      "dominant_first_word_rate": 0.1590909090909091,
      "dominant_response_rate": 0.056818181818181816
    },
    {
      "label": "d20_b_bootstrap@300051",
      "path": "runs/d20_showcase/d20_c_chat/d20_b_bootstrap/final.pt",
      "target_tokens": 300051,
      "language_ppl": 13.018369333213563,
      "chat_nll": 0.48617335107116244,
      "behavior_accuracy": 0.2624113475177305,
      "first_token_accuracy": 0.3,
      "dominant_first_word_rate": 0.13068181818181818,
      "dominant_response_rate": 0.05113636363636364
    }
  ],
  "pareto_front_unconstrained": [
    {
      "label": "d20_b_bootstrap@0",
      "path": "runs/d20_showcase/d20_c_chat/d20_b_bootstrap/final.pt",
      "target_tokens": 0,
      "language_ppl": 12.346700531928073,
      "chat_nll": 8.307723474431288,
      "behavior_accuracy": 0.02127659574468085,
      "first_token_accuracy": 0.0,
      "dominant_first_word_rate": 0.3352272727272727,
      "dominant_response_rate": 0.017045454545454544
    },
    {
      "label": "d20_b_bootstrap@75141",
      "path": "runs/d20_showcase/d20_c_chat/d20_b_bootstrap/final.pt",
      "target_tokens": 75141,
      "language_ppl": 14.803841332228563,
      "chat_nll": 0.5034808784678735,
      "behavior_accuracy": 0.3049645390070922,
      "first_token_accuracy": 0.32222222222222224,
      "dominant_first_word_rate": 0.16477272727272727,
      "dominant_response_rate": 0.056818181818181816
    },
    {
      "label": "d20_b_bootstrap@250350",
      "path": "runs/d20_showcase/d20_c_chat/d20_b_bootstrap/final.pt",
      "target_tokens": 250350,
      "language_ppl": 13.253008020233327,
      "chat_nll": 0.4494036901157533,
      "behavior_accuracy": 0.28368794326241137,
      "first_token_accuracy": 0.3,
      "dominant_first_word_rate": 0.1590909090909091,
      "dominant_response_rate": 0.056818181818181816
    },
    {
      "label": "d20_b_bootstrap@300051",
      "path": "runs/d20_showcase/d20_c_chat/d20_b_bootstrap/final.pt",
      "target_tokens": 300051,
      "language_ppl": 13.018369333213563,
      "chat_nll": 0.48617335107116244,
      "behavior_accuracy": 0.2624113475177305,
      "first_token_accuracy": 0.3,
      "dominant_first_word_rate": 0.13068181818181818,
      "dominant_response_rate": 0.05113636363636364
    }
  ],
  "eligible_on_front": 4,
  "selected": {
    "label": "d20_b_bootstrap@75141",
    "path": "runs/d20_showcase/d20_c_chat/d20_b_bootstrap/final.pt",
    "target_tokens": 75141,
    "language_ppl": 14.803841332228563,
    "chat_nll": 0.5034808784678735,
    "behavior_accuracy": 0.3049645390070922,
    "first_token_accuracy": 0.32222222222222224,
    "dominant_first_word_rate": 0.16477272727272727,
    "dominant_response_rate": 0.056818181818181816
  },
  "rejections": {
    "d20_c_segment_001@1000403": [
      "positional collapse: dominant first word 0.426"
    ],
    "d20_c_segment_002@0": [
      "positional collapse: dominant first word 0.426"
    ],
    "d20_c_segment_002@401172": [
      "positional collapse: dominant first word 0.381"
    ],
    "d20_c_segment_002@601271": [
      "positional collapse: dominant first word 0.375"
    ],
    "d20_c_segment_002@801429": [
      "positional collapse: dominant first word 0.477"
    ],
    "d20_c_segment_002@1002604": [
      "positional collapse: dominant first word 0.426"
    ],
    "d20_c_segment_002@1202667": [
      "positional collapse: dominant first word 0.381"
    ],
    "d20_c_segment_002@1403784": [
      "positional collapse: dominant first word 0.455"
    ],
    "d20_c_segment_002@1604482": [
      "positional collapse: dominant first word 0.682"
    ],
    "d20_c_segment_002@1805059": [
      "positional collapse: dominant first word 0.540"
    ],
    "d20_c_segment_002@2000336": [
      "positional collapse: dominant first word 0.591"
    ]
  }
}
```

## Known failures and divergences

- between 0 and 0 assistant tokens, chat NLL improved (7.5227 -> 3.8473) while behavioural accuracy fell (0.262 -> 0.064)
- between 75141 and 100148 assistant tokens, chat NLL improved (0.5035 -> 0.4879) while behavioural accuracy fell (0.305 -> 0.234)
- between 150222 and 175254 assistant tokens, chat NLL improved (0.4764 -> 0.4751) while behavioural accuracy fell (0.284 -> 0.241)
- between 400632 and 401172 assistant tokens, chat NLL improved (4.2469 -> 3.7305) while behavioural accuracy fell (0.135 -> 0.092)
- between 401172 and 601271 assistant tokens, chat NLL improved (3.7305 -> 3.6823) while behavioural accuracy fell (0.092 -> 0.085)
- between 802534 and 1000403 assistant tokens, chat NLL improved (3.9238 -> 3.8473) while behavioural accuracy fell (0.085 -> 0.064)
- between 1002604 and 1202667 assistant tokens, chat NLL improved (3.6007 -> 3.5567) while behavioural accuracy fell (0.099 -> 0.092)
- between 1403784 and 1604482 assistant tokens, chat NLL improved (3.5197 -> 3.4906) while behavioural accuracy fell (0.099 -> 0.085)
- between 1805059 and 2000336 assistant tokens, chat NLL improved (3.4561 -> 3.4329) while behavioural accuracy fell (0.099 -> 0.085)

### Rejected checkpoints

- `d20_c_segment_001@1000403`: positional collapse: dominant first word 0.426
- `d20_c_segment_002@0`: positional collapse: dominant first word 0.426
- `d20_c_segment_002@401172`: positional collapse: dominant first word 0.381
- `d20_c_segment_002@601271`: positional collapse: dominant first word 0.375
- `d20_c_segment_002@801429`: positional collapse: dominant first word 0.477
- `d20_c_segment_002@1002604`: positional collapse: dominant first word 0.426
- `d20_c_segment_002@1202667`: positional collapse: dominant first word 0.381
- `d20_c_segment_002@1403784`: positional collapse: dominant first word 0.455
- `d20_c_segment_002@1604482`: positional collapse: dominant first word 0.682
- `d20_c_segment_002@1805059`: positional collapse: dominant first word 0.540
- `d20_c_segment_002@2000336`: positional collapse: dominant first word 0.591

## Fixed demonstration prompts

- **Hello!** → Hello! How can I help you today?
- **What is your name?** → I'm sorry for the tonight. I'm Don't worry, I'll be happy to help you with a more concise and I am wr
- **Are you human?** → There are twenty, I'm Don't forget to there's a revised version of the sentence, I am C
- **What color is the sky?** → The sun islining like the sun's shining brightly, you are a beautiful day.
- **Is the sky green?** → There are twenty to the sunshine. Is there anything else Is there anything else Is there anything else Is?
- **Do dogs live in the ocean?** → The ocean is a received in the ocean.
- **What is a computer?** → There are a good idea that the computer is a complexity and the computer of the computer of the computer is
- **Give me three fruits.** → Welcome to my friend, I'm Don't forget. It's a great day, and I am excited to explore the world around us.
- **My name is Matheus.** → Hello! How can I help you today?
- **What is my name?** → I'm sorry for the tonight. I'm Don't worry, I'lln't you have a suggestions. We're not a bit
- **I had a difficult day today.** → Hello! However, I can help you today?
- **Can you tell me something cheerful?** → I'm sorry for a moment, I'm still curious about the to meet you. I'm here. I'm glad you're looking for something to do you

## Plots

- `language_nll_vs_tokens.png`
- `language_ppl_vs_tokens.png`
- `chat_nll_vs_assistant_tokens.png`
- `first_token_accuracy_vs_tokens.png`
- `behavior_vs_tokens.png`
- `retention_vs_tokens.png`
- `retention_vs_behavior.png`
- `chat_nll_vs_behavior.png`
- `pareto_ppl_chat.png`

## Limitations

- A ~20M model is not comparable to commercial systems and this run does not claim otherwise.
- The architecture has no depth knob, so scale here means width, vocabulary and memory width. The Chua-Yang core stays small in storage because it is reused at every evolution step.
- The direct receptive field is the cell count; longer prompts reach the decision only through the associative state.
- Showcase sampling (temperature > 0) is never used for the metrics above; all reported numbers are greedy.
