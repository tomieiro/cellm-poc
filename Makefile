.DEFAULT_GOAL := build
SHELL := /bin/bash

UV ?= uv
EXTRA ?= cpu
PYTHON ?= $(UV) run --locked --extra $(EXTRA) python
DEVICE ?= cpu
OUT ?= runs/continuity
SMOKE_OUT ?= runs/smoke
CONFIG ?= experiments/continuity.json
ARTEFACTS ?= artefacts
TEXT8 ?= $(ARTEFACTS)/text8/text8
TEXT8_OUT ?= runs/text8
TEXT8_STEPS ?= 20000
TEXT8_RUNGS ?= ABCDEFGH
TEXT8_PARALLEL ?= 3
D4_OUT ?= runs/tinystories-d4
D4_CORPUS ?= $(ARTEFACTS)/tinystories-d4/language_raw.jsonl
D4_TOKENIZER ?= $(ARTEFACTS)/tinystories-d4/tokenizer.json
LAMP_OUT ?= runs/lamp.json
CHECKPOINT ?=
LENGTH ?= 2048
REPEATS ?= 5
DECODE_TOKENS ?= 8
THREADS ?= 2
SEED ?= 42
CONTEXTS ?= 256 2048 8192
BENCHMODE ?=
BENCH_OUT ?= $(OUT)/benchmarks

.PHONY: setup build test smoke continuity benchmark benchmarks history text8 tinystories-d4 d4-check lamp verify-artefacts

setup:
	$(UV) sync --locked --extra $(EXTRA)

build:
	$(UV) build --no-sources

test:
	PYTHONPATH=src OMP_NUM_THREADS=$(THREADS) $(PYTHON) -m unittest discover -s tests -v

smoke:
	PYTHONPATH=src $(PYTHON) -m cellm_poc.experiment --config experiments/smoke.json --output "$(SMOKE_OUT)" --device "$(DEVICE)" --threads "$(THREADS)"

continuity:
	PYTHONPATH=src $(PYTHON) -m cellm_poc.experiment --config "$(CONFIG)" --data "$(TRAIN)" "$(VALIDATION)" "$(TEST)" --output "$(OUT)" --device "$(DEVICE)" --threads "$(THREADS)"

benchmark:
	PYTHONPATH=src $(PYTHON) -m cellm_poc.benchmark "$(CHECKPOINT)" --device "$(DEVICE)" --length "$(LENGTH)" --repeats "$(REPEATS)" --decode-tokens "$(DECODE_TOKENS)" --threads "$(THREADS)" $(if $(BENCHMODE),--mode "$(BENCHMODE)")

benchmarks:
	test -f "$(OUT)/manifest.json"
	mkdir "$(BENCH_OUT)"
	set -e; for length in $(CONTEXTS); do \
	  for arm in carry reset transformer; do \
	    $(MAKE) -s benchmark CHECKPOINT="$(OUT)/$(SEED)-$$arm.pt" LENGTH="$$length" > "$(BENCH_OUT)/$$arm-$$length.json"; \
	  done; \
	done

history:
	PYTHONPATH=src $(PYTHON) -m cellm_poc.history "$(ARTEFACTS)"

text8:
	printf '%s  %s\n' 3bea1919949baf155f99411df5fada7e "$(TEXT8)" | md5sum --check -
	mkdir -p "$(TEXT8_OUT)"
	PYTHONPATH=historical/text8/src OMP_NUM_THREADS=$(THREADS) $(PYTHON) -m celllm.ladder --data "$(TEXT8)" --steps "$(TEXT8_STEPS)" --rungs "$(TEXT8_RUNGS)" --parallel-seeds "$(TEXT8_PARALLEL)" --device "$(DEVICE)" --out "$(TEXT8_OUT)/results.json" --checkpoint-dir "$(TEXT8_OUT)/checkpoints"

tinystories-d4:
	PYTHONPATH=src $(PYTHON) experiments/replay_d4.py "$(ARTEFACTS)" --corpus "$(D4_CORPUS)" --tokenizer "$(D4_TOKENIZER)" --output "$(D4_OUT)" --device "$(DEVICE)"

d4-check:
	PYTHONPATH=src $(PYTHON) experiments/replay_d4.py "$(ARTEFACTS)" --check

lamp:
	PYTHONPATH=src $(PYTHON) experiments/replay_lamp.py "$(ARTEFACTS)" "$(LAMP_OUT)"

verify-artefacts:
	cd "$(ARTEFACTS)" && sha256sum --check SHA256SUMS
