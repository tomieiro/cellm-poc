.DEFAULT_GOAL := build
SHELL := /bin/bash

UV ?= uv
EXTRA ?= cpu
PYTHON ?= $(UV) run --locked --extra $(EXTRA) python
DEVICE ?= cpu
OUT ?= runs/continuity
SMOKE_OUT ?= runs/smoke
CONFIG ?= experiments/continuity.json
CHECKPOINT ?=
LENGTH ?= 2048
REPEATS ?= 5
DECODE_TOKENS ?= 8
THREADS ?= 2
SEED ?= 42
CONTEXTS ?= 256 2048 8192
BENCHMODE ?=
BENCH_OUT ?= $(OUT)/benchmarks

.PHONY: setup build test smoke continuity benchmark benchmarks

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
