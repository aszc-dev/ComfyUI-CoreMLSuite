# ComfyUI-CoreMLSuite — tiered test/bench dispatcher (Phase 4).
#
# Tiers (see MODERNIZATION_SPEC.md):
#   Tier 0 (unit):  framework-free pure-logic tests, run anywhere in seconds.
#   Tier 1 (smoke): macOS-ARM, no ANE, no full model — converts a synthetic
#                   micro-UNet to catch coremltools / ml-stable-diffusion
#                   API breakage in minutes.
#   Tier 2 (m2):    real ANE on Apple Silicon; integration + bench.
#
# COMFY_DIR defaults to the canonical custom-node layout (two dirs up from here).
# PY defaults to the project's uv-managed venv interpreter.

COMFY_DIR ?= $(realpath $(CURDIR)/../..)
PY ?= $(CURDIR)/.venv/bin/python
PYTEST ?= $(PY) -m pytest

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
IS_MACOS_ARM := $(filter Darwin,$(UNAME_S))$(filter arm64,$(UNAME_M))

.PHONY: help test-unit test-smoke test-m2 bench bench-rerun ci-tier0 ci-tier1 clean check-macos-arm

help:
	@echo "ComfyUI-CoreMLSuite — make targets"
	@echo ""
	@echo "  test-unit       Tier 0: pure-logic pytest, runs anywhere, seconds"
	@echo "  test-smoke      Tier 1: synthetic micro-UNet ct.convert + load (macOS-ARM, minutes)"
	@echo "  test-m2         Tier 2: pytest -m m2 against a real Core ML UNet (Apple Silicon + ANE)"
	@echo "  bench           Run bench/run.py against a converted .mlmodelc"
	@echo ""
	@echo "Vars: COMFY_DIR (default: $(COMFY_DIR)), PY (default: $(PY))"

check-macos-arm:
	@if [ -z "$(IS_MACOS_ARM)" ]; then \
		echo "this target requires macOS on Apple Silicon (got $(UNAME_S)/$(UNAME_M))"; \
		exit 2; \
	fi

# Tier 0 — Linux-safe pure logic. Should not import comfy/coremltools.
test-unit:
	$(PYTEST) -m unit tests/

# Tier 1 — macOS-ARM smoke. Converts a synthetic UNet through coremltools to
# catch API breakage without needing a real SD checkpoint or the ANE.
test-smoke: check-macos-arm
	$(PYTEST) -m smoke tests/

# Tier 2 — full Apple Silicon path: integration + m2 golden + bench.
# Requires a converted .mlmodelc (see bench/scripts/convert_sd15.py).
test-m2: check-macos-arm
	$(PYTEST) -m m2 tests/

# Bench harness. Override MODEL=/path/to/.mlmodelc for an explicit model.
MODEL ?= $(COMFY_DIR)/models/unet/v1-5-pruned-emaonly_1x512x512_se_unet.mlmodelc
COMPUTE_UNITS ?= CPU_AND_NE CPU_AND_GPU
REPEATS ?= 30
ASSUMED_STEPS ?= 20
bench: check-macos-arm
	$(PY) bench/run.py \
		--model "$(MODEL)" \
		--compute-units $(COMPUTE_UNITS) \
		--repeats $(REPEATS) \
		--assumed-steps $(ASSUMED_STEPS)

# What CI actually invokes — same as test-unit but echoes the env capture
# alongside so failed runs land with diagnostics.
ci-tier0:
	@echo "## env (Tier 0)" && $(PY) --version && uv pip freeze --python "$(PY)" 2>/dev/null | head -50 || true
	$(MAKE) test-unit

ci-tier1: check-macos-arm
	@echo "## env (Tier 1)" && $(PY) --version && uv pip freeze --python "$(PY)" 2>/dev/null | head -50 || true
	$(MAKE) test-smoke

clean:
	rm -rf .pytest_cache tests/m2/_latest_generated.png pytestdebug.log
