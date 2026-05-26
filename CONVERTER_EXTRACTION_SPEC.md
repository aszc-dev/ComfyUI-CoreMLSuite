# ComfyUI-CoreMLSuite — Converter Extraction Spec for Claude Code

> **Companion to `MODERNIZATION_SPEC.md`.** That spec hardens the repo and (Phase 3)
> splits the *inference* math from the framework. **This** spec splits the *conversion*
> path (`safetensors → CoreML`) out into a standalone, `comfy`-free, pip-installable
> package that CoreMLSuite then depends on — and that other projects (incl. on-device
> iOS tooling) can reuse.
>
> **Same discipline as the modernization spec:** safety-net first, behavior-preserving
> until told otherwise, one phase = one branch = one PR, `STOP — VALIDATE` gate between
> every phase, golden-latent as the regression anchor. `[M2]` = needs macOS/Apple Silicon;
> `[M2-ANE]` = needs the Neural Engine. Everything else must run on plain Linux/CI.

---

## 0. How to work (read first — non-negotiable)

1. **Behavior-preserving until Phase E6.** Phases E1–E5 must not change image output, node
   names, `INPUT_TYPES` field names, or `NODE_CLASS_MAPPINGS` keys. The node graph is the
   public contract; saved user-workflow JSON breaks if these change.
2. **The conversion package produces an artifact and stops there.** Its job ends at a written
   `.mlpackage` / `.mlmodelc` on disk. It must NOT import `comfy`, `folder_paths`, or
   `comfy_extras`, and must NOT know ComfyUI's `models/unet` layout. Paths are *inputs*.
3. **The runtime loader stays in the suite.** The loader is the **local** `coreml_suite.coreml_model.CoreMLModel`
   — a thin wrapper over `coremltools.models.MLModel` (NOT Apple's
   `python_coreml_stable_diffusion.coreml_model.CoreMLModel`, which is no longer used; see #58).
   It *runs* a compiled model in Python — a desktop/Python inference concern, not a conversion
   concern. It is NOT moved into the package. (On iOS the `.mlmodelc` is loaded natively; the
   package's output is the deliverable, not a Python runner.)
4. **Decouple in-repo before splitting repos.** Phases E1–E4 create the package *inside this
   repo* and prove equivalence. The physical second-repo split is Phase E5, only after the
   golden latent is proven identical. Do not create a second repository before Gate E4 passes.
5. **Reuse the existing regression anchor.** The golden latent / PSNR anchor from
   `MODERNIZATION_SPEC.md` Phase 2 is the cross-cutting proof for every gate here. If it is not
   yet captured, capture it first (it is a prerequisite for E2 onward).
6. **No new runtime dependencies** without flagging in the gate report (name, why, license, size).
7. **A failing gate means stop and report**, not work around into the next phase.
8. **Tooling is `uv`, not bare `pip`/`venv`.** Every environment/install/lock step uses the
   project's `uv` toolchain: `uv venv`, `uv pip install`, `uv pip install -e .`, `uv lock`,
   `uv run pytest`, `uv export`/`uv pip freeze` for baselines. Where this spec says "fresh venv",
   read "`uv venv` + `uv pip install`". Reserve `uv pip` (not `pip`) inside that venv too.
9. **The package is the single source of truth for *what is possible*; the node is a thin,
   discovery-driven frontend.** See the "Interface contract" pillar below — this is the
   maintainer's hard requirement and it overrides the earlier (now-rescinded) "freeze the
   dropdown list" instruction.

---

## Interface contract (the maintainer's hard requirement) — read before any phase

Two coupled guarantees must hold once the package is split out:

**(A) Updating the converter must NOT require updating CoreMLSuite.**
This is satisfied by treating the package's public surface as a versioned contract:
- `convert(...)` and `compile_model(...)` are **keyword-only with defaults** for everything
  past the genuinely-required positionals (`ckpt_path`, `model_version`, `out_path`). New
  capabilities are added as new keyword args with defaults, so an old Suite's call still
  validates against a newer package. **Never** reorder or rename existing parameters.
- `compose_out_name` (the `.mlpackage` filename = the cache key) **moves into the package** and
  is versioned with it. The Suite must not carry its own copy; if the package changes the naming
  scheme that is a **major** bump (old cached artifacts stop resolving).

**(B) CoreMLSuite must be able to list *new* conversion types WITHOUT a Suite code change or
version bump.** Today the node hardcodes its dropdowns:
```python
"model_version": ([ModelVersion.SD15.name, ModelVersion.SDXL.name],),  # hand-typed, also INCOMPLETE (no LCM / SDXL_REFINER)
"attention_implementation": (list(ATTENTION_IMPLEMENTATIONS),),         # from coreml_suite.attention
"quantize_nbits": (list(QUANT_NBITS_VALUES), {"default": "none"}),      # from coreml_suite.core.naming
```
These are replaced by **runtime discovery calls into the package**, evaluated inside
`INPUT_TYPES` (ComfyUI re-evaluates `INPUT_TYPES` on every plugin load):
```python
import coreml_diffusion
"model_version":            (coreml_diffusion.list_model_versions(),),
"attention_implementation": (coreml_diffusion.list_attention_impls(),),
"quantize_nbits":           (coreml_diffusion.list_quant_modes(), {"default": "none"}),
```
Effect: `uv pip install -U coreml_diffusion` + ComfyUI restart surfaces any newly-added type in the old
plugin's dropdown — **no Suite edit, no Suite version bump.** This is the requirement.

**The cost, stated honestly (accept this trade-off explicitly at Gate E0):**
- The Suite becomes a "dumb" frontend; the package is the sole authority on what conversions
  exist. The Suite can no longer guarantee its saved workflows are valid against *arbitrary*
  future package versions.
- Therefore the package's discovery identifiers (`ModelVersion` values, attn-impl strings, quant
  modes) are an **ADDITIVE-ONLY contract**: the package may *add* identifiers freely (minor bump,
  no Suite change); **removing or renaming an identifier is a breaking change requiring a MAJOR
  bump and a migration note**, because a saved workflow JSON references these strings verbatim.
  Without this rule, "no version bump" silently becomes "randomly broken workflows."
- `INPUT_TYPES` must **fail soft** when the package is missing/old: wrap the discovery calls so a
  missing `coreml_diffusion` (or an old one lacking a `list_*` function) yields a sane fallback list and a
  logged warning, instead of the node failing to register and disappearing from the menu.

**Discovery API the package must expose (stable names):**
```python
coreml_diffusion.list_model_versions() -> list[str]   # VERIFIED ones only, e.g. ["SD15","SDXL"] today (.name — see seam.md)
coreml_diffusion.list_attention_impls() -> list[str]  # ["SPLIT_EINSUM","SPLIT_EINSUM_V2","ORIGINAL"]
coreml_diffusion.list_quant_modes() -> list[str]      # ["none","8","6","4"]
coreml_diffusion.CONTRACT_VERSION: str                # bump rules above; Suite may log/compare it
```
These return the *display strings already used today*, so existing workflows keep validating.

**Verification status is a PACKAGE property, not a node hardcode (maintainer's intent).**
The Suite wants to expose *every model the converter can verifiably convert*. Today `lcm` and
`sdxl_refiner` are absent from the converter node not because the Suite chooses to hide them, but
because they lack a full golden/PSNR verification. So the gating lives in the package as a status:
```python
from enum import Enum
class Status(Enum):
    VERIFIED = "verified"          # has a golden anchor + passing [M2-ANE] check
    EXPERIMENTAL = "experimental"  # convertible but not yet anchored/verified

# internal registry, single source of truth.
# KEY by ModelVersion enum MEMBER so list_* can emit .name. Keying by the lowercase
# .value string returns ["sd15",...], which the node reverses via ModelVersion[...] -> KeyError.
_MODEL_STATUS = {ModelVersion.SD15: Status.VERIFIED, ModelVersion.SDXL: Status.VERIFIED,
                 ModelVersion.SDXL_REFINER: Status.EXPERIMENTAL, ModelVersion.LCM: Status.EXPERIMENTAL}

def list_model_versions(include_experimental: bool = False) -> list[str]:
    return [v.name for v, s in _MODEL_STATUS.items()   # .name -> "SD15","SDXL"; node reverses with ModelVersion[...]
            if s is Status.VERIFIED or (include_experimental and s is Status.EXPERIMENTAL)]
```
Consequence: **promoting a model to VERIFIED in the package expands the Suite's dropdown with no
Suite change and no Suite bump** — exactly the requirement. The act of verification (E-LCM
produces an LCM golden anchor; same later for refiner) is what flips the status. The Suite's
converter node calls `list_model_versions()` (verified-only); a power-user/CLI path may pass
`include_experimental=True`. Promotion VERIFIED-from-EXPERIMENTAL is additive (minor bump);
demotion or removal is breaking (major bump + note).

---

## Naming & layout (chosen — frozen at Gate E0)

**Distribution name (PyPI):** `coreml-diffusion`. **Import name (Python):** `coreml_diffusion`.
(PyPI normalizes `-`/`_`; the distribution uses the hyphen, the importable module the underscore.)
Availability checked: both `coreml-diffusion` and the near variants were free on PyPI at E0.

**Why this name (the positioning it encodes):** the project's niche is *diffusion models on Apple
Neural Engine via CoreML, inside ComfyUI and on-device* — **not** Stable Diffusion specifically.
`sd*` was rejected because it falsely narrows scope to SD; `coreml-diffusion` keeps `coreml` on the
front for discoverability while `diffusion` honestly states the scope (SD/SDXL/LCM today, Flux and
other diffusion architectures later) **without** promising arbitrary non-diffusion torch models,
whose tracing/shape/sample-input pipeline differs. The name must not be re-narrowed to SD in
future docs. ANE is the *differentiator* (documented in the README), but `coreml` was chosen over
`ane` in the name for search discoverability per maintainer decision.

Target package layout (framework-free — zero `comfy` imports):

```
coreml_diffusion/
  __init__.py        # public API surface (see "Public API" below)
  model_version.py   # ModelVersion enum — the SINGLE source of truth, no comfy
  attention.py       # ATTENTION_IMPLEMENTATIONS tuple (from coreml_suite/attention.py) + apply_attention_implementation
  pipeline.py        # get_pipeline (from_single_file), get_unet (cml UNet from ref unet)
  unet.py            # UNet2DConditionModelLCM (moved from coreml_suite/lcm/unet.py)
  inputs.py          # get_sample_input, lcm_inputs, sdxl_inputs,
                     #   get_encoder_hidden_states_shape, get_coreml_inputs, get_inputs_spec
  controlnet.py      # add_cnet_support (conversion-side residual SHAPE calc only)
  convert.py         # convert_unet, convert (orchestration), convert_to_coreml, load_coreml_model
  compile.py         # compile_coreml_model
  quantize.py        # (Phase E6 / MODERNIZATION Phase 6 lands here) palettization 4/6/8-bit
  cli.py             # console entry point: `coreml-diffusion convert ...`
  pyproject.toml     # standalone packaging (at E5)
```

What stays in `coreml_suite/` (the ComfyUI side, thinned):
- `nodes.py` — still owns **name-encoding** (`out_name` construction), path resolution via
  `folder_paths`, the node `INPUT_TYPES`/mappings, and wrapping the result in `CoreMLModel`.
- `models.py`, `latents.py`, `controlnet.py` (inference parts), `lcm/utils.py`, `config.py`
  (inference config build) — untouched by this spec except the import-source of `ModelVersion`.

### Public API (the contract `coreml_diffusion` exposes)
```python
from coreml_diffusion import ModelVersion, convert, compile_model, compose_out_name
from coreml_diffusion import list_model_versions, list_attention_impls, list_quant_modes, CONTRACT_VERSION

# Mirror the CURRENT converter.py signature, made keyword-only past the required positionals
# and with paths/device injected (no folder_paths, no comfy.model_management):
# convert(ckpt_path, model_version, out_path, *,
#         batch_size=1, sample_size=(64, 64), controlnet_support=False,
#         lora_weights=None, attn_impl=list_attention_impls()[0], config_path=None,
#         quantize_nbits="none", device=None) -> None   # side effect: writes out_path
#   (current convert() returns None and writes via convert_unet → coreml_unet.save; keep that,
#    or change to `return out_path` as a deliberate, documented improvement — pick one at E0.)
# compile_model(src_path, out_dir, final_name) -> str   # returns compiled .mlmodelc path
```
Note: `convert` takes an **explicit `out_path`** — no `folder_paths`. `device` is injected
(defaults to torch's default device). `compose_out_name` lives here (cache-key contract) and the
node imports it from the package. The `list_*` discovery functions back the node's dropdowns.

---

## The import chains to cut (root cause inventory) — REVISED against current code

> **State note (verified):** the code moved on since the original draft. Several chains are
> already cut. Re-verify each line by `grep` before acting; do not assume the original draft.

**Already done (verify, then skip):**
- ✅ `converter.py` already imports `from coreml_suite.model_version import ModelVersion`, and
  `model_version.py` is **clean** (`from enum import Enum` only — zero comfy). The old
  "converter → config → comfy" chain is **already broken**. `config.py` still imports comfy, but
  it is **inference-side** (`get_model_config` via `supported_models_base`/`latent_formats`) —
  *not* on the conversion path. Do **not** treat `config.py` as a converter dependency.
- ✅ `converter.py` now uses `diffusers.UNet2DConditionModel.from_single_file` and a local
  `CoreMLUNetWrapper` (in `coreml_suite/conversion/unet.py`) — it is **no longer** importing the
  Apple `python_coreml_stable_diffusion.unet.UNet2DConditionModel*` internals on the main path.
  A `coreml_suite/conversion/` subpackage already exists (`attention`, `shapes`, `trace`, `unet`).
- ✅ Name-encoding already extracted to `coreml_suite/core/naming.py` (`compose_out_name`,
  `lora_names_from_params`, `ATTN_SUFFIX`, `QUANT_NBITS_VALUES`) **with characterization tests**
  (`tests/unit/test_characterization_out_name.py`). The pure-naming split is done.
- ✅ Quantization is **already implemented** in `converter.py` (`quantize_nbits`, k-means
  `palettize_weights`) and surfaced as an optional node input. Phase E6 is therefore *move*, not
  *build* (see revised E6).

**Still to cut (the real remaining work):**
1. `coreml_suite/converter.py::get_out_path` → `from folder_paths import get_folder_paths`.
   Main converter still reaches into ComfyUI's model dir. **Cut: `out_path` is an injected arg;
   `folder_paths` resolution moves up into the node** (the node already computes `out_name`).
2. `coreml_suite/lcm/converter.py` → still has its **own** `from folder_paths import
   get_folder_paths` (`get_out_path`) and (per original draft) `comfy.model_management`. Verify
   the current LCM file and cut both: inject `out_path` and `device`.
3. Global mutation of the attention impl: confirm where it now lives. Main path appears to route
   through `coreml_suite/conversion/attention.apply_attention_implementation` (cleaner than the
   old global), but `lcm/converter.py` may still set a module global at import. **Ensure the
   package sets attention per-call, never at import time.**
4. **Duplication LCM vs main:** `lcm/converter.py` still carries its own copies of
   `convert_to_coreml`, `load_coreml_model`, `get_out_path`, `get_sample_input` (the LCM variant
   takes a `scheduler` arg), and hardcodes `SimianLuo/LCM_Dreamshaper_v7`. **Dedupe into the
   single `coreml_diffusion` implementation;** the HF-hardcode consolidation is the *behavior-changing*
   part → deferred to optional **E-LCM**, not E1–E5.
5. **`compose_out_name` ownership:** currently in `coreml_suite/core/naming.py` and called by the
   node. Per the Interface-contract pillar it must **move into the package** (it is the cache-key
   contract) and the node must import it from `coreml_diffusion`, not keep a copy.

---

## Phase E0 — Seam decision & inventory (no code change)

**Objective:** lock the cut line, the interface contract, and naming so later phases don't drift.

### Tasks
1. Produce `docs/extraction/seam.md`: a table of every symbol in `converter.py`,
   `lcm/converter.py`, `lcm/unet.py`, **plus the already-extracted `conversion/` subpackage
   (`attention`, `shapes`, `trace`, `unet`) and `core/naming.py`**, classified
   **CONVERSION → coreml_diffusion** vs **STAYS (comfy/node)**. Note which are already framework-free.
2. ~~Confirm the current `python_coreml_stable_diffusion` footprint.~~ **DONE (seam.md §6):
   footprint is ZERO** — no runtime imports anywhere; only a docstring mention in
   `core/__init__.py:4`. Main path uses `diffusers` + local `CoreMLUNetWrapper`; the runtime
   `CoreMLModel` (STAYS in suite) is a local coremltools wrapper, not Apple's. No shape/attn helper
   comes from Apple (local `conversion/shapes.py`, `conversion/attention.py`).
3. **Decide the interface contract concretely (the maintainer's hard requirement):**
   - Discovery functions `list_model_versions / list_attention_impls / list_quant_modes` live in
     the package and return today's display strings verbatim. Node `INPUT_TYPES` calls them.
   - `ModelVersion` values, attn-impl strings, quant modes are **ADDITIVE-ONLY** across package
     versions; removal/rename = MAJOR bump + migration note. Write this into the package's
     versioning policy doc now.
   - `compose_out_name` moves to the package; node imports it (no copy). Confirm the
     characterization tests in `test_characterization_out_name.py` will be re-pointed, not
     duplicated.
   - **Resolve the `model_version` dropdown question (maintainer decided):** the Suite exposes
     *every model the converter can verifiably convert*. `lcm` and `sdxl_refiner` are absent today
     only because they lack a golden/PSNR verification — **not** because the node hardcodes a
     short list. Encode this as a **status registry in the package** (`VERIFIED` vs
     `EXPERIMENTAL`); `list_model_versions()` returns VERIFIED-only by default. The converter node
     calls it plainly. Promoting LCM/refiner to VERIFIED (after E-LCM / a refiner anchor) expands
     the dropdown with **no Suite change**. Do NOT add permanent per-node filtering — the gate is
     verification status, owned by the package.
4. ~~Confirm the `ml-stable-diffusion` git dep is pinned.~~ **N/A — already removed (#58).** Verified:
   zero `python_coreml_stable_diffusion` imports in the repo; `CoreMLModel` is now a local
   coremltools wrapper; the dep is absent from `pyproject.toml`/`requirements.txt`. No SHA to pin.

### STOP — VALIDATE (Gate E0)
```
## Gate E0 report
- seam.md committed: <path>; symbol counts (move / stay / already-framework-free)
- python_coreml_stable_diffusion usage (verified by grep): conversion=<list> runtime=<list>
- Discovery API signatures frozen: list_model_versions (verified-only) / list_attention_impls / list_quant_modes
- Status registry decided: sd15+sdxl=VERIFIED, lcm+sdxl_refiner=EXPERIMENTAL (gated, not hidden)
- Additive-only contract policy doc written (incl. promotion=minor, demotion/removal=major): <path>
- model_version dropdown: expose all (incl. LCM/REFINER) / filtered per node — DECISION: <...>
- compose_out_name move-not-copy confirmed; tests re-point plan: <...>
- LCM consolidation deferred to optional E-LCM: YES/NO
- ml-stable-diffusion: N/A — already removed (#58), not a dependency (was: pin-or-BLOCKER)
- Package name in-repo: coreml_diffusion (final PyPI name deferred to E5)
```

---

## Phase E1 — Establish `coreml_diffusion` package + discovery API (mostly verification)

**Objective:** stand up the package namespace and the discovery surface. Much of the comfy-chain
cut is **already done** — this phase mostly *verifies* that and adds the discovery functions.

### Tasks
1. **Verify (don't redo):** `coreml_suite/model_version.py` is already clean (`Enum` only). Confirm
   `import coreml_suite.model_version` works with **no comfy** (`uv run python -c "..."` in a
   comfy-free `uv venv`). If true, E1's original "extract ModelVersion" task is already satisfied.
2. Create the `coreml_diffusion/` package skeleton with `__init__.py` exporting the **discovery API**
   backed by the *existing* sources of truth for now (re-export `ModelVersion`, the
   `ATTENTION_IMPLEMENTATIONS` tuple, and `QUANT_NBITS_VALUES`) so values are byte-identical:
   ```python
   def list_model_versions(): return [v.name for v in ModelVersion]   # .name -> "SD15" (node reverses via ModelVersion[...]; .value KeyErrors)
   def list_attention_impls(): return list(ATTENTION_IMPLEMENTATIONS)
   def list_quant_modes(): return list(QUANT_NBITS_VALUES)
   CONTRACT_VERSION = "1.0"
   ```
   (At this stage `coreml_diffusion` may live inside the repo and import from `coreml_suite.*`; the
   physical move of implementation happens in E2. The point of E1 is to freeze the *contract*.)
3. **Decided (`.name`):** the node renders `ModelVersion.SD15.name` (`"SD15"`) and reverses the
   dropdown string via `ModelVersion[model_version]` (name lookup, `nodes.py:286`). Discovery API
   therefore returns `.name`; `.value` (`"sd15"`) would `KeyError`. Recorded in `seam.md` §5.

### Acceptance criteria
- `uv run python -c "import coreml_diffusion; print(coreml_diffusion.list_model_versions(), coreml_diffusion.list_quant_modes())"`
  works in a **comfy-free** `uv venv` and prints today's exact strings.
- Existing characterization tests pass unchanged.
- No node behavior change yet (node still uses its current hardcoded lists in E1).

### STOP — VALIDATE (Gate E1)
```
## Gate E1 report
- model_version.py confirmed comfy-free (uv, no comfy): PASS/FAIL
- coreml_diffusion.list_* returns byte-identical strings to current dropdowns: YES/NO (show values)
- .name vs .value decision for model_version discovery: <...>
- CONTRACT_VERSION set; additive-only policy linked: <path>
- Characterization tests unchanged & green (uv run pytest): YES/NO
```

---

## Phase E2 — Move conversion code into `coreml_diffusion` (in-repo, dedup, behavior-preserving)

**Objective:** physically relocate the conversion mechanics into the framework-free package,
collapsing the two duplicate converters into one, with paths/device injected.

### Tasks
1. Move into `coreml_diffusion/`: `pipeline.py` (`get_pipeline`, `get_unet`), `unet.py`
   (`UNet2DConditionModelLCM`), `inputs.py` (sample/lcm/sdxl input builders +
   `get_encoder_hidden_states_shape` + `get_coreml_inputs` + `get_inputs_spec`),
   `controlnet.py` (`add_cnet_support`), `convert.py` (`convert_unet`, `convert`,
   `convert_to_coreml`, `load_coreml_model`), `compile.py` (`compile_coreml_model`).
2. **Dedupe LCM vs main** (the real remaining duplication): delete `lcm/converter.py`'s copies of
   `convert_to_coreml` / `load_coreml_model` / `get_out_path` / `get_sample_input` (LCM variant
   carries a `scheduler` arg — fold that into the shared `get_sample_input` as an optional param)
   in favor of the single `coreml_diffusion` implementation. The main path's helpers
   (`get_unet`/`get_encoder_hidden_states_shape`/`get_coreml_inputs`/`convert_unet`/`convert`) and
   the `conversion/` subpackage (`attention`, `shapes`, `trace`, `unet`) move as-is.
3. **Inject paths**: replace `get_out_path`'s `folder_paths` reach-in with an injected `out_path`
   argument on `convert(...)`; `folder_paths` resolution moves up into the node (which already
   computes `out_name`). No `folder_paths` import anywhere in `coreml_diffusion`.
4. **Inject device** where the LCM path used `comfy.model_management` (verify it still does):
   `convert(..., device=None)`, default to torch's default device.
5. **Attention per-call, never at import:** main path already routes through
   `conversion/attention.apply_attention_implementation` — keep that. If `lcm/converter.py` still
   sets any module global at import, remove it; the package sets attention from the `attn_impl`
   arg inside `convert`.
6. **Move `compose_out_name` into the package** (`coreml_diffusion/naming.py`); re-point
   `test_characterization_out_name.py` imports to `coreml_diffusion.naming` — assertions and values
   unchanged. The node will import it from the package in E3.
7. Leave **thin shims** in `coreml_suite/converter.py` and `coreml_suite/lcm/converter.py` that
   re-export from `coreml_diffusion`, preserving the old call signatures the nodes use (nodes untouched
   this phase). Shims map comfy `folder_paths`/device into package args.

### Acceptance criteria
- `uv run pytest -m unit` (Tier 0) imports `coreml_diffusion.*` with **no comfy / no MPS** and is green on Linux.
- The dedup leaves exactly one implementation of each previously-duplicated function.
- Characterization tests pass unchanged after the `compose_out_name` re-point.
- `[M2]` A real SD1.5 conversion via the shim still produces a loadable model.
- `[M2-ANE]` **Golden latent identical / within tolerance** to the MODERNIZATION Phase 2 anchor
  (same seed/prompt) — proves the move + dedup changed nothing.

### STOP — VALIDATE (Gate E2 — first regression gate)
```
## Gate E2 report
- Tier 0 import of coreml_diffusion without comfy/MPS (uv run): PASS/FAIL
- LCM/main duplicated funcs collapsed to one (list old→new): <map>
- compose_out_name moved to package; char-tests re-pointed & green: YES/NO
- Paths injected (no folder_paths in package): confirmed
- Device injected (no comfy.model_management in package): confirmed
- Attention set per-call, not at import (both main & lcm): confirmed
- [M2-ANE] Golden latent vs Phase-2 anchor: identical / within tol <x> / DIVERGED (STOP)
- Node INPUT_TYPES / mappings untouched: confirmed (diff)
```
**If the golden latent diverged at all, STOP and report — do not continue.**

---

## Phase E3 — Thin the nodes onto the package (behavior-preserving)

**Objective:** remove the shims; have the ComfyUI nodes call `coreml_diffusion` directly, keeping the
node contract byte-identical.

### Tasks
1. `CoreMLConverter.convert` (in `coreml_suite/nodes.py`): keep the `folder_paths`-based path
   resolution **in the node**; import `compose_out_name` from `coreml_diffusion` (not `coreml_suite.core`);
   call `coreml_diffusion.convert(...)` and `coreml_diffusion.compile_model(...)` directly; wrap the compiled path
   in `CoreMLModel`.
2. **Wire the dropdowns to discovery (the maintainer's hard requirement).** Replace the hardcoded
   `INPUT_TYPES` lists with fail-soft discovery calls:
   ```python
   def _discover(fn, fallback):
       try:
           import coreml_diffusion
           return getattr(coreml_diffusion, fn)()
       except Exception as e:        # missing/old package, or import error
           logger.warning(f"coreml_diffusion.{fn} unavailable ({e}); using fallback {fallback}")
           return fallback
   ...
   "model_version":            (_discover("list_model_versions", ["SD15", "SDXL"]),),
   "attention_implementation": (_discover("list_attention_impls", ["SPLIT_EINSUM","SPLIT_EINSUM_V2","ORIGINAL"]),),
   "quantize_nbits":           (_discover("list_quant_modes", ["none","8","6","4"]), {"default": "none"}),
   ```
   This is what makes "update the package → new types appear in the old node, no Suite bump" true.
3. `COREML_CONVERT_LCM` (in `coreml_suite/lcm/nodes.py`): route through `coreml_diffusion` for the shared
   mechanics. **Keep the existing LCM behavior/HF-hardcode for now** — consolidation is optional E-LCM.
4. Delete the now-dead `coreml_suite/converter.py` / `coreml_suite/lcm/converter.py` shims (or
   reduce to a one-line re-export if anything external imports them — grep first).

### Acceptance criteria
- `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` keys: **unchanged** (diff `__init__.py`).
- Every `INPUT_TYPES` **field name** unchanged. Dropdown **values**: the discovery calls must
  return **a superset of** today's values, with every previously-present value still present and
  spelled identically (additive-only). *(This deliberately replaces the original spec's
  "values must be byte-identical/frozen" criterion — the maintainer requires the list be
  extensible at runtime. Frozen-field-names + additive-only-values is the new contract.)*
- With `coreml_diffusion` **absent**, the node still registers and shows the fallback lists (fail-soft).
- `[M2-ANE]` Golden latent still identical to the Phase-2 anchor.
- `[M2-ANE]` The committed e2e workflow `tests/integration/...` still passes (PSNR > 25).

### STOP — VALIDATE (Gate E3)
```
## Gate E3 report
- Node mappings diff: empty (confirmed)
- INPUT_TYPES field-names diff: empty (confirmed)
- Dropdown values: superset of prior, all prior values still present & identical: YES/NO (show)
- Fail-soft with coreml_diffusion absent (node still registers): PASS/FAIL
- compose_out_name now imported from coreml_diffusion (no node-side copy): confirmed
- [M2-ANE] Golden latent vs anchor: identical / within tol / DIVERGED (STOP)
- [M2-ANE] e2e workflow PSNR: <value> (> 25?)
- Dead converter shims removed / reduced: <list>
```

---

## Phase E4 — Standalone packaging & CLI (still in-repo)

**Objective:** make `coreml_diffusion` independently installable and usable without ComfyUI, with a CLI
suitable for the planned article and for on-device/iOS conversion workflows.

### Tasks
1. Add `coreml_diffusion/pyproject.toml`: name (working `coreml_diffusion`), `requires-python`, dependencies
   = `coremltools` (pinned to the MODERNIZATION-validated version), `diffusers`, `transformers`,
   `peft`, `omegaconf`, `numpy`, `torch`. **No `ml-stable-diffusion`** (already removed in #58, see
   §0.3) and **no comfy**. Suite pins `transformers>=4.44`/`peft>=0.13`/`omegaconf>=2.3` today;
   grep-confirm each is on the conversion path before listing it. A `[project.scripts]` entry:
   `coreml-diffusion = "coreml_diffusion.cli:main"`.
2. `coreml_diffusion/cli.py`: `coreml-diffusion convert --ckpt PATH --model-version sd15 --out PATH
   [--height --width --batch-size --attn-impl --controlnet --lora NAME:STRENGTH ... --config PATH]`
   and `coreml-diffusion compile --src PATH --out-dir DIR --name NAME`. Mirrors `convert()`/`compile_model()`.
3. Tier-0 Linux tests for the CLI **arg→call mapping** (mock the heavy `convert`); the real
   convert remains `[M2]`. Add a `[M2]` smoke test: convert a tiny synthetic UNet end-to-end.
4. README for the package: install, CLI usage, "produce a `.mlpackage`/`.mlmodelc` for use in a
   Swift/iOS app", and the ANE positioning note (low-power, GPU-free, embeddable; SD1.5/SDXL on
   ANE, **not** a Flux-speed claim).

### Acceptance criteria
- Fresh `python -m venv` + `uv pip install ./coreml-diffusion` (no ComfyUI present) imports and runs
  `coreml-diffusion --help` and the arg-mapping tests on Linux.
- `[M2]` `coreml-diffusion convert` produces a model file identical (golden) to the node path.

### STOP — VALIDATE (Gate E4)
```
## Gate E4 report
- uv pip install ./coreml-diffusion in comfy-free venv: PASS/FAIL (log)
- CLI arg→call tests (Tier 0, Linux): green
- [M2] CLI-produced model golden vs node-produced model: identical / DIVERGED
- Package deps list (with pinned SHAs/versions + licenses):
- New runtime deps vs suite before: <none / list>
```
**This is the gate that proves the package stands alone. Do not split repos before it passes.**

---

## Phase E5 — Physical split into a second repository

**Objective:** move `coreml_diffusion/` to its own repo; CoreMLSuite depends on it by pinned version.

### Tasks
1. Create the new repo (maintainer action — agent prepares the tree, not the GitHub repo).
   Choose final distributable name; rename imports if changed (single sweep, recorded).
2. CoreMLSuite `pyproject.toml` / `requirements.txt`: replace the conversion-only deps with a
   pinned dependency on the new package (`coreml_diffusion==<version>` from PyPI, or `git+...@<tag>`
   until first PyPI release). (There is no `git+...ml-stable-diffusion` line to remove — already
   gone since #58.)
3. ~~Keep `python_coreml_stable_diffusion` for the loader.~~ **Void.** The loader is the local
   `coreml_suite/coreml_model.py` over `coremltools`; the suite keeps `coremltools` as a direct dep
   for it. No Apple lib involved.
4. Set up the new repo's CI: Tier 0 on Linux (import + arg-mapping + input-shape math),
   `[M2]`/`[M2-ANE]` on a self-hosted/macOS-ARM runner reusing the golden-latent anchor.
5. Versioning: SemVer; first release `0.1.0`. Document the compatibility matrix
   (coreml_diffusion ↔ coremltools version ↔ diffusers version). No ml-stable-diffusion axis.

### Acceptance criteria
- CoreMLSuite installs in a fresh venv pulling the new package; e2e workflow still passes `[M2-ANE]`.
- New repo CI green on Linux (Tier 0) and `[M2-ANE]` golden latent matches the anchor.
- No conversion code remains in CoreMLSuite (grep: no `ct.convert`, no `from_single_file`,
  no `torch.jit.trace`).

### STOP — VALIDATE (Gate E5)
```
## Gate E5 report
- New repo tree prepared: <path/branch>; final package name: <name>
- Suite depends on package by pinned version: <spec>
- Suite e2e [M2-ANE] PSNR after split: <value> (> 25?)
- Conversion code fully absent from suite: confirmed (grep output)
- Compatibility matrix documented: <link>
- First release tag: 0.1.0
```

---

## Phase E6 — Quantization travels WITH the conversion code (already implemented → move)

**Objective:** quantization is **already implemented** (k-means `palettize_weights` in
`converter.py`, `quantize_nbits` node input, `_q<bits>` filename suffix, README tradeoff table).
There is nothing to *build*. It simply **moves with the conversion code in E2** as part of
`convert_unet`. This phase is a checkpoint that it survived the extraction intact, plus exposing
it through the CLI.

### Tasks
1. Confirm the palettization block moved cleanly into `coreml_diffusion` (lives in `convert.py` or a
   `quantize.py` helper called from `convert_unet`). Default `"none"` stays byte-identical.
2. Expose via CLI flag `--quantize {none,8,6,4}` (E4 already lists this) and via
   `list_quant_modes()` discovery (E1/E3).
3. The existing README tradeoff table (SD1.5 1×512×512 SPLIT_EINSUM: none/8/6/4 → size/ms/PSNR)
   moves to the package README. Re-confirm one row `[M2-ANE]` so the article can cite a live number.

### Acceptance criteria
- Default (`none`) output byte-identical to pre-extraction (covered by the E2/E3 golden latent).
- `coreml-diffusion convert --quantize 4` produces a `_q4` artifact matching the node's `_q4` artifact `[M2]`.
- `list_quant_modes()` drives the node dropdown (no hardcoded copy remains).

### STOP — VALIDATE (Gate E6)
```
## Gate E6 report
- Palettization relocated into coreml_diffusion, called from convert_unet: confirmed
- Default none output identical (golden): YES/NO
- [M2] CLI --quantize {8,6,4} artifacts match node artifacts: YES/NO
- Tradeoff table in package README with at least one re-confirmed [M2-ANE] row: <link>
```

---

## Phase E-LCM — FIRST task after the split: clean up LCM + verify → promote (behavior-changing, gated)

> Promoted from "optional, someday" to **the first thing after E5**, per maintainer intent: the
> Suite should expose every verifiably-convertible model, and LCM is the obvious first cleanup.

Two coupled goals:
1. **Consolidate the LCM path.** Make the LCM node use the unified `from_single_file` path in
   `coreml_diffusion.convert(model_version=LCM, ...)` instead of the hardcoded `SimianLuo/LCM_Dreamshaper_v7`
   HF download; drop the duplicated LCM helpers (already deduped in E2). **Behavior change** ⇒
   capture an LCM golden anchor *before* the change, then prove within-tolerance after.
2. **Verify → promote.** Once the LCM conversion has a passing `[M2-ANE]` golden anchor, flip
   `_MODEL_STATUS["lcm"] = Status.VERIFIED` **in the package** (minor bump). The Suite's dropdown
   gains `lcm` automatically — no Suite change, no Suite bump. This is the end-to-end proof that
   the discovery contract works as designed.

Repeat the same recipe for `sdxl_refiner` when it gets an anchor (separate small gate). Do NOT
bundle E-LCM into E1–E5; it changes behavior and must stand on its own golden.

### STOP — VALIDATE (Gate E-LCM)
```
## Gate E-LCM report
- LCM golden anchor captured BEFORE change: <path/hash>
- LCM node now uses unified from_single_file path; HF hardcode removed: confirmed
- [M2-ANE] LCM golden after change: identical / within tol <x> / DIVERGED (STOP)
- Status flipped lcm→VERIFIED in package (minor bump <ver>): confirmed
- Suite dropdown now lists lcm with NO Suite code change / NO Suite bump: confirmed (diff empty)
- LCM node accepts a checkpoint arg now (documented breaking-ish UI note): <link>
```

---

## Article deliverable (after E4)

Once the CLI exists and stands alone, the "convert a Comfy/A1111 workflow into an on-device iOS
app" write-up becomes a clean tutorial: `coreml-diffusion convert` → `.mlmodelc` → load in Swift/CoreML.
Frame the niche honestly per the README note above (ANE feasibility & power, not raw Flux speed).

---

## Quick reference: extraction gate discipline

```
E0  Seam decision, interface contract, discovery API frozen → Gate E0  (cut line + additive-only policy?)
E1  Stand up coreml_diffusion + discovery API (mostly verify)        → Gate E1  (list_* byte-identical, comfy-free?)
E2  Move conversion code, dedup LCM/main, inject paths/device→ Gate E2  (golden identical? duplicates gone?) ← first regression gate
E3  Thin nodes onto package + wire discovery dropdowns       → Gate E3  (field-names frozen, values additive, fail-soft, golden identical?)
E4  Standalone packaging + CLI (uv)                          → Gate E4  (uv pip install w/o comfy? CLI golden?) ← proves it stands alone
E5  Physical second-repo split                               → Gate E5  (suite depends on pkg? conversion absent?)
E-LCM  FIRST post-split: clean up LCM, verify → promote      → Gate E-LCM (LCM golden? dropdown gains lcm w/ no Suite bump?)
E6  Quantization checkpoint (already built → moved in E2)    → Gate E6  (default identical? CLI quant matches?)
(refiner)  same recipe as E-LCM when an anchor exists        → own small gate (promote sdxl_refiner→VERIFIED)
```

**Interface-contract invariants (the maintainer's hard requirement), restated:**
- Package API is keyword-only-with-defaults past the required positionals → converter updates
  don't force Suite updates.
- Node dropdowns are discovery-driven (`coreml_diffusion.list_*`) + fail-soft → new conversion types
  appear in the old plugin with `uv pip install -U coreml_diffusion`, **no Suite code change, no bump**.
- Discovery identifiers are **additive-only**; removal/rename = MAJOR bump + migration note.
- `compose_out_name` (cache key) lives in the package, single copy.


**Golden rule (inherited): never cross a gate with a failing acceptance criterion.
Stop, report, wait. The golden latent is the single source of truth that the extraction
changed nothing.**
