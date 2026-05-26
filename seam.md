# Conversion Extraction — Seam Inventory (`docs/extraction/seam.md`)

> **Gate E0 deliverable.** Symbol-by-symbol cut line between the future `coreml_diffusion`
> package (CONVERSION) and what stays in `coreml_suite` (the ComfyUI side).
>
> **Confidence legend:**
> - ✅ **verified** — read directly from the current source in this repo.
> - 🔍 **confirm** — inferred / partially seen; Claude Code must `grep`-verify before acting.
>
> **Cut rule:** a symbol goes to `coreml_diffusion` iff it participates in producing the `.mlpackage`
> artifact AND can be made free of `comfy` / `folder_paths` / `comfy_extras`. The runtime
> *loader* that **runs** a compiled model stays in the suite.

---

## 1. File-level map

| File | Side | Status | Note |
|---|---|---|---|
| `coreml_suite/model_version.py` | **coreml_diffusion** | ✅ | Already `Enum`-only, zero comfy. Becomes pkg source of truth. |
| `coreml_suite/attention.py` | **coreml_diffusion** | ✅ | `ATTENTION_IMPLEMENTATIONS` tuple; pure constant. |
| `coreml_suite/core/naming.py` | **coreml_diffusion** | ✅ | `compose_out_name` = cache-key contract. Move (not copy). |
| `coreml_suite/converter.py` | **coreml_diffusion** (mostly) | ✅ | Main conversion. One symbol stays-adjacent: `get_out_path` (folder_paths) is replaced by injected `out_path`. |
| `coreml_suite/conversion/attention.py` | **coreml_diffusion** | ✅ | `apply_attention_implementation`. Imports `logging`,`torch` only — no comfy. |
| `coreml_suite/conversion/shapes.py` | **coreml_diffusion** | ✅ | `conv2d_output_shape`. Pure math, no imports. |
| `coreml_suite/conversion/trace.py` | **coreml_diffusion** | ✅ | Imports `types.MethodType`, `diffusers...Transformer2DModel` only — torch/diffusers. |
| `coreml_suite/conversion/unet.py` | **coreml_diffusion** | ✅ | `CoreMLUNetWrapper`. Imports `torch` only — no comfy. |
| `coreml_suite/lcm/converter.py` | **coreml_diffusion** (after dedup) | ✅ | Dup helpers deleted; `MODEL_VERSION` HF-hardcode (L22) → E-LCM. `folder_paths` (L111) + `comfy.model_management` (L54) confirmed present → CUT. |
| `coreml_suite/lcm/unet.py` | **coreml_diffusion** | ✅ | `UNet2DConditionModelLCM(UNet2DConditionModel)`. diffusers-only, no comfy. |
| `coreml_suite/config.py` | **STAYS** | ✅ | Imports `comfy.supported_models_base`/`latent_formats`/`model_detection`. **Inference-side** (`get_model_config`), NOT conversion. |
| `coreml_suite/coreml_model.py` | **STAYS** | ✅ | `CoreMLModel` = runtime loader (runs `.mlpackage`). Desktop/Python inference; not used on iOS. |
| `coreml_suite/nodes.py` | **STAYS** | ✅ | Nodes; will call `coreml_diffusion` + own `folder_paths` path resolution + discovery dropdowns. |
| `coreml_suite/lcm/nodes.py` | **STAYS** | ✅ | `COREML_CONVERT_LCM` node. |
| `coreml_suite/models.py` | **STAYS** | ✅ | Inference: `add_sdxl_model_options`, `is_sdxl`, `get_model_patcher`, `get_latent_image`. |
| `coreml_suite/latents.py` | **STAYS** | ✅ | Inference chunking (MODERNIZATION Phase 3 target, not this spec). |
| `coreml_suite/controlnet.py` | **STAYS** | ✅ | Inference-side controlnet. Distinct from converter `add_cnet_support`. |
| `coreml_suite/lcm/utils.py` | **STAYS** | ✅ | `add_lcm_model_options`, `lcm_patch`, `is_lcm`; imports `comfy_extras`. Inference. |
| `coreml_suite/logger.py` | **both / copy** | ✅ | Trivial. Package gets its own logger; suite keeps its. |

---

## 2. Symbol-level: `coreml_suite/converter.py` (main conversion)

| Symbol | Side | Status | Cut action |
|---|---|---|---|
| `DEFAULT_TRACE_TIMESTEP`, `TEXT_TOKEN_SEQUENCE_LENGTH` | coreml_diffusion | ✅ | Move as-is (module constants). |
| `get_unet(model_version, ref_unet, attention_implementation)` | coreml_diffusion | ✅ | Move. Uses `conversion.{trace,attention,unet}`. No comfy. |
| `get_encoder_hidden_states_shape(ref_unet, batch_size)` | coreml_diffusion | ✅ | Move. Reads `ref_unet.config.cross_attention_dim`. Pure. |
| `get_coreml_inputs(sample_inputs)` | coreml_diffusion | ✅ | Move. `ct.TensorType` build. |
| `load_coreml_model(out_path)` | coreml_diffusion | ✅ | Move. `ct.models.MLModel(out_path)`. (Dedup target vs LCM copy.) |
| `convert_to_coreml(submodule, ts_module, inputs, names, out_path)` | coreml_diffusion | ✅ | Move. `ct.convert(...)`. (Dedup target vs LCM copy.) |
| `get_sample_input(batch, ehs_shape, sample_shape)` | coreml_diffusion | ✅ | Move. **Merge** with LCM variant (LCM passes extra `scheduler` → optional param). |
| `lcm_inputs(sample_unet_inputs)` | coreml_diffusion | ✅ | Move. Adds `timestep_cond`. |
| `sdxl_inputs(sample_unet_inputs, ref_unet, model_version)` | coreml_diffusion | ✅ | Move. `time_ids`/`text_embeds`/`add_embeds`. |
| `add_cnet_support(sample_shape, ref_unet)` | coreml_diffusion | ✅ | Move. Builds `additional_residual_*` inputs from unet block channels. |
| `convert_unet(ref_unet, model_version, unet_out_path, ...)` | coreml_diffusion | ✅ | Move. Orchestrates trace→convert→**quant (palettize)**→save. Quant travels here (E6). |
| `convert(ckpt_path, model_version, unet_out_path, ...)` | coreml_diffusion | ✅ | Move. **Make kw-only past `ckpt_path,model_version,out_path`** (contract). Validates `attn_impl`. |
| `load_unet(ckpt_path, config_path)` | coreml_diffusion | ✅ | Move. `UNet2DConditionModel.from_single_file`. |
| `get_out_path(submodule_name, model_name)` | **STAYS (node)** | ✅ | Uses `folder_paths.get_folder_paths`. **Delete from converter; node resolves path and passes `out_path` in.** |

**Apple `python_coreml_stable_diffusion` footprint on this path:** ✅ **none.** Verified by grep:
zero imports in `converter.py` / `conversion/*`. Main path uses `diffusers` +
local `CoreMLUNetWrapper`. (And the runtime `CoreMLModel` is now a local coremltools wrapper too —
see §6 stale-spec note.)

---

## 3. Symbol-level: `coreml_suite/lcm/converter.py` (LCM — dedup + defer)

| Symbol | Side | Status | Cut action |
|---|---|---|---|
| `load_coreml_model` (LCM copy) | DELETE | ✅ | Duplicate of main. Remove; use `coreml_diffusion.load_coreml_model`. |
| `convert_to_coreml` (LCM copy) | DELETE | ✅ | Duplicate of main. Remove. |
| `get_out_path` (LCM copy, folder_paths) | DELETE | ✅ | Duplicate + comfy. Remove; node injects `out_path`. |
| `get_sample_input(..., scheduler)` (LCM copy) | MERGE → coreml_diffusion | ✅ | Fold `scheduler` into shared `get_sample_input` as optional param. |
| `MODEL_NAME` (= LCM_Dreamshaper) | **E-LCM** | ✅ | HF hardcode. Removing it is the behavior change → E-LCM, not E2. |
| `convert(out_path, sample_size, batch_size, controlnet_support)` (LCM, L190) | coreml_diffusion (via unified) | ✅ | Route through `coreml_diffusion.convert(model_version=LCM, ...)` in E-LCM. |
| `from comfy.model_management import get_torch_device` (L54, in `get_scheduler`) | **CUT** | ✅ | Confirmed present. Inject `device`. |
| module-global attention set at import | n/a | ✅ | **No module global.** Attention already per-call: `get_unets` (L36) calls `apply_attention_implementation(ref_unet, "SPLIT_EINSUM")`. No `ATTENTION_IMPLEMENTATION_IN_EFFECT` anywhere in repo. (Note: LCM hardcodes `"SPLIT_EINSUM"` — pass `attn_impl` through in dedup.) |

---

## 4. Symbol-level: `coreml_suite/core/naming.py` → `coreml_diffusion/naming.py`

| Symbol | Side | Status | Cut action |
|---|---|---|---|
| `compose_out_name(...)` | coreml_diffusion | ✅ | **Move** (cache-key contract). Node imports from pkg. |
| `lora_names_from_params(...)` | coreml_diffusion | ✅ | Move. |
| `ATTN_SUFFIX` dict | coreml_diffusion | ✅ | Move. |
| `QUANT_NBITS_VALUES` | coreml_diffusion | ✅ | Move; backs `list_quant_modes()`. |
| `tests/unit/test_characterization_out_name.py` | re-point | ✅ | Change import to `coreml_diffusion.naming`. Assertions/values **unchanged**. |

---

## 5. Discovery API + status registry (new in `coreml_diffusion/__init__.py`)

```python
from enum import Enum

class Status(Enum):
    VERIFIED = "verified"          # has a golden anchor + passing [M2-ANE] check
    EXPERIMENTAL = "experimental"  # convertible, not yet anchored/verified

# Single source of truth. Suite gates on this, NOT on a hardcoded node list.
# KEY by ModelVersion enum MEMBER (not a bare string) so list_model_versions can
# emit .name — see the .name decision below. Keying by the lowercase .value string
# (as an earlier draft of this block did) returns ["sd15",...], which the node then
# reverses via ModelVersion[...] → KeyError. Do NOT key by .value.
_MODEL_STATUS = {
    ModelVersion.SD15:         Status.VERIFIED,
    ModelVersion.SDXL:         Status.VERIFIED,
    ModelVersion.SDXL_REFINER: Status.EXPERIMENTAL,   # → VERIFIED after a refiner golden anchor
    ModelVersion.LCM:          Status.EXPERIMENTAL,   # → VERIFIED after E-LCM golden anchor
}

def list_model_versions(include_experimental: bool = False) -> list[str]:
    return [v.name for v, s in _MODEL_STATUS.items()   # .name → "SD15","SDXL" (see decision)
            if s is Status.VERIFIED or (include_experimental and s is Status.EXPERIMENTAL)]

def list_attention_impls() -> list[str]:   # from attention.ATTENTION_IMPLEMENTATIONS
    ...
def list_quant_modes() -> list[str]:       # from naming.QUANT_NBITS_VALUES
    ...

CONTRACT_VERSION = "1.0"
# Additive-only: adding an id or promoting EXPERIMENTAL→VERIFIED = minor bump (Suite unaffected).
# Removing/renaming an id, or demoting VERIFIED→EXPERIMENTAL = MAJOR bump + migration note.
```

**Decision check (`.name` vs `.value`): RESOLVED → `.name`.** ✅
Verified in current source:
- Node renders `ModelVersion.SD15.name` / `ModelVersion.SDXL.name` → `"SD15"`, `"SDXL"`
  (`nodes.py:224-225`).
- Node reverses the dropdown string with `model_version = ModelVersion[model_version]`
  (`nodes.py:286`) — i.e. **lookup by NAME**. Feeding it a `.value` (`"sd15"`) raises `KeyError`.
- Enum values are lowercase (`model_version.py`: `SD15="sd15"`, `SDXL="sdxl"`,
  `SDXL_REFINER="sdxl_refiner"`, `LCM="lcm"`).
- `compose_out_name` does NOT consume the model_version string (grep of `core/naming.py` empty) —
  no coupling there, so no constraint from that side.

**Decision:** `list_model_versions()` returns `.name` (uppercase). Saved workflows store `"SD15"`,
node already validates them via `ModelVersion[...]`. The `_MODEL_STATUS` block above was corrected
to key by enum member and emit `.name`. **The earlier `v.value` form was a latent bug.**

---

## 6. `python_coreml_stable_diffusion` split (Gate E0 line to fill by grep)

| Use | Side | Status |
|---|---|---|
| `coreml_model.CoreMLModel` (runs compiled model) | **STAYS** (suite runtime) | ✅ — **local class**, not Apple's |
| `unet.UNet2DConditionModel*` internals | **gone** — `converter.py:319` uses `diffusers.UNet2DConditionModel.from_single_file` | ✅ |
| `AttentionImplementations` enum | gone — local `apply_attention_implementation` + `attention.py` tuple | ✅ |
| `calculate_conv2d_output_shape` | gone — replaced by `conversion/shapes.conv2d_output_shape` | ✅ |

> ### ⚠️ SPEC IS STALE: `ml-stable-diffusion` is already fully removed
> Commit #58 ("replace apple/ml-stable-diffusion with native diffusers conversion") already did
> the de-Apple work. Verified now:
> - **Zero** `python_coreml_stable_diffusion` runtime imports anywhere in `coreml_suite` (only a
>   docstring mention at `core/__init__.py:4`).
> - `coreml_suite/coreml_model.py:8` `CoreMLModel` is a **local** wrapper over
>   `coremltools.models.MLModel` (`coreml_model.py:22`) — it does **not** import Apple's class.
> - `ml-stable-diffusion` / `python_coreml_stable_diffusion` appears in **neither** `pyproject.toml`
>   **nor** `requirements.txt`. It is not a dependency at all.
>
> **Consequences for the spec (correct these in CONVERTER_EXTRACTION_SPEC.md):**
> - §0.3 premise ("runtime loader = `python_coreml_stable_diffusion.coreml_model.CoreMLModel`,
>   stays in suite") is **wrong**: the loader is already the local `coreml_model.CoreMLModel`. The
>   "stays in suite" conclusion still holds; the identity does not.
> - **Gate E0 item "ml-stable-diffusion pinned SHA — BLOCKER if unpinned" is MOOT** — there is no
>   such dep to pin. Mark it N/A, not BLOCKER.
> - **E4/E5 dependency lists must drop `git+...ml-stable-diffusion@<sha>`.** Package runtime deps
>   are: `coremltools`, `diffusers`, `peft` (LoRA), `omegaconf` (config), `numpy`, `torch`. Confirm
>   `peft`/`omegaconf` actually used before listing (grep at E4).
> - The "keep `python_coreml_stable_diffusion` as a suite dep for the loader" instruction in E5 is
>   **void** — coremltools backs the loader.

---

## 7. Pre-flight checklist before E1 (run these greps)

```
grep -rn "import comfy"            coreml_suite/conversion coreml_suite/converter.py coreml_suite/lcm/converter.py coreml_suite/lcm/unet.py
grep -rn "folder_paths"            coreml_suite/converter.py coreml_suite/lcm/converter.py
grep -rn "model_management"        coreml_suite/lcm
grep -rn "python_coreml_stable_diffusion" coreml_suite
grep -rn "ATTENTION_IMPLEMENTATION_IN_EFFECT" coreml_suite
grep -rn "SimianLuo\|LCM_Dreamshaper" coreml_suite/lcm
```
Every 🔍 above resolves to ✅ or a correction once these run. Do not start moving code (E2)
with any 🔍 unresolved on the CONVERSION side.

**STATUS (run 2026-05-26): all 🔍 resolved.** Summary of what the greps found:
- `conversion/*`, `lcm/unet.py`: comfy-free (torch/diffusers only). ✅
- `converter.py`: only comfy reach-in is `folder_paths` in `get_out_path` (L91-94) → inject `out_path`.
- `lcm/converter.py`: `folder_paths` (L111-114) + `comfy.model_management.get_torch_device` (L54)
  → cut both. Dup helpers (`load_coreml_model`,`convert_to_coreml`,`get_out_path`,`get_sample_input`)
  confirmed → dedup E2. `MODEL_VERSION="SimianLuo/LCM_Dreamshaper_v7"` (L22) → E-LCM.
- No attention module-global anywhere (`ATTENTION_IMPLEMENTATION_IN_EFFECT` absent); already per-call.
  LCM hardcodes `"SPLIT_EINSUM"` in `get_unets` — thread `attn_impl` through during dedup.
- `.name` vs `.value`: **decided `.name`** (node reverses via `ModelVersion[...]`). §5 corrected.
- `ml-stable-diffusion`: **already gone** (#58). §6 stale-spec note added — fix the spec's E0/E4/E5
  dep + pinning items.

Two grep blind-spots to note (the checklist above doesn't cover them, but cheap to add): the
`folder_paths` grep only scans the two converter files — also grep `coreml_suite/lcm/utils.py`
(it imports `comfy.model_management` at L3, but it's inference/STAYS, so fine) and confirm no other
`conversion/` file grew a comfy import since.
