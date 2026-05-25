# Spike 1 — MultiFunction models

**Question (from spec):** Can one `.mlpackage` hold multiple batch sizes /
controlnet-on-off variants sharing weights, replacing the
"reconvert-per-config / encode-in-filename" approach?

**Verdict:** Prototype for **controlnet-on/off only**. Use enumerated shapes
(spike 2), not multifunction, for batch size. Hard-gated behind **macOS 15 /
iOS 18**. Medium effort, medium payoff.

## What it is

MultiFunction packs several "functions" into one `mlprogram` `.mlpackage` and
**deduplicates identical weights across them**, so disk and in-memory footprint
shrink rather than just bundling N models.

- Introduced: **coremltools 8.0** (2024-09-16). Unchanged through 9.0.
- Runtime minimum: **macOS 15 / iOS 18** (enforced — `save_multifunction` forces
  `spec_version = max(spec_version, iOS18)`).
- `mlprogram` only (all of this suite's `.mlpackage`s already are).

Build a multifunction package:

```python
from coremltools.utils import MultiFunctionDescriptor, save_multifunction

desc = MultiFunctionDescriptor()
desc.add_function(model_path="unet.mlpackage",
                  source_function_name="main", target_function_name="plain")
desc.add_function(model_path="unet_cn.mlpackage",
                  source_function_name="main", target_function_name="controlnet")
desc.default_function_name = "plain"          # mandatory
save_multifunction(desc, "unet_multi.mlpackage")
```

Load a specific function (the save-time `CPU_ONLY` default does **not** bind
runtime compute units — reload with the unit you want):

```python
import coremltools as ct
m = ct.models.MLModel("unet_multi.mlpackage",
                      function_name="controlnet",
                      compute_units=ct.ComputeUnit.CPU_AND_NE)
```

Swift runtime selection: `MLModelConfiguration.functionName`.

## Does it actually share weights?

Yes, genuinely — not just a bundle. `save_multifunction` runs the MIL pass
`const_deduplication` across functions. Dedup is **value-based (weight-hash)**,
not reference-based: two *independently converted* packages dedup their
bit-identical tensors at merge time.

**Critical caveat for this suite:** dedup only fires if the shared tensors are
**byte-identical**. Different palettization/quantization (this repo's
`quantize_nbits` k-means path, `converter.py:309-325`) between two functions
defeats the hash match — you silently get a bundle, not a shared-weight model,
with no error. To benefit, convert the shared UNet trunk with identical
precision for both functions.

Apple's documented benefit is **smaller size + shared in-memory weights** (load
base once, switch functions cheaply). No Apple source claims an ANE *compute*
speedup beyond that memory/load win.

## How the two targets map

| Target | Fit | Why |
|--------|-----|-----|
| **controlnet-on / controlnet-off** | **Good** | Structurally the adapter case Apple pushes (WWDC24 10159/10161): shared UNet trunk + extra control inputs in one function. Different I/O signatures per function are fully supported. This is the prototype candidate. |
| **different batch sizes** | **Awkward — wrong tool** | Batch is a *shape* difference, not a graph/weight difference. The idiomatic answer is `EnumeratedShapes` on a single function (spike 2): same weights, multiple shapes, no merge step, works on older OS targets. |

Current per-config grounding: `converter.py:288-289` adds controlnet inputs via
`add_cnet_support`; `naming.py:22-63` encodes `_cn` into the filename cache key.
A 2-function package would replace the two separate `*_cn` / non-`cn` artifacts
with one — but only for macOS 15+ users.

## Pipeline support

`python_coreml_stable_diffusion` / `apple/ml-stable-diffusion` has **no
multifunction support** — it still emits separate `.mlpackage`s per component
and per config (`--unet-batch-one`, separate `*_control-unet.mlpackage`). You
would convert each variant as today, then add your **own** post-merge step
calling `MultiFunctionDescriptor`/`save_multifunction`.

## Effort / risk

- **Hard OS gate (macOS 15 / iOS 18).** Single biggest adoption risk — many Mac
  SD users lag on OS. Likely need to keep the per-config path as fallback, which
  erodes the simplicity win. The suite currently targets macOS 13.
- **Dedup defeated by mismatched quant** between functions (see above).
- `default_function_name` is mandatory; `source_function_name` is almost always
  `"main"` for `ct.convert` output — verify per package.
- Spec version is forced up to the max of inputs (≥ iOS18); cannot keep a lower
  target after merge.

## Recommendation

Low priority. If pursued: prototype **controlnet-on/off as a two-function
package**, converting the shared trunk with identical precision so dedup fires;
gate behind a macOS 15+ check; keep per-config reconvert as fallback. Do **not**
use multifunction for batch size — that is spike 2's job. Measure ANE engagement
and first-load compile time on real hardware before committing — those are the
make-or-break unknowns.

## Unverified (needs [M2-ANE])

- Real-world ANE behavior of a merged SD UNet — no public report either way.
- Whether Core ML compiles a multifunction package per-function or whole-package
  on first load (SD UNet first-load compile is minutes — could worsen startup).
- Exact `@available` version string for Swift `MLModelConfiguration.functionName`
  (property confirmed; version inferred iOS 18 from the feature gate).

## Sources

- coremltools MultiFunction guide — https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html
- `coremltools/models/utils.py` source — https://github.com/apple/coremltools/blob/main/coremltools/models/utils.py
- coremltools 8.0 release notes — https://github.com/apple/coremltools/releases/tag/8.0
- WWDC24 10159 (Bring your ML/AI models to Apple silicon) — https://developer.apple.com/videos/play/wwdc2024/10159/
- WWDC24 10161 (Deploy ML/AI models on-device with Core ML) — https://developer.apple.com/videos/play/wwdc2024/10161/
- `MLModelConfiguration.functionName` — https://developer.apple.com/documentation/coreml/mlmodelconfiguration/functionname
- apple/ml-stable-diffusion `torch2coreml.py` — https://github.com/apple/ml-stable-diffusion/blob/main/python_coreml_stable_diffusion/torch2coreml.py
