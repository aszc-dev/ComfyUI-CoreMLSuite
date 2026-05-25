# Spike 2 — Flexible / enumerated shapes

**Question (from spec):** Can enumerated/dynamic shapes +
`materialize_dynamic_shape_mlmodel` remove the fixed 512×512 / per-size
reconversion and simplify the `chunk_batch` padding?

**Verdict:** Use `ct.EnumeratedShapes` over a small set of **spatial** latent
sizes to kill per-size reconversion while keeping the ANE. Keep `chunk_batch`
for the **batch** dimension. Medium effort, medium–high payoff. The batch logic
is only partially eliminable and not worth the OS-gate cost.

## The three mechanisms

```python
import coremltools as ct

# (a) RangeDim — continuous bounded range per dim
shape = ct.Shape(shape=(1, 4,
        ct.RangeDim(lower_bound=64, upper_bound=128, default=64),
        ct.RangeDim(lower_bound=64, upper_bound=128, default=64)))

# (b) EnumeratedShapes — fixed finite set (≤128 shapes)
shape = ct.EnumeratedShapes(
        shapes=[[1,4,64,64],[1,4,96,96],[1,4,128,128]], default=[1,4,64,64])

# (c) fixed — what converter.py:80 uses today
ct.TensorType(name="sample", shape=(1,4,64,64))
```

## The ANE crux (confirmed, not inferred)

coremltools maintainers confirmed in issue
[#2370](https://github.com/apple/coremltools/issues/2370) (tested coremltools
8.0 / macOS 15 / M3 Pro):

- **`EnumeratedShapes` → runs every listed shape on the ANE.** This is the only
  flexible option Apple endorses for an ANE model, and the only one their own
  SD repo points to.
- **`RangeDim` → only the *default* shape runs on ANE.** Non-default shapes fall
  to GPU/CPU (measured ~10× slower). The `ReshapeFrequency.Infrequent`
  optimization hint (iOS 17.4+) lets RangeDim use the ANE for arbitrary shapes,
  but **each new shape triggers an ANE recompile** (latency spike).

coremltools FAQ, verbatim: *"Use `EnumeratedShapes` for best performance ...
The converted model will run on the NE, unless the conversion introduces dynamic
layers not supported on the NE, such as converting a static reshape to a fully
dynamic reshape."*

## `materialize_dynamic_shape_mlmodel`

Alternative path: convert once with symbolic dims, then bake out N fixed-shape
**functions** into one weight-shared package.

```python
from coremltools.utils import materialize_dynamic_shape_mlmodel
materialize_dynamic_shape_mlmodel(
    dynamic_shape_mlmodel,
    {"size_64": {"sample": (1,4,64,64)}, "size_96": {"sample": (1,4,96,96)}},
    "unet_materialized.mlpackage")
```

- Introduced coremltools **8.0**. Tensor I/O only (SD UNet is `MLMultiArray` — OK).
- Deployment target: materializing **>1 function** (or non-`main` source) emits a
  multifunction model → forced to **iOS 18 / macOS 15**. Materializing exactly
  **one** function from `"main"` stays unifunction at the **original** target
  (macOS 13 stays viable for the single-shape case).
- Materialized functions are fully static → run on ANE like normal fixed-shape
  models, with shared weights across functions.

## Does this kill `chunk_batch` or just per-size reconversion?

Current grounding:
- `converter.py:266-271` converts for a fixed `(batch, C, H, W)` `sample_shape`.
- `latents.py:10-35` `chunk_batch` pads/splits the **batch** dim to the fixed
  converted batch size.

| Approach | ANE | Kills per-size reconv? | Kills `chunk_batch`? | Min target | Effort |
|----------|-----|------------------------|----------------------|------------|--------|
| `EnumeratedShapes` (spatial only) | Yes (all shapes) | **Yes** | No | macOS 13 | Med — `group_norm` hurdle |
| `EnumeratedShapes` (spatial × batch) | Yes | Yes | Partially | **macOS 15** (multi-input) | Med–High |
| `RangeDim` | Default only / `Infrequent`=recompile | Yes but slow | Partially | iOS 17.4 (hint) | Low code, bad ANE |
| `materialize_dynamic_shape_mlmodel` | Yes (static funcs) | Yes (1 package) | No | macOS 15 if >1 func | Med |

- **Per-size (spatial H/W) reconversion → realistically eliminable** via
  `EnumeratedShapes` over a curated size set, keeping the ANE.
- **`chunk_batch` batch padding → only partially eliminable, higher risk.**
  Reasons: (1) batch must co-vary across multiple UNet inputs (`sample`,
  `encoder_hidden_states`, `timestep`, SDXL/LCM/cnet extras), which hits the
  **pre-iOS 18 single-`EnumeratedShapes`-input limit** — a hard wall unless you
  target macOS 15+; (2) enumeration only covers the exact batch sizes you list,
  so you still need `chunk_batch`-style fallback for unlisted counts;
  (3) batch × spatial multiplies the enumerated-shape count, inflating compile
  time and package size.

## What breaks / risk

- **`group_norm` conversion failure** is documented for flexible-shape SD UNet
  (ml-stable-diffusion [#70](https://github.com/apple/ml-stable-diffusion/issues/70)):
  flexible shapes failed at `group_norm`, and `SPLIT_EINSUM` with non-standard
  dims caused kernel panics. Flexible *spatial* dims on the SD UNet are known to
  be fiddly, not turnkey. (This suite defaults to `SPLIT_EINSUM` —
  `converter.py:339`.)
- One **unreproduced** report in #2370 of `EnumeratedShapes` on `mlprogram`
  running only the default shape on ANE; the maintainer could not reproduce.
  Treat the documented "all shapes on ANE" as expected but **benchmark on target
  hardware** — this is the single biggest verification gap.

## Recommendation

Highest-value direction of the three packaging spikes. If pursued:
`EnumeratedShapes` over a curated set of latent **spatial** sizes (e.g. 64²/96²/
128²), keep batch fixed at the CFG-doubled value, keep `chunk_batch` for batch.
Stays on macOS 13. Budget time for the `group_norm` conversion hurdle and
**prove ANE residency on real hardware first** before removing any per-size
reconversion. `materialize_dynamic_shape_mlmodel` is a clean fallback for
"ship N fixed sizes in one weight-shared package" but forces macOS 15 once you
bake >1 function.

## Unverified (needs [M2-ANE])

- `EnumeratedShapes`-on-`mlprogram` ANE residency for a *real* SD UNet
  (the #2370 regression — unreproduced by maintainer).
- That materialized functions run on ANE (strongly implied; no benchmark found).
- `.mlmodelc` size / compile-time growth per enumerated shape (qualitative only).
- Whether batch-as-flexible-dim is ANE-safe for the SD UNet (no API prohibition,
  no positive confirmation).

## Sources

- coremltools flexible inputs guide — https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html
- coremltools #2370 (Flexible shapes on Neural Engine) — https://github.com/apple/coremltools/issues/2370
- coremltools #2386 (docs fix) — https://github.com/apple/coremltools/pull/2386
- `materialize_dynamic_shape_mlmodel` source — https://github.com/apple/coremltools/blob/main/coremltools/models/utils.py
- coremltools 8.0 release notes — https://github.com/apple/coremltools/releases/tag/8.0
- ml-stable-diffusion #70 (flexible-shape group_norm failure) — https://github.com/apple/ml-stable-diffusion/issues/70
