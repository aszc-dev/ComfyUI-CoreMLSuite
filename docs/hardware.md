# Hardware & Compute Units

This page explains how the suite maps to Apple Silicon hardware, the difference
between ANE, GPU, and MPS, and how to choose a compute unit and attention
implementation.

## ANE vs GPU vs MPS

Three terms get conflated:

- **ANE (Apple Neural Engine)** — a dedicated ML accelerator on Apple Silicon.
  Only Core ML can target it; PyTorch cannot. This is the whole reason the suite
  exists.
- **GPU** — the Metal GPU. Reachable both by Core ML (as a compute unit) and by
  PyTorch (via MPS).
- **MPS (Metal Performance Shaders)** — PyTorch's GPU backend on macOS. This is
  the path standard ComfyUI nodes use.

**This suite uses Core ML compute units only — it never runs the UNet through
PyTorch/MPS.** Consequently `PYTORCH_ENABLE_MPS_FALLBACK` has no effect on these
nodes. It may still matter for the rest of your ComfyUI graph (CLIP, VAE,
samplers on non-Core ML models), but not for Core ML inference itself.

Rough performance picture (SD1.5, maintainer- and user-reported):

- ANE is meaningfully faster than MPS — on the order of **50–100%** for SD1.5.
- Core ML on the GPU is only marginally faster than PyTorch/MPS.

So the speedup comes from the Neural Engine, which means it depends on being able
to actually run on the ANE (see [attention implementations](#attention-implementations)
and the [SDXL caveat](#sdxl-and-the-ane)).

## Compute units

The **compute unit** is set on the loader/converter node and tells Core ML which
hardware to use. It is applied when the model is loaded
(`coreml_suite/coreml_model.py:22`), not during conversion.

| Value | Hardware | Best paired with |
|---|---|---|
| `CPU_AND_NE` (default) | CPU + Neural Engine | `SPLIT_EINSUM` / `SPLIT_EINSUM_V2` |
| `CPU_AND_GPU` | CPU + Metal GPU | `ORIGINAL` |
| `CPU_ONLY` | CPU only | fallback / debugging |
| `ALL` | all available hardware | rarely optimal — see below |

Notes:

- Every option includes the CPU; there is no GPU-and-ANE-without-CPU combination.
- `NE` in `CPU_AND_NE` is the Neural Engine (Apple's enum spells it `NE`, not
  `ANE`).
- **`CPU_AND_NE` is often faster than `ALL`.** Letting Core ML use everything can
  be *slower* on non-Max chips, where memory bandwidth is the bottleneck. Try
  `CPU_AND_NE` first for SD1.5.

## Attention implementations

Chosen at conversion time on the **Convert Checkpoint to Core ML** node. It
decides whether the model can run on the ANE:

- **`SPLIT_EINSUM`** — ANE-friendly attention. Use for the Neural Engine.
- **`SPLIT_EINSUM_V2`** — a variant; in practice ≈ `SPLIT_EINSUM` for most users.
- **`ORIGINAL`** — standard attention. Runs on the GPU, not the ANE.

The implementation and the compute unit must agree: a `SPLIT_EINSUM` model wants
`CPU_AND_NE`; an `ORIGINAL` model wants `CPU_AND_GPU`.

## Which should I pick?

| Scenario | Attention | Compute unit |
|---|---|---|
| SD1.5 at 512×512 | `SPLIT_EINSUM` | `CPU_AND_NE` |
| SD1.5 at larger sizes (e.g. 768) | `ORIGINAL` | `CPU_AND_GPU` |
| SDXL / SDXL Turbo | `ORIGINAL` | `CPU_AND_GPU` |

### Resolution crossover

ANE shines at small latents; the GPU scales better as resolution grows. In user
benchmarks:

- At **512×512**, ANE + `SPLIT_EINSUM` wins by roughly **10%** over the GPU path.
- At **768×512**, GPU + `ORIGINAL` pulls ahead by roughly **10%**, and the larger
  image is about 2× slower overall.

If you mostly work at 512×512, convert with `SPLIT_EINSUM` and load on
`CPU_AND_NE`. If you routinely go larger, an `ORIGINAL` + GPU model may be
faster.

### SDXL and the ANE

SDXL (and SDXL Turbo) **cannot run on the ANE** — the dual-text-encoder UNet
exceeds what the Neural Engine path supports. SDXL therefore runs at roughly
MPS-equivalent speed with no ANE speedup. If a Core ML SDXL workflow feels no
faster than the standard nodes, this is why. Convert SDXL with `ORIGINAL` and
load with `CPU_AND_GPU` or `CPU_ONLY`. See
[limitations](limitations.md) for the full picture.
