# Limitations & Support Matrix

## Support matrix

| Feature | Status | Notes |
|---|---|---|
| SD1.5 | ✅ Full | ANE via `SPLIT_EINSUM`; the primary, fastest path |
| SDXL / SDXL Turbo | ⚠️ Partial | GPU only (no ANE), no speedup; possible quality loss vs source. Don't run Turbo at 1024² |
| SD2.1 | ❌ Unsupported | |
| Inpainting checkpoints (9-channel) | ❌ Unsupported | |
| ControlNet | ✅ Supported | Convert the checkpoint with `controlnet_support = True` |
| LoRA | ⚠️ Experimental | Inconsistent per-LoRA; baked at conversion, immutable afterward |
| LCM | ⚠️ Experimental | Full-distill LCM checkpoints auto-detected by the converter |
| SVD | ❌ Not supported | |
| AnimateDiff | ❌ Not supported | Motion modules need pre-conversion injection; not feasible today |
| IPAdapter | ❌ Not supported | Needs a real `MODEL` the Core ML wrapper can't provide |
| Core ML Adapter | ⚠️ Experimental | Works for many nodes; fails for merges/IPAdapter/etc. |

## Fixed input/output shapes

A Core ML model is converted for one specific resolution and batch size. To work
at a different size, re-convert with the new width/height (conversion is cheap and
cached by name). This is also why detailers and latent-upscale workflows that
rescale mid-graph break — see [troubleshooting](troubleshooting.md).

There is experimental support for flexible shapes via
[EnumeratedShapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html#select-from-predetermined-shapes),
but it is **much slower** — user benchmarks show roughly **5×** the per-iteration
time on every run, not just the first. Fixed-shape models per resolution are the
practical choice.

## SDXL on the Neural Engine

SDXL and SDXL Turbo cannot run on the ANE — the dual-text-encoder UNet exceeds the
supported Neural Engine path. They run on the GPU at roughly MPS-equivalent speed,
so Core ML offers no speed advantage for SDXL, and converted output may look
degraded versus the safetensors original (an upstream conversion artifact). Use
`ORIGINAL` + `CPU_AND_GPU`. See [hardware](hardware.md).

## Experimental Core ML Adapter

The Adapter wraps a Core ML model to look like a standard ComfyUI `MODEL`, which
covers many standard and custom nodes. But it can't fully emulate a real model:
operations that need genuine `MODEL` internals — model merges, IPAdapter, some
LoRA flows, detailers — generally won't work, and the model's
fixed input shapes aren't validated, so mismatches error at runtime. Prefer the
native Core ML Sampler when you don't need the `MODEL` type.

## Prompt length

Core ML enforces a hard 77-token prompt limit with no auto-chunking. Split long
prompts across multiple CLIP Text Encode nodes and merge with Conditioning
(Combine).
