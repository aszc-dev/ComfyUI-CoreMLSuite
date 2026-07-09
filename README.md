# Core ML Suite for ComfyUI

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that run
Stable Diffusion UNets as [Core ML](https://developer.apple.com/documentation/coreml)
models on Apple Silicon (M1/M2/M3). Core ML can use the Apple Neural Engine
(ANE), which is unavailable to PyTorch — on an M2 Pro 32 GB, SD1.5 at 512×512
generates roughly **1.5–2× faster** than the standard PyTorch/MPS path.

You convert a Stable Diffusion checkpoint to a Core ML model with the nodes in
this suite, then sample from it like any other ComfyUI workflow.

> [!IMPORTANT]
> **Convert your own checkpoints — that is the only supported path.** This
> suite uses its own input dimensions, naming convention, and metadata
> (produced by the [coreml-diffusion](https://github.com/aszc-dev/coreml-diffusion)
> package). Pre-converted Core ML models from elsewhere (e.g. the
> coreml-community Hugging Face org) are **not** supported. Conversion is cheap
> and runs on your machine, so there is no need to download Core ML models.

## Installation

### ComfyUI-Manager (recommended)

Open **Manager → Install Custom Nodes**, search for `Core ML`, click
**Install**, and restart ComfyUI.

### Manual

```bash
cd /path/to/comfyui/custom_nodes
git clone https://github.com/aszc-dev/ComfyUI-CoreMLSuite.git
cd ComfyUI-CoreMLSuite
pip install -r requirements.txt
```

Dependencies (`coreml-diffusion`, `coremltools`, `numpy`, `diffusers`) install
from PyPI. PyTorch is intentionally **not** pinned — it is provided by your
ComfyUI host, and a hard cap here would downgrade it and break ComfyUI.

## Quickstart

1. Put a SD1.5 checkpoint in `models/checkpoints`.
2. Add the **Convert Checkpoint to Core ML** node, select the checkpoint, and
   queue once. It writes a `.mlpackage` to `models/unet` (cached by name — it
   won't reconvert next time).
3. Sample with the **Core ML Sampler** node, decoding the latent with a normal
   VAE Decode. CLIP and VAE come from standard ComfyUI nodes.

See [docs/workflows.md](docs/workflows.md) for complete example graphs (txt2img,
ControlNet, LoRA, LCM, SDXL).

## Which compute unit should I pick?

The **compute unit** selects the hardware Core ML runs on. Pair it with the
attention implementation chosen at conversion time:

| Model | Convert with | Load with | Runs on |
|---|---|---|---|
| SD1.5 @ 512×512 | `SPLIT_EINSUM` | `CPU_AND_NE` | Neural Engine (fastest) |
| SD1.5 @ larger sizes | `ORIGINAL` | `CPU_AND_GPU` | GPU |
| SDXL | `ORIGINAL` | `CPU_AND_GPU` | GPU (ANE unsupported) |

`CPU_AND_NE` is usually the fastest option for SD1.5 — often faster than `ALL`.
This suite uses Core ML compute units only; it never touches PyTorch MPS, so
`PYTORCH_ENABLE_MPS_FALLBACK` is irrelevant to these nodes. Full reasoning and
benchmarks: [docs/hardware.md](docs/hardware.md).

## Documentation

- [Hardware & compute units](docs/hardware.md) — ANE vs GPU vs MPS, attention
  implementations, which to choose.
- [Nodes](docs/nodes.md) — full reference for every node.
- [Conversion](docs/conversion.md) — how conversion works, caching,
  quantization.
- [Example workflows](docs/workflows.md) — annotated example graphs.
- [FAQ](docs/faq.md) — answers to common questions.
- [Troubleshooting](docs/troubleshooting.md) — common errors and fixes.
- [Limitations & support matrix](docs/limitations.md) — what is and isn't
  supported.

## Glossary

- **Core ML** — Apple's on-device machine-learning framework.
- **`.mlpackage`** — the Core ML model format this suite produces and loads.
- **ANE** — Apple Neural Engine, a hardware accelerator for ML.
- **Compute unit** — which hardware Core ML uses (`CPU_AND_NE`, `CPU_AND_GPU`,
  `CPU_ONLY`, `ALL`).
- **Attention implementation** — `SPLIT_EINSUM` / `SPLIT_EINSUM_V2` (ANE-friendly)
  or `ORIGINAL` (GPU-friendly), chosen at conversion.

> [!IMPORTANT]
> **Breaking change in 2.0.0.** The converted Core ML UNet now takes
> `encoder_hidden_states` in the native `diffusers` layout
> `(batch, tokens, hidden)` instead of the previous `(batch, hidden, 1, tokens)`.
> Models converted with earlier versions are not compatible and must be
> re-converted.

## Acknowledgements

The conversion pipeline began as an adaptation of Apple's
[ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion), which
pioneered running Stable Diffusion on the Neural Engine. It has since diverged
and no longer depends on that package: UNet conversion runs natively on
`diffusers`' `UNet2DConditionModel`, the ANE attention path (`SPLIT_EINSUM`,
`SPLIT_EINSUM_V2`) is reimplemented as standalone `diffusers` attention
processors, and the toolchain tracks current ComfyUI (NumPy 2, Torch 2.7+,
coremltools 9, Python 3.12+). Conversion now lives in the separate
[coreml-diffusion](https://github.com/aszc-dev/coreml-diffusion) package.

## Support

Questions or suggestions? Open an
[issue](https://github.com/aszc-dev/ComfyUI-CoreMLSuite/issues).
