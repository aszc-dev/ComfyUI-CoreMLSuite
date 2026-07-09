# FAQ

## What's the difference between ANE, GPU, and MPS, and which do I pick?

ANE is the Neural Engine (Core ML only), GPU is the Metal GPU (Core ML or
PyTorch), MPS is PyTorch's GPU backend. This suite uses **Core ML compute units
only** and never touches MPS. Short answer: SD1.5 at 512×512 → convert
`SPLIT_EINSUM`, load `CPU_AND_NE`; larger sizes or SDXL → convert `ORIGINAL`,
load `CPU_AND_GPU`. Full reasoning: [hardware](hardware.md).

## Do I still need `PYTORCH_ENABLE_MPS_FALLBACK=1`?

Not for these nodes — Core ML inference doesn't use PyTorch MPS. It may still
matter for other parts of your ComfyUI graph, but it has no effect on Core ML
sampling.

## Why is my Core ML SDXL workflow no faster than the default nodes?

Because **SDXL can't run on the ANE** — the speedup comes from the Neural Engine,
and SDXL falls back to the GPU, running at roughly MPS-equivalent speed. This is a
known limitation, not a misconfiguration. The ANE benefit is real for SD1.5. See
[limitations](limitations.md).

## Where do I get Core ML models?

You convert them yourself — that's the only supported path. See
[conversion](conversion.md). Downloaded Core ML models (e.g. coreml-community) use
different dimensions/metadata and are not supported.

## Is conversion run every time I queue, or once?

Once. Parameters are encoded in the output filename, so an already-converted model
is reused and conversion is skipped. Convert once, then load the `.mlpackage`. See
[conversion → caching](conversion.md#one-time-conversion-and-name-based-caching).

## Does a converted model produce the same output as the original?

With the default `quantize_nbits = none`, the converted UNet output matches the
source within numerical rounding (the golden test in `tests/m2/test_golden_image.py`
gates on PSNR ≥ 20 dB on the decoded image). Quantization (`8`/`6`/`4`) introduces
measured, bounded drift — see the [PSNR table](conversion.md#quantization). For
bit-identical output, keep `none`.

## Are `.mlpackage` models safe to use?

`.mlpackage` is a declarative Core ML model format — it carries weights and a
compute graph, not arbitrary executable code or Python pickle, so its safety
profile is comparable to `safetensors`. In practice this matters little here,
since the only supported models are ones you convert locally from your own
checkpoints.

## Are LoRAs reliable?

Partially. Some LoRAs convert cleanly; others produce poor or broken output —
there's no firm rule, so test per-LoRA. LoRA weights and `strength_model` are
baked in at conversion and can't be changed afterward; for some LCM-LoRA cases the
[Core ML Adapter](nodes.md#core-ml-adapter-experimental-coremlmodeladapter) path
is more reliable. Treat LoRA support as experimental. See
[troubleshooting](troubleshooting.md).

## Does the experimental Adapter cost performance vs the Core ML Sampler?

Yes, a little. The Adapter wraps the model in a ComfyUI `ModelPatcher` so standard
samplers work, which adds per-step interface overhead the native Core ML Sampler
avoids. Use the native sampler unless you specifically need a `MODEL` (e.g.
`ModelSamplingDiscrete` for LCM LoRAs).

## Which Python versions work?

Python 3.12 or newer (`requires-python >=3.12`). Older 3.12 install failures
came from the now-removed `ml-stable-diffusion` build, not from this suite.

## Long prompts crash my workflow

Core ML has a hard **77-token** prompt limit and doesn't auto-chunk long prompts.
Split the prompt across multiple CLIP Text Encode nodes and merge with Conditioning
(Combine). See [troubleshooting](troubleshooting.md).
