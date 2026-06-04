# Troubleshooting

## `Expected shape … got …` / latent size mismatch

The most common error. A Core ML model has **fixed** input dimensions — a model
converted for 512×512 expects a 64×64 latent and rejects any other size (batch
size is handled and doesn't matter; only width/height are fixed).

**Fix:** set your Empty Latent (or upstream latent) to exactly the resolution the
model was converted for, or re-convert at the size you want.

## Old `.mlmodelc` model, or `metadata.json` not found

This suite no longer produces or loads `.mlmodelc`; the loader lists `.mlpackage`
only. Models from an older version (or downloaded community models) with a
`.mlmodelc` structure won't load.

**Fix:** re-convert the checkpoint with **Convert Checkpoint to Core ML**. No
Xcode or `coremlcompiler` is required — that dependency was removed.

## Prompt too long (`Expected size 154 but got 77`, or a crash)

Core ML enforces a hard **77-token** prompt limit and does not auto-chunk like
A1111/ComfyUI.

**Fix:** split the prompt across multiple CLIP Text Encode nodes and merge them
with **Conditioning (Combine)**.

## `cannot import name 'ModelSamplingDiscreteLCM'`

A ComfyUI refactor renamed this symbol.

**Fix:** update the suite (fixed in PR #29) and re-run
`pip install -r requirements.txt`.

## LoRA loader `ImportError`

`peft` became a required dependency.

**Fix:** `pip install -r requirements.txt`. This recurs after ComfyUI-Manager
updates if requirements aren't reinstalled.

## ControlNet has no effect

ControlNet support is baked at conversion. If the checkpoint was converted with
`controlnet_support = False`, ControlNet does nothing.

**Fix:** re-convert with `controlnet_support = True`. The ControlNet model itself
needs no conversion, and `.fp16.safetensors` vs `.safetensors` makes no
difference.

## LoRAs produce garbage

LoRA support is inconsistent — some work, some don't, with no firm rule. Test
per-LoRA. For some LCM-LoRA setups, routing through the
[Core ML Adapter](nodes.md#core-ml-adapter-experimental-coremlmodeladapter) is
more reliable than the basic loader path. Remember weights are baked at conversion
and can't be changed afterward.

## FaceDetailer / detailers error on size

Detailers rescale latents internally (e.g. 512 → 1024), which breaks the model's
fixed input shape.

**Fix:** use the `CoreMLDetailerHookProvider` node to pin the detailer's internal
size to the model's converted resolution. Note it only offers preset sizes, so
non-standard resolutions may not be selectable.

## Inpainting checkpoint errors (`tensor size 9 vs 4`)

SD1.5 inpainting checkpoints use a 9-channel input and are **not supported**. This
error is expected, not a bug.

## Errors mentioning `python_coreml_stable_diffusion` or `ml-stable-diffusion`

You're on a stale install. That dependency was removed; old install scripts tried
`pip install git+…/ml-stable-diffusion.git`, which fails on modern Python.

**Fix:** reinstall the current suite (`pip install -r requirements.txt`, which
pulls `coreml-diffusion` from PyPI).

## `all input tensors must be on the same device (mps:0 and cpu)` / ControlNet residual shape `(2,…) vs (1,…)`

Old bugs that have been fixed.

**Fix:** update to the latest version.
