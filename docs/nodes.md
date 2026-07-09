# Node Reference

All nodes live in the **Core ML Suite** category. Right-click the canvas →
**Add Node → Core ML Suite**, or double-click and search.

| Display name | Class | Purpose |
|---|---|---|
| Load Core ML UNet | `CoreMLUNetLoader` | Load a converted `.mlpackage` |
| Core ML Sampler | `CoreMLSampler` | Sample (KSampler-style) |
| Core ML Sampler (Advanced) | `CoreMLSamplerAdvanced` | Sample (KSamplerAdvanced-style) |
| Core ML Adapter (Experimental) | `CoreMLModelAdapter` | Wrap as a standard `MODEL` |
| Load LoRA to use with Core ML | `Core ML LoRA Loader` | Bake LoRA(s) at conversion |
| Convert Checkpoint to Core ML | `Core ML Converter` | Convert a checkpoint |

---

## Load Core ML UNet (`CoreMLUNetLoader`)

![Load Core ML UNet](../assets/unet_loader.png?raw=true)

Loads a converted `.mlpackage` from `models/unet` and outputs a `coreml_model`
for the samplers. Only `.mlpackage` files are listed — this suite no longer uses
`.mlmodelc`.

- **Inputs**
  - `coreml_name` — the `.mlpackage` to load from `models/unet`.
  - `compute_unit` — hardware to run on: `CPU_AND_NE` (default), `CPU_AND_GPU`,
    `CPU_ONLY`, `ALL`. See [hardware](hardware.md).
- **Output**
  - `coreml_model` — for the Core ML Sampler or Adapter.

---

## Core ML Sampler (`CoreMLSampler`)

![Core ML Sampler](../assets/sampler.png?raw=true)

Generates a latent from a Core ML model. Behaves like the standard KSampler and
outputs a `LATENT` you can decode or feed downstream.

- **Inputs**
  - `coreml_model` — output of the loader or a converter.
  - `latent_image` *(optional)* — must match the model's input size. If omitted,
    a suitable empty latent is created. Provide one for img2img.
  - `negative` *(optional)* — required for normal models; optional for LCM.
  - Remaining inputs (`seed`, `steps`, `cfg`, `sampler_name`, `scheduler`,
    `positive`, `denoise`) match the KSampler.
- **Output**
  - `LATENT` — decode with a VAE Decode, or use downstream.

---

## Core ML Sampler (Advanced) (`CoreMLSamplerAdvanced`)

The KSamplerAdvanced counterpart of the Core ML Sampler — same Core ML input,
plus the advanced sampling controls. Use it for partial denoising, fixed noise,
and multi-stage (e.g. SDXL base → refiner) workflows.

- **Inputs**
  - `coreml_model` — output of the loader or a converter.
  - `add_noise`, `noise_seed`, `start_at_step`, `end_at_step`,
    `return_with_leftover_noise` — as in KSamplerAdvanced.
  - `steps`, `cfg`, `sampler_name`, `scheduler`, `positive` — as usual.
  - `latent_image` *(optional)*, `negative` *(optional, required for non-LCM)*.
- **Output**
  - `LATENT`.

---

## Core ML Adapter (Experimental) (`CoreMLModelAdapter`)

![Core ML Adapter](../assets/adapter.png?raw=true)

Wraps a Core ML model so it presents as a standard ComfyUI `MODEL`, letting you
feed it to the normal KSampler and many other nodes (e.g. `ModelSamplingDiscrete`
for LCM LoRAs).

- **Input**
  - `coreml_model`.
- **Output**
  - `MODEL` — a Core ML model wrapped as a ComfyUI model.

> [!NOTE]
> Experimental. The wrapper presents a `MODEL` interface but cannot fully
> emulate one — model merges, IPAdapter, and similar advanced uses generally
> won't work, and the model's fixed input shapes are not validated, so mismatched
> inputs error at runtime. The native Core ML Sampler is faster when you don't
> need the `MODEL` type. See the [FAQ](faq.md) and [limitations](limitations.md).

---

## Load LoRA to use with Core ML (`Core ML LoRA Loader`)

![LoRA Loader](../assets/lora_loader.png?raw=true)

Collects LoRA name + `strength_model` to bake into the model at conversion, and
applies the LoRA to CLIP (which is not part of the Core ML path). Chain multiple
loaders for multiple LoRAs.

Because a converted model is immutable, the baked weights and `strength_model`
**cannot** be changed afterward — changing them means re-converting. `strength_clip`
only affects CLIP and can be changed freely. After conversion, when loading with
`CoreMLUNetLoader`, apply the same LoRAs to CLIP manually (see
[workflows](workflows.md)).

- **Inputs**
  - `lora_name`, `strength_model`, `strength_clip`.
  - `clip` — from `CLIPLoader` / `CheckpointLoaderSimple` or another LoRA loader.
  - `lora_params` *(optional)* — chain from another LoRA loader.
- **Outputs**
  - `CLIP` — with the LoRA applied.
  - `lora_params` — pass to the converter or the next LoRA loader.

> [!NOTE]
> LoRA support is experimental and inconsistent — some LoRAs convert cleanly,
> others produce poor results. Test per-LoRA. See [troubleshooting](troubleshooting.md).

---

## Convert Checkpoint to Core ML (`Core ML Converter`)

![Checkpoint Converter](../assets/checkpoint_converter.png?raw=true)

Converts a checkpoint from `models/checkpoints` to a Core ML `.mlpackage` in
`models/unet`. The model version (SD1.5, SDXL, SDXL refiner, or full-distill
LCM) is auto-detected from the checkpoint's architecture — there is no version
dropdown. The conversion parameters are encoded in the output name, so an
already-converted model is reused instead of re-converted. See
[conversion](conversion.md) for details.

- **Inputs**
  - `ckpt_name` — checkpoint in `models/checkpoints`.
  - `height`, `width` — target image size; any positive multiple of 8 (default
    512). The model's input size is fixed at these values.
  - `batch_size` — default 1; raise to convert a batch-capable model.
  - `attention_implementation` — `SPLIT_EINSUM` / `SPLIT_EINSUM_V2` (ANE) or
    `ORIGINAL` (GPU). See [hardware](hardware.md).
  - `compute_unit` — used only when loading the result; does not affect
    conversion.
  - `controlnet_support` — set `True` to make the model usable with ControlNet
    (default `False`).
  - `quantize_nbits` *(optional)* — `none` (default), `8`, `6`, `4`. See
    [conversion → quantization](conversion.md#quantization).
  - `lora_params` *(optional)* — from the LoRA loader, to bake LoRAs in.
- **Output**
  - `coreml_model`.

> [!NOTE]
> Some checkpoints need a custom config `.yaml`. Place it in `models/configs`
> named like the checkpoint (e.g. `juggernaut.safetensors` →
> `juggernaut.yaml`); it is loaded automatically during conversion.

> [!NOTE]
> Full-distill LCM checkpoints (e.g.
> [LCM_Dreamshaper_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7)) are
> detected and converted like any other checkpoint. When sampling an LCM model,
> set `sampler_name` to `lcm` and `scheduler` to `sgm_uniform`.
