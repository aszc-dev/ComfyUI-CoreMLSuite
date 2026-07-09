# Example Workflows

> [!NOTE]
> The models referenced are examples — substitute your own. Every workflow
> starts from a checkpoint you convert yourself (see [conversion](conversion.md));
> there is no Core ML model to download.

## Basic txt2img

Convert a SD1.5 checkpoint, then sample from it. CLIP and VAE come from standard
ComfyUI nodes — either loaded separately or pulled from the checkpoint.

1. Place a SD1.5 checkpoint in `models/checkpoints` (e.g.
   [v1-5-pruned-emaonly](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors)).
2. **Convert Checkpoint to Core ML** → queue once → a `.mlpackage` lands in
   `models/unet`.
3. **Load Core ML UNet** (or wire the converter output straight in) →
   **Core ML Sampler** → **VAE Decode**.

**CLIP and VAE from the checkpoint:**

![Core ML UNet + checkpoint](../assets/unet+sampler+checkpoint.png?raw=true)

**CLIP and VAE loaded separately** — use any SD1.5-compatible
[CLIP](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/text_encoder/model.safetensors)
and [VAE](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/vae/diffusion_pytorch_model.safetensors),
placed in `models/clip` and `models/vae`:

![Core ML UNet + CLIP + VAE](../assets/unet+sampler+clip+vae.png?raw=true)

## ControlNet

Convert the checkpoint with `controlnet_support = True`, then wire a standard
ComfyUI ControlNet. The ControlNet model itself needs no conversion. Place it in
`models/controlnet` (e.g.
[control_v11p_sd15_scribble](https://huggingface.co/lllyasviel/control_v11p_sd15_scribble/blob/main/diffusion_pytorch_model.fp16.safetensors)).

![Core ML UNet + ControlNet](../assets/unet+sampler+controlnet.png?raw=true)

## Checkpoint conversion

The minimal conversion graph. See
[Convert Checkpoint to Core ML](nodes.md#convert-checkpoint-to-core-ml-core-ml-converter).

![Checkpoint converter](../assets/basic_conversion.png?raw=true)

## Conversion with LoRA

Bake LoRA(s) into the model at conversion. Read the
[LoRA caveats](nodes.md#load-lora-to-use-with-core-ml-core-ml-lora-loader) first
— baked weights are immutable, and support is inconsistent per-LoRA.

![Checkpoint converter + LoRA](../assets/conversion+lora.png?raw=true)

## LCM LoRA conversion

Chain multiple LoRA loaders to use several LoRAs with one model.

> [!IMPORTANT]
> Here the model goes through the **Core ML Adapter** and `ModelSamplingDiscrete`
> into the standard ComfyUI KSampler (not the Core ML Sampler).
> `ModelSamplingDiscrete` is required to sample LCM LoRAs correctly.

![Multiple LoRAs](../assets/conversion+lcm_lora.png?raw=true)

## Loading a model with baked LoRAs

Load a model that already has LoRAs baked in. CLIP must be loaded separately and
passed through the same LoRA nodes used at conversion. Since `lora_name` and
`strength_model` are baked in, they need not be passed to the loader.

> [!IMPORTANT]
> As above, the model goes through the Core ML Adapter + `ModelSamplingDiscrete`
> into the standard KSampler.

![Loader + LoRA](../assets/loader+lcm_lora.png?raw=true)

## LCM conversion with ControlNet

Convert a full-distill LCM checkpoint (e.g.
[LCM_Dreamshaper_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7)) with
the standard **Convert Checkpoint to Core ML** node — the LCM architecture is
auto-detected. Use it with or without ControlNet. When sampling, set
`sampler_name` to `lcm` and `scheduler` to `sgm_uniform`.

![LCM + ControlNet](../assets/lcm+controlnet.png?raw=true)

## SDXL Base + Refiner

A basic SDXL graph. Add LoRAs and ControlNets as in the SD1.5 examples; the
refiner step is optional.

Models:
[base + text encoders](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0),
[refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0),
[VAE](https://huggingface.co/stabilityai/sdxl-vae).

> [!IMPORTANT]
> SDXL does not run on the ANE. Convert with `ORIGINAL` and load with
> `CPU_AND_GPU` (or `CPU_ONLY`). If loading hangs on `CPU_AND_NE`, that is the
> cause. See [limitations](limitations.md).

![SDXL](../assets/sdxl_conversion.png?raw=true)
