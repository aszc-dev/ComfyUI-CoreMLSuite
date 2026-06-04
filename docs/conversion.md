# Conversion

## Conversion is the only supported path

You always start from a Stable Diffusion checkpoint (`.safetensors` / `.ckpt`)
and convert it with the **Convert Checkpoint to Core ML** (or **Convert LCM**)
node. Pre-converted Core ML models from elsewhere are not supported, because:

- The suite uses its own input **dimensions**, **naming convention**, and
  **metadata**, all produced by the
  [coreml-diffusion](https://github.com/aszc-dev/coreml-diffusion) package.
- Apple's `ml-stable-diffusion` (which most community Core ML models target) is
  effectively obsolete, and the layouts differ (see the 2.0.0
  `encoder_hidden_states` change in the README).
- Conversion is cheap and one-time, so there is no value in maintaining
  backwards compatibility with foreign formats.

The output is always a **`.mlpackage`**. The suite no longer compiles to
`.mlmodelc` (it didn't work with the inference backend), so there is **no Xcode
or `coremlcompiler` dependency**.

## One-time conversion and name-based caching

Conversion runs **once**, not on every queue. The converter encodes all
conversion parameters into the output filename (via `coreml_diffusion.compose_out_name`,
called in `coreml_suite/nodes.py:313`):

- checkpoint name, `batch_size`, `width`, `height`
- `controlnet_support`, `attention_implementation`
- baked LoRA names, `quantize_nbits`

The result is written as `<encoded-name>_unet.mlpackage` in `models/unet`. If a
file with that name already exists, it is reused and conversion is skipped. Change
any parameter → new name → new conversion; keep them the same → the cached model
is loaded instantly.

This is why the recommended workflow is **convert once, then load**: run the
converter a single time, then in day-to-day use load the `.mlpackage` with the
**Load Core ML UNet** node. (You can also leave the converter node in the graph;
it short-circuits to the cached file.)

> [!NOTE]
> The converter relies on the filename to decide whether to reconvert. If you
> rename the `.mlpackage`, it will be converted again. You can otherwise rename
> it freely if the auto-generated name is too long.

## Quantization

Both converter nodes accept an optional `quantize_nbits` dropdown that runs
k-means weight palettization (`coremltools.optimize.coreml.palettize_weights`) on
the UNet before saving.

Values: `none` (default — no quantization, identical output and filename to
before), `8`, `6`, `4`. The number is appended to the `.mlpackage` stem as
`_q<bits>`, so quantized and unquantized variants coexist on disk and in cache.

### SD1.5 1×512×512 SPLIT_EINSUM tradeoffs (M2 Pro, ANE)

Measured with 20 UNet forward passes at a fixed seed:

| nbits | size (MB) | size vs none | fwd median (ms) | PSNR vs `none` (dB) |
|---|---:|---:|---:|---:|
| none | 1641 | 1.000 | 197.1 | — |
| 8    |  822 | 0.501 | 186.6 | 53.5 |
| 6    |  617 | 0.376 | 183.0 | 40.2 |
| 4    |  412 | 0.251 | 179.8 | 27.5 |

PSNR here is computed on the raw `noise_pred` output of a single UNet forward at a
fixed seed, not on the final decoded image — it isolates quantization drift from
sampler/VAE noise. Final-image PSNR is comfortably higher (the sampler averages
over many steps).

### Recommended settings per chip / RAM

- **8 GB (M1/M2 base):** `nbits=4`. ~4× smaller, still loads, 27 dB is visually
  identical at SD1.5 sizes.
- **16 GB (M1/M2/M3 Pro):** `nbits=6` — the sweet spot, ~2.7× smaller, 40 dB, no
  perceptible quality drop.
- **32 GB+ (Max / Ultra):** `nbits=8` for a safety margin, or `none` for
  bit-identical output (golden testing).

The default stays `none`, so existing workflows produce byte-for-byte identical
output.

## Where conversion lives

The conversion engine was extracted into the standalone
[coreml-diffusion](https://github.com/aszc-dev/coreml-diffusion) PyPI package.
The nodes in this suite resolve ComfyUI paths and call into it; node names,
inputs, and outputs are unchanged, so the split has effectively no user-facing
impact beyond `pip install` pulling one more dependency.

One detail: the LCM converter still imports `diffusers` directly (in
`coreml_suite/lcm/converter.py`) to download the hardcoded LCM model from Hugging
Face. This is an internal note, not something you need to act on.
