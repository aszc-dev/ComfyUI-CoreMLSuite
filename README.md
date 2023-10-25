# Core ML Suite for ComfyUI

## Overview

This repository contains a set of custom nodes for ComfyUI that allow you to use Core ML models in your ComfyUI
workflows. The models can be obtained [here](https://huggingface.co/coreml-community), or you can
convert your own models using [coremltools](https://github.com/apple/ml-stable-diffusion).

The main motivation behind using Core ML models in ComfyUI is to allow you to utilize the ANE (Apple Neural Engine)
on Apple Silicon (M1/M2) machines to improve performance.

While testing on M2 Pro 32GB, the ANE (`CPU_AND_NE` option) was able to speed up the inference by a factor
of ~1.5-2x.

### Features

- Loading Core ML Unet models
- Support for ControlNet
- Support for ANE (Apple Neural Engine)
- Support for CPU and GPU
- Support for `mlmodelc` and `mlpackage` files

> [!NOTE]
> The main downside of using Core ML models is the initial compilation/loading time. For best results, please use the
> compiled models (`.mlmodelc` files) instead of the `.mlpackage` files.

> [!NOTE]  
> This repository is a work in progress and will be updated with more nodes and features in the future.

## Installation

To install the custom nodes, you can clone this repository ComfyUI into the `custom_nodes` directory of your ComfyUI.
Alternatively, you can download the repository as a zip file and extract it into the `custom_nodes` directory.

```bash
cd /path/to/comfyui/custom_nodes
git clone https://github.com/aszc-dev/ComfyUI-CoreMLSuite.git
```

Then, use pip or other package manager to install the dependencies:

```bash
cd /path/to/comfyui/custom_nodes/ComfyUI-CoreMLSuite
pip install -r requirements.txt
```

## Usage

### Available Nodes

#### CoreML UNet Loader (`CoreMLUnetLoader`)

![CoreMLUnetLoader](https://github.com/aszc-dev/ComfyUI-CoreMLSuite/assets/24932801/2bd10f73-4103-4860-894c-b6a6e56c6546)

This node allows you to load a Core ML UNet model and use it in your ComfyUI workflow. Place the converted
.mlpackage or .mlmodelc file in ComfyUI's models/unet directory and use the node to load the model. The output of the
node is a `MODEL` object similar to standard ComfyUI models.  
Additionally, you can select the Compute Unit, that will be used to run the model. The default is `CPU_AND_NE`, which
gives the best results. You may, however, want to use `CPU_AND_GPU`, `ALL` or `CPU_ONLY` for experimentation.
> [!NOTE]  
> To enable ControlNet in your workflow you need to use a Core ML model specifically converted for ControlNet.
> - If you use an unsupported model, the ControlNet input will be ignored.
> - If you use a model that supports ControlNet, but do not provide a ControlNet input (this includes setting
    start_percent and end_percent to values other than 0 and 1 respectively), the model will use random noise
    as ControlNet input.

## Limitations

- Due to the nature of Core ML models, the inputs and outputs of the models are fixed and cannot be changed[^1] once the
  model is converted. This means that the nodes in this repository are not as flexible as the standard ComfyUI nodes.
  You need to use latent images of the same size as the input of the model (512x512 is the default for SD1.5). You can
  also convert the model to a different input size using tools available in
  [apple/ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion) repository.
- For now, only Stable Diffusion v1.5 is supported.
- LoRA is not supported yet.

[^1]: Unless [EnumeratedShapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html#select-from-predetermined-shapes)
is used during conversion. Needs more testing.

## Support

Feel free to open an issue if you have any questions or suggestions.
