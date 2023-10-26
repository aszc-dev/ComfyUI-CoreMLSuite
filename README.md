# Core ML Suite for ComfyUI

## Overview

Welcome! I've developed a set of custom nodes for ComfyUI that allows you to use Core ML models in your ComfyUI
workflows.
These models are designed to leverage the Apple Neural Engine (ANE) on Apple Silicon (M1/M2) machines, 
thereby enhancing your workflows and improving performance.

If you're not sure how to obtain these models, you can download them
[here](https://huggingface.co/coreml-community) or convert your own models using
[coremltools](https://github.com/apple/ml-stable-diffusion).

In simple terms, think of Core ML models as a tool that can help your ComfyUI work faster and more efficiently.
For instance, during my tests on an M2 Pro 32GB machine,
the use of Core ML models sped up the generation of 512x512 images by a factor
of approximately 1.5 to 2 times.

## Getting Started

To start using custom nodes in your ComfyUI, follow these simple steps:

1. Clone or download this repository: You can do this directly into the custom_nodes directory of your ComfyUI.
2. Install the dependencies: You'll need to use a package manager like pip to do this.

That's it! You're now ready to start enhancing your ComfyUI workflows with Core ML models.
- Check [Installation](#installation) for more details on installation.
- Check [How to use](#how-to-use) for more details on how to use the custom nodes.
- Check [Example Workflows](#example-workflows) for some example workflows.


## Glossary

- **Core ML**: A machine learning framework developed by Apple. It's used to run machine learning models on Apple
  devices.
- **Core ML Model**: A machine learning model that can be run on Apple devices using Core ML.
- **mlmodelc**: A compiled Core ML model. This is the recommended format for Core ML models.
- **mlpackage**: A Core ML model packaged in a directory. This is the default format for Core ML models.
- **ANE**: Apple Neural Engine. A hardware accelerator for machine learning tasks on Apple devices.
- **Compute Unit**: A Core ML option that allows you to specify the hardware on which the model should run.
  - **CPU_AND_ANE**: A Core ML compute unit option that allows the model to run on both the CPU and ANE. This is the
    default option.
  - **CPU_AND_GPU**: A Core ML compute unit option that allows the model to run on both the CPU and GPU.
  - **CPU_ONLY**: A Core ML compute unit option that allows the model to run on the CPU only.
  - **ALL**: A Core ML compute unit option that allows the model to run on all available hardware.
- **CLIP**: Contrastive Language-Image Pre-training. A model that learns visual concepts from natural language 
  supervision. It's used as a text encoder in Stable Diffusion.
- **VAE**: Variational Autoencoder. A model that learns a latent representation of images. It's used as a prior in 
  Stable Diffusion.
- **Checkpoint**: A file that contains the weights of a model. It's used to load models in Stable Diffusion.

> [!NOTE]
> Note on Compute Units:
> For the model to run on the ANE, the model must be converted with the `--cross-attention SPLIT_EINSUM` option.
> Models converted with `--cross-attention ORIGINAL` will run on GPU instead of ANE.

## Features
These custom nodes come with a host of features, including:
- Loading Core ML Unet models
- Support for ControlNet
- Support for ANE (Apple Neural Engine)
- Support for CPU and GPU
- Support for `mlmodelc` and `mlpackage` files

> [!NOTE]
> Please note that using Core ML models can take a bit longer to load initially. 
> For the best experience, I recommend using the compiled models 
> (.mlmodelc files) instead of the .mlpackage files.

> [!NOTE]  
> This repository will continue to be updated with more nodes and features over time.

## Installation

The installation process is simple!

1. Clone this repository into the custom_nodes directory of your ComfyUI. If you're not sure how to do this, you can
   download the repository as a zip file and extract it into the same directory.
    ```bash
    cd /path/to/comfyui/custom_nodes
    git clone https://github.com/aszc-dev/ComfyUI-CoreMLSuite.git
    ```
2. Next, install the required dependencies using pip or another package manager:

    ```bash
    cd /path/to/comfyui/custom_nodes/ComfyUI-CoreMLSuite
    pip install -r requirements.txt
    ```

## How to use

Once you've installed the custom nodes, you can start using them in your ComfyUI workflows.
To do this, you need to add the nodes to your workflow. You can do this by double-clicking on the workflow canvas and
selecting the nodes from the list of available nodes. You can also use the search bar to find the nodes.
The list of available nodes is given below. You can also find the nodes in the `CoreML Suite` category.

### Available Nodes

#### CoreML UNet Loader (`CoreMLUnetLoader`)

![CoreMLUnetLoader](https://github.com/aszc-dev/ComfyUI-CoreMLSuite/assets/24932801/2bd10f73-4103-4860-894c-b6a6e56c6546)

This node allows you to load a Core ML UNet model and use it in your ComfyUI workflow. Place the converted
.mlpackage or .mlmodelc file in ComfyUI's `models/unet` directory and use the node to load the model. The output of the
node is a `MODEL` object similar to standard ComfyUI models.
> [!NOTE]  
> Some models are designed to support ControlNet. If you're using such a model,
> make sure to provide a ControlNet input; otherwise, the model will use random noise as ControlNet input.

### Example Workflows

> [!NOTE]
> The models used are just an example. Feel free to experiment with different models and see what works best for you.


#### Basic txt2img with Core ML UNet loader
This is a basic txt2img workflow that uses the Core ML UNet loader to load a Core ML UNet model. The CLIP and VAE models
are loaded using the standard ComfyUI nodes. In the first example, the text encoder (CLIP) and VAE models are loaded 
separately. In the second example, the text encoder and VAE models are loaded from the checkpoint file. Note that you can use any CLIP or VAE model
as long as it's compatible with Stable Diffusion v1.5.

1. **Loading text encoder (CLIP) and VAE models separately**
   - This workflow uses CLIP and VAE models available
   [here](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/text_encoder/model.safetensors) and 
   [here](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/vae/diffusion_pytorch_model.safetensors). 
   Once downloaded, place the models in the`models/clip` and `models/vae` directories respectively. 
   - The Core ML UNet model is available 
   [here](https://huggingface.co/coreml-community/coreml-stable-diffusion-v1-5_cn/blob/main/split_einsum/stable-diffusion-_v1-5_split-einsum_cn.zip).
   Once downloaded, place the model in the `models/unet` directory.
![coreml-unet+clip+vae](https://github.com/aszc-dev/ComfyUI-CoreMLSuite/assets/24932801/ff7b8d75-37ea-4da9-a258-829edd6eb1b7)


2. **Loading text encoder (CLIP) and VAE models from checkpoint file**
   - This workflow loads the CLIP and VAE models from the checkpoint file available
   [here](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors). 
   Once downloaded, place the model in the`models/checkpoints` directory.
   - The Core ML UNet model is available 
   [here](https://huggingface.co/coreml-community/coreml-stable-diffusion-v1-5_cn/blob/main/split_einsum/stable-diffusion-_v1-5_split-einsum_cn.zip). 
   Once downloaded, place the model in the `models/unet` directory.
![coreml-unet+checkpoint](https://github.com/aszc-dev/ComfyUI-CoreMLSuite/assets/24932801/0c8b4f65-9bde-4b0d-936b-5bb27023d2ce)


#### ControlNet with Core ML UNet loader 

(Coming soon)

## Limitations

- Core ML models are fixed in terms of their inputs and outputs. 
This means you'll need to use latent images of the same size as the input of the model (512x512 is the default for SD1.5). 
However, you can convert the model to a different input size using tools available 
in the [apple/ml-stable-diffusion](https://github.com/apple/ml-stable-diffusion) repository.
- For now, only Stable Diffusion v1.5 is supported.
- LoRA is not supported yet.

[^1]: Unless [EnumeratedShapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html#select-from-predetermined-shapes)
is used during conversion. Needs more testing.

## Support

I'm here to help! If you have any questions or suggestions, don't hesitate to open an issue and I'll do my best 
to assist you.
