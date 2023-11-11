import os

import torch
from coremltools import ComputeUnit
from python_coreml_stable_diffusion.coreml_model import CoreMLModel

from comfy.model_management import get_torch_device
from comfy_extras.nodes_model_advanced import ModelSamplingDiscreteLCM, LCM
from coreml_suite.lcm import converter as lcm_converter


class COREML_CONVERT_LCM:
    """Converts a LCM model to Core ML."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("INT", {"default": 512, "min": 512, "max": 768, "step": 8}),
                "width": ("INT", {"default": 512, "min": 512, "max": 768, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "compute_unit": (
                    [
                        ComputeUnit.CPU_AND_NE.name,
                        ComputeUnit.CPU_AND_GPU.name,
                        ComputeUnit.ALL.name,
                        ComputeUnit.CPU_ONLY.name,
                    ],
                ),
                "controlnet_support": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("COREML_UNET",)
    RETURN_NAMES = ("coreml_model",)
    FUNCTION = "convert"

    def convert(self, height, width, batch_size, compute_unit, controlnet_support):
        """Converts a LCM model to Core ML.

        Args:
            height (int): Height of the target image.
            width (int): Width of the target image.
            batch_size (int): Batch size.
            compute_unit (str): Compute unit to use when loading the model.

        Returns:
            coreml_model: The converted Core ML model.

        The converted model is also saved to "models/unet" directory and
        can be loaded with the "LCMCoreMLLoaderUNet" node.
        """
        h = height
        w = width
        sample_size = (h // 8, w // 8)
        batch_size = batch_size
        cn_support_str = "_cn" if controlnet_support else ""

        out_name = f"{lcm_converter.MODEL_NAME}_{batch_size}x{w}x{h}{cn_support_str}"

        out_path = lcm_converter.get_out_path("unet", f"{out_name}")

        if not os.path.exists(out_path):
            lcm_converter.convert(
                out_path=out_path,
                sample_size=sample_size,
                batch_size=batch_size,
                controlnet_support=controlnet_support,
            )
        target_path = lcm_converter.compile_model(out_path=out_path, out_name=out_name)

        return (CoreMLModel(target_path, compute_unit, "compiled"),)


def get_w_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    see https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
    Args:
    timesteps: torch.Tensor: generate embedding vectors at these timesteps
    embedding_dim: int: dimension of the embeddings to generate
    dtype: data type of the generated embeddings

    Returns:
    embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


def model_function_wrapper(w_embedding):
    def wrapper(model_function, params):
        x = params["input"]
        t = params["timestep"]
        c = params["c"]

        context = c.get("c_crossattn")

        if context is None:
            return torch.zeros_like(x)

        return model_function(x, t, **c, timestep_cond=w_embedding)

    return wrapper


def lcm_patch(model):
    m = model.clone()
    sampling_type = LCM
    sampling_base = ModelSamplingDiscreteLCM

    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass

    model_sampling = ModelSamplingAdvanced()
    m.add_object_patch("model_sampling", model_sampling)

    return m


def add_lcm_model_options(model_patcher, cfg, latent_image):
    mp = model_patcher.clone()

    latent = latent_image["samples"].to(get_torch_device())
    batch_size = latent.shape[0]
    dtype = latent.dtype
    device = get_torch_device()

    w = torch.tensor(cfg).repeat(batch_size)
    w_embedding = get_w_embedding(w, embedding_dim=256).to(device=device, dtype=dtype)

    model_options = {
        "model_function_wrapper": model_function_wrapper(w_embedding),
        "sampler_cfg_function": lambda x: x["cond"].to(device),
    }
    mp.model_options |= model_options

    return mp
