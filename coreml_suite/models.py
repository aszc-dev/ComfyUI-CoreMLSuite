"""Framework-coupled glue between Core ML UNets and ComfyUI's sampler stack.

Pure math (CoreMLInputs, SDXL detection, time_ids/text_embeds assembly,
sdxl_model_function_wrapper) lives in coreml_suite.core.*.
This module is what touches comfy.*: model_base, ModelPatcher, the
diffusion_model wrapper, and the maintainer-facing add_sdxl_model_options
adapter.
"""
import torch

from comfy import model_base
from comfy.model_management import get_torch_device
from comfy.model_patcher import ModelPatcher

from coreml_suite.config import get_model_config, ModelVersion
from coreml_suite.core.inputs import CoreMLInputs
from coreml_suite.core.latents import merge_chunks
from coreml_suite.core.sdxl import (
    build_sdxl_text_embeds,
    build_sdxl_time_ids,
    is_sdxl,
    is_sdxl_base,
    is_sdxl_refiner,
    sdxl_model_function_wrapper,
)
from coreml_suite.lcm.utils import is_lcm
from coreml_suite.logger import logger

__all__ = [
    "CoreMLInputs",
    "CoreMLModelWrapper",
    "CoreMLModelWrapperLCM",
    "add_sdxl_model_options",
    "get_latent_image",
    "get_model_patcher",
    "is_sdxl",
    "is_sdxl_base",
    "is_sdxl_refiner",
    "sdxl_model_function_wrapper",
]


class CoreMLModelWrapper:
    def __init__(self, coreml_model):
        self.coreml_model = coreml_model
        self.dtype = torch.float16

    def __call__(self, x, t, context, control, transformer_options=None, **kwargs):
        inputs = CoreMLInputs(x, t, context, control, **kwargs)
        input_list = inputs.chunks(self.expected_inputs)

        chunked_out = [
            self.get_torch_outputs(
                self.coreml_model(**input_kwargs.coreml_kwargs(self.expected_inputs)),
                x.device,
            )
            for input_kwargs in input_list
        ]
        merged_out = merge_chunks(chunked_out, x.shape)

        return merged_out

    @staticmethod
    def get_torch_outputs(model_output, device):
        return torch.from_numpy(model_output["noise_pred"]).to(device)

    @property
    def expected_inputs(self):
        return self.coreml_model.expected_inputs

    @property
    def is_lcm(self):
        return is_lcm(self.coreml_model)

    @property
    def is_sdxl_base(self):
        return is_sdxl_base(self.coreml_model)

    @property
    def is_sdxl_refiner(self):
        return is_sdxl_refiner(self.coreml_model)

    @property
    def config(self):
        if self.is_sdxl_base:
            return get_model_config(ModelVersion.SDXL)

        if self.is_sdxl_refiner:
            return get_model_config(ModelVersion.SDXL_REFINER)

        return get_model_config(ModelVersion.SD15)


class CoreMLModelWrapperLCM(CoreMLModelWrapper):
    def __init__(self, coreml_model):
        super().__init__(coreml_model)
        self.config = None


def add_sdxl_model_options(model_patcher, positive, negative):
    mp = model_patcher.clone()

    pos_dict = positive[0][1]
    neg_dict = negative[0][1]

    is_base = model_patcher.model.diffusion_model.is_sdxl_base
    is_refiner = model_patcher.model.diffusion_model.is_sdxl_refiner

    time_ids = build_sdxl_time_ids(
        pos_dict, neg_dict, is_base=is_base, is_refiner=is_refiner
    )
    text_embeds = build_sdxl_text_embeds(
        pos_dict["pooled_output"], neg_dict["pooled_output"]
    )

    mp.model_options |= {
        "model_function_wrapper": sdxl_model_function_wrapper(
            time_ids, text_embeds, is_refiner
        ),
    }
    return mp


def get_latent_image(coreml_model, latent_image):
    if latent_image is not None:
        return latent_image

    logger.warning("No latent image provided, using empty tensor.")
    expected = coreml_model.expected_inputs["sample"]["shape"]
    batch_size = max(expected[0] // 2, 1)
    latent_image = {"samples": torch.zeros(batch_size, *expected[1:])}
    return latent_image


def get_model_patcher(coreml_model):
    wrapped_model = CoreMLModelWrapper(coreml_model)

    if wrapped_model.is_sdxl_base:
        model = model_base.SDXL(wrapped_model.config, device=get_torch_device())
    elif wrapped_model.is_sdxl_refiner:
        model = model_base.SDXLRefiner(wrapped_model.config, device=get_torch_device())
    else:
        model = model_base.BaseModel(wrapped_model.config, device=get_torch_device())

    model.diffusion_model = wrapped_model
    model_patcher = ModelPatcher(model, get_torch_device(), None)
    return model_patcher
