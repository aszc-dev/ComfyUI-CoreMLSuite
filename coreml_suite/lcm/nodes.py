import os

import torch
from coremltools import ComputeUnit
from diffusers import LCMScheduler
from python_coreml_stable_diffusion.coreml_model import CoreMLModel

import comfy.utils
import latent_preview
from comfy import model_base
from comfy.model_management import get_torch_device
from comfy.model_patcher import ModelPatcher
from coreml_suite.config import get_model_config
from coreml_suite.lcm import converter as lcm_converter
from coreml_suite.lcm.sampler import CoreMLSamplerLCM
from coreml_suite.logger import logger
from coreml_suite.models import CoreMLModelWrapper
from coreml_suite.nodes import CoreMLSampler


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


class COREML_SAMPLER_LCM(CoreMLSampler):
    @classmethod
    def INPUT_TYPES(s):
        old_required = CoreMLSampler.INPUT_TYPES()["required"].copy()
        old_required["steps"][1]["default"] = 4
        old_required.pop("negative")
        old_required.pop("sampler_name")
        old_required.pop("scheduler")
        new_required = {"coreml_model": ("COREML_UNET",)}
        return {
            "required": new_required | old_required,
            "optional": {"latent_image": ("LATENT",)},
        }

    CATEGORY = "Core ML Suite"

    def sample(
        self,
        coreml_model,
        seed,
        steps,
        cfg,
        positive,
        latent_image=None,
        denoise=1.0,
        **kwargs,
    ):
        scheduler = LCMScheduler.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7", subfolder="scheduler"
        )

        model_config = get_model_config()
        wrapped_model = CoreMLModelWrapper(coreml_model)
        model = model_base.BaseModel(model_config, device=get_torch_device())
        model.diffusion_model = wrapped_model
        model_patcher = ModelPatcher(model, get_torch_device(), None)

        if latent_image is None:
            logger.warning("No latent image provided, using empty tensor.")
            expected = coreml_model.expected_inputs["sample"]["shape"]
            latent_image = {"samples": torch.zeros(*expected).to(get_torch_device())}

        x0_output = {}
        callback = latent_preview.prepare_callback(model_patcher, steps, x0_output)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        sampler = CoreMLSamplerLCM(scheduler)
        samples = sampler.sample(
            model_patcher,
            seed,
            steps,
            cfg,
            positive,
            latent_image=latent_image,
            denoise=denoise,
            callback=callback,
            disable_pbar=disable_pbar,
        )

        out = latent_image.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent_image.copy()
            out_denoised["samples"] = model.process_latent_out(
                x0_output["x0"].to(get_torch_device())
            )
        else:
            out_denoised = out
        return out, out_denoised
