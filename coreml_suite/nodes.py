import os

import torch
from coremltools import ComputeUnit
from python_coreml_stable_diffusion.coreml_model import CoreMLModel

import folder_paths
from comfy import model_base
from comfy.model_management import get_torch_device
from comfy.model_patcher import ModelPatcher
from coreml_suite.lcm.nodes import add_lcm_model_options, lcm_patch
from coreml_suite.logger import logger
from nodes import KSampler

from coreml_suite.models import CoreMLModelWrapper
from coreml_suite.config import get_model_config


class CoreMLSampler(KSampler):
    @classmethod
    def INPUT_TYPES(s):
        old_required = KSampler.INPUT_TYPES()["required"].copy()
        old_required.pop("model")
        old_required.pop("latent_image")
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
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image=None,
        denoise=1.0,
    ):
        model_patcher = self.get_model_patcher(coreml_model)
        latent_image = self.get_latent_image(coreml_model, latent_image)

        if is_lcm(coreml_model):
            positive[0][1]["control_apply_to_uncond"] = False
            model_patcher = add_lcm_model_options(model_patcher, cfg, latent_image)
            model_patcher = lcm_patch(model_patcher)

        return super().sample(
            model_patcher,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise,
        )

    def get_latent_image(self, coreml_model, latent_image):
        if latent_image is not None:
            return latent_image

        logger.warning("No latent image provided, using empty tensor.")
        expected = coreml_model.expected_inputs["sample"]["shape"]
        batch_size = max(expected[0] // 2, 1)
        latent_image = {"samples": torch.zeros(batch_size, *expected[1:])}
        return latent_image

    def get_model_patcher(self, coreml_model):
        model_config = get_model_config()
        wrapped_model = CoreMLModelWrapper(coreml_model)
        model = model_base.BaseModel(model_config, device=get_torch_device())
        model.diffusion_model = wrapped_model
        model_patcher = ModelPatcher(model, get_torch_device(), None)
        return model_patcher


class CoreMLLoader:
    PACKAGE_DIRNAME = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "coreml_name": (list(s.coreml_filenames().keys()),),
                "compute_unit": (
                    [
                        ComputeUnit.CPU_AND_NE.name,
                        ComputeUnit.CPU_AND_GPU.name,
                        ComputeUnit.ALL.name,
                        ComputeUnit.CPU_ONLY.name,
                    ],
                ),
            }
        }

    FUNCTION = "load"
    CATEGORY = "Core ML Suite"

    @classmethod
    def coreml_filenames(cls):
        extensions = (".mlmodelc", ".mlpackage")
        all_paths = folder_paths.get_filename_list_(cls.PACKAGE_DIRNAME)[1]
        coreml_paths = folder_paths.filter_files_extensions(all_paths, extensions)

        return {os.path.split(p)[-1]: p for p in coreml_paths}

    def load(self, coreml_name, compute_unit):
        logger.info(f"Loading {coreml_name} to {compute_unit}")

        coreml_path = self.coreml_filenames()[coreml_name]

        sources = "compiled" if coreml_name.endswith(".mlmodelc") else "packages"

        return (CoreMLModel(coreml_path, compute_unit, sources),)


class CoreMLLoaderUNet(CoreMLLoader):
    PACKAGE_DIRNAME = "unet"
    RETURN_TYPES = ("COREML_UNET",)
    RETURN_NAMES = ("coreml_model",)


class CoreMLModelAdapter:
    """
    Adapter Node to use CoreML models as Comfy models. This is an experimental
    feature and may not work as expected.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "coreml_model": ("COREML_UNET",),
            }
        }

    RETURN_TYPES = ("MODEL",)

    FUNCTION = "wrap"
    CATEGORY = "Core ML Suite"

    def wrap(self, coreml_model):
        model_config = get_model_config()
        wrapped_model = CoreMLModelWrapper(coreml_model)
        model = model_base.BaseModel(model_config, device=get_torch_device())
        model.diffusion_model = wrapped_model
        model_patcher = ModelPatcher(model, get_torch_device(), None)
        return (model_patcher,)


def is_lcm(coreml_model):
    return "timestep_cond" in coreml_model.expected_inputs
