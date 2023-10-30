import os

from coremltools import ComputeUnit
from python_coreml_stable_diffusion.coreml_model import CoreMLModel

import folder_paths
from comfy.model_management import get_torch_device
from comfy.model_patcher import ModelPatcher
from coreml_suite.logger import logger
from coreml_suite.latents import reshape_latent_image
from nodes import KSampler

from coreml_suite.models import CoreMLModelWrapper, get_model_config


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
        sample_shape = coreml_model.expected_inputs["sample"]["shape"]
        latent_image = reshape_latent_image(latent_image, sample_shape)
        latent_image["samples"] = latent_image["samples"][0:1]

        model_config = get_model_config()
        wrapped_model = CoreMLModelWrapper(model_config, coreml_model)
        model = ModelPatcher(wrapped_model, get_torch_device(), None)

        return super().sample(
            model,
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

        return self._load(coreml_path, compute_unit, sources)

    def _load(self, coreml_path, compute_unit, sources):
        return (CoreMLModel(coreml_path, compute_unit, sources),)


class CoreMLLoaderUNet(CoreMLLoader):
    PACKAGE_DIRNAME = "unet"
    RETURN_TYPES = ("COREML_UNET",)
    RETURN_NAMES = ("coreml_model",)
