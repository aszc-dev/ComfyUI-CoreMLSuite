import os

import torch
from coremltools import ComputeUnit
from python_coreml_stable_diffusion.coreml_model import CoreMLModel

import folder_paths
from comfy import model_base
from comfy.model_management import get_torch_device
from comfy.model_patcher import ModelPatcher
from coreml_suite import COREML_NODE
from coreml_suite import converter
from coreml_suite.lcm.utils import add_lcm_model_options, lcm_patch, is_lcm
from coreml_suite.logger import logger
from coreml_suite.lora import load_lora
from nodes import KSampler

from coreml_suite.models import CoreMLModelWrapper
from coreml_suite.config import get_model_config


class CoreMLSampler(COREML_NODE, KSampler):
    @classmethod
    def INPUT_TYPES(s):
        old_required = KSampler.INPUT_TYPES()["required"].copy()
        old_required.pop("model")
        old_required.pop("negative")
        old_required.pop("latent_image")
        new_required = {"coreml_model": ("COREML_UNET",)}
        return {
            "required": new_required | old_required,
            "optional": {"negative": ("CONDITIONING",), "latent_image": ("LATENT",)},
        }

    def sample(
        self,
        coreml_model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative=None,
        latent_image=None,
        denoise=1.0,
    ):
        model_patcher = self.get_model_patcher(coreml_model)
        latent_image = self.get_latent_image(coreml_model, latent_image)

        if is_lcm(coreml_model):
            negative = [[None, {}]]
            positive[0][1]["control_apply_to_uncond"] = False
            model_patcher = add_lcm_model_options(model_patcher, cfg, latent_image)
            model_patcher = lcm_patch(model_patcher)
        else:
            assert (
                negative is not None
            ), "Negative conditioning is optional only for LCM models."

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


class CoreMLLoader(COREML_NODE):
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


class CoreMLModelAdapter(COREML_NODE):
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


class COREML_CONVERT(COREML_NODE):
    """Converts a LCM model to Core ML."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "height": ("INT", {"default": 512, "min": 512, "max": 2048, "step": 8}),
                "width": ("INT", {"default": 512, "min": 512, "max": 2048, "step": 8}),
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
            },
            "optional": {
                "lora_stack": ("LORA_STACK",),
            },
        }

    RETURN_TYPES = ("COREML_UNET", "CLIP")
    RETURN_NAMES = ("coreml_model", "CLIP")
    FUNCTION = "convert"

    def convert(
        self,
        ckpt_name,
        height,
        width,
        batch_size,
        compute_unit,
        controlnet_support,
        lora_stack=None,
    ):
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
        lora_str = (
            "_" + "_".join(lora_param[0].split(".")[0] for lora_param in lora_stack)
            if lora_stack
            else ""
        )

        out_name = (
            f"{ckpt_name.split('.')[0]}{lora_str}_{batch_size}x{w}x{h}{cn_support_str}"
        )

        unet_out_path = converter.get_out_path("unet", f"{out_name}")
        unet_out_path = unet_out_path.replace(" ", "_")

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        lora_stack = lora_stack or []
        lora_paths = [
            folder_paths.get_full_path("loras", lora[0]) for lora in lora_stack
        ]

        converter.convert(
            ckpt_path=ckpt_path,
            unet_out_path=unet_out_path,
            sample_size=sample_size,
            batch_size=batch_size,
            controlnet_support=controlnet_support,
            lora_paths=lora_paths,
        )
        unet_target_path = converter.compile_model(
            out_path=unet_out_path, out_name=out_name, submodule_name="unet"
        )

        clip = load_lora(lora_stack, ckpt_name)

        return CoreMLModel(unet_target_path, compute_unit, "compiled"), clip
