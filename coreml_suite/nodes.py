import os

import torch
from coremltools import ComputeUnit
from python_coreml_stable_diffusion.coreml_model import CoreMLModel
from python_coreml_stable_diffusion.unet import AttentionImplementations

import folder_paths
from comfy import model_base
from comfy.model_management import get_torch_device
from comfy.model_patcher import ModelPatcher
from coreml_suite import COREML_NODE
from coreml_suite import converter
from coreml_suite.lcm.utils import add_lcm_model_options, lcm_patch, is_lcm
from coreml_suite.logger import logger
from nodes import KSampler, LoraLoader, KSamplerAdvanced

from coreml_suite.models import (
    CoreMLModelWrapper,
    add_sdxl_model_options,
    is_sdxl,
    get_model_patcher,
    get_latent_image,
)
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
        model_patcher = get_model_patcher(coreml_model)
        latent_image = get_latent_image(coreml_model, latent_image)

        if is_lcm(coreml_model):
            negative = [[None, {}]]
            positive[0][1]["control_apply_to_uncond"] = False
            model_patcher = add_lcm_model_options(model_patcher, cfg, latent_image)
            model_patcher = lcm_patch(model_patcher)
        else:
            assert (
                negative is not None
            ), "Negative conditioning is optional only for LCM models."

        if is_sdxl(coreml_model):
            model_patcher = add_sdxl_model_options(model_patcher, positive, negative)

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


class CoreMLSamplerAdvanced(COREML_NODE, KSamplerAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        old_required = KSamplerAdvanced.INPUT_TYPES()["required"].copy()
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
        add_noise,
        noise_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        negative=None,
        latent_image=None,
        denoise=1.0,
    ):
        model_patcher = get_model_patcher(coreml_model)
        latent_image = get_latent_image(coreml_model, latent_image)

        if is_lcm(coreml_model):
            negative = [[None, {}]]
            positive[0][1]["control_apply_to_uncond"] = False
            model_patcher = add_lcm_model_options(model_patcher, cfg, latent_image)
            model_patcher = lcm_patch(model_patcher)
        else:
            assert (
                negative is not None
            ), "Negative conditioning is optional only for LCM models."

        if is_sdxl(coreml_model):
            model_patcher = add_sdxl_model_options(model_patcher, positive, negative)

        return super().sample(
            model_patcher,
            add_noise,
            noise_seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            start_at_step,
            end_at_step,
            return_with_leftover_noise,
            denoise,
        )


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
                "attention_implementation": (
                    [
                        AttentionImplementations.SPLIT_EINSUM.name,
                        AttentionImplementations.SPLIT_EINSUM_V2.name,
                        AttentionImplementations.ORIGINAL.name,
                    ],
                ),
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
                "lora_params": ("LORA_PARAMS",),
            },
        }

    RETURN_TYPES = ("COREML_UNET",)
    RETURN_NAMES = ("coreml_model",)
    FUNCTION = "convert"

    def convert(
        self,
        ckpt_name,
        height,
        width,
        batch_size,
        attention_implementation,
        compute_unit,
        controlnet_support,
        lora_params=None,
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
        lora_params = lora_params or {}
        lora_params = [(k, v[0]) for k, v in lora_params.items()]
        lora_params = sorted(lora_params, key=lambda lora: lora[0])
        lora_weights = [(self.lora_path(lora[0]), lora[1]) for lora in lora_params]

        h = height
        w = width
        sample_size = (h // 8, w // 8)
        batch_size = batch_size
        cn_support_str = "_cn" if controlnet_support else ""
        lora_str = (
            "_" + "_".join(lora_param[0].split(".")[0] for lora_param in lora_params)
            if lora_params
            else ""
        )

        attn_str = (
            "_"
            + {"SPLIT_EINSUM": "se", "SPLIT_EINSUM_V2": "se2", "ORIGINAL": "orig"}[
                attention_implementation
            ]
        )

        out_name = f"{ckpt_name.split('.')[0]}{lora_str}_{batch_size}x{w}x{h}{cn_support_str}{attn_str}"
        out_name = out_name.replace(" ", "_")

        logger.info(f"Converting {ckpt_name} to {out_name}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Width: {w}, Height: {h}")
        logger.info(f"ControlNet support: {controlnet_support}")
        logger.info(f"Attention implementation: {attention_implementation}")

        if lora_params:
            logger.info(f"LoRAs used:")
            for lora_param in lora_params:
                logger.info(f"  {lora_param[0]} - strength: {lora_param[1]}")

        unet_out_path = converter.get_out_path("unet", f"{out_name}")
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        config_filename = ckpt_name.split(".")[0] + ".yaml"
        config_path = folder_paths.get_full_path("configs", config_filename)
        if config_path:
            logger.info(f"Using config file {config_path}")

        converter.convert(
            ckpt_path=ckpt_path,
            unet_out_path=unet_out_path,
            sample_size=sample_size,
            batch_size=batch_size,
            controlnet_support=controlnet_support,
            lora_weights=lora_weights,
            attn_impl=attention_implementation,
            config_path=config_path,
        )
        unet_target_path = converter.compile_model(
            out_path=unet_out_path, out_name=out_name, submodule_name="unet"
        )

        return (CoreMLModel(unet_target_path, compute_unit, "compiled"),)

    @staticmethod
    def lora_path(lora_name):
        return folder_paths.get_full_path("loras", lora_name)


class COREML_LOAD_LORA(COREML_NODE, LoraLoader):
    @classmethod
    def INPUT_TYPES(s):
        required = LoraLoader.INPUT_TYPES()["required"].copy()
        required.pop("model")
        return {
            "required": required,
            "optional": {"lora_params": ("LORA_PARAMS",)},
        }

    RETURN_TYPES = ("CLIP", "LORA_PARAMS")
    RETURN_NAMES = ("CLIP", "lora_params")

    def load_lora(
        self, clip, lora_name, strength_model, strength_clip, lora_params=None
    ):
        _, lora_clip = super().load_lora(
            None, clip, lora_name, strength_model, strength_clip
        )

        lora_params = lora_params or {}
        lora_params[lora_name] = (strength_model, strength_clip)

        return lora_clip, lora_params
