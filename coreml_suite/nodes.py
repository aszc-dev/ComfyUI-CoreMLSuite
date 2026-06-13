import os

from coremltools import ComputeUnit

import folder_paths
from coreml_suite import COREML_NODE
from coreml_suite.coreml_model import CoreMLModel
from coreml_suite.lcm.utils import add_lcm_model_options, lcm_patch, is_lcm
from coreml_suite.logger import logger
from nodes import KSampler, LoraLoader, KSamplerAdvanced

from coreml_suite.models import (
    add_sdxl_model_options,
    is_sdxl,
    get_model_patcher,
    get_latent_image,
)


def _discover(fn_name, fallback):
    """Populate a converter dropdown from coreml_diffusion's discovery API.

    Fails soft: if the package is missing, too old to expose ``fn_name``, or
    errors, the node still registers with the fallback list instead of vanishing
    from the menu. Evaluated on every INPUT_TYPES call, so installing a newer
    coreml_diffusion surfaces new conversion types with no Suite change.
    """
    try:
        import coreml_diffusion

        return getattr(coreml_diffusion, fn_name)()
    except Exception as exc:  # missing/old package, import error, etc.
        logger.warning(
            f"coreml_diffusion.{fn_name} unavailable ({exc}); "
            f"using fallback {fallback}"
        )
        return fallback


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
        extensions = (".mlpackage",)
        all_paths = folder_paths.get_filename_list_(cls.PACKAGE_DIRNAME)[1]
        coreml_paths = folder_paths.filter_files_extensions(all_paths, extensions)

        return {os.path.split(p)[-1]: p for p in coreml_paths}

    def load(self, coreml_name, compute_unit):
        logger.info(f"Loading {coreml_name} to {compute_unit}")

        coreml_path = self.coreml_filenames()[coreml_name]

        return (CoreMLModel(coreml_path, compute_unit),)


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
        model_patcher = get_model_patcher(coreml_model)
        return (model_patcher,)


class CoreMLConverter(COREML_NODE):
    """Converts a Stable Diffusion checkpoint (UNet) to Core ML.

    The model version (SD15 / SDXL / SDXL refiner / LCM) is auto-detected from
    the checkpoint's architecture, so there is no version dropdown — one node
    converts every supported family, including full-distill LCM.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "height": ("INT", {"default": 512, "min": 8, "step": 8}),
                "width": ("INT", {"default": 512, "min": 8, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "attention_implementation": (
                    _discover(
                        "list_attention_impls",
                        ["SPLIT_EINSUM", "SPLIT_EINSUM_V2", "ORIGINAL"],
                    ),
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
                # k-means weight palettization. Kept optional so workflows
                # that omit it still validate — ComfyUI rejects a prompt that
                # omits any `required` input. When omitted it defaults to
                # "none", identical to unquantized behavior and filename, so
                # existing cached .mlpackages still resolve.
                "quantize_nbits": (
                    _discover("list_quant_modes", ["none", "8", "6", "4"]),
                    {"default": "none"},
                ),
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
        quantize_nbits="none",
        lora_params=None,
    ):
        """Converts a checkpoint's UNet to Core ML.

        Args:
            ckpt_name (str): Checkpoint to convert; its model version is
                auto-detected from the weights.
            height (int): Height of the target image.
            width (int): Width of the target image.
            batch_size (int): Batch size.
            compute_unit (str): Compute unit to use when loading the model.

        Returns:
            coreml_model: The converted Core ML model.

        The converted model is also saved to "models/unet" directory and
        can be loaded with the "Load Core ML UNet" node.
        """
        lora_params = lora_params or {}
        lora_params = [(k, v[0]) for k, v in lora_params.items()]
        lora_params = sorted(lora_params, key=lambda lora: lora[0])
        lora_weights = [(self.lora_path(lora[0]), lora[1]) for lora in lora_params]

        h = height
        w = width
        sample_size = (h // 8, w // 8)
        import coreml_diffusion

        out_name = coreml_diffusion.compose_out_name(
            ckpt_name=ckpt_name,
            batch_size=batch_size,
            width=w,
            height=h,
            controlnet_support=controlnet_support,
            attention_implementation=attention_implementation,
            lora_names=coreml_diffusion.lora_names_from_params(lora_params),
            quantize_nbits=quantize_nbits,
        )

        logger.info(f"Converting {ckpt_name} to {out_name}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Width: {w}, Height: {h}")
        logger.info(f"ControlNet support: {controlnet_support}")
        logger.info(f"Attention implementation: {attention_implementation}")

        if lora_params:
            logger.info("LoRAs used:")
            for lora_param in lora_params:
                logger.info(f"  {lora_param[0]} - strength: {lora_param[1]}")

        # Resolve the ComfyUI models/unet path here (a node concern); the package
        # takes the output path as an injected argument.
        unet_path = folder_paths.get_folder_paths("unet")[0]
        unet_out_path = os.path.join(unet_path, f"{out_name}_unet.mlpackage")
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        config_filename = ckpt_name.split(".")[0] + ".yaml"
        config_path = folder_paths.get_full_path("configs", config_filename)
        if config_path:
            logger.info(f"Using config file {config_path}")

        coreml_diffusion.convert(
            ckpt_path,
            None,  # model_version auto-detected from the checkpoint
            unet_out_path,
            sample_size=sample_size,
            batch_size=batch_size,
            controlnet_support=controlnet_support,
            lora_weights=lora_weights,
            attn_impl=attention_implementation,
            config_path=config_path,
            quantize_nbits=quantize_nbits,
        )
        return (CoreMLModel(unet_out_path, compute_unit),)

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
