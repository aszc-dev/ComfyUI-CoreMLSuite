"""LCM conversion node (comfy-side orchestration).

E-LCM removed the bespoke trace/convert pipeline that was hardcoded to
``SimianLuo/LCM_Dreamshaper_v7``; conversion now goes through the unified
``coreml_diffusion.convert(model_version=LCM, ...)`` path, which accepts any
full-distill LCM checkpoint (diffusers-layout UNet dumps and LDM single
files). LCM-LoRA merged checkpoints have plain SD1.5 architecture and are
rejected by the package with a pointer to the standard converter node.

What stays here is comfy-side wiring: checkpoint selection (with a
convenience auto-download of the canonical SimianLuo single file), output
path resolution under ComfyUI's ``models/unet``, and the loaded-model return.
"""

import os

from coremltools import ComputeUnit

import folder_paths
from coreml_suite import COREML_NODE
from coreml_suite.coreml_model import CoreMLModel
from coreml_suite.logger import logger

DEFAULT_LCM_REPO = "SimianLuo/LCM_Dreamshaper_v7"
DEFAULT_LCM_FILE = "LCM_Dreamshaper_v7_4k.safetensors"
DEFAULT_LCM_OPTION = f"{DEFAULT_LCM_FILE} (auto-download)"


class COREML_CONVERT_LCM(COREML_NODE):
    """Converts a full-distill LCM checkpoint to Core ML."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # The sentinel first: it is the default, so workflows saved
                # before this input existed keep the old auto-download
                # behavior when reloaded.
                "ckpt_name": (
                    [DEFAULT_LCM_OPTION]
                    + folder_paths.get_filename_list("checkpoints"),
                ),
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

    def convert(
        self, ckpt_name, height, width, batch_size, compute_unit, controlnet_support
    ):
        """Converts a full-distill LCM checkpoint's UNet to Core ML.

        Args:
            ckpt_name (str): Checkpoint to convert — a file from the
                checkpoints folder, or the default entry which downloads the
                canonical SimianLuo single file into the Hugging Face cache.
            height (int): Height of the target image.
            width (int): Width of the target image.
            batch_size (int): Batch size.
            compute_unit (str): Compute unit to use when loading the model.
            controlnet_support (bool): Add ControlNet residual inputs.

        Returns:
            coreml_model: The converted Core ML model.

        The converted model is saved to the "models/unet" directory (skipped
        when the same conversion already exists) and can also be loaded with
        the "Load Core ML UNet" node.
        """
        import coreml_diffusion

        ckpt_path = self._resolve_checkpoint(ckpt_name)

        out_name = coreml_diffusion.compose_out_name(
            ckpt_name=os.path.basename(ckpt_path),
            batch_size=batch_size,
            width=width,
            height=height,
            controlnet_support=controlnet_support,
            attention_implementation="SPLIT_EINSUM",
        )
        unet_dir = folder_paths.get_folder_paths("unet")[0]
        out_path = os.path.join(unet_dir, f"{out_name}_unet.mlpackage")

        logger.info(f"Converting {ckpt_name} to {out_name}")

        coreml_diffusion.convert(
            ckpt_path,
            coreml_diffusion.ModelVersion.LCM,
            out_path,
            sample_size=(height // 8, width // 8),
            batch_size=batch_size,
            controlnet_support=controlnet_support,
            attn_impl="SPLIT_EINSUM",
        )
        return (CoreMLModel(out_path, compute_unit),)

    @staticmethod
    def _resolve_checkpoint(ckpt_name):
        if ckpt_name == DEFAULT_LCM_OPTION:
            from huggingface_hub import hf_hub_download

            logger.info(f"Resolving {DEFAULT_LCM_REPO}/{DEFAULT_LCM_FILE}")
            return hf_hub_download(DEFAULT_LCM_REPO, DEFAULT_LCM_FILE)
        return folder_paths.get_full_path("checkpoints", ckpt_name)
