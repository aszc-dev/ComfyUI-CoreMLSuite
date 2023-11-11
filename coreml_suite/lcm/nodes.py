import os

from coremltools import ComputeUnit
from python_coreml_stable_diffusion.coreml_model import CoreMLModel

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
