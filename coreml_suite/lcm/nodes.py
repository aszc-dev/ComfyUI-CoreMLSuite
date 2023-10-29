import os

from coreml_suite.lcm import lcm_converter


class CoreMLConverterLCM:
    """Converts a LCM model to Core ML."""

    RETURN_TYPES = ("COMBO",)
    RETURN_NAMES = ("model_name",)
    FUNCTION = "convert"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("INT", {"default": 512, "min": 512, "max": 768, "step": 8}),
                "width": ("INT", {"default": 512, "min": 512, "max": 768, "step": 8}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 64}),
            }
        }

    def convert(self, height, width, batch_size):
        """Converts a LCM model to Core ML.

        Args:
            height (int): Height of the target image.
            width (int): Width of the target image.
            batch_size (int): Batch size.

        Returns:
            MODEL: The converted Core ML model.

        The converted model is also saved to "models/unet" directory and
        can be loaded with the "LCMCoreMLLoaderUNet" node.
        """
        h = height
        w = width
        sample_size = (h // 8, w // 8)
        batch_size = batch_size

        out_name = f"{lcm_converter.MODEL_NAME}_{w}x{h}_batch{batch_size}"

        out_path = lcm_converter.get_out_path("unet", f"{out_name}")

        if not os.path.exists(out_path):
            lcm_converter.convert(
                out_path=out_path, sample_size=sample_size, batch_size=batch_size
            )
        target_path = lcm_converter.compile_model(out_path=out_path, out_name=out_name)

        return (target_path.split("/")[-1],)
