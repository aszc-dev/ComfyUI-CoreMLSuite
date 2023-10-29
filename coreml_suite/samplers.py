import torch
from torchvision.transforms.functional import resize

from comfy.model_management import get_torch_device
from comfy.model_patcher import ModelPatcher
from coreml_suite.logger import logger
from nodes import KSampler

from coreml_suite.models import CoreMLModelWrapper, get_model_config


def reshape_latent_image(latent_image, target_shape):
    if latent_image is None:
        logger.warning("No latent image provided, using zeros.")
        return {"samples": torch.zeros(target_shape)}

    if latent_image["samples"].shape == target_shape:
        return latent_image

    logger.warning(
        "Latent image shape does not match model input shape,"
        " resizing to match models expected input shape."
    )
    resized = resize(latent_image["samples"], target_shape[-2:])
    return {"samples": resized}


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

    CATEGORY = "CoreML Suite"

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
