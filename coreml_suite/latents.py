import torch
from torchvision.transforms.functional import resize

from coreml_suite.logger import logger


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
