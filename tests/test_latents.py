import pytest

import torch

from coreml_suite.samplers import reshape_latent_image


def test_fix_latents_no_latent_image():
    reshaped = reshape_latent_image(None, (2, 4, 64, 64))
    assert reshaped["samples"].shape == (2, 4, 64, 64)


@pytest.mark.parametrize(
    "latent_shape", [(2, 4, 64, 64), (2, 4, 128, 128), (2, 4, 32, 32), (2, 4, 128, 64)]
)
def test_reshape_latents(latent_shape):
    latent_image = {"samples": torch.zeros(latent_shape)}
    reshaped = reshape_latent_image(latent_image, (2, 4, 64, 64))
    assert reshaped["samples"].shape == (2, 4, 64, 64)
