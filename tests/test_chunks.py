from unittest import mock

import pytest

import torch

from comfy.model_management import get_torch_device
from coreml_suite.latents import chunk_batch, merge_chunks
from coreml_suite.controlnet import chunk_control
from coreml_suite.models import (
    CoreMLModelWrapper,
    get_model_config,
    CoreMLModelWrapperLCM,
    CoreMLInputs,
)


@pytest.fixture
def expected_inputs():
    expected = {
        "sample": {"shape": (2, 4, 64, 64)},
        "timestep": {"shape": (2,)},
        "timestep_cond": {"shape": (2, 256)},
        "encoder_hidden_states": {"shape": (2, 768, 1, 77)},
        "additional_residual_0": {"shape": (2, 320, 64, 64)},
        "additional_residual_1": {"shape": (2, 640, 32, 32)},
    }
    return expected


@pytest.fixture
def model_config():
    return get_model_config()


@pytest.mark.parametrize("batch_size", [1, 2, 4, 5, 9])
def test_batch_chunking(batch_size):
    latent_image = torch.randn(batch_size, 4, 64, 64).to(get_torch_device())
    target_shape = (4, 4, 64, 64)

    chunked = chunk_batch(latent_image, target_shape)

    for chunk in chunked:
        assert chunk.shape == target_shape

    if batch_size % target_shape[0] != 0:
        assert chunked[-1][batch_size % target_shape[0] :].sum() == 0


@pytest.mark.parametrize("batch_size", [1, 2, 4, 5, 9])
def test_merge_chunks(batch_size):
    input_tensor = torch.randn(batch_size, 4, 64, 64).to(get_torch_device())
    target_shape = (4, 4, 64, 64)
    chunked = chunk_batch(input_tensor, target_shape)

    merged = merge_chunks(chunked, input_tensor.shape)

    assert merged.shape == input_tensor.shape
    assert torch.equal(input_tensor, merged)


@pytest.fixture
def inputs():
    x = torch.randn(1, 4, 64, 64).to(get_torch_device())
    t = torch.randn([1]).to(get_torch_device())
    c_crossattn = torch.randn(1, 77, 768).to(get_torch_device())
    control = {
        "output": [
            torch.randn(1, 320, 64, 64).to(get_torch_device()),
            torch.randn(1, 640, 32, 32).to(get_torch_device()),
        ],
    }
    timestep_cond = torch.randn(1, 256).to(get_torch_device())

    return CoreMLInputs(x, t, c_crossattn, control, timestep_cond=timestep_cond)


@pytest.mark.parametrize(
    "b, target_size, num_chunks",
    [
        (1, 2, 1),
        (1, 1, 1),
        (2, 2, 1),
        (3, 2, 2),
        (4, 2, 2),
        (5, 3, 2),
        (9, 4, 3),
    ],
)
def test_chunking_controlnet(b, target_size, num_chunks):
    cn = {
        "output": [
            torch.randn(b, 320, 64, 64).to(get_torch_device()),
            torch.randn(b, 640, 32, 32).to(get_torch_device()),
        ],
        "middle": [
            torch.randn(b, 1280, 8, 8).to(get_torch_device()),
        ],
    }

    chunked = chunk_control(cn, target_size)

    assert len(chunked) == num_chunks
    for chunk in chunked:
        assert chunk["output"][0].shape == (target_size, 320, 64, 64)
        assert chunk["output"][1].shape == (target_size, 640, 32, 32)
        assert chunk["middle"][0].shape == (target_size, 1280, 8, 8)


def test_chunking_no_control():
    cn = None
    target_size = 2

    chunked = chunk_control(cn, target_size)

    assert chunked == [None, None]


def test_chunking_inputs(expected_inputs, inputs):
    chunked = inputs.chunks(expected_inputs)

    assert len(chunked) == 1

    assert chunked[0].x.shape == (2, 4, 64, 64)
    assert chunked[0].t.shape == (2,)
    assert chunked[0].context.shape == (2, 77, 768)
    assert chunked[0].control["output"][0].shape == (2, 320, 64, 64)
    assert chunked[0].control["output"][1].shape == (2, 640, 32, 32)
    assert chunked[0].ts_cond.shape == (2, 256)
