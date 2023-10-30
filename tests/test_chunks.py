import pytest

import torch

from coreml_suite.latents import chunk_batch, merge_chunks
from coreml_suite.controlnet import chunk_control


@pytest.mark.parametrize("batch_size", [2, 4, 5, 9])
def test_batch_chunking(batch_size):
    latent_image = torch.randn(batch_size, 4, 64, 64)
    target_shape = (4, 4, 64, 64)

    chunked = chunk_batch(latent_image, target_shape)

    for chunk in chunked:
        assert chunk.shape == target_shape

    if batch_size % target_shape[0] != 0:
        assert chunked[-1][batch_size % target_shape[0] :].sum() == 0


@pytest.mark.parametrize("batch_size", [2, 4, 5, 9])
def test_merge_chunks(batch_size):
    input_tensor = torch.randn(batch_size, 4, 64, 64)
    target_shape = (4, 4, 64, 64)
    chunked = chunk_batch(input_tensor, target_shape)

    merged = merge_chunks(chunked, input_tensor.shape)

    assert merged.shape == input_tensor.shape
    assert torch.equal(input_tensor, merged)


def test_chunking_controlnet():
    cn = {
        "output": [torch.randn(4, 4, 64, 64), torch.randn(4, 4, 128, 128)],
        "middle": [torch.randn(4, 4, 256, 256)],
    }
    target_batch_size = 2
    num_chunks = cn["output"][0].shape[0] // target_batch_size

    chunked = chunk_control(cn, num_chunks)

    for chunk in chunked:
        assert chunk["output"][0].shape == (target_batch_size, 4, 64, 64)
        assert chunk["output"][1].shape == (target_batch_size, 4, 128, 128)
        assert chunk["middle"][0].shape == (target_batch_size, 4, 256, 256)


def test_chunking_no_control():
    cn = None
    num_chunks = 2

    chunked = chunk_control(cn, num_chunks)

    assert chunked == [None, None]
