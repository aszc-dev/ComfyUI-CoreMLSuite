"""Characterization tests for coreml_suite.latents.

Locks the *current* behavior of chunk_batch / merge_chunks — including the
zero-pad regions and the truncation in merge — so a refactor
cannot silently shift either contract.
"""
import pytest
import torch

from coreml_suite.core.latents import chunk_batch, merge_chunks


@pytest.fixture(autouse=True)
def _deterministic_seed():
    torch.manual_seed(0)


def _const_tensor(batch, *rest):
    return torch.arange(batch * 4 * 8 * 8, dtype=torch.float32).reshape(batch, 4, 8, 8)


# ---------- chunk_batch ------------------------------------------------------


def test_chunk_batch_passthrough_when_shape_matches():
    x = _const_tensor(2)
    out = chunk_batch(x, (2, 4, 8, 8))
    assert len(out) == 1
    # passthrough: the same object identity is returned (no copy).
    assert out[0] is x


def test_chunk_batch_pads_single_chunk_when_input_smaller():
    """batch=1, target=2 -> one padded chunk; the second row is exact zero."""
    x = _const_tensor(1)
    out = chunk_batch(x, (2, 4, 8, 8))
    assert len(out) == 1
    assert out[0].shape == (2, 4, 8, 8)
    assert torch.equal(out[0][0], x[0])
    assert torch.equal(out[0][1], torch.zeros(4, 8, 8))


def test_chunk_batch_splits_exact_multiple():
    """batch=4, target=2 -> two chunks, no padding."""
    x = _const_tensor(4)
    out = chunk_batch(x, (2, 4, 8, 8))
    assert len(out) == 2
    assert out[0].shape == (2, 4, 8, 8)
    assert out[1].shape == (2, 4, 8, 8)
    assert torch.equal(out[0], x[:2])
    assert torch.equal(out[1], x[2:])


def test_chunk_batch_pads_remainder_chunk():
    """batch=5, target=2 -> chunks=[x[0:2], x[2:4]] then [x[4], 0]."""
    x = _const_tensor(5)
    out = chunk_batch(x, (2, 4, 8, 8))
    assert len(out) == 3
    assert torch.equal(out[0], x[0:2])
    assert torch.equal(out[1], x[2:4])
    last = out[-1]
    assert last.shape == (2, 4, 8, 8)
    assert torch.equal(last[0], x[4])
    # The remainder row is zero-padded; lock that exact contract.
    assert torch.equal(last[1], torch.zeros(4, 8, 8))
    assert last[1].sum() == 0


@pytest.mark.parametrize(
    "batch_size,target,expected_chunks",
    [
        (1, 4, 1),
        (3, 2, 2),
        (5, 3, 2),
        (9, 4, 3),
    ],
)
def test_chunk_batch_pad_region_is_zero(batch_size, target, expected_chunks):
    x = _const_tensor(batch_size)
    out = chunk_batch(x, (target, 4, 8, 8))
    assert len(out) == expected_chunks
    mod = batch_size % target
    if mod == 0 and batch_size >= target:
        return
    last = out[-1]
    pad_rows = target - (mod if (mod != 0 and batch_size >= target) else batch_size)
    pad_region = last[-pad_rows:]
    assert torch.equal(pad_region, torch.zeros_like(pad_region))


# ---------- merge_chunks -----------------------------------------------------


def test_merge_chunks_exact_concat():
    x = _const_tensor(4)
    chunks = chunk_batch(x, (2, 4, 8, 8))
    merged = merge_chunks(chunks, x.shape)
    assert merged.shape == x.shape
    assert torch.equal(merged, x)


def test_merge_chunks_truncates_padding():
    """Round-trip with a padded last chunk drops the pad rows."""
    x = _const_tensor(5)
    chunks = chunk_batch(x, (2, 4, 8, 8))
    merged = merge_chunks(chunks, x.shape)
    assert merged.shape == x.shape
    assert torch.equal(merged, x)


def test_merge_chunks_singleton_returns_equal_copy_when_shape_matches():
    """A singleton chunk list still goes through torch.cat, so we get a new
    tensor equal to the input — locked here because a refactor might be tempted
    to short-circuit and accidentally return the same object."""
    x = _const_tensor(2)
    out = merge_chunks([x], x.shape)
    assert torch.equal(out, x)
    assert out is not x
