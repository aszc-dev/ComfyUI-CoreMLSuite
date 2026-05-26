"""Characterization tests for coreml_suite.controlnet.

Locks shapes + dtypes + zero-fill behavior of expand_inputs / no_control /
extract_residual_kwargs / chunk_control. These pure helpers feed the Core ML
UNet's additional_residual_N inputs; any drift here silently breaks
ControlNet-based workflows.
"""
import numpy as np
import pytest
import torch

from coreml_suite.core.controlnet import (
    chunk_control,
    expand_inputs,
    extract_residual_kwargs,
    no_control,
)


@pytest.fixture(autouse=True)
def _deterministic_seed():
    torch.manual_seed(0)
    np.random.seed(0)


SD15_RESIDUAL_SPEC = {
    "additional_residual_0": {"shape": (2, 320, 64, 64)},
    "additional_residual_1": {"shape": (2, 640, 32, 32)},
    "additional_residual_2": {"shape": (2, 1280, 8, 8)},
}
NON_RESIDUAL_SPEC = {
    "sample": {"shape": (2, 4, 64, 64)},
    "encoder_hidden_states": {"shape": (2, 77, 768)},
}


# ---------- expand_inputs ----------------------------------------------------


def test_expand_inputs_doubles_singleton_numpy():
    inputs = {"a": np.ones((1, 4), dtype=np.float32)}
    out = expand_inputs(inputs)
    assert out["a"].shape == (2, 4)
    assert np.array_equal(out["a"], np.ones((2, 4)))


def test_expand_inputs_doubles_singleton_torch():
    inputs = {"a": torch.ones(1, 4)}
    out = expand_inputs(inputs)
    assert out["a"].shape == (2, 4)
    assert torch.equal(out["a"], torch.ones(2, 4))


def test_expand_inputs_doubles_singleton_list():
    inputs = {"a": [42]}
    out = expand_inputs(inputs)
    assert out["a"] == [42, 42]


def test_expand_inputs_skips_already_batched():
    """batch > 1 inputs are returned unchanged (same object identity)."""
    arr = np.ones((2, 4), dtype=np.float32)
    tensor = torch.ones(3, 4)
    lst = [1, 2]
    out = expand_inputs({"a": arr, "b": tensor, "c": lst})
    assert out["a"] is arr
    assert out["b"] is tensor
    assert out["c"] is lst


def test_expand_inputs_preserves_unknown_value_types():
    # Strings/None pass through untouched — locks current permissive contract.
    inputs = {"s": "hello", "none": None, "int": 7}
    out = expand_inputs(inputs)
    assert out == {"s": "hello", "none": None, "int": 7}


# ---------- no_control -------------------------------------------------------


def test_no_control_returns_zero_fp16_for_residuals():
    out = no_control({**SD15_RESIDUAL_SPEC, **NON_RESIDUAL_SPEC})
    # Only additional_residual_* keys are produced.
    assert set(out.keys()) == set(SD15_RESIDUAL_SPEC.keys())
    for key, spec in SD15_RESIDUAL_SPEC.items():
        arr = out[key]
        assert arr.shape == spec["shape"]
        assert arr.dtype == np.float16
        assert np.all(arr == 0)


def test_no_control_returns_empty_when_no_residuals():
    out = no_control(NON_RESIDUAL_SPEC)
    assert out == {}


# ---------- extract_residual_kwargs -----------------------------------------


def test_extract_residual_kwargs_empty_when_model_has_no_residual_inputs():
    out = extract_residual_kwargs(NON_RESIDUAL_SPEC, control={"output": [], "middle": []})
    assert out == {}


def test_extract_residual_kwargs_none_control_returns_no_control_shapes():
    out = extract_residual_kwargs(SD15_RESIDUAL_SPEC, control=None)
    assert set(out.keys()) == set(SD15_RESIDUAL_SPEC.keys())
    for key, spec in SD15_RESIDUAL_SPEC.items():
        assert out[key].shape == spec["shape"]
        assert out[key].dtype == np.float16
        assert np.all(out[key] == 0)


def test_extract_residual_kwargs_flattens_output_then_middle_and_casts_fp16():
    """output residuals come first (indexed 0..N-1), then middle residuals
    (indexed N..M-1). Values come out of CPU as fp16 numpy arrays."""
    control = {
        "output": [torch.ones(2, 320, 64, 64) * 0.5, torch.ones(2, 640, 32, 32) * 2.0],
        "middle": [torch.ones(2, 1280, 8, 8) * -1.0],
    }
    out = extract_residual_kwargs(SD15_RESIDUAL_SPEC, control)
    assert set(out.keys()) == {"additional_residual_0", "additional_residual_1", "additional_residual_2"}
    assert out["additional_residual_0"].shape == (2, 320, 64, 64)
    assert out["additional_residual_1"].shape == (2, 640, 32, 32)
    assert out["additional_residual_2"].shape == (2, 1280, 8, 8)
    for arr in out.values():
        assert arr.dtype == np.float16
    # Locked order: index 0 == first output residual (0.5), index 2 == middle (-1.0).
    assert np.allclose(out["additional_residual_0"], 0.5)
    assert np.allclose(out["additional_residual_1"], 2.0)
    assert np.allclose(out["additional_residual_2"], -1.0)


# ---------- chunk_control ----------------------------------------------------


def test_chunk_control_none_returns_list_of_nones_with_length_target():
    """`no_control` path: when there's no control, you get [None] * target_size
    (NOT [None, None] regardless of target — this is the contract today)."""
    assert chunk_control(None, 1) == [None]
    assert chunk_control(None, 2) == [None, None]
    assert chunk_control(None, 4) == [None, None, None, None]


@pytest.mark.parametrize(
    "batch,target,expected_chunks",
    [(1, 2, 1), (2, 2, 1), (3, 2, 2), (4, 2, 2), (5, 3, 2), (9, 4, 3)],
)
def test_chunk_control_shapes_after_chunking(batch, target, expected_chunks):
    cn = {
        "output": [
            torch.randn(batch, 320, 64, 64),
            torch.randn(batch, 640, 32, 32),
        ],
        "middle": [torch.randn(batch, 1280, 8, 8)],
    }
    chunks = chunk_control(cn, target)
    assert len(chunks) == expected_chunks
    for c in chunks:
        assert c["output"][0].shape == (target, 320, 64, 64)
        assert c["output"][1].shape == (target, 640, 32, 32)
        assert c["middle"][0].shape == (target, 1280, 8, 8)


def test_chunk_control_preserves_keys_order():
    """Output dicts contain exactly {"output", "middle"} in that order."""
    cn = {
        "output": [torch.zeros(2, 4, 4, 4)],
        "middle": [torch.zeros(2, 4, 4, 4)],
    }
    chunks = chunk_control(cn, 2)
    assert list(chunks[0].keys()) == ["output", "middle"]


def test_chunk_control_zero_pads_remainder():
    """A batch=3, target=2 split puts the third row alongside a zero row."""
    cn = {
        "output": [torch.arange(3 * 4).reshape(3, 1, 2, 2).float()],
        "middle": [torch.arange(3 * 4).reshape(3, 1, 2, 2).float()],
    }
    chunks = chunk_control(cn, 2)
    assert len(chunks) == 2
    last_out = chunks[-1]["output"][0]
    # First row is the original third row; second row is padding zeros.
    assert torch.equal(last_out[0], cn["output"][0][2])
    assert torch.equal(last_out[1], torch.zeros(1, 2, 2))
