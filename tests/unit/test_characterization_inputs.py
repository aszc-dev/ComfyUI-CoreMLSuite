"""Characterization tests for coreml_suite.models.CoreMLInputs.

Locks the shape transforms applied by chunks() and coreml_kwargs() for the
four model variants the suite supports: SD1.5, LCM (SD1.5 + timestep_cond),
SDXL base (time_ids len 6), and SDXL refiner (time_ids len 5).

These contracts feed the Core ML UNet at runtime; if a refactor silently
re-shapes them, generation breaks.
"""
import numpy as np
import pytest
import torch

from coreml_suite.core.inputs import CoreMLInputs


@pytest.fixture(autouse=True)
def _deterministic_seed():
    torch.manual_seed(0)
    np.random.seed(0)


# ---------- expected_inputs fixtures (mirror real model expectations) -------

SD15_EXPECTED = {
    "sample": {"shape": (2, 4, 64, 64)},
    "timestep": {"shape": (2,)},
    "encoder_hidden_states": {"shape": (2, 77, 768)},
}

SD15_WITH_CN = {
    **SD15_EXPECTED,
    "additional_residual_0": {"shape": (2, 320, 64, 64)},
    "additional_residual_1": {"shape": (2, 640, 32, 32)},
}

LCM_EXPECTED = {
    **SD15_EXPECTED,
    "timestep_cond": {"shape": (2, 256)},
}

SDXL_BASE_EXPECTED = {
    "sample": {"shape": (2, 4, 128, 128)},
    "timestep": {"shape": (2,)},
    "encoder_hidden_states": {"shape": (2, 77, 2048)},
    "time_ids": {"shape": (2, 6)},
    "text_embeds": {"shape": (2, 1280)},
}

SDXL_REFINER_EXPECTED = {
    "sample": {"shape": (2, 4, 128, 128)},
    "timestep": {"shape": (2,)},
    "encoder_hidden_states": {"shape": (2, 77, 1280)},
    "time_ids": {"shape": (2, 5)},
    "text_embeds": {"shape": (2, 1280)},
}


def _sd15_inputs(batch=1, with_control=False, with_ts_cond=False):
    x = torch.randn(batch, 4, 64, 64)
    t = torch.full((batch,), 999.0)
    context = torch.randn(batch, 77, 768)
    control = None
    if with_control:
        control = {
            "output": [torch.randn(batch, 320, 64, 64), torch.randn(batch, 640, 32, 32)],
            "middle": [],
        }
    kwargs = {}
    if with_ts_cond:
        kwargs["timestep_cond"] = torch.randn(batch, 256)
    return CoreMLInputs(x, t, context, control, **kwargs)


def _sdxl_inputs(batch=1, refiner=False):
    x = torch.randn(batch, 4, 128, 128)
    t = torch.full((batch,), 999.0)
    ctx_dim = 1280 if refiner else 2048
    context = torch.randn(batch, 77, ctx_dim)
    time_ids_dim = 5 if refiner else 6
    time_ids = torch.randn(batch, time_ids_dim)
    text_embeds = torch.randn(batch, 1280)
    return CoreMLInputs(
        x, t, context, control=None, time_ids=time_ids, text_embeds=text_embeds
    )


# ---------- coreml_kwargs ---------------------------------------------------


def test_coreml_kwargs_sd15_shapes_and_fp16():
    out = _sd15_inputs(batch=1).coreml_kwargs(SD15_EXPECTED)
    assert set(out.keys()) == {"sample", "encoder_hidden_states", "timestep"}
    assert out["sample"].shape == (1, 4, 64, 64)
    assert out["sample"].dtype == np.float16
    # encoder_hidden_states keeps Comfy's native (b, seq, dim) layout.
    assert out["encoder_hidden_states"].shape == (1, 77, 768)
    assert out["encoder_hidden_states"].dtype == np.float16
    assert out["timestep"].shape == (1,)
    assert out["timestep"].dtype == np.float16


def test_coreml_kwargs_sd15_with_controlnet_emits_residuals():
    inputs = _sd15_inputs(batch=1, with_control=True)
    out = inputs.coreml_kwargs(SD15_WITH_CN)
    assert "additional_residual_0" in out
    assert "additional_residual_1" in out
    assert out["additional_residual_0"].shape == (1, 320, 64, 64)
    assert out["additional_residual_1"].shape == (1, 640, 32, 32)


def test_coreml_kwargs_sd15_without_controlnet_zero_fills_residuals():
    inputs = _sd15_inputs(batch=1, with_control=False)
    out = inputs.coreml_kwargs(SD15_WITH_CN)
    assert np.all(out["additional_residual_0"] == 0)
    assert np.all(out["additional_residual_1"] == 0)


def test_coreml_kwargs_lcm_adds_timestep_cond():
    inputs = _sd15_inputs(batch=1, with_ts_cond=True)
    out = inputs.coreml_kwargs(LCM_EXPECTED)
    assert "timestep_cond" in out
    assert out["timestep_cond"].shape == (1, 256)
    assert out["timestep_cond"].dtype == np.float16


def test_coreml_kwargs_lcm_skips_timestep_cond_when_not_provided():
    """timestep_cond is only forwarded when the input supplied one — even if
    the model's expected_inputs lists it."""
    inputs = _sd15_inputs(batch=1, with_ts_cond=False)
    out = inputs.coreml_kwargs(LCM_EXPECTED)
    assert "timestep_cond" not in out


def test_coreml_kwargs_sdxl_base_emits_time_ids_and_text_embeds():
    out = _sdxl_inputs(batch=1, refiner=False).coreml_kwargs(SDXL_BASE_EXPECTED)
    assert out["time_ids"].shape == (1, 6)
    assert out["text_embeds"].shape == (1, 1280)
    assert out["time_ids"].dtype == np.float16
    assert out["text_embeds"].dtype == np.float16


def test_coreml_kwargs_sdxl_refiner_uses_len5_time_ids():
    out = _sdxl_inputs(batch=1, refiner=True).coreml_kwargs(SDXL_REFINER_EXPECTED)
    assert out["time_ids"].shape == (1, 5)


# ---------- chunks ----------------------------------------------------------


def test_chunks_sd15_pad_to_batch2_returns_one_chunk():
    chunked = _sd15_inputs(batch=1).chunks(SD15_EXPECTED)
    assert len(chunked) == 1
    c = chunked[0]
    assert c.x.shape == (2, 4, 64, 64)
    assert c.t.shape == (2,)
    # context shape: (b, seq, dim) padded along batch dim.
    assert c.context.shape == (2, 77, 768)
    assert c.control is None
    assert c.ts_cond is None
    assert c.time_ids is None
    assert c.text_embeds is None


def test_chunks_sd15_with_controlnet_chunks_residuals_too():
    chunked = _sd15_inputs(batch=1, with_control=True).chunks(SD15_EXPECTED)
    assert len(chunked) == 1
    cn = chunked[0].control
    assert cn is not None
    assert cn["output"][0].shape == (2, 320, 64, 64)
    assert cn["output"][1].shape == (2, 640, 32, 32)


def test_chunks_lcm_carries_timestep_cond_per_chunk():
    chunked = _sd15_inputs(batch=1, with_ts_cond=True).chunks(LCM_EXPECTED)
    assert len(chunked) == 1
    assert chunked[0].ts_cond is not None
    assert chunked[0].ts_cond.shape == (2, 256)


def test_chunks_sdxl_base_propagates_time_ids_and_text_embeds():
    chunked = _sdxl_inputs(batch=1, refiner=False).chunks(SDXL_BASE_EXPECTED)
    assert len(chunked) == 1
    c = chunked[0]
    assert c.time_ids is not None and c.time_ids.shape == (2, 6)
    assert c.text_embeds is not None and c.text_embeds.shape == (2, 1280)


def test_chunks_sdxl_refiner_uses_len5_time_ids():
    chunked = _sdxl_inputs(batch=1, refiner=True).chunks(SDXL_REFINER_EXPECTED)
    assert chunked[0].time_ids.shape == (2, 5)


def test_chunks_sdxl_synthesizes_zero_time_ids_when_caller_omits():
    """If the model expects time_ids but caller passed nothing, the suite
    fabricates a zero-filled tensor. Lock that fallback."""
    x = torch.randn(1, 4, 128, 128)
    t = torch.full((1,), 999.0)
    context = torch.randn(1, 77, 2048)
    inputs = CoreMLInputs(x, t, context, control=None)
    chunked = inputs.chunks(SDXL_BASE_EXPECTED)
    assert chunked[0].time_ids.shape == (2, 6)
    assert torch.equal(chunked[0].time_ids, torch.zeros(2, 6))
    assert chunked[0].text_embeds.shape == (2, 1280)
    assert torch.equal(chunked[0].text_embeds, torch.zeros(2, 1280))


def test_chunks_splits_batch_into_multiple_target2_chunks():
    """batch=5 with target_batch=2 -> 3 chunks (last padded)."""
    chunked = _sd15_inputs(batch=5).chunks(SD15_EXPECTED)
    assert len(chunked) == 3
    for c in chunked:
        assert c.x.shape == (2, 4, 64, 64)
        assert c.context.shape == (2, 77, 768)
    # Last chunk's second batch row is the zero-pad.
    assert torch.equal(chunked[-1].x[1], torch.zeros(4, 64, 64))


def test_chunks_timestep_is_broadcast_from_first_value():
    """t is rebuilt from t[0] across all chunks: locks current behavior that
    discards any per-row timestep variation."""
    x = torch.randn(2, 4, 64, 64)
    t = torch.tensor([42.0, 99.0])  # the second value will be lost
    context = torch.randn(2, 77, 768)
    inputs = CoreMLInputs(x, t, context, control=None)
    chunked = inputs.chunks(SD15_EXPECTED)
    assert chunked[0].t.shape == (2,)
    assert torch.equal(chunked[0].t, torch.full((2,), 42.0))
