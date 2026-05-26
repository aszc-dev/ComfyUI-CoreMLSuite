"""Characterization tests for the SDXL options math.

The SDXL time_ids / text_embeds math lives in
coreml_suite.core.sdxl as pure builders. The framework adapter
add_sdxl_model_options lives in models.py; here we just lock the pure
math.
"""
import inspect

import pytest
import torch

from coreml_suite.core.sdxl import (
    build_sdxl_text_embeds,
    build_sdxl_time_ids,
    sdxl_model_function_wrapper,
)


@pytest.fixture(autouse=True)
def _deterministic_seed():
    torch.manual_seed(0)


# ---------- build_sdxl_time_ids: base (len 6) -------------------------------


def test_build_time_ids_base_defaults():
    out = build_sdxl_time_ids({}, {}, is_base=True, is_refiner=False)
    expected = torch.tensor([[768, 768, 0, 0, 768, 768], [768, 768, 0, 0, 768, 768]])
    assert out.shape == (2, 6)
    assert torch.equal(out, expected)


def test_build_time_ids_base_respects_overrides():
    pos = {"height": 1024, "width": 512, "crop_h": 8, "crop_w": 4,
           "target_height": 1024, "target_width": 1024}
    neg = {"height": 256, "width": 256, "crop_h": 0, "crop_w": 0,
           "target_height": 256, "target_width": 256}
    out = build_sdxl_time_ids(pos, neg, is_base=True, is_refiner=False)
    expected = torch.tensor([[1024, 512, 8, 4, 1024, 1024], [256, 256, 0, 0, 256, 256]])
    assert torch.equal(out, expected)


# ---------- build_sdxl_time_ids: refiner (len 5) ----------------------------


def test_build_time_ids_refiner_defaults():
    out = build_sdxl_time_ids({}, {}, is_base=False, is_refiner=True)
    expected = torch.tensor([[768, 768, 0, 0, 6.0], [768, 768, 0, 0, 2.5]])
    assert out.shape == (2, 5)
    assert torch.equal(out, expected)


def test_build_time_ids_refiner_respects_aesthetic_score():
    pos = {"aesthetic_score": 8.5}
    neg = {"aesthetic_score": 1.5}
    out = build_sdxl_time_ids(pos, neg, is_base=False, is_refiner=True)
    expected = torch.tensor([[768, 768, 0, 0, 8.5], [768, 768, 0, 0, 1.5]])
    assert torch.equal(out, expected)


# ---------- build_sdxl_time_ids: edge case ----------------------------------


def test_build_time_ids_neither_base_nor_refiner_returns_len4():
    out = build_sdxl_time_ids({}, {}, is_base=False, is_refiner=False)
    assert out.shape == (2, 4)


# ---------- build_sdxl_text_embeds ------------------------------------------


def test_text_embeds_concat_pos_then_neg():
    pos = torch.full((1, 1280), 1.0)
    neg = torch.full((1, 1280), -1.0)
    out = build_sdxl_text_embeds(pos, neg)
    assert out.shape == (2, 1280)
    assert torch.equal(out[0], pos[0])
    assert torch.equal(out[1], neg[0])


# ---------- sdxl_model_function_wrapper closure -----------------------------


def test_wrapper_captures_time_ids_text_embeds_refiner_via_closure():
    time_ids = torch.zeros(2, 6)
    text_embeds = torch.zeros(2, 1280)
    wrapper = sdxl_model_function_wrapper(time_ids, text_embeds, refiner=False)
    closure = inspect.getclosurevars(wrapper).nonlocals
    assert closure["time_ids"] is time_ids
    assert closure["text_embeds"] is text_embeds
    assert closure["refiner"] is False


def test_wrapper_returns_zero_when_context_missing():
    """When c_crossattn is None the wrapper short-circuits to zeros_like(x).
    Locked here because the refactor mustn't change this default."""
    wrapper = sdxl_model_function_wrapper(torch.zeros(2, 6), torch.zeros(2, 1280))
    x = torch.randn(2, 4, 16, 16)
    out = wrapper(
        model_function=lambda *a, **kw: pytest.fail("model_function must not run"),
        params={"input": x, "timestep": torch.zeros(2), "c": {}},
    )
    assert torch.equal(out, torch.zeros_like(x))


def test_wrapper_refiner_truncates_context_to_g_clip():
    """refiner=True slices c_crossattn[:, :, 768:] before forwarding."""
    captured = {}

    def fake_model(x, t, **c):
        captured["context_shape"] = c["c_crossattn"].shape
        captured["time_ids_shape"] = c["time_ids"].shape
        return x

    wrapper = sdxl_model_function_wrapper(
        torch.zeros(2, 5), torch.zeros(2, 1280), refiner=True
    )
    x = torch.randn(2, 4, 16, 16)
    context = torch.randn(2, 77, 2048)  # 768 + 1280 dims
    wrapper(
        model_function=fake_model,
        params={"input": x, "timestep": torch.zeros(2), "c": {"c_crossattn": context}},
    )
    assert captured["context_shape"] == (2, 77, 1280)
    assert captured["time_ids_shape"] == (2, 5)
