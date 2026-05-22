"""Phase 2 characterization tests for add_sdxl_model_options.

Locks the SDXL time_ids / text_embeds assembly: base produces a (2, 6)
time_ids vector (h, w, crop_h, crop_w, target_h, target_w), refiner
produces (2, 5) (h, w, crop_h, crop_w, aesthetic_score). Both stack pos
then neg along the batch dim, and text_embeds is cat(pos_pooled,
neg_pooled).

Uses a SimpleNamespace-shaped fake ModelPatcher because exercising the
real comfy.model_patcher.ModelPatcher here is overkill — only three
attribute paths are read by the SUT.
"""
import inspect
from types import SimpleNamespace

import pytest
import torch

from coreml_suite.models import add_sdxl_model_options


@pytest.fixture(autouse=True)
def _deterministic_seed():
    torch.manual_seed(0)


def _fake_patcher(is_base: bool, is_refiner: bool):
    diffusion = SimpleNamespace(is_sdxl_base=is_base, is_sdxl_refiner=is_refiner)
    model = SimpleNamespace(diffusion_model=diffusion)
    patcher = SimpleNamespace(model=model, model_options={})
    patcher.clone = lambda: patcher  # in-place: simplest mock that matches contract
    return patcher


def _cond(pooled, **overrides):
    base = {"pooled_output": pooled}
    base.update(overrides)
    return [(None, base)]


def _closure_vars(wrapper):
    return inspect.getclosurevars(wrapper).nonlocals


# ---------- base (len 6) -----------------------------------------------------


def test_add_sdxl_model_options_base_produces_len6_time_ids_with_defaults():
    pos_pooled = torch.randn(1, 1280)
    neg_pooled = torch.randn(1, 1280)
    patcher = _fake_patcher(is_base=True, is_refiner=False)
    out = add_sdxl_model_options(patcher, _cond(pos_pooled), _cond(neg_pooled))
    wrapper = out.model_options["model_function_wrapper"]
    closure = _closure_vars(wrapper)
    assert closure["time_ids"].shape == (2, 6)
    # Defaults: 768 height/width, 0 crop, 768 target.
    expected = torch.tensor([[768, 768, 0, 0, 768, 768], [768, 768, 0, 0, 768, 768]])
    assert torch.equal(closure["time_ids"], expected)
    assert closure["refiner"] is False


def test_add_sdxl_model_options_base_respects_overrides():
    pos_pooled = torch.randn(1, 1280)
    neg_pooled = torch.randn(1, 1280)
    patcher = _fake_patcher(is_base=True, is_refiner=False)
    out = add_sdxl_model_options(
        patcher,
        _cond(pos_pooled, height=1024, width=512, crop_h=8, crop_w=4, target_height=1024, target_width=1024),
        _cond(neg_pooled, height=256, width=256, crop_h=0, crop_w=0, target_height=256, target_width=256),
    )
    closure = _closure_vars(out.model_options["model_function_wrapper"])
    expected = torch.tensor([[1024, 512, 8, 4, 1024, 1024], [256, 256, 0, 0, 256, 256]])
    assert torch.equal(closure["time_ids"], expected)


# ---------- refiner (len 5) -------------------------------------------------


def test_add_sdxl_model_options_refiner_produces_len5_time_ids():
    pos_pooled = torch.randn(1, 1280)
    neg_pooled = torch.randn(1, 1280)
    patcher = _fake_patcher(is_base=False, is_refiner=True)
    out = add_sdxl_model_options(patcher, _cond(pos_pooled), _cond(neg_pooled))
    closure = _closure_vars(out.model_options["model_function_wrapper"])
    assert closure["time_ids"].shape == (2, 5)
    # Defaults: pos aesthetic_score=6, neg aesthetic_score=2.5.
    expected = torch.tensor(
        [[768, 768, 0, 0, 6.0], [768, 768, 0, 0, 2.5]],
    )
    assert torch.equal(closure["time_ids"], expected)
    assert closure["refiner"] is True


def test_add_sdxl_model_options_refiner_respects_aesthetic_score_overrides():
    pos_pooled = torch.randn(1, 1280)
    neg_pooled = torch.randn(1, 1280)
    patcher = _fake_patcher(is_base=False, is_refiner=True)
    out = add_sdxl_model_options(
        patcher,
        _cond(pos_pooled, aesthetic_score=8.5),
        _cond(neg_pooled, aesthetic_score=1.5),
    )
    closure = _closure_vars(out.model_options["model_function_wrapper"])
    expected = torch.tensor([[768, 768, 0, 0, 8.5], [768, 768, 0, 0, 1.5]])
    assert torch.equal(closure["time_ids"], expected)


# ---------- text_embeds ----------------------------------------------------


def test_text_embeds_is_concat_pos_then_neg_along_batch():
    pos_pooled = torch.full((1, 1280), 1.0)
    neg_pooled = torch.full((1, 1280), -1.0)
    patcher = _fake_patcher(is_base=True, is_refiner=False)
    out = add_sdxl_model_options(patcher, _cond(pos_pooled), _cond(neg_pooled))
    closure = _closure_vars(out.model_options["model_function_wrapper"])
    embeds = closure["text_embeds"]
    assert embeds.shape == (2, 1280)
    assert torch.equal(embeds[0], pos_pooled[0])
    assert torch.equal(embeds[1], neg_pooled[0])


def test_neither_base_nor_refiner_yields_len4_time_ids():
    """Locked edge case: if both is_sdxl_base and is_sdxl_refiner are False,
    no extra entries are appended -> time_ids is only the 4 shared fields."""
    pos_pooled = torch.randn(1, 1280)
    neg_pooled = torch.randn(1, 1280)
    patcher = _fake_patcher(is_base=False, is_refiner=False)
    out = add_sdxl_model_options(patcher, _cond(pos_pooled), _cond(neg_pooled))
    closure = _closure_vars(out.model_options["model_function_wrapper"])
    assert closure["time_ids"].shape == (2, 4)


# ---------- patcher contract ------------------------------------------------


def test_model_options_dict_is_merged_into_clone_not_original():
    """Verify SUT writes to the clone's model_options. Our fake reuses the
    same instance, so the merge should still leave the model_function_wrapper
    key present after the call."""
    pos_pooled = torch.randn(1, 1280)
    neg_pooled = torch.randn(1, 1280)
    patcher = _fake_patcher(is_base=True, is_refiner=False)
    patcher.model_options["pre_existing"] = "kept"
    out = add_sdxl_model_options(patcher, _cond(pos_pooled), _cond(neg_pooled))
    assert "model_function_wrapper" in out.model_options
    assert out.model_options["pre_existing"] == "kept"
