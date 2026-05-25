"""Characterization tests for the .mlpackage filename composition.

The filename composition is the pure
coreml_suite.core.naming.compose_out_name function. CoreMLConverter.convert
calls it; testing the pure function avoids monkey-patching heavy converter
internals just to capture the string.
"""
import pytest

from coreml_suite.core.naming import compose_out_name, lora_names_from_params


# ---------- attention suffixes ----------------------------------------------


@pytest.mark.parametrize(
    "attn_name,suffix",
    [
        ("SPLIT_EINSUM", "se"),
        ("SPLIT_EINSUM_V2", "se2"),
        ("ORIGINAL", "orig"),
    ],
)
def test_attention_suffix(attn_name, suffix):
    out = compose_out_name(
        ckpt_name="dreamshaper_8.safetensors",
        batch_size=1, width=512, height=512,
        controlnet_support=False,
        attention_implementation=attn_name,
    )
    assert out == f"dreamshaper_8_1x512x512_{suffix}"


# ---------- batch / size ----------------------------------------------------


def test_includes_batch_and_size():
    out = compose_out_name(
        ckpt_name="dreamshaper_8.safetensors",
        batch_size=4, width=768, height=1024,
        controlnet_support=False,
        attention_implementation="SPLIT_EINSUM",
    )
    assert out == "dreamshaper_8_4x768x1024_se"


# ---------- ControlNet ------------------------------------------------------


def test_appends_cn_suffix_when_controlnet_support_true():
    out = compose_out_name(
        ckpt_name="dreamshaper_8.safetensors",
        batch_size=1, width=512, height=512,
        controlnet_support=True,
        attention_implementation="SPLIT_EINSUM",
    )
    assert out == "dreamshaper_8_1x512x512_cn_se"


# ---------- ckpt name massage -----------------------------------------------


def test_drops_extension_at_first_period():
    out = compose_out_name(
        ckpt_name="my.checkpoint.v2.safetensors",
        batch_size=1, width=512, height=512,
        controlnet_support=False,
        attention_implementation="SPLIT_EINSUM",
    )
    assert out == "my_1x512x512_se"


def test_replaces_spaces_with_underscores():
    out = compose_out_name(
        ckpt_name="dream shaper 8.safetensors",
        batch_size=1, width=512, height=512,
        controlnet_support=False,
        attention_implementation="SPLIT_EINSUM",
    )
    assert out == "dream_shaper_8_1x512x512_se"


# ---------- LoRA suffixes ---------------------------------------------------


def test_single_lora():
    out = compose_out_name(
        ckpt_name="dreamshaper_8.safetensors",
        batch_size=1, width=512, height=512,
        controlnet_support=False,
        attention_implementation="SPLIT_EINSUM",
        lora_names=["epi_noiseoffset.safetensors"],
    )
    assert out == "dreamshaper_8_epi_noiseoffset_1x512x512_se"


def test_multiple_loras_sorted():
    out = compose_out_name(
        ckpt_name="dreamshaper_8.safetensors",
        batch_size=1, width=512, height=512,
        controlnet_support=False,
        attention_implementation="SPLIT_EINSUM",
        lora_names=["zoom.safetensors", "alpha.safetensors", "moody.safetensors"],
    )
    assert out == "dreamshaper_8_alpha_moody_zoom_1x512x512_se"


def test_lora_plus_controlnet():
    out = compose_out_name(
        ckpt_name="dreamshaper_8.safetensors",
        batch_size=1, width=512, height=512,
        controlnet_support=True,
        attention_implementation="SPLIT_EINSUM",
        lora_names=["a.safetensors"],
    )
    assert out == "dreamshaper_8_a_1x512x512_cn_se"


# ---------- sdxl combinations -----------------------------------------------


def test_sdxl_1024_original_gpu():
    out = compose_out_name(
        ckpt_name="sd_xl_base_1.0.safetensors",
        batch_size=1, width=1024, height=1024,
        controlnet_support=False,
        attention_implementation="ORIGINAL",
    )
    assert out == "sd_xl_base_1_1x1024x1024_orig"


# ---------- lora_names_from_params helper ----------------------------------


def test_lora_names_from_params_sorts_by_name():
    names = lora_names_from_params([
        ("zebra.safetensors", 1.0),
        ("apple.safetensors", 0.5),
        ("mango.safetensors", 0.7),
    ])
    assert names == ["apple.safetensors", "mango.safetensors", "zebra.safetensors"]


def test_lora_names_from_params_empty_list():
    assert lora_names_from_params([]) == []


# ---------- quantize_nbits suffix ------------------------------------------


def test_quantize_nbits_none_appends_nothing():
    """'none' is the default and must keep the unquantized filename so
    existing cached .mlpackages still resolve."""
    out = compose_out_name(
        ckpt_name="dreamshaper_8.safetensors",
        batch_size=1, width=512, height=512,
        controlnet_support=False,
        attention_implementation="SPLIT_EINSUM",
        quantize_nbits="none",
    )
    assert out == "dreamshaper_8_1x512x512_se"


@pytest.mark.parametrize("nbits,suffix", [("4", "_q4"), ("6", "_q6"), ("8", "_q8")])
def test_quantize_nbits_appends_q_suffix(nbits, suffix):
    out = compose_out_name(
        ckpt_name="dreamshaper_8.safetensors",
        batch_size=1, width=512, height=512,
        controlnet_support=False,
        attention_implementation="SPLIT_EINSUM",
        quantize_nbits=nbits,
    )
    assert out == f"dreamshaper_8_1x512x512_se{suffix}"


def test_quantize_nbits_with_controlnet_and_lora():
    out = compose_out_name(
        ckpt_name="dreamshaper_8.safetensors",
        batch_size=1, width=512, height=512,
        controlnet_support=True,
        attention_implementation="SPLIT_EINSUM",
        lora_names=["a.safetensors"],
        quantize_nbits="6",
    )
    assert out == "dreamshaper_8_a_1x512x512_cn_se_q6"


def test_quantize_nbits_invalid_raises():
    import pytest as _pytest
    with _pytest.raises(ValueError, match="quantize_nbits"):
        compose_out_name(
            ckpt_name="x.safetensors",
            batch_size=1, width=512, height=512,
            controlnet_support=False,
            attention_implementation="SPLIT_EINSUM",
            quantize_nbits="16",  # not in {none, 8, 6, 4}
        )
