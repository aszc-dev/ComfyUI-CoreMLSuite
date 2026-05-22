"""Phase 2 characterization tests for CoreMLConverter.out_name composition.

out_name is encoded into the .mlpackage filename and therefore drives the
"have we converted this combo already?" cache check. Drift here silently
invalidates user caches and breaks workflow node references.

Tests intercept converter.get_out_path to capture the composed string,
and stub out the heavy conversion + Core ML model load.
"""
import pytest

import folder_paths
from coreml_suite import converter
from coreml_suite import nodes as nodes_mod
from coreml_suite.nodes import CoreMLConverter


@pytest.fixture
def capture_out_name(monkeypatch):
    captured = {}

    def fake_get_out_path(submodule, name):
        captured["submodule"] = submodule
        captured["out_name"] = name
        return f"/tmp/fake/{name}.mlpackage"

    monkeypatch.setattr(converter, "get_out_path", fake_get_out_path)
    monkeypatch.setattr(converter, "convert", lambda **kwargs: None)
    monkeypatch.setattr(
        converter,
        "compile_model",
        lambda out_path, out_name, submodule_name: f"/tmp/fake/{out_name}_{submodule_name}.mlmodelc",
    )
    monkeypatch.setattr(
        folder_paths,
        "get_full_path",
        lambda kind, name: f"/tmp/ckpts/{name}" if kind == "checkpoints" else None,
    )
    monkeypatch.setattr(nodes_mod, "CoreMLModel", lambda *a, **kw: object())
    return captured


def _convert(
    *,
    ckpt_name="dreamshaper_8.safetensors",
    model_version="SD15",
    height=512,
    width=512,
    batch_size=1,
    attention_implementation="SPLIT_EINSUM",
    compute_unit="CPU_AND_NE",
    controlnet_support=False,
    lora_params=None,
):
    node = CoreMLConverter()
    node.convert(
        ckpt_name=ckpt_name,
        model_version=model_version,
        height=height,
        width=width,
        batch_size=batch_size,
        attention_implementation=attention_implementation,
        compute_unit=compute_unit,
        controlnet_support=controlnet_support,
        lora_params=lora_params,
    )


# ---------- basic golden strings --------------------------------------------


@pytest.mark.parametrize(
    "attn_name,suffix",
    [
        ("SPLIT_EINSUM", "se"),
        ("SPLIT_EINSUM_V2", "se2"),
        ("ORIGINAL", "orig"),
    ],
)
def test_out_name_attention_suffix(capture_out_name, attn_name, suffix):
    _convert(attention_implementation=attn_name)
    assert capture_out_name["out_name"] == f"dreamshaper_8_1x512x512_{suffix}"


def test_out_name_includes_batch_and_size(capture_out_name):
    _convert(batch_size=4, width=768, height=1024)
    assert capture_out_name["out_name"] == "dreamshaper_8_4x768x1024_se"


def test_out_name_appends_cn_suffix_when_controlnet_support_true(capture_out_name):
    _convert(controlnet_support=True)
    assert capture_out_name["out_name"] == "dreamshaper_8_1x512x512_cn_se"


def test_out_name_drops_dot_extension_only_at_first_period(capture_out_name):
    """`ckpt_name.split('.')[0]` — first '.' wins; locked behaviour."""
    _convert(ckpt_name="my.checkpoint.v2.safetensors")
    assert capture_out_name["out_name"] == "my_1x512x512_se"


def test_out_name_replaces_spaces_with_underscores(capture_out_name):
    _convert(ckpt_name="dream shaper 8.safetensors")
    assert capture_out_name["out_name"] == "dream_shaper_8_1x512x512_se"


# ---------- LoRA suffixes ---------------------------------------------------


def test_out_name_with_single_lora(capture_out_name, monkeypatch):
    monkeypatch.setattr(
        folder_paths,
        "get_full_path",
        lambda kind, name: f"/tmp/{kind}/{name}",
    )
    _convert(lora_params={"epi_noiseoffset.safetensors": (0.8,)})
    assert (
        capture_out_name["out_name"]
        == "dreamshaper_8_epi_noiseoffset_1x512x512_se"
    )


def test_out_name_with_multiple_loras_sorted(capture_out_name, monkeypatch):
    """LoRAs are sorted by name then joined with '_' — locks the order."""
    monkeypatch.setattr(
        folder_paths,
        "get_full_path",
        lambda kind, name: f"/tmp/{kind}/{name}",
    )
    _convert(
        lora_params={
            "zoom.safetensors": (1.0,),
            "alpha.safetensors": (0.5,),
            "moody.safetensors": (0.3,),
        }
    )
    assert (
        capture_out_name["out_name"]
        == "dreamshaper_8_alpha_moody_zoom_1x512x512_se"
    )


def test_out_name_lora_plus_controlnet(capture_out_name, monkeypatch):
    monkeypatch.setattr(
        folder_paths,
        "get_full_path",
        lambda kind, name: f"/tmp/{kind}/{name}",
    )
    _convert(
        lora_params={"a.safetensors": (1.0,)},
        controlnet_support=True,
    )
    assert capture_out_name["out_name"] == "dreamshaper_8_a_1x512x512_cn_se"


# ---------- sdxl combinations -----------------------------------------------


def test_out_name_sdxl_1024(capture_out_name):
    _convert(
        ckpt_name="sd_xl_base_1.0.safetensors",
        model_version="SDXL",
        width=1024,
        height=1024,
        attention_implementation="ORIGINAL",
        compute_unit="CPU_AND_GPU",
    )
    assert capture_out_name["out_name"] == "sd_xl_base_1_1x1024x1024_orig"


# ---------- submodule path --------------------------------------------------


def test_get_out_path_invoked_with_unet_submodule(capture_out_name):
    _convert()
    assert capture_out_name["submodule"] == "unet"
