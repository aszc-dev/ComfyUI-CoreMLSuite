"""Compatibility shim.

Conversion mechanics moved to ``coreml_diffusion.convert`` in extraction phase
E2. This shim preserves the call surface ``coreml_suite.nodes`` uses so the node
is untouched this phase; E3 re-points the node onto ``coreml_diffusion`` directly.

``get_out_path`` stays here on purpose: resolving the ComfyUI ``models/unet``
directory via ``folder_paths`` is a node/runtime concern, not a conversion one.
E3 folds it into the node itself.
"""
import os

from coreml_suite.attention import ATTENTION_IMPLEMENTATIONS


def get_out_path(submodule_name, model_name):
    from folder_paths import get_folder_paths

    fname = f"{model_name}_{submodule_name}.mlpackage"
    unet_path = get_folder_paths(submodule_name)[0]
    out_path = os.path.join(unet_path, fname)
    return out_path


def convert(
    ckpt_path,
    model_version,
    unet_out_path,
    batch_size=1,
    sample_size=(64, 64),
    controlnet_support=False,
    lora_weights=None,
    attn_impl=ATTENTION_IMPLEMENTATIONS[0],
    config_path=None,
    quantize_nbits="none",
):
    # Imported lazily: coreml_diffusion.convert pulls coremltools/diffusers, which
    # must not load until an actual conversion runs.
    from coreml_diffusion.convert import convert as _convert

    return _convert(
        ckpt_path,
        model_version,
        unet_out_path,
        batch_size=batch_size,
        sample_size=sample_size,
        controlnet_support=controlnet_support,
        lora_weights=lora_weights,
        attn_impl=attn_impl,
        config_path=config_path,
        quantize_nbits=quantize_nbits,
    )
