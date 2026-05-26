"""LCM-specific conversion orchestration (comfy-side).

E2 deduped the generic helpers (input building, Core ML export, residual-shape
calc) into ``coreml_diffusion.convert`` — this file now imports them instead of
carrying near-identical copies. What stays here is the genuinely LCM-specific
path: the hardcoded ``SimianLuo/LCM_Dreamshaper_v7`` download and the scheduler
that supplies the trace timestep. Consolidating that into the unified
``coreml_diffusion.convert(model_version=LCM, ...)`` path is a behavior change
deferred to E-LCM (it needs its own golden anchor).

``get_scheduler`` keeps using ``comfy.model_management`` because it runs on the
comfy side; the conversion package itself stays comfy-free.
"""
import gc
import logging
import os

import torch
from diffusers import UNet2DConditionModel, LCMScheduler
from diffusers.loaders import LoraLoaderMixin

from coreml_diffusion.conversion.attention import apply_attention_implementation
from coreml_diffusion.conversion.unet import CoreMLUNetWrapper
from coreml_diffusion.convert import (
    add_cnet_support,
    convert_to_coreml,
    get_coreml_inputs,
    get_encoder_hidden_states_shape,
    get_inputs_spec,
    get_sample_input,
    lcm_inputs,
)
from coreml_diffusion import ModelVersion

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MODEL_VERSION = "SimianLuo/LCM_Dreamshaper_v7"
MODEL_NAME = MODEL_VERSION.split("/")[-1] + "_4k"


def get_unets():
    ref_unet = UNet2DConditionModel.from_pretrained(
        MODEL_VERSION,
        subfolder="unet",
        device_map=None,
        low_cpu_mem_usage=False,
    )

    cml_unet = CoreMLUNetWrapper(
        apply_attention_implementation(ref_unet.eval(), "SPLIT_EINSUM"),
        ModelVersion.LCM,
    )

    return cml_unet, ref_unet


def get_scheduler():
    from comfy.model_management import get_torch_device

    scheduler = LCMScheduler.from_pretrained(MODEL_VERSION, subfolder="scheduler")
    scheduler.set_timesteps(50, get_torch_device(), 50)
    return scheduler


def get_out_path(submodule_name, model_name):
    from folder_paths import get_folder_paths

    fname = f"{model_name}_{submodule_name}.mlpackage"
    unet_path = get_folder_paths(submodule_name)[0]
    out_path = os.path.join(unet_path, fname)
    return out_path


def convert(
    out_path: str,
    batch_size: int = 1,
    sample_size: tuple[int, int] = (64, 64),
    controlnet_support: bool = False,
    lora_paths: list[str] = None,
):
    lora_paths = lora_paths or []
    coreml_unet, ref_unet = get_unets()

    for lora_path in lora_paths:
        lora_sd, network_alphas = LoraLoaderMixin.lora_state_dict(lora_path)
        LoraLoaderMixin.load_lora_into_unet(lora_sd, network_alphas, ref_unet)
        ref_unet.fuse_lora()

    sample_shape = (
        batch_size,  # B
        ref_unet.config.in_channels,  # C
        sample_size[0],  # H
        sample_size[1],  # W
    )

    encoder_hidden_states_shape = get_encoder_hidden_states_shape(ref_unet, batch_size)

    scheduler = get_scheduler()

    sample_inputs = get_sample_input(
        batch_size, encoder_hidden_states_shape, sample_shape, scheduler=scheduler
    )
    sample_inputs |= lcm_inputs(sample_inputs)

    if controlnet_support:
        sample_inputs |= add_cnet_support(sample_shape, ref_unet)

    sample_inputs_spec = get_inputs_spec(sample_inputs)

    logger.info(f"Sample UNet inputs spec: {sample_inputs_spec}")
    logger.info("JIT tracing..")
    traced_unet = torch.jit.trace(
        coreml_unet, example_inputs=list(sample_inputs.values())
    )
    logger.info("Done.")

    coreml_sample_inputs = get_coreml_inputs(sample_inputs)

    coreml_unet = convert_to_coreml(
        "unet", traced_unet, coreml_sample_inputs, ["noise_pred"], out_path
    )

    del traced_unet
    gc.collect()

    coreml_unet.save(out_path)
    logger.info(f"Saved unet into {out_path}")


if __name__ == "__main__":
    h = 512
    w = 512
    sample_size = (h // 8, w // 8)
    batch_size = 4

    cn_support_str = "_cn" if True else ""

    out_name = f"{MODEL_NAME}_{batch_size}x{w}x{h}{cn_support_str}"

    out_path = get_out_path("unet", f"{out_name}")
    if not os.path.exists(out_path):
        convert(out_path=out_path, sample_size=sample_size, batch_size=batch_size)
