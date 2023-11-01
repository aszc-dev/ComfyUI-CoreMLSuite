import os
import shutil
import logging
import time
import gc

import numpy as np
import torch
from diffusers import UNet2DConditionModel
from python_coreml_stable_diffusion.unet import (
    UNet2DConditionModel as CoreMLUNet2DConditionModel,
)
from transformers import CLIPTextModel
import coremltools as ct

from folder_paths import get_folder_paths
from coreml_suite.lcm.lcm_scheduler import LCMScheduler

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MODEL_VERSION = "SimianLuo/LCM_Dreamshaper_v7"
MODEL_NAME = MODEL_VERSION.split("/")[-1] + "_4k"

import python_coreml_stable_diffusion.unet as unet

unet.ATTENTION_IMPLEMENTATION_IN_EFFECT = unet.AttentionImplementations.SPLIT_EINSUM


def get_unets():
    ref_unet = UNet2DConditionModel.from_pretrained(
        MODEL_VERSION,
        subfolder="unet",
        device_map=None,
        low_cpu_mem_usage=False,
    )

    ref_config = ref_unet.config

    cml_unet = CoreMLUNet2DConditionModel().eval()
    cml_unet.load_state_dict(ref_unet.state_dict(), strict=False)

    return cml_unet, ref_unet


def get_encoder_hidden_states_shape(unet_config, batch_size):
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_VERSION, subfolder="text_encoder"
    )

    text_token_sequence_length = text_encoder.config.max_position_embeddings
    hidden_size = (text_encoder.config.hidden_size,)

    encoder_hidden_states_shape = (
        batch_size,
        unet_config.cross_attention_dim or hidden_size,
        1,
        text_token_sequence_length,
    )

    return encoder_hidden_states_shape


def get_scheduler():
    scheduler = LCMScheduler(
        beta_start=0.00085,
        beta_end=0.0120,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(50, 50)
    return scheduler


def get_coreml_inputs(sample_inputs):
    coreml_sample_unet_inputs = {
        k: v.numpy().astype(np.float16) for k, v in sample_inputs.items()
    }
    return [
        ct.TensorType(
            name=k,
            shape=v.shape,
            dtype=v.numpy().dtype if isinstance(v, torch.Tensor) else v.dtype,
        )
        for k, v in coreml_sample_unet_inputs.items()
    ]


def load_coreml_model(out_path):
    logger.info(f"Loading model from {out_path}")

    start = time.time()
    coreml_model = ct.models.MLModel(out_path)
    logger.info(f"Loading {out_path} took {time.time() - start:.1f} seconds")

    return coreml_model


def convert_to_coreml(
    submodule_name, torchscript_module, sample_inputs, output_names, out_path
):
    if os.path.exists(out_path):
        logger.info(f"Skipping export because {out_path} already exists")
        coreml_model = load_coreml_model(out_path)
    else:
        logger.info(f"Converting {submodule_name} to CoreML..")
        coreml_model = ct.convert(
            torchscript_module,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
            inputs=sample_inputs,
            outputs=[
                ct.TensorType(name=name, dtype=np.float32) for name in output_names
            ],
            skip_model_load=True,
        )

        del torchscript_module
        gc.collect()

    return coreml_model


def get_out_path(submodule_name, model_name):
    fname = f"{model_name}_{submodule_name}.mlpackage"
    unet_path = get_folder_paths(submodule_name)[0]
    out_path = os.path.join(unet_path, fname)
    return out_path


def compile_coreml_model(source_model_path, output_dir, final_name):
    """Compiles Core ML models using the coremlcompiler utility from Xcode toolchain"""
    target_path = os.path.join(output_dir, f"{final_name}.mlmodelc")
    if os.path.exists(target_path):
        logger.warning(f"Found existing compiled model at {target_path}! Skipping..")
        return target_path

    logger.info(f"Compiling {source_model_path}")
    source_model_name = os.path.basename(os.path.splitext(source_model_path)[0])

    os.system(f"xcrun coremlcompiler compile {source_model_path} {output_dir}")
    compiled_output = os.path.join(output_dir, f"{source_model_name}.mlmodelc")
    shutil.move(compiled_output, target_path)

    return target_path


def get_sample_input(batch_size, encoder_hidden_states_shape, sample_shape, scheduler):
    sample_unet_inputs = dict(
        [
            ("sample", torch.rand(*sample_shape)),
            (
                "timestep",
                torch.tensor([scheduler.timesteps[0].item()] * (batch_size)).to(
                    torch.float32
                ),
            ),
            ("encoder_hidden_states", torch.rand(*encoder_hidden_states_shape)),
        ]
    )
    return sample_unet_inputs


def get_unet_inputs_spec(sample_unet_inputs):
    sample_unet_inputs_spec = {
        k: (v.shape, v.dtype) for k, v in sample_unet_inputs.items()
    }
    return sample_unet_inputs_spec


def add_cnet_support(sample_shape, reference_unet):
    from python_coreml_stable_diffusion.unet import calculate_conv2d_output_shape

    additional_residuals_shapes = []

    batch_size = sample_shape[0]
    h, w = sample_shape[2:]

    # conv_in
    out_h, out_w = calculate_conv2d_output_shape(
        h,
        w,
        reference_unet.conv_in,
    )
    additional_residuals_shapes.append(
        (batch_size, reference_unet.conv_in.out_channels, out_h, out_w)
    )

    # down_blocks
    for down_block in reference_unet.down_blocks:
        additional_residuals_shapes += [
            (batch_size, resnet.out_channels, out_h, out_w)
            for resnet in down_block.resnets
        ]
        if hasattr(down_block, "downsamplers") and down_block.downsamplers is not None:
            for downsampler in down_block.downsamplers:
                out_h, out_w = calculate_conv2d_output_shape(
                    out_h, out_w, downsampler.conv
                )
            additional_residuals_shapes.append(
                (
                    batch_size,
                    down_block.downsamplers[-1].conv.out_channels,
                    out_h,
                    out_w,
                )
            )

    # mid_block
    additional_residuals_shapes.append(
        (batch_size, reference_unet.mid_block.resnets[-1].out_channels, out_h, out_w)
    )

    additional_inputs = {}
    for i, shape in enumerate(additional_residuals_shapes):
        sample_residual_input = torch.rand(*shape)
        additional_inputs[f"additional_residual_{i}"] = sample_residual_input

    return additional_inputs


def convert(
    out_path: str,
    batch_size: int = 1,
    sample_size: tuple[int, int] = (64, 64),
    controlnet_support: bool = False,
):
    coreml_unet, ref_unet = get_unets()

    sample_shape = (
        batch_size,  # B
        ref_unet.config.in_channels,  # C
        sample_size[0],  # H
        sample_size[1],  # W
    )

    encoder_hidden_states_shape = get_encoder_hidden_states_shape(
        ref_unet.config, batch_size
    )

    scheduler = get_scheduler()

    sample_inputs = get_sample_input(
        batch_size, encoder_hidden_states_shape, sample_shape, scheduler
    )

    if controlnet_support:
        sample_inputs |= add_cnet_support(sample_shape, ref_unet)

    sample_inputs_spec = get_unet_inputs_spec(sample_inputs)

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


def compile_model(out_path, out_name):
    # Compile the model
    target_path = compile_coreml_model(
        out_path, get_folder_paths("unet")[0], f"{out_name}_unet"
    )
    logger.info(f"Compiled {out_path} to {target_path}")
    return target_path


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
    compile_model(out_path=out_path, out_name=out_name)
