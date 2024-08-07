import gc
import os
import shutil
import time
from typing import Union

import coremltools as ct
import numpy as np
import python_coreml_stable_diffusion.unet
import torch
from diffusers import (
    StableDiffusionPipeline,
    LatentConsistencyModelPipeline,
    StableDiffusionXLPipeline,
)
from python_coreml_stable_diffusion.unet import (
    UNet2DConditionModel,
    UNet2DConditionModelXL,
    AttentionImplementations,
)

from coreml_suite.config import ModelVersion
from coreml_suite.lcm.unet import UNet2DConditionModelLCM
from coreml_suite.logger import logger
from folder_paths import get_folder_paths


class StableDiffusionLCMPipeline(LatentConsistencyModelPipeline):
    pass


MODEL_TYPE_TO_UNET_CLS = {
    ModelVersion.SD15: UNet2DConditionModel,
    ModelVersion.SDXL: UNet2DConditionModelXL,
    ModelVersion.LCM: UNet2DConditionModelLCM,
}

MODEL_TYPE_TO_PIPE_CLS = {
    ModelVersion.SD15: StableDiffusionPipeline,
    ModelVersion.SDXL: StableDiffusionXLPipeline,
    ModelVersion.LCM: StableDiffusionLCMPipeline,
}


def get_unet(model_type: ModelVersion, ref_pipe):
    ref_unet = ref_pipe.unet

    unet_cls = MODEL_TYPE_TO_UNET_CLS[model_type]
    cml_unet = unet_cls.from_config(ref_unet.config).eval()
    cml_unet.load_state_dict(ref_unet.state_dict(), strict=False)

    return cml_unet


def get_encoder_hidden_states_shape(ref_pipe, batch_size):
    text_encoder = (
        ref_pipe.text_encoder_2
        if hasattr(ref_pipe, "text_encoder_2")
        else ref_pipe.text_encoder
    )

    text_token_sequence_length = text_encoder.config.max_position_embeddings
    hidden_size = (text_encoder.config.hidden_size,)

    encoder_hidden_states_shape = (
        batch_size,
        ref_pipe.unet.config.cross_attention_dim or hidden_size,
        1,
        text_token_sequence_length,
    )

    return encoder_hidden_states_shape


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
                torch.tensor([scheduler.timesteps[0].item()] * batch_size).to(
                    torch.float32
                ),
            ),
            ("encoder_hidden_states", torch.rand(*encoder_hidden_states_shape)),
        ]
    )
    return sample_unet_inputs


def lcm_inputs(sample_unet_inputs):
    batch_size = sample_unet_inputs["sample"].shape[0]
    return {"timestep_cond": torch.randn(batch_size, 256).to(torch.float32)}


def sdxl_inputs(sample_unet_inputs, ref_pipe):
    sample_shape = sample_unet_inputs["sample"].shape
    batch_size = sample_shape[0]
    h = sample_shape[2] * 8
    w = sample_shape[3] * 8
    original_size = (h, w)
    crops_coords_top_left = (0, 0)

    is_refiner = (
        hasattr(ref_pipe.config, "requires_aesthetics_score")
        and ref_pipe.config.requires_aesthetics_score
    )

    if is_refiner:
        aesthetic_score = (6.0,)
        time_ids_list = list(original_size + crops_coords_top_left + aesthetic_score)
    else:
        target_size = (h, w)
        time_ids_list = list(original_size + crops_coords_top_left + target_size)

    time_ids = torch.tensor(time_ids_list).repeat(batch_size, 1).to(torch.int64)
    text_embeds_shape = (batch_size, ref_pipe.text_encoder_2.config.hidden_size)

    return {
        "time_ids": time_ids,
        "text_embeds": torch.randn(*text_embeds_shape).to(torch.float32),
    }


def get_inputs_spec(inputs):
    inputs_spec = {k: (v.shape, v.dtype) for k, v in inputs.items()}
    return inputs_spec


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


def convert_unet(
    ref_pipe,
    model_version: ModelVersion,
    unet_out_path: str,
    batch_size: int = 1,
    sample_size: tuple[int, int] = (64, 64),
    controlnet_support: bool = False,
):
    coreml_unet = get_unet(model_version, ref_pipe)
    ref_unet = ref_pipe.unet

    sample_shape = (
        batch_size,  # B
        ref_unet.config.in_channels,  # C
        sample_size[0],  # H
        sample_size[1],  # W
    )

    encoder_hidden_states_shape = get_encoder_hidden_states_shape(ref_pipe, batch_size)

    scheduler = ref_pipe.scheduler
    scheduler.set_timesteps(50)

    sample_inputs = get_sample_input(
        batch_size, encoder_hidden_states_shape, sample_shape, scheduler
    )

    if model_version == ModelVersion.LCM:
        sample_inputs |= lcm_inputs(sample_inputs)

    if model_version == ModelVersion.SDXL:
        sample_inputs |= sdxl_inputs(sample_inputs, ref_pipe)

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
        "unet", traced_unet, coreml_sample_inputs, ["noise_pred"], unet_out_path
    )

    del traced_unet
    gc.collect()

    coreml_unet.save(unet_out_path)
    logger.info(f"Saved unet into {unet_out_path}")


def convert(
    ckpt_path: str,
    model_version: ModelVersion,
    unet_out_path: str,
    batch_size: int = 1,
    sample_size: tuple[int, int] = (64, 64),
    controlnet_support: bool = False,
    lora_weights: list[tuple[Union[str, os.PathLike], float]] = None,
    attn_impl: str = AttentionImplementations.SPLIT_EINSUM.name,
    config_path: str = None,
):
    if os.path.exists(unet_out_path):
        logger.info(f"Found existing model at {unet_out_path}! Skipping..")
        return

    python_coreml_stable_diffusion.unet.ATTENTION_IMPLEMENTATION_IN_EFFECT = (
        AttentionImplementations(attn_impl)
    )

    ref_pipe = get_pipeline(ckpt_path, config_path, model_version)

    for i, lora_weight in enumerate(lora_weights or []):
        lora_path, strength = lora_weight
        adapter_name = f"lora_{i}"
        ref_pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        ref_pipe.set_adapters([adapter_name], adapter_weights=[strength])
        ref_pipe.fuse_lora()

    convert_unet(
        ref_pipe,
        model_version,
        unet_out_path,
        batch_size,
        sample_size,
        controlnet_support,
    )


def get_pipeline(ckpt_path, config_path, model_version):
    pipe_cls = MODEL_TYPE_TO_PIPE_CLS[model_version]
    ref_pipe = pipe_cls.from_single_file(ckpt_path, original_config_file=config_path)
    return ref_pipe


def compile_model(out_path, out_name, submodule_name):
    # Compile the model
    target_path = compile_coreml_model(
        out_path, get_folder_paths(submodule_name)[0], f"{out_name}_{submodule_name}"
    )
    logger.info(f"Compiled {out_path} to {target_path}")
    return target_path
