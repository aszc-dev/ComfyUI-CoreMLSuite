import numpy as np
import torch

from comfy import model_base
from comfy.model_management import get_torch_device
from comfy.model_patcher import ModelPatcher
from coreml_suite.config import get_model_config
from coreml_suite.controlnet import extract_residual_kwargs, chunk_control
from coreml_suite.latents import chunk_batch, merge_chunks
from coreml_suite.logger import logger


class CoreMLModelWrapper:
    def __init__(self, coreml_model):
        self.coreml_model = coreml_model
        self.dtype = torch.float16

    def __call__(self, x, t, context, control, transformer_options=None, **kwargs):
        inputs = CoreMLInputs(x, t, context, control, **kwargs)
        input_list = inputs.chunks(self.expected_inputs)

        chunked_out = [
            self.get_torch_outputs(
                self.coreml_model(**input_kwargs.coreml_kwargs(self.expected_inputs)),
                x.device,
            )
            for input_kwargs in input_list
        ]
        merged_out = merge_chunks(chunked_out, x.shape)

        return merged_out

    @staticmethod
    def get_torch_outputs(model_output, device):
        return torch.from_numpy(model_output["noise_pred"]).to(device)

    @property
    def expected_inputs(self):
        return self.coreml_model.expected_inputs


class CoreMLModelWrapperLCM(CoreMLModelWrapper):
    def __init__(self, coreml_model):
        super().__init__(coreml_model)
        self.config = None


class CoreMLInputs:
    def __init__(self, x, t, context, control, **kwargs):
        self.x = x
        self.t = t
        self.context = context
        self.control = control
        self.time_ids = kwargs.get("time_ids")
        self.text_embeds = kwargs.get("text_embeds")
        self.ts_cond = kwargs.get("timestep_cond")

    def coreml_kwargs(self, expected_inputs):
        sample = self.x.cpu().numpy().astype(np.float16)

        context = self.context.cpu().numpy().astype(np.float16)
        context = context.transpose(0, 2, 1)[:, :, None, :]

        t = self.t.cpu().numpy().astype(np.float16)

        model_input_kwargs = {
            "sample": sample,
            "encoder_hidden_states": context,
            "timestep": t,
        }
        residual_kwargs = extract_residual_kwargs(expected_inputs, self.control)
        model_input_kwargs |= residual_kwargs

        # LCM
        if self.ts_cond is not None:
            model_input_kwargs["timestep_cond"] = (
                self.ts_cond.cpu().numpy().astype(np.float16)
            )

        # SDXL
        if "text_embeds" in expected_inputs:
            model_input_kwargs["text_embeds"] = (
                self.text_embeds.cpu().numpy().astype(np.float16)
            )
        if "time_ids" in expected_inputs:
            model_input_kwargs["time_ids"] = (
                self.time_ids.cpu().numpy().astype(np.float16)
            )

        return model_input_kwargs

    def chunks(self, expected_inputs):
        sample_shape = expected_inputs["sample"]["shape"]
        timestep_shape = expected_inputs["timestep"]["shape"]
        hidden_shape = expected_inputs["encoder_hidden_states"]["shape"]
        context_shape = (hidden_shape[0], hidden_shape[3], hidden_shape[1])

        chunked_x = chunk_batch(self.x, sample_shape)
        ts = list(torch.full((len(chunked_x), timestep_shape[0]), self.t[0]))
        chunked_context = chunk_batch(self.context, context_shape)

        chunked_control = [None] * len(chunked_x)
        if self.control is not None:
            chunked_control = chunk_control(self.control, sample_shape[0])

        chunked_ts_cond = [None] * len(chunked_x)
        if self.ts_cond is not None:
            ts_cond_shape = expected_inputs["timestep_cond"]["shape"]
            chunked_ts_cond = chunk_batch(self.ts_cond, ts_cond_shape)

        chunked_time_ids = [None] * len(chunked_x)
        if expected_inputs["time_ids"] is not None:
            time_ids_shape = expected_inputs["time_ids"]["shape"]
            if self.time_ids is None:
                self.time_ids = torch.zeros(len(chunked_x), *time_ids_shape[1:]).to(
                    self.x.device
                )
            chunked_time_ids = chunk_batch(self.time_ids, time_ids_shape)

        chunked_text_embeds = [None] * len(chunked_x)
        if expected_inputs["text_embeds"] is not None:
            text_embeds_shape = expected_inputs["text_embeds"]["shape"]
            if self.text_embeds is None:
                self.text_embeds = torch.zeros(
                    len(chunked_x), *text_embeds_shape[1:]
                ).to(self.x.device)
            chunked_text_embeds = chunk_batch(self.text_embeds, text_embeds_shape)

        return [
            CoreMLInputs(
                x,
                t,
                context,
                control,
                timestep_cond=ts_cond,
                time_ids=time_ids,
                text_embeds=text_embeds,
            )
            for x, t, context, control, ts_cond, time_ids, text_embeds in zip(
                chunked_x,
                ts,
                chunked_context,
                chunked_control,
                chunked_ts_cond,
                chunked_time_ids,
                chunked_text_embeds,
            )
        ]


def is_sdxl(coreml_model):
    return (
        "time_ids" in coreml_model.expected_inputs
        and "text_embeds" in coreml_model.expected_inputs
    )


def sdxl_model_function_wrapper(time_ids, text_embeds):
    def wrapper(model_function, params):
        x = params["input"]
        t = params["timestep"]
        c = params["c"]

        context = c.get("c_crossattn")

        if context is None:
            return torch.zeros_like(x)

        return model_function(x, t, **c, time_ids=time_ids, text_embeds=text_embeds)

    return wrapper


def add_sdxl_model_options(model_patcher, positive, negative):
    mp = model_patcher.clone()

    pos_dict = positive[0][1]
    neg_dict = negative[0][1]

    pos_time_ids = [
        pos_dict.get("height", 768),
        pos_dict.get("width", 768),
        pos_dict.get("crop_h", 0),
        pos_dict.get("crop_w", 0),
    ]

    neg_time_ids = [
        neg_dict.get("height", 768),
        neg_dict.get("width", 768),
        neg_dict.get("crop_h", 0),
        neg_dict.get("crop_w", 0),
    ]

    if model_patcher.model.diffusion_model.expected_inputs["time_ids"]["shape"][1] == 6:
        base_pos_time_ids = [
            pos_dict.get("target_height", 768),
            pos_dict.get("target_width", 768),
        ]
        pos_time_ids += base_pos_time_ids

        base_neg_time_ids = [
            neg_dict.get("target_height", 768),
            neg_dict.get("target_width", 768),
        ]
        neg_time_ids += base_neg_time_ids

    else:
        refiner_pos_time_ids = [
            pos_dict.get("aesthetic_score", 6),
        ]
        pos_time_ids += refiner_pos_time_ids

        refiner_neg_time_ids = [
            neg_dict.get("aesthetic_score", 2.5),
        ]
        neg_time_ids += refiner_neg_time_ids

    time_ids = torch.tensor([pos_time_ids, neg_time_ids])

    text_embeds = torch.cat((pos_dict["pooled_output"], neg_dict["pooled_output"]))

    model_options = {
        "model_function_wrapper": sdxl_model_function_wrapper(time_ids, text_embeds),
    }
    mp.model_options |= model_options

    return mp


def get_latent_image(coreml_model, latent_image):
    if latent_image is not None:
        return latent_image

    logger.warning("No latent image provided, using empty tensor.")
    expected = coreml_model.expected_inputs["sample"]["shape"]
    batch_size = max(expected[0] // 2, 1)
    latent_image = {"samples": torch.zeros(batch_size, *expected[1:])}
    return latent_image


def get_model_patcher(coreml_model):
    model_config = get_model_config()
    wrapped_model = CoreMLModelWrapper(coreml_model)

    if is_sdxl(coreml_model):
        model = model_base.SDXL(model_config, device=get_torch_device())
    else:
        model = model_base.BaseModel(model_config, device=get_torch_device())

    model.diffusion_model = wrapped_model
    model_patcher = ModelPatcher(model, get_torch_device(), None)
    return model_patcher
