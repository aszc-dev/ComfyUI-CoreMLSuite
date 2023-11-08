import numpy as np
import torch

from comfy import supported_models_base
from comfy.latent_formats import SD15

from coreml_suite.controlnet import extract_residual_kwargs, chunk_control
from coreml_suite.latents import chunk_batch, merge_chunks


def get_model_config():
    # TODO: This is a dummy model config, but it should be enough to
    #  get the model to load - implement a proper model config
    model_config = supported_models_base.BASE({})
    model_config.latent_format = SD15()
    model_config.unet_config = {
        "disable_unet_model_creation": True,
        "num_res_blocks": 2,
        "attention_resolutions": [1, 2, 4],
        "channel_mult": [1, 2, 4, 4],
        "transformer_depth": [1, 1, 1, 0],
    }
    return model_config


class CoreMLModelWrapper:
    def __init__(self, coreml_model):
        self.coreml_model = coreml_model
        self.dtype = torch.float16

    def __call__(self, x, t, context, control, transformer_options, **kwargs):
        chunked_in = self.chunk_inputs(
            x, t, context, control, kwargs.get("timestep_cond")
        )
        input_list = [
            self.get_np_input_kwargs(*chunked) for chunked in zip(*chunked_in)
        ]

        chunked_out = [
            self.get_torch_outputs(self.coreml_model(**input_kwargs), x.device)
            for input_kwargs in input_list
        ]
        merged_out = merge_chunks(chunked_out, x.shape)

        return merged_out

    def get_np_input_kwargs(self, x, t, context, control, ts_cond=None):
        sample = x.cpu().numpy().astype(np.float16)

        context = context.cpu().numpy().astype(np.float16)
        context = context.transpose(0, 2, 1)[:, :, None, :]

        t = t.cpu().numpy().astype(np.float16)

        model_input_kwargs = {
            "sample": sample,
            "encoder_hidden_states": context,
            "timestep": t,
        }
        residual_kwargs = extract_residual_kwargs(self.coreml_model, control)
        model_input_kwargs |= residual_kwargs

        if ts_cond is not None:
            model_input_kwargs["timestep_cond"] = (
                ts_cond.cpu().numpy().astype(np.float16)
            )

        return model_input_kwargs

    def chunk_inputs(self, x, t, context, control, ts_cond=None):
        sample_shape = self.expected_inputs["sample"]["shape"]
        timestep_shape = self.expected_inputs["timestep"]["shape"]
        hidden_shape = self.expected_inputs["encoder_hidden_states"]["shape"]
        context_shape = (hidden_shape[0], hidden_shape[3], hidden_shape[1])

        chunked_x = chunk_batch(x, sample_shape)
        ts = list(torch.full((len(chunked_x), timestep_shape[0]), t[0]))
        chunked_context = chunk_batch(context, context_shape)

        chunked_control = [None] * len(chunked_x)
        if control is not None:
            chunked_control = chunk_control(control, sample_shape[0])

        chunked_ts_cond = [None] * len(chunked_x)
        if ts_cond is not None:
            ts_cond_shape = self.expected_inputs["timestep_cond"]["shape"]
            chunked_ts_cond = chunk_batch(ts_cond, ts_cond_shape)

        return chunked_x, ts, chunked_context, chunked_control, chunked_ts_cond

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
