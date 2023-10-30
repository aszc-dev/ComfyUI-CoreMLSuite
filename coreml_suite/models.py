import numpy as np
import torch


from comfy import supported_models_base
from comfy.latent_formats import SD15
from comfy.model_base import BaseModel

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


class CoreMLModelWrapper(BaseModel):
    def __init__(self, model_config, coreml_model):
        super().__init__(model_config)
        self.diffusion_model = coreml_model

    def apply_model(
        self,
        x,
        t,
        c_concat=None,
        c_crossattn=None,
        c_adm=None,
        control=None,
        transformer_options={},
    ):
        sample_shape = self.diffusion_model.expected_inputs["sample"]["shape"]

        chunked_x = chunk_batch(x, sample_shape)
        ts = t.chunk(len(chunked_x), dim=0)
        chunked_context = c_crossattn.chunk(len(chunked_x), dim=0)
        chunked_control = chunk_control(control, len(chunked_x))

        chunked_out = [
            self._apply_model(
                x, t, c_concat, c_crossattn, c_adm, control, transformer_options
            )
            for x, t, c_crossattn, control in zip(
                chunked_x, ts, chunked_context, chunked_control
            )
        ]

        merged_out = merge_chunks(chunked_out, x.shape)
        return merged_out

    def get_dtype(self):
        # Hardcoding torch-compatible dtype (used for memory allocation)
        return torch.float16

    def _apply_model(
        self,
        x,
        t,
        c_concat=None,
        c_crossattn=None,
        c_adm=None,
        control=None,
        transformer_options={},
    ):
        sample = x.cpu().numpy().astype(np.float16)

        context = c_crossattn.cpu().numpy().astype(np.float16)
        context = context.transpose(0, 2, 1)[:, :, None, :]

        t = t.cpu().numpy().astype(np.float16)

        model_input_kwargs = {
            "sample": sample,
            "encoder_hidden_states": context,
            "timestep": t,
        }
        residual_kwargs = extract_residual_kwargs(self.diffusion_model, control)
        model_input_kwargs |= residual_kwargs
        # model_input_kwargs = expand_inputs(model_input_kwargs)

        np_out = self.diffusion_model(**model_input_kwargs)["noise_pred"]
        return torch.from_numpy(np_out).to(x.device)

    @property
    def expected_inputs(self):
        return self.diffusion_model.expected_inputs
