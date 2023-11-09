import numpy as np
import torch

from coreml_suite.controlnet import extract_residual_kwargs, chunk_control
from coreml_suite.latents import chunk_batch, merge_chunks


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

        if self.ts_cond is not None:
            model_input_kwargs["timestep_cond"] = (
                self.ts_cond.cpu().numpy().astype(np.float16)
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

        return [
            CoreMLInputs(x, t, context, control, timestep_cond=ts_cond)
            for x, t, context, control, ts_cond in zip(
                chunked_x, ts, chunked_context, chunked_control, chunked_ts_cond
            )
        ]
