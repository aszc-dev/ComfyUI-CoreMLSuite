import numpy as np
import torch

from python_coreml_stable_diffusion.coreml_model import CoreMLModel

from comfy.model_base import BaseModel

from .utils import expand_inputs, extract_residual_kwargs


class CoreMLModelWrapper(BaseModel):
    def __init__(self, model_config, mlpackage_path, compute_unit,
                 sources="packages"):
        super().__init__(model_config)
        self.diffusion_model = CoreMLModel(mlpackage_path, compute_unit,
                                           sources)

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, c_adm=None,
                    control=None, transformer_options={}):
        sample = x.cpu().numpy().astype(np.float16)

        context = c_crossattn.cpu().numpy().astype(np.float16)
        context = context.transpose(0, 2, 1)[:, :, None, :]

        t = t.cpu().numpy().astype(np.float16)

        model_input_kwargs = {
            "sample": sample,
            "encoder_hidden_states": context,
            "timestep": t,
        }
        residual_kwargs = extract_residual_kwargs(self.diffusion_model,
                                                  control)
        model_input_kwargs |= residual_kwargs
        model_input_kwargs = expand_inputs(model_input_kwargs)

        np_out = self.diffusion_model(**model_input_kwargs)["noise_pred"]
        return torch.from_numpy(np_out).to(x.device)

    def get_dtype(self):
        # Hardcoding torch-compatible dtype (used for memory allocation)
        return torch.float16
