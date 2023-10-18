from itertools import chain

import numpy as np
import torch

from .logger import logger


def expand_inputs(inputs):
    expanded = inputs.copy()
    for k, v in inputs.items():
        if isinstance(v, np.ndarray):
            expanded[k] = np.concatenate([v] * 2) if v.shape[0] == 1 else v
        elif isinstance(v, torch.Tensor):
            expanded[k] = torch.cat([v] * 2) if v.shape[0] == 1 else v
        elif isinstance(v, list):
            expanded[k] = v * 2 if len(v) == 1 else v
        elif isinstance(v, dict):
            expand_inputs(v)
    return expanded


def extract_residual_kwargs(model, control):
    if ("additional_residual_0" not in model.expected_inputs.keys()):
        return {}
    if control is None:
        return no_control(model)

    residual_kwargs = {
        "additional_residual_{}".format(i): r.cpu().numpy().astype(
            np.float16)
        for i, r in
        enumerate(chain(control["output"], control["middle"]))
    }
    return residual_kwargs


def no_control(model):
    # Dirty hack to get the expected input shape when doing partial ControlNet
    # 0.18215 is the latent scale factor (IDK, it kinda works)
    # TODO: Find a better way to do this or tweak the values

    logger.warning("No ControlNet input, despite the model supports it. "
                   "Using random noise as ControlNet residuals. "
                   "For better results, please use a ControlNet or a model "
                   "that does not support ControlNet.")
    residuals_names = [name for name in model.expected_inputs.keys()
                       if name.startswith("additional_residual")]
    residual_kwargs = {
        "additional_residual_{}".format(i): 0.18215 * torch.randn(
            *model.expected_inputs["additional_residual_{}".format(i)][
                "shape"]).cpu().numpy().astype(dtype=np.float16)
        for i in range(len(residuals_names))
    }
    return residual_kwargs
