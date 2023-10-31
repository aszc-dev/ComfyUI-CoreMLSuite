from itertools import chain
from math import ceil

import numpy as np
import torch

from coreml_suite.latents import chunk_batch
from coreml_suite.logger import logger


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
    if "additional_residual_0" not in model.expected_inputs.keys():
        return {}
    if control is None:
        return no_control(model)

    residual_kwargs = {
        "additional_residual_{}".format(i): r.cpu().numpy().astype(np.float16)
        for i, r in enumerate(chain(control["output"], control["middle"]))
    }
    return residual_kwargs


def no_control(model):
    expected = model.expected_inputs
    residual_kwargs = {
        k: torch.zeros(*expected[k]["shape"]).cpu().numpy().astype(dtype=np.float16)
        for k in model.expected_inputs.keys()
        if k.startswith("additional_residual")
    }
    return residual_kwargs


def chunk_control(cn, target_size):
    if cn is None:
        return [None] * target_size

    num_chunks = ceil(cn["output"][0].shape[0] / target_size)

    out = [{"output": [], "middle": []} for _ in range(num_chunks)]

    for k, v in cn.items():
        for i, x in enumerate(v):
            chunks = chunk_batch(x, (target_size, *x.shape[1:]))
            for j, chunk in enumerate(chunks):
                out[j][k].append(chunk)

    return out
