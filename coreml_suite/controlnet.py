from itertools import chain
from math import ceil

import numpy as np
import torch

from coreml_suite.latents import chunk_batch


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


def extract_residual_kwargs(expected_inputs, control):
    if "additional_residual_0" not in expected_inputs.keys():
        return {}
    if control is None:
        return no_control(expected_inputs)

    residual_kwargs = {
        "additional_residual_{}".format(i): r.cpu().numpy().astype(np.float16)
        for i, r in enumerate(chain(control["output"], control["middle"]))
    }
    return residual_kwargs


def no_control(expected_inputs):
    shapes_dict = {
        k: v["shape"] for k, v in expected_inputs.items() if k.startswith("additional")
    }
    residual_kwargs = {
        k: torch.zeros(*shape).cpu().numpy().astype(dtype=np.float16)
        for k, shape in shapes_dict.items()
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
