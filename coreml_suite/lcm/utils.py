import torch

from comfy.model_management import get_torch_device
from comfy_extras.nodes_model_advanced import ModelSamplingDiscreteDistilled, LCM


def is_lcm(coreml_model):
    return "timestep_cond" in coreml_model.expected_inputs


def get_w_embedding(w, embedding_dim=512, dtype=torch.float32):
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


def model_function_wrapper(w_embedding):
    def wrapper(model_function, params):
        x = params["input"]
        t = params["timestep"]
        c = params["c"]

        context = c.get("c_crossattn")

        if context is None:
            return torch.zeros_like(x)

        return model_function(x, t, **c, timestep_cond=w_embedding)

    return wrapper


def lcm_patch(model):
    m = model.clone()
    sampling_type = LCM
    sampling_base = ModelSamplingDiscreteDistilled

    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass

    model_sampling = ModelSamplingAdvanced()
    m.add_object_patch("model_sampling", model_sampling)

    return m


def add_lcm_model_options(model_patcher, cfg, latent_image):
    mp = model_patcher.clone()

    latent = latent_image["samples"].to(get_torch_device())
    batch_size = latent.shape[0]
    dtype = latent.dtype
    device = get_torch_device()

    w = torch.tensor(cfg).repeat(batch_size)
    w_embedding = get_w_embedding(w, embedding_dim=256).to(device=device, dtype=dtype)

    model_options = {
        "model_function_wrapper": model_function_wrapper(w_embedding),
        "sampler_cfg_function": lambda x: x["cond"].to(device),
    }
    mp.model_options |= model_options

    return mp
