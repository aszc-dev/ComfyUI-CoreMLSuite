import numpy as np
import torch

from comfy import samplers
from comfy.model_management import get_torch_device
from comfy.sample import sample_custom, prepare_noise


class CoreMLSamplerLCM:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def sample(
        self,
        model_patcher,
        seed,
        steps,
        cfg,
        positive,
        latent_image,
        denoise=1.0,
        callback=None,
        disable_pbar=False,
    ):
        positive[0][1]["control_apply_to_uncond"] = False

        latent = latent_image["samples"].to(get_torch_device())

        batch_size = latent.shape[0]
        dtype = latent.dtype
        device = get_torch_device()

        w = torch.tensor(cfg).repeat(batch_size)
        w_embedding = self.get_w_embedding(w, embedding_dim=256).to(
            device=device, dtype=dtype
        )

        model_options = {
            "model_function_wrapper": model_function_wrapper(w_embedding),
            "sampler_cfg_function": lambda x: x["cond"].to(device),
        }
        model_patcher.model_options |= model_options

        self.prepare_timesteps(denoise, device, steps)
        all_sigmas, sigmas = self.get_sigmas(steps, denoise)

        model_patcher.model.model_sampling.set_sigmas(all_sigmas)
        sigma_to_timestep = {
            s.item(): t for s, t in zip(sigmas, self.scheduler.timesteps)
        }
        model_patcher.model.model_sampling.timestep = lambda x: sigma_to_timestep[
            x[0].item()
        ].expand(1)

        noise_mask = latent_image.get("noise_mask")
        batch_inds = latent_image.get("batch_index")
        noise = prepare_noise(latent, seed, batch_inds)

        sampler = samplers.ksampler("ddpm")()
        samples = sample_custom(
            model_patcher,
            noise,
            cfg,
            sampler,
            sigmas,
            positive,
            (),
            latent,
            noise_mask,
            callback,
            disable_pbar,
            seed,
        )
        return samples

    def get_sigmas(self, steps, denoise):
        alphas_cumprod = self.scheduler.alphas_cumprod
        sigmas = np.asarray(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5)
        skipping_step = len(sigmas) // steps
        s = sigmas[::-skipping_step][:steps]
        if len(s) == steps:
            s = np.append(s, 0.0).astype(np.float32)
        sigmas = torch.from_numpy(sigmas).to(get_torch_device())
        return (sigmas, torch.from_numpy(s.copy()).to(get_torch_device()))

    def prepare_timesteps(self, denoise, device, steps):
        lcm_origin_steps = 50
        self.scheduler.num_inference_steps = steps
        c = self.scheduler.config.num_train_timesteps // lcm_origin_steps
        lcm_origin_timesteps = (
            np.asarray(list(range(1, int(lcm_origin_steps * denoise) + 1))) * c - 1
        )
        skipping_step = len(lcm_origin_timesteps) // steps
        timesteps = lcm_origin_timesteps[::-skipping_step][:steps]
        timesteps = torch.from_numpy(timesteps.copy()).to(device)
        self.scheduler.timesteps = timesteps

    def get_w_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        see https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
        Args:
        timesteps: torch.Tensor: generate embedding vectors at these timesteps
        embedding_dim: int: dimension of the embeddings to generate
        dtype: data type of the generated embeddings

        Returns:
        embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
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
