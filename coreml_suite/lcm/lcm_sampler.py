import numpy as np
import torch

import latent_preview
from comfy import model_base, samplers
from comfy.model_management import get_torch_device
from comfy.model_patcher import ModelPatcher
from comfy.sample import sample_custom, prepare_noise

from coreml_suite.lcm.lcm_scheduler import LCMScheduler
from coreml_suite.logger import logger
from coreml_suite.models import CoreMLModelWrapper
from coreml_suite.nodes import CoreMLSampler
from coreml_suite.config import get_model_config


class CoreMLSamplerLCM(CoreMLSampler):
    @classmethod
    def INPUT_TYPES(s):
        old_required = CoreMLSampler.INPUT_TYPES()["required"].copy()
        old_required["steps"][1]["default"] = 4
        old_required.pop("negative")
        old_required.pop("sampler_name")
        old_required.pop("scheduler")
        new_required = {"coreml_model": ("COREML_UNET",)}
        return {
            "required": new_required | old_required,
            "optional": {"latent_image": ("LATENT",)},
        }

    CATEGORY = "Core ML Suite"

    def __init__(self):
        self.scheduler = LCMScheduler.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7", subfolder="scheduler"
        )

    def sample(
        self,
        coreml_model,
        seed,
        steps,
        cfg,
        positive,
        latent_image=None,
        denoise=1.0,
        **kwargs,
    ):
        model_config = get_model_config()
        wrapped_model = CoreMLModelWrapper(coreml_model)
        model = model_base.BaseModel(model_config, device=get_torch_device())
        model.diffusion_model = wrapped_model
        model_patcher = ModelPatcher(model, get_torch_device(), None)

        positive[0][1]["control_apply_to_uncond"] = False

        if latent_image is None:
            logger.warning("No latent image provided, using empty tensor.")
            expected = coreml_model.expected_inputs["sample"]["shape"]
            latent_image = {"samples": torch.zeros(*expected).to(get_torch_device())}

        latent = latent_image["samples"].to(get_torch_device())

        x0_output = {}
        callback = latent_preview.prepare_callback(model_patcher, steps, x0_output)

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

        batch_inds = latent_image.get("batch_index")
        noise = prepare_noise(latent, seed, batch_inds)

        self.prepare_timesteps(denoise, device, steps)
        # self.scheduler.set_timesteps(steps, 50, device)

        all_sigmas, sigmas = self.get_sigmas(steps, denoise)
        model_patcher.model.model_sampling.set_sigmas(all_sigmas)
        sigma_to_timestep = {
            s.item(): t for s, t in zip(sigmas, self.scheduler.timesteps)
        }
        model_patcher.model.model_sampling.timestep = lambda x: sigma_to_timestep[
            x[0].item()
        ].expand(1)

        noise_mask = latent_image.get("noise_mask")

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
        )

        out = latent_image.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent_image.copy()
            out_denoised["samples"] = model.process_latent_out(
                x0_output["x0"].to(device)
            )
        else:
            out_denoised = out
        return (out, out_denoised)

    def get_sigmas(self, steps):
        alphas_cumprod = self.scheduler.alphas_cumprod
        sigmas = np.asarray(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5)
        skipping_step = len(sigmas) // steps
        s = sigmas[::-skipping_step][:steps]
        if len(s) == steps:
            s = np.append(s, sigmas[0])
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
