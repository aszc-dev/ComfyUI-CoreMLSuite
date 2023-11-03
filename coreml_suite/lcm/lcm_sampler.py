import os

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

import latent_preview
from comfy.model_management import get_torch_device
from comfy.model_patcher import ModelPatcher
from coreml_suite.lcm.lcm_scheduler import LCMScheduler
from coreml_suite.logger import logger
from coreml_suite.models import get_model_config, CoreMLModelWrapperLCM
from coreml_suite.nodes import CoreMLSampler


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
            os.path.join(os.path.dirname(__file__), "scheduler_config.json")
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
        wrapped_model = CoreMLModelWrapperLCM(model_config, coreml_model)
        patched_model = ModelPatcher(wrapped_model, get_torch_device(), None)

        if latent_image is None:
            logger.warning("No latent image provided, using empty tensor.")
            expected = coreml_model.expected_inputs["sample"]["shape"]
            latent_image = {"samples": torch.zeros(*expected).to(get_torch_device())}

        positive = positive[0][0]

        callback = latent_preview.prepare_callback(patched_model, steps, None)
        torch.manual_seed(seed)

        return self._sample(
            patched_model, steps, cfg, positive, latent_image, denoise, callback
        )

    def _sample(
        self, model, steps, cfg, positive, latent_image, denoise, callback=None
    ):
        device = get_torch_device()
        batch_size = latent_image["samples"].shape[0]

        prompt_embeds = self.prepare_prompt_embeds(batch_size, positive)

        timesteps = self.prepare_timesteps(denoise, device, steps)

        latents = self.prepare_latents(latent_image, device)

        w = torch.tensor(cfg).repeat(batch_size)
        w_embedding = self.get_w_embedding(w, embedding_dim=256).to(
            device=device, dtype=latents.dtype
        )

        # LCM MultiStep Sampling Loop:
        iterator = tqdm(timesteps, desc="Core ML LCM Sampler", total=steps)
        for i, t in enumerate(iterator):
            ts = torch.full((batch_size,), t, device=device, dtype=torch.float16)

            model_pred = model.model(
                latents,
                ts,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=w_embedding,
            )[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents, denoised = self.scheduler.step(
                model_pred, i, t, latents, return_dict=False
            )

            if callback:
                callback(i, denoised, latents, steps)

        denoised = denoised.to(get_torch_device())

        return ({"samples": denoised / 0.1825},)

    def prepare_prompt_embeds(self, batch_size, positive):
        bs_embed, seq_len, _ = positive.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = positive.repeat(1, batch_size, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * batch_size, seq_len, -1)
        return prompt_embeds

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
        timesteps = self.scheduler.timesteps
        return timesteps

    def prepare_latents(self, latent_image, device):
        latent = latent_image["samples"].to(device) * 0.1825
        latent = latent.to(torch.float16)

        if not torch.any(latent):
            latents = torch.randn(latent.shape, dtype=torch.float16).to(device)
            latents *= self.scheduler.init_noise_sigma
            return latents

        batch_size = latent.shape[0]

        burned = randn_tensor(latent.shape, device=device, dtype=torch.float16)
        noise = randn_tensor(latent.shape, device=device, dtype=torch.float16)

        latent_timestep = self.scheduler.timesteps[:1].repeat(batch_size)
        latents = self.scheduler.add_noise(latent, noise, latent_timestep)

        return latents

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
