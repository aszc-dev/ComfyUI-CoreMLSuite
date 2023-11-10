import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

import latent_preview
from comfy import model_base, samplers
from comfy.model_management import get_torch_device
from comfy.model_patcher import ModelPatcher
from comfy.sample import sample_custom, prepare_noise
from comfy.samplers import (
    sampling_function,
    Sampler,
    KSamplerX0Inpaint,
)
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

        # sampler = LCMSampler(self.scheduler)
        sampler = samplers.ksampler("ddpm")()

        all_sigmas, sigmas = self.get_sigmas(steps)
        model_patcher.model.model_sampling.set_sigmas(all_sigmas)
        sigma_to_timestep = {
            s.item(): t for s, t in zip(sigmas, self.scheduler.timesteps)
        }
        model_patcher.model.model_sampling.timestep = lambda x: sigma_to_timestep[
            x[0].item()
        ].expand(1)

        noise_mask = latent_image.get("noise_mask")

        positive[0][1]["control_apply_to_uncond"] = False

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

        #
        # model, positive, _, _, _ = prepare_sampling(
        #     model_patcher, latent_image["samples"].shape, positive, (), None
        # )
        #
        # pre_run_control(model, positive)
        #
        # return self._sample(
        #     model, steps, cfg, positive, latent_image, denoise, callback
        # )

    def get_sigmas(self, steps):
        alphas_cumprod = self.scheduler.alphas_cumprod
        sigmas = np.asarray(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5)
        skipping_step = len(sigmas) // steps
        s = sigmas[::-skipping_step][:steps]
        if len(s) == steps:
            s = np.append(s, sigmas[0])
        sigmas = torch.from_numpy(sigmas).to(get_torch_device())
        return (sigmas, torch.from_numpy(s.copy()).to(get_torch_device()))

    def _sample(
        self, model, steps, cfg, positive, latent_image, denoise, callback=None
    ):
        device = get_torch_device()
        batch_size = latent_image["samples"].shape[0]

        timesteps = self.prepare_timesteps(denoise, device, steps)

        latents = self.prepare_latents(latent_image, device)

        # LCM MultiStep Sampling Loop:
        iterator = tqdm(timesteps, desc="Core ML LCM Sampler", total=steps)
        for i, t in enumerate(iterator):
            ts = torch.full((batch_size,), t, device=device, dtype=torch.float16)

            model_pred = sampling_function(
                model.diffusion_model,
                latents,
                ts,
                None,
                positive,
                denoise,
                model_options,
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents, denoised = self.scheduler.step(
                model_pred, i, t, latents, return_dict=False
            )

            if callback:
                callback(i, denoised.float(), latents, steps)

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


class LCMSampler(Sampler):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        extra_args["denoise_mask"] = denoise_mask
        model_k = KSamplerX0Inpaint(model_wrap)
        model_k.latent_image = latent_image
        model_k.noise = noise

        if self.max_denoise(model_wrap, sigmas):
            noise = noise * torch.sqrt(1.0 + sigmas[0] ** 2.0)
        else:
            noise = noise * sigmas[0]

        k_callback = None
        total_steps = len(sigmas) - 1

        if latent_image is not None:
            noise += latent_image

        samples = self._sample(model_k, noise, sigmas, extra_args, callback)
        return samples

    def _sample(self, model, noise, sigmas, extra_args, callback):
        batch_size = noise.shape[0]
        model_options = extra_args["model_options"]
        positive = extra_args["cond"]
        cond_scale = extra_args["cond_scale"]
        denoise_mask = extra_args["denoise_mask"]

        timesteps = self.scheduler.timesteps
        sample = noise

        # LCM MultiStep Sampling Loop:
        iterator = tqdm(timesteps, desc="Core ML LCM Sampler", total=len(timesteps))
        for i, t in enumerate(iterator):
            ss = torch.full(
                (batch_size,), sigmas[i], device=t.device, dtype=sigmas.dtype
            )
            # 1. get previous step value
            prev_timeindex = i + 1
            if prev_timeindex < len(timesteps):
                prev_timestep = timesteps[prev_timeindex]
            else:
                prev_timestep = t

            # 2. compute alphas, betas
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )

            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            # 3. Get scalings for boundary conditions
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(
                t
            )

            model_output = model(
                sample, ss, None, positive, cond_scale, denoise_mask, model_options
            )
            pred_x0 = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()

            # 4. Denoise model output using boundary conditions
            denoised = c_out * pred_x0 + c_skip * sample

            # 5. Sample z ~ N(0, I), For MultiStep Inference
            # Noise is not used for one-step sampling.
            if len(timesteps) > 1:
                noise = torch.randn(sample.shape).to(sample.device)
                sample = (
                    alpha_prod_t_prev.sqrt() * denoised
                    + beta_prod_t_prev.sqrt() * noise
                )
            else:
                sample = denoised

            # # # compute the previous noisy sample x_t -> x_t-1
            # noise, denoised = self.scheduler.step(
            #     model_pred, i, t, noise, return_dict=False
            # )

            if callback:
                callback(i, denoised.float(), noise, len(timesteps))

        return denoised
