import os

import numpy as np
import torch

import comfy.utils
import latent_preview
from comfy.model_management import get_torch_device
from comfy.model_patcher import ModelPatcher
from coreml_suite.lcm.lcm_pipeline import LatentConsistencyModelPipeline
from coreml_suite.lcm.lcm_scheduler import LCMScheduler
from coreml_suite.logger import logger
from coreml_suite.models import get_model_config, CoreMLModelWrapperLCM
from coreml_suite.nodes import CoreMLSampler


class CoreMLSamplerLCM_Simple:
    def __init__(self):
        self.scheduler = LCMScheduler.from_pretrained(
            os.path.join(os.path.dirname(__file__), "scheduler_config.json")
        )
        self.pipe = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "coreml_model": ("COREML_UNET",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.5,
                        "round": 0.01,
                    },
                ),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 64}),
                "positive_prompt": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(
        self,
        coreml_model,
        seed,
        steps,
        cfg,
        positive_prompt,
        num_images,
    ):
        height = coreml_model.expected_inputs["sample"]["shape"][2] * 8
        width = coreml_model.expected_inputs["sample"]["shape"][3] * 8

        model_config = get_model_config()
        wrapped_model = CoreMLModelWrapperLCM(model_config, coreml_model)

        if self.pipe is None:
            self.pipe = LatentConsistencyModelPipeline.from_pretrained(
                pretrained_model_name_or_path="SimianLuo/LCM_Dreamshaper_v7",
                scheduler=self.scheduler,
                safety_checker=None,
            )

            self.pipe.to(torch_device=get_torch_device(), torch_dtype=torch.float16)

        coreml_unet = wrapped_model
        coreml_unet.config = self.pipe.unet.config

        self.pipe.unet = coreml_unet

        torch.manual_seed(seed)

        result = self.pipe(
            prompt=positive_prompt,
            width=width,
            height=height,
            guidance_scale=cfg,
            num_inference_steps=steps,
            num_images_per_prompt=num_images,
            lcm_origin_steps=50,
            output_type="np",
        ).images

        images_tensor = torch.from_numpy(result)

        return (images_tensor,)


class CoreMLSamplerLCM(CoreMLSampler):
    @classmethod
    def INPUT_TYPES(s):
        old_required = CoreMLSampler.INPUT_TYPES()["required"].copy()
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

        torch.manual_seed(seed)

        return self._sample(patched_model, steps, cfg, positive, latent_image, denoise)

    def _sample(self, model, steps, cfg, positive, latent_image, denoise):
        batch_size = latent_image["samples"].shape[0]

        bs_embed, seq_len, _ = positive.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = positive.repeat(1, batch_size, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * batch_size, seq_len, -1)

        device = get_torch_device()
        # callback = latent_preview.prepare_callback(model, steps, None)

        # Prepare timesteps
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

        # Prepare latent variable
        latents = self.prepare_latents(latent_image, device)

        # LCM MultiStep Sampling Loop:
        progress_bar = comfy.utils.ProgressBar(total=steps)
        for i, t in enumerate(timesteps):
            ts = torch.full((batch_size,), t, device=device, dtype=torch.float16)

            # model prediction (v-prediction, eps, x)
            model_pred = model.model(
                latents,
                ts,
                encoder_hidden_states=prompt_embeds,
            )[0]

            # model_pred *= cfg

            # compute the previous noisy sample x_t -> x_t-1
            latents, denoised = self.scheduler.step(
                model_pred, i, t, latents, return_dict=False
            )

            # # call the callback, if provided
            # if i == len(timesteps) - 1:
            # callback(i, t, latents, steps)

        denoised = denoised.to(get_torch_device())

        return ({"samples": denoised / 0.1825},)

    def prepare_latents(self, latent_image, device):
        latent = latent_image["samples"]
        if not torch.any(latent):
            latents = torch.randn(latent.shape, dtype=torch.float16).to(device)
            latents *= self.scheduler.init_noise_sigma
            return latents

        batch_size = latent.shape[0]
        noise = torch.randn(latent.shape, dtype=torch.float16).cpu()
        latent_timestep = self.scheduler.timesteps[:1].repeat(batch_size)
        latents = self.scheduler.add_noise(latent, noise, latent_timestep)

        return latents
