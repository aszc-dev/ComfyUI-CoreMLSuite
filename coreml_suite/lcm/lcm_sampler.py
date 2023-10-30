import os
import time

import torch

from comfy.model_management import get_torch_device
from coreml_suite.lcm.lcm_pipeline import LatentConsistencyModelPipeline
from coreml_suite.lcm.lcm_scheduler import LCMScheduler
from coreml_suite.models import get_model_config, CoreMLModelWrapperLCM


class CoreMLSamplerLCM:
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
                "height": ("INT", {"default": 512, "min": 512, "max": 768}),
                "width": ("INT", {"default": 512, "min": 512, "max": 768}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 64}),
                "use_fp16": ("BOOLEAN", {"default": True}),
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
        height,
        width,
        num_images,
        use_fp16,
    ):

        model_config = get_model_config()
        wrapped_model = CoreMLModelWrapperLCM(model_config, coreml_model)

        if self.pipe is None:
            self.pipe = LatentConsistencyModelPipeline.from_pretrained(
                pretrained_model_name_or_path="SimianLuo/LCM_Dreamshaper_v7",
                scheduler=self.scheduler,
                safety_checker=None,
            )

            if use_fp16:
                self.pipe.to(torch_device=get_torch_device(), torch_dtype=torch.float16)
            else:
                self.pipe.to(torch_device=get_torch_device(), torch_dtype=torch.float32)

            coreml_unet = wrapped_model
            coreml_unet.config = self.pipe.unet.config

            self.pipe.unet = coreml_unet

        torch.manual_seed(seed)
        start_time = time.time()

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

        print("LCM inference time: ", time.time() - start_time, "seconds")
        images_tensor = torch.from_numpy(result)

        return (images_tensor,)
