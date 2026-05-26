import torch


class CoreMLUNetWrapper(torch.nn.Module):
    """Adapt diffusers UNet inputs to CoreMLSuite's stable Core ML contract."""

    def __init__(self, unet, model_version):
        super().__init__()
        self.unet = unet
        self.model_version = model_version

    def forward(self, sample, timestep, encoder_hidden_states, *extra_inputs):
        input_index = 0
        timestep_cond = None
        if self._is_lcm:
            timestep_cond = extra_inputs[input_index]
            input_index += 1

        added_cond_kwargs = None
        if self._is_sdxl:
            time_ids = extra_inputs[input_index]
            text_embeds = extra_inputs[input_index + 1]
            input_index += 2
            added_cond_kwargs = {
                "time_ids": time_ids,
                "text_embeds": text_embeds,
            }

        additional_residuals = extra_inputs[input_index:]
        down_residuals = None
        mid_residual = None
        if additional_residuals:
            down_residuals = tuple(additional_residuals[:-1])
            mid_residual = additional_residuals[-1]

        outputs = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            timestep_cond=timestep_cond,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_residuals,
            mid_block_additional_residual=mid_residual,
            return_dict=False,
        )
        return outputs[0]

    @property
    def _is_lcm(self):
        return self.model_version.name == "LCM"

    @property
    def _is_sdxl(self):
        return self.model_version.name in {"SDXL", "SDXL_REFINER"}
