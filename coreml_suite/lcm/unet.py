from overrides import overrides
from python_coreml_stable_diffusion.unet import UNet2DConditionModel, TimestepEmbedding


class UNet2DConditionModelLCM(UNet2DConditionModel):
    def __init__(
        self,
        time_cond_proj_dim=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        timestep_input_dim = self.config.block_out_channels[0]
        time_embed_dim = self.config.block_out_channels[0] * 4

        time_embedding = TimestepEmbedding(
            timestep_input_dim, time_embed_dim, cond_proj_dim=time_cond_proj_dim
        )
        self.time_embedding = time_embedding

    @overrides(check_signature=False)
    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        timestep_cond,
        *additional_residuals,
    ):
        # 0. Project (or look-up) time embeddings
        t_emb = self.time_proj(timestep)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 1. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if (
                hasattr(downsample_block, "attentions")
                and downsample_block.attentions is not None
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if additional_residuals:
            new_down_block_res_samples = ()
            for i, down_block_res_sample in enumerate(down_block_res_samples):
                down_block_res_sample = down_block_res_sample + additional_residuals[i]
                new_down_block_res_samples += (down_block_res_sample,)
            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states
        )

        if additional_residuals:
            sample = sample + additional_residuals[-1]

        # 5. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            if (
                hasattr(upsample_block, "attentions")
                and upsample_block.attentions is not None
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return (sample,)
