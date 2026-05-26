from types import MethodType

from diffusers.models.transformers.transformer_2d import Transformer2DModel


def prepare_unet_for_coreml_trace(unet):
    for module in unet.modules():
        if isinstance(module, Transformer2DModel):
            module._operate_on_continuous_inputs = MethodType(
                _operate_on_continuous_inputs,
                module,
            )
            module._get_output_for_continuous_inputs = MethodType(
                _get_output_for_continuous_inputs,
                module,
            )
    return unet


def _operate_on_continuous_inputs(self, hidden_states):
    hidden_states = self.norm(hidden_states)

    if not self.use_linear_projection:
        hidden_states = self.proj_in(hidden_states)
        inner_dim = self.inner_dim
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
    else:
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj_in(hidden_states)

    return hidden_states, inner_dim


def _get_output_for_continuous_inputs(
    self,
    hidden_states,
    residual,
    batch_size,
    height,
    width,
    inner_dim,
):
    if not self.use_linear_projection:
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size,
            inner_dim,
            height,
            width,
        )
        hidden_states = self.proj_out(hidden_states)
    else:
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size,
            inner_dim,
            height,
            width,
        )

    return hidden_states + residual
