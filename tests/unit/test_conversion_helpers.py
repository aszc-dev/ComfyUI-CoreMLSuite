from types import SimpleNamespace

import torch

from coreml_diffusion.conversion.attention import (
    SplitEinsumAttnProcessor,
    SplitEinsumV2AttnProcessor,
    apply_attention_implementation,
    split_einsum,
    split_einsum_v2,
)
from coreml_diffusion.conversion.shapes import conv2d_output_shape
from coreml_diffusion.conversion.unet import CoreMLUNetWrapper


class RecordingUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.call = None

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        timestep_cond=None,
        added_cond_kwargs=None,
        down_block_additional_residuals=None,
        mid_block_additional_residual=None,
        return_dict=True,
        **kwargs,
    ):
        self.call = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep_cond": timestep_cond,
            "added_cond_kwargs": added_cond_kwargs,
            "down_block_additional_residuals": down_block_additional_residuals,
            "mid_block_additional_residual": mid_block_additional_residual,
            "return_dict": return_dict,
        }
        return (sample + 1,)


def test_conv2d_output_shape_matches_torch_conv2d_contract():
    conv = torch.nn.Conv2d(
        4,
        8,
        kernel_size=(3, 5),
        stride=(2, 3),
        padding=(1, 2),
        dilation=(1, 2),
    )

    assert conv2d_output_shape(17, 19, conv) == (9, 5)


def test_unet_wrapper_passes_context_through_for_sd15():
    unet = RecordingUNet()
    wrapper = CoreMLUNetWrapper(unet, SimpleNamespace(name="SD15"))

    sample = torch.randn(2, 4, 8, 8)
    timestep = torch.randn(2)
    context = torch.randn(2, 77, 768)

    out = wrapper(sample, timestep, context)

    assert torch.equal(out, sample + 1)
    assert unet.call["encoder_hidden_states"] is context
    assert unet.call["return_dict"] is False


def test_unet_wrapper_routes_lcm_sdxl_and_controlnet_inputs():
    unet = RecordingUNet()
    wrapper = CoreMLUNetWrapper(unet, SimpleNamespace(name="LCM"))

    sample = torch.randn(1, 4, 8, 8)
    timestep = torch.randn(1)
    context = torch.randn(1, 77, 768)
    timestep_cond = torch.randn(1, 256)
    down_residual = torch.randn(1, 320, 8, 8)
    mid_residual = torch.randn(1, 1280, 1, 1)

    wrapper(sample, timestep, context, timestep_cond, down_residual, mid_residual)

    assert unet.call["timestep_cond"] is timestep_cond
    assert len(unet.call["down_block_additional_residuals"]) == 1
    assert unet.call["down_block_additional_residuals"][0] is down_residual
    assert unet.call["mid_block_additional_residual"] is mid_residual


def test_unet_wrapper_routes_sdxl_added_conditioning():
    unet = RecordingUNet()
    wrapper = CoreMLUNetWrapper(unet, SimpleNamespace(name="SDXL"))

    sample = torch.randn(1, 4, 8, 8)
    timestep = torch.randn(1)
    context = torch.randn(1, 77, 2048)
    time_ids = torch.randn(1, 6)
    text_embeds = torch.randn(1, 1280)

    wrapper(sample, timestep, context, time_ids, text_embeds)

    assert unet.call["added_cond_kwargs"]["time_ids"] is time_ids
    assert unet.call["added_cond_kwargs"]["text_embeds"] is text_embeds


def test_split_einsum_matches_original_attention_math():
    torch.manual_seed(0)
    batch = 2
    heads = 3
    dim_head = 4
    sequence = 16
    channels = heads * dim_head
    q = torch.randn(batch, channels, 1, sequence)
    k = torch.randn(batch, channels, 1, sequence)
    v = torch.randn(batch, channels, 1, sequence)

    expected = _original_attention(q, k, v, None, heads, dim_head)

    # split-einsum reorders the float32 reductions vs the reference, so equality
    # only holds up to rounding; the drift exceeds allclose's default atol on
    # some BLAS backends (e.g. Linux x86 CI).
    assert torch.allclose(split_einsum(q, k, v, None, heads, dim_head), expected, atol=1e-6)
    assert torch.allclose(split_einsum_v2(q, k, v, None, heads, dim_head), expected, atol=1e-6)


def test_split_einsum_v2_chunked_path_matches_original_attention_math():
    torch.manual_seed(0)
    batch = 1
    heads = 2
    dim_head = 2
    sequence = 512
    channels = heads * dim_head
    q = torch.randn(batch, channels, 1, sequence)
    k = torch.randn(batch, channels, 1, sequence)
    v = torch.randn(batch, channels, 1, sequence)

    expected = _original_attention(q, k, v, None, heads, dim_head)

    assert torch.allclose(
        split_einsum_v2(q, k, v, None, heads, dim_head),
        expected,
        atol=1e-6,
    )


def test_apply_attention_implementation_sets_split_processors():
    unet = RecordingProcessorUNet()

    assert apply_attention_implementation(unet, "ORIGINAL") is unet
    assert unet.processor is None

    apply_attention_implementation(unet, "SPLIT_EINSUM")
    assert isinstance(unet.processor, SplitEinsumAttnProcessor)

    apply_attention_implementation(unet, "SPLIT_EINSUM_V2")
    assert isinstance(unet.processor, SplitEinsumV2AttnProcessor)


class RecordingProcessorUNet:
    def __init__(self):
        self.processor = None

    def set_attn_processor(self, processor):
        self.processor = processor


def _original_attention(q, k, v, mask, heads, dim_head):
    batch = q.size(0)
    mh_q = q.view(batch, heads, dim_head, -1)
    mh_k = k.view(batch, heads, dim_head, -1)
    mh_v = v.view(batch, heads, dim_head, -1)

    weights = torch.einsum("bhcq,bhck->bhqk", mh_q, mh_k)
    weights = weights * (dim_head**-0.5)
    if mask is not None:
        weights = weights + mask
    weights = weights.softmax(dim=3)

    attn = torch.einsum("bhqk,bhck->bhcq", weights, mh_v)
    return attn.contiguous().view(batch, heads * dim_head, 1, -1)
