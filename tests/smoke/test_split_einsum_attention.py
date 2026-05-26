import platform

import pytest
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor

from coreml_diffusion.conversion.attention import (
    SplitEinsumAttnProcessor,
    SplitEinsumV2AttnProcessor,
)


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Tier 1 requires macOS on Apple Silicon",
)


@pytest.mark.parametrize(
    "processor",
    [
        SplitEinsumAttnProcessor(),
        SplitEinsumV2AttnProcessor(),
    ],
)
def test_split_einsum_processor_matches_diffusers_attention(processor):
    torch.manual_seed(0)
    reference = Attention(query_dim=32, heads=4, dim_head=8, dropout=0.0)
    reference.set_processor(AttnProcessor())

    candidate = Attention(query_dim=32, heads=4, dim_head=8, dropout=0.0)
    candidate.load_state_dict(reference.state_dict())
    candidate.set_processor(processor)

    hidden_states = torch.randn(2, 17, 32)
    encoder_hidden_states = torch.randn(2, 11, 32)

    expected = reference(hidden_states, encoder_hidden_states=encoder_hidden_states)
    actual = candidate(hidden_states, encoder_hidden_states=encoder_hidden_states)

    assert torch.allclose(actual, expected, atol=1e-5)
