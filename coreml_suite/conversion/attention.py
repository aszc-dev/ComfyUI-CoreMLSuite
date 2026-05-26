import logging

import torch

logger = logging.getLogger(__name__)

CHUNK_SIZE = 512


def apply_attention_implementation(unet, attention_implementation):
    if attention_implementation == "ORIGINAL":
        return unet

    if attention_implementation == "SPLIT_EINSUM":
        unet.set_attn_processor(SplitEinsumAttnProcessor())
        return unet

    if attention_implementation == "SPLIT_EINSUM_V2":
        unet.set_attn_processor(SplitEinsumV2AttnProcessor())
        return unet

    raise ValueError(f"Unsupported attention implementation: {attention_implementation}")


class SplitEinsumAttnProcessor:
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        return _attention_forward(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            temb,
            split_einsum,
        )


class SplitEinsumV2AttnProcessor:
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        return _attention_forward(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            temb,
            split_einsum_v2,
        )


def _attention_forward(
    attn,
    hidden_states,
    encoder_hidden_states,
    attention_mask,
    temb,
    attention_fn,
):
    residual = hidden_states

    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
    else:
        batch_size, _, channel = hidden_states.shape
        height = None
        width = None

    batch_size, key_sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(
            attention_mask,
            key_sequence_length,
            batch_size,
        )
        attention_mask = _prepare_split_einsum_mask(
            attention_mask,
            batch_size,
            attn.heads,
            key_sequence_length,
        )

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    batch_size = query.shape[0]
    dim_head = attn.inner_kv_dim // attn.heads

    query = _linear_projection_to_bchw(query)
    key = _linear_projection_to_bchw(key)
    value = _linear_projection_to_bchw(value)

    hidden_states = attention_fn(
        query,
        key,
        value,
        attention_mask,
        attn.heads,
        dim_head,
    )
    hidden_states = hidden_states.squeeze(2).transpose(1, 2)
    hidden_states = hidden_states.reshape(batch_size, -1, attn.inner_dim)

    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(
            batch_size,
            channel,
            height,
            width,
        )

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor
    return hidden_states


def split_einsum(q, k, v, mask, heads, dim_head):
    q_heads = _split_heads(q, heads, dim_head)
    k = k.transpose(1, 3)
    k_heads = [
        k[:, :, :, head_idx * dim_head : (head_idx + 1) * dim_head]
        for head_idx in range(heads)
    ]
    v_heads = _split_heads(v, heads, dim_head)

    weights = [
        torch.einsum("bchq,bkhc->bkhq", query, key) * (dim_head**-0.5)
        for query, key in zip(q_heads, k_heads)
    ]
    if mask is not None:
        weights = [weight + mask for weight in weights]

    weights = [weight.softmax(dim=1) for weight in weights]
    outputs = [
        torch.einsum("bkhq,bchk->bchq", weight, value)
        for weight, value in zip(weights, v_heads)
    ]
    return torch.cat(outputs, dim=1)


def split_einsum_v2(q, k, v, mask, heads, dim_head):
    query_length = q.size(3)
    num_chunks = query_length // CHUNK_SIZE
    if num_chunks == 0:
        logger.info(
            "SPLIT_EINSUM_V2 query sequence is shorter than %s; using SPLIT_EINSUM.",
            CHUNK_SIZE,
        )
        return split_einsum(q, k, v, mask, heads, dim_head)

    q_heads = _split_heads(q, heads, dim_head)
    q_chunks = [
        [
            head[..., chunk_idx * CHUNK_SIZE : (chunk_idx + 1) * CHUNK_SIZE]
            for chunk_idx in range(num_chunks)
        ]
        for head in q_heads
    ]

    k = k.transpose(1, 3)
    k_heads = [
        k[:, :, :, head_idx * dim_head : (head_idx + 1) * dim_head]
        for head_idx in range(heads)
    ]
    v_heads = _split_heads(v, heads, dim_head)

    head_outputs = []
    for query_chunks, key, value in zip(q_chunks, k_heads, v_heads):
        chunk_outputs = []
        for query_chunk in query_chunks:
            weights = torch.einsum("bchq,bkhc->bkhq", query_chunk, key)
            weights = weights * (dim_head**-0.5)
            if mask is not None:
                weights = weights + mask
            weights = weights.softmax(dim=1)
            chunk_outputs.append(torch.einsum("bkhq,bchk->bchq", weights, value))
        head_outputs.append(torch.cat(chunk_outputs, dim=3))

    return torch.cat(head_outputs, dim=1)


def _split_heads(x, heads, dim_head):
    return [
        x[:, head_idx * dim_head : (head_idx + 1) * dim_head, :, :]
        for head_idx in range(heads)
    ]


def _linear_projection_to_bchw(x):
    return x.transpose(1, 2).unsqueeze(2)


def _prepare_split_einsum_mask(mask, batch_size, heads, key_sequence_length):
    if mask.ndim == 2:
        mask = mask[:, None, :]
    if mask.shape[0] == batch_size * heads:
        mask = mask.reshape(batch_size, heads, -1, key_sequence_length)
        mask = mask[:, 0]
    if mask.ndim == 3:
        mask = mask[:, :, None, None]
    return mask
