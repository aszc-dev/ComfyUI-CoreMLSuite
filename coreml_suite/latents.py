import torch

from comfy.model_management import get_torch_device


def chunk_batch(input_tensor, target_shape):
    if input_tensor.shape == target_shape:
        return [input_tensor]

    batch_size = input_tensor.shape[0]
    target_batch_size = target_shape[0]

    num_chunks = batch_size // target_batch_size
    if num_chunks == 0:
        padding = torch.zeros(target_batch_size - batch_size, *target_shape[1:]).to(
            get_torch_device()
        )
        return [torch.cat((input_tensor, padding), dim=0)]

    mod = batch_size % target_batch_size
    if mod != 0:
        chunks = list(torch.chunk(input_tensor[:-mod], num_chunks))
        padding = torch.zeros(target_batch_size - mod, *target_shape[1:]).to(
            get_torch_device()
        )
        padded = torch.cat((input_tensor[-mod:], padding), dim=0)
        chunks.append(padded)
        return chunks

    chunks = list(torch.chunk(input_tensor, num_chunks))
    return chunks


def merge_chunks(chunks, orig_shape):
    merged = torch.cat(chunks, dim=0)
    if merged.shape == orig_shape:
        return merged
    return merged[: orig_shape[0]]
