import torch


def chunk_batch(latent_image, target_shape):
    if latent_image.shape == target_shape:
        return [latent_image]

    batch_size = latent_image.shape[0]
    target_batch_size = target_shape[0]

    num_chunks = batch_size // target_batch_size
    if num_chunks == 0:
        padding = torch.zeros(target_batch_size - batch_size, *target_shape[1:])
        return [torch.cat((latent_image, padding), dim=0)]

    mod = batch_size % target_batch_size
    if mod != 0:
        chunks = list(torch.chunk(latent_image[:-mod], num_chunks))
        padding = torch.zeros(target_batch_size - mod, *target_shape[1:])
        padded = torch.cat((latent_image[-mod:], padding), dim=0)
        chunks.append(padded)
        return [chunk for chunk in chunks]

    chunks = list(torch.chunk(latent_image, num_chunks))

    return [chunk for chunk in chunks]


def merge_chunks(chunks, orig_shape):
    merged = torch.cat(chunks, dim=0)
    if merged.shape == orig_shape:
        return merged
    return merged[: orig_shape[0]]
