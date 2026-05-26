def conv2d_output_shape(height, width, conv):
    """Return the spatial output shape for a torch.nn.Conv2d-like module."""
    kernel_h, kernel_w = _pair(conv.kernel_size)
    stride_h, stride_w = _pair(conv.stride)
    pad_h, pad_w = _pair(conv.padding)
    dilation_h, dilation_w = _pair(conv.dilation)

    out_h = _conv_output_dim(height, kernel_h, stride_h, pad_h, dilation_h)
    out_w = _conv_output_dim(width, kernel_w, stride_w, pad_w, dilation_w)
    return out_h, out_w


def _conv_output_dim(size, kernel, stride, padding, dilation):
    return ((size + (2 * padding) - (dilation * (kernel - 1)) - 1) // stride) + 1


def _pair(value):
    if isinstance(value, tuple):
        return value
    return value, value
