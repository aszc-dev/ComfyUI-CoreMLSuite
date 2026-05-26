"""Core ML conversion helpers.

The conversion approach originates from Apple's ml-stable-diffusion
(https://github.com/apple/ml-stable-diffusion). This implementation has since
diverged: it runs natively on diffusers' UNet2DConditionModel with its own
SPLIT_EINSUM / SPLIT_EINSUM_V2 attention processors and no longer depends on
that package. The intent is to keep iterating on these methods independently
while tracking current tooling.
"""
