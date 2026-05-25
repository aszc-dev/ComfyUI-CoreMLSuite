"""Compatibility shim — re-exports from coreml_suite.core.controlnet."""
from coreml_suite.core.controlnet import (
    chunk_control,
    expand_inputs,
    extract_residual_kwargs,
    no_control,
)

__all__ = [
    "chunk_control",
    "expand_inputs",
    "extract_residual_kwargs",
    "no_control",
]
