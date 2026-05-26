"""Compatibility shim.

The name-encoding contract moved to ``coreml_diffusion.naming`` in extraction
phase E2 (it is the ``.mlpackage`` cache key, owned by the conversion package).
Re-exported here so ``coreml_suite.nodes`` keeps importing from the old path
until E3 re-points the node to ``coreml_diffusion`` directly.
"""
from coreml_diffusion.naming import (  # noqa: F401
    ATTN_SUFFIX,
    QUANT_NBITS_VALUES,
    compose_out_name,
    lora_names_from_params,
)
