"""coreml_diffusion — framework-free Core ML diffusion conversion (discovery surface).

Phase E1 stands up the package namespace and its versioned discovery API. The
conversion *implementation* still lives in ``coreml_suite`` for now; E2 physically
moves the mechanics here and E3 thins the ComfyUI nodes onto this package.

This module MUST stay free of ``comfy`` / ``folder_paths`` / ``comfy_extras``. It
re-exports only from already-framework-free ``coreml_suite`` modules
(``model_version``, ``attention``, ``core.naming``), whose package ``__init__``
files import no comfy, so ``import coreml_diffusion`` works in a comfy-free env.

Discovery contract (the maintainer's hard requirement): the ComfyUI node populates
its dropdowns by calling ``list_*`` here, so installing a newer ``coreml_diffusion``
surfaces new conversion types in the old node with no Suite change and no Suite
version bump. The identifiers returned here are an ADDITIVE-ONLY contract:

- adding an identifier, or promoting EXPERIMENTAL -> VERIFIED  => minor bump
- removing/renaming an identifier, or demoting VERIFIED        => MAJOR bump + note

because a saved workflow JSON references these strings verbatim.
"""
from enum import Enum

from coreml_suite.model_version import ModelVersion
from coreml_suite.attention import ATTENTION_IMPLEMENTATIONS
from coreml_diffusion.naming import QUANT_NBITS_VALUES, compose_out_name

__all__ = [
    "ModelVersion",
    "Status",
    "list_model_versions",
    "list_attention_impls",
    "list_quant_modes",
    "CONTRACT_VERSION",
    "compose_out_name",
    "convert",
]


class Status(Enum):
    VERIFIED = "verified"          # has a golden anchor + passing [M2-ANE] check
    EXPERIMENTAL = "experimental"  # convertible, not yet anchored/verified


# Single source of truth for which conversions the Suite may surface. The Suite
# gates on this status, NOT on a hardcoded node list: promoting a model to
# VERIFIED expands the node's dropdown with no Suite change.
#
# Keyed by the ModelVersion MEMBER so ``list_model_versions`` can emit ``.name``
# ("SD15", "SDXL"). The node reverses the dropdown string via ``ModelVersion[...]``
# (name lookup, nodes.py), so emitting ``.value`` ("sd15") would raise KeyError on
# every saved workflow. See seam.md §5.
_MODEL_STATUS = {
    ModelVersion.SD15: Status.VERIFIED,
    ModelVersion.SDXL: Status.VERIFIED,
    ModelVersion.SDXL_REFINER: Status.EXPERIMENTAL,  # -> VERIFIED after a refiner golden anchor
    ModelVersion.LCM: Status.EXPERIMENTAL,           # -> VERIFIED after E-LCM golden anchor
}


def list_model_versions(include_experimental: bool = False) -> list[str]:
    """Model versions by ``.name`` (e.g. ``["SD15", "SDXL"]``).

    Returns VERIFIED versions only by default — the converter node calls this
    plainly. A power-user/CLI path may pass ``include_experimental=True`` to also
    list convertible-but-unanchored versions.
    """
    return [
        version.name
        for version, status in _MODEL_STATUS.items()
        if status is Status.VERIFIED
        or (include_experimental and status is Status.EXPERIMENTAL)
    ]


def list_attention_impls() -> list[str]:
    """Supported attention implementations, e.g. ``["SPLIT_EINSUM", ...]``."""
    return list(ATTENTION_IMPLEMENTATIONS)


def list_quant_modes() -> list[str]:
    """Palettization modes, e.g. ``["none", "8", "6", "4"]`` ("none" = unquantized)."""
    return list(QUANT_NBITS_VALUES)


# Discovery-contract version. Bump per the additive-only rules in this module's
# docstring and CONVERTER_EXTRACTION_SPEC.md "Interface contract".
CONTRACT_VERSION = "1.0"


def __getattr__(name):
    """Lazily expose the heavy conversion entrypoint.

    ``convert`` pulls coremltools/diffusers, so importing it eagerly would drag
    the Mac/heavy stack into every ``import coreml_diffusion`` and break the
    Tier-0 (Linux, framework-free) lane. Resolve it only on first access.
    """
    if name == "convert":
        from coreml_diffusion.convert import convert as _convert

        return _convert
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
