"""Gate: prove the Tier-0 lane is framework-free.

In a pure `pytest -m unit` run, none of the banned runtime modules
(comfy, coremltools, python_coreml_stable_diffusion, folder_paths,
nodes, comfy_extras, diffusers, diffusionkit) may be in sys.modules
after collection. If they are, a tests/unit/ file is transitively
pulling them in and the Tier-0 promise — "runs on Linux with no Mac
stack" — is broken.

When other tiers (m2 / integration) are also collected, comfy is
expected in sys.modules (integration imports it deliberately), so the
check is skipped in mixed runs — Tier-0 purity is only meaningful when
nothing else is loaded.
"""
import sys

import pytest

BANNED_ROOTS = {
    "comfy",
    "comfy_extras",
    "coremltools",
    "python_coreml_stable_diffusion",
    "folder_paths",
    "nodes",
    "diffusers",
    "diffusionkit",
}


def test_no_framework_modules_loaded_by_unit_tier(request):
    markexpr = request.config.option.markexpr
    if markexpr != "unit":
        pytest.skip(
            "purity gate only meaningful in a pure `-m unit` run "
            f"(got markexpr={markexpr!r}); other tiers are expected to "
            "import comfy/coremltools."
        )
    loaded = {name for name in sys.modules if name.split(".")[0] in BANNED_ROOTS}
    assert not loaded, (
        f"Tier-0 leakage: these framework modules are in sys.modules after "
        f"collecting tests/unit/: {sorted(loaded)}. Pure-core promise broken."
    )
