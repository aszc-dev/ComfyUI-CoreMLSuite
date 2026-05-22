"""Pytest bootstrap for ComfyUI-CoreMLSuite tests.

- Adds the ComfyUI checkout to sys.path so production modules that
  transitively import `comfy.*` resolve when pytest is invoked from this
  package's root. Phase 3 will split pure logic into a comfy-free core and
  this hack can go.
- Auto-applies tier markers based on the directory a test lives in, so
  individual files don't have to repeat @pytest.mark.unit / .m2.
- Skips the maintainer's in-progress test scaffolds so they don't break
  collection (they reference modules / venv layouts that aren't part of
  Phase 2 scope).
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_DIR = REPO_ROOT.parents[1]

for p in (str(COMFY_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


# Skip WIP test scaffolds left in the tree by the maintainer; they import
# modules (coreml_suite.experiments, convert_apple) that are not part of
# Phase 2 scope.
collect_ignore_glob = [
    "unit/test_experiments.py",
    "unit/test_unet_conversion.py",
    "unit/standalone_test.py",
]


_TIER_BY_DIR = {
    "tests/unit": "unit",
    "tests/m2": "m2",
    "tests/integration": "m2",
    "tests/smoke": "smoke",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        path = str(item.fspath).replace("\\", "/")
        for fragment, marker in _TIER_BY_DIR.items():
            if f"/{fragment}/" in path:
                item.add_marker(getattr(pytest.mark, marker))
                break
