"""Pytest bootstrap for ComfyUI-CoreMLSuite tests.

- Adds the ComfyUI checkout to sys.path so the framework-coupled modules
  that transitively import `comfy.*` resolve when pytest is invoked from
  this package's root.
- Auto-applies tier markers based on the directory a test lives in, so
  individual files don't have to repeat @pytest.mark.unit / .m2.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_DIR = REPO_ROOT.parents[1]

for p in (str(COMFY_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


_TIER_BY_DIR = {
    "tests/unit": "unit",
    "tests/m2": "m2",
    "tests/integration": "m2",
    "tests/smoke": "smoke",
}

# When the user asks for a single tier (-m unit / -m m2), skip the other
# directories at collection time. Tier-0 cannot afford to import tests/m2
# files because they pull in PIL + ComfyUI runtime which Linux CI won't have.
_TIER_DIRS = {
    "unit": ("/tests/unit/",),
    "m2": ("/tests/m2/", "/tests/integration/"),
    "smoke": ("/tests/smoke/",),
}


def pytest_ignore_collect(collection_path, config):
    expr = config.option.markexpr
    if expr not in _TIER_DIRS:
        return None
    allowed = _TIER_DIRS[expr]
    rel = str(collection_path).replace("\\", "/")
    if "/tests/" not in rel:
        return None
    # Always allow tests/ root + the tier's own dirs.
    if rel.endswith("/tests"):
        return None
    if any(frag in rel + "/" for frag in allowed):
        return None
    return True


def pytest_collection_modifyitems(config, items):
    for item in items:
        path = str(item.fspath).replace("\\", "/")
        for fragment, marker in _TIER_BY_DIR.items():
            if f"/{fragment}/" in path:
                item.add_marker(getattr(pytest.mark, marker))
                break
