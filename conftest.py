"""Top-level conftest: prevent pytest from importing the repo-root
__init__.py (the ComfyUI custom-node entry point pulls in comfy + nodes,
which breaks the Tier-0 'no-framework' promise)."""
collect_ignore = ["__init__.py"]
