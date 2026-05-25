"""Framework-free pure-logic core of ComfyUI-CoreMLSuite.

Modules under this package must NOT import `comfy`, `coremltools`,
`python_coreml_stable_diffusion`, `folder_paths`, `nodes`, or any other
ComfyUI / Apple runtime. Only `numpy` and `torch` are allowed.

The thin adapters in `coreml_suite.{latents,controlnet,models}` keep the
old public import paths working so `coreml_suite/nodes.py` and downstream
ComfyUI workflows are unchanged.
"""
