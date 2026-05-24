#!/usr/bin/env python3
"""bench/scripts/convert_sd15.py — Phase 1 baseline UNet conversion.

Converts a SD1.5 checkpoint to a Core ML UNet (.mlmodelc) for the bench harness.
Bypasses the ComfyUI node graph and calls coreml_suite.converter directly so
the conversion is reproducible from a single command.

Run from the repo root with the project venv:
    .venv/bin/python bench/scripts/convert_sd15.py

Environment overrides:
    COMFY_DIR=...                       (default: ../..)
    CKPT_NAME=v1-5-pruned-emaonly.safetensors
    ATTN=SPLIT_EINSUM                   (SPLIT_EINSUM | SPLIT_EINSUM_V2 | ORIGINAL)
    HEIGHT=512 WIDTH=512 BATCH_SIZE=1
    CONTROLNET=0
"""
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("convert_sd15")

REPO_ROOT = Path(__file__).resolve().parents[2]
COMFY_DIR = Path(os.environ.get("COMFY_DIR", REPO_ROOT.parents[1])).resolve()

if str(COMFY_DIR) not in sys.path:
    sys.path.insert(0, str(COMFY_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import folder_paths  # noqa: E402  (ComfyUI module — needs sys.path above)

# Ensure ComfyUI's folder_paths is pointed at the real install for checkpoints.
folder_paths.base_path = str(COMFY_DIR)
folder_paths.add_model_folder_path("checkpoints", str(COMFY_DIR / "models" / "checkpoints"))
folder_paths.add_model_folder_path("unet", str(COMFY_DIR / "models" / "unet"))

from coreml_suite import converter  # noqa: E402
from coreml_suite.config import ModelVersion  # noqa: E402


def main() -> int:
    ckpt_name = os.environ.get("CKPT_NAME", "v1-5-pruned-emaonly.safetensors")
    attn = os.environ.get("ATTN", "SPLIT_EINSUM")
    height = int(os.environ.get("HEIGHT", "512"))
    width = int(os.environ.get("WIDTH", "512"))
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    controlnet = os.environ.get("CONTROLNET", "0") not in ("0", "false", "False", "")
    quantize_nbits = os.environ.get("QUANT_NBITS", "none")

    ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
    if not ckpt_path:
        log.error("checkpoint not found: %s under %s", ckpt_name, COMFY_DIR / "models" / "checkpoints")
        return 2

    from coreml_suite.core.naming import compose_out_name

    out_name = compose_out_name(
        ckpt_name=ckpt_name,
        batch_size=batch_size,
        width=width,
        height=height,
        controlnet_support=controlnet,
        attention_implementation=attn,
        quantize_nbits=quantize_nbits,
    )
    unet_out_path = converter.get_out_path("unet", out_name)

    log.info("repo_root=%s comfy_dir=%s", REPO_ROOT, COMFY_DIR)
    log.info("ckpt=%s out_name=%s", ckpt_path, out_name)
    log.info(
        "attn=%s size=%dx%d batch=%d controlnet=%s quant=%s",
        attn, width, height, batch_size, controlnet, quantize_nbits,
    )

    converter.convert(
        ckpt_path=ckpt_path,
        model_version=ModelVersion.SD15,
        unet_out_path=unet_out_path,
        batch_size=batch_size,
        sample_size=(height // 8, width // 8),
        controlnet_support=controlnet,
        lora_weights=[],
        attn_impl=attn,
        config_path=None,
        quantize_nbits=quantize_nbits,
    )

    target_path = converter.compile_model(out_path=unet_out_path, out_name=out_name, submodule_name="unet")
    log.info("compiled: %s", target_path)
    print(target_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
