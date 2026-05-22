#!/usr/bin/env python3
"""bench/scripts/smoke_image.py — Phase 1 known-good-workflow image smoke test.

Posts a Stable Diffusion 1.5 workflow (CLIP → Core ML UNet → VAE → PNG) to a
locally running ComfyUI server, waits for the queue to drain, then copies the
generated CoreML + MPS reference images into bench/results/smoke/<gitsha>/
and writes a small report.json with the PSNR between them. This is the Phase 1
"known-good workflow still produces an image [M2-ANE]" artifact.

Prereqs:
  - ComfyUI server running locally on $COMFY_HOST:$COMFY_PORT (default
    http://localhost:8188), started against the project .venv so it picks up
    the pinned ml-stable-diffusion + Core ML Suite nodes.
  - The Core ML UNet has already been converted (see convert_sd15.py); the
    workflow's Core ML Converter node will skip if the .mlmodelc exists.

Run from the repo root:
    .venv/bin/python bench/scripts/smoke_image.py
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
COMFY_DIR = Path(os.environ.get("COMFY_DIR", REPO_ROOT.parents[1])).resolve()
COMFY_HOST = os.environ.get("COMFY_HOST", "localhost")
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"

CKPT_NAME = os.environ.get("CKPT_NAME", "v1-5-pruned-emaonly.safetensors")
WORKFLOW_PATH = Path(os.environ.get(
    "WORKFLOW_PATH",
    REPO_ROOT / "tests" / "integration" / "workflows" / "e2e-1.5-basic-conversion.json",
))
TIMEOUT_S = int(os.environ.get("TIMEOUT_S", "1800"))  # 30min cap


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "nogit"


def http_get_json(path: str) -> dict:
    with urllib.request.urlopen(f"{COMFY_URL}{path}", timeout=60) as r:
        return json.loads(r.read().decode())


def http_post_json(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFY_URL}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    # Long timeout: first POST blocks while the server warms model loaders +
    # the Core ML compile-on-load pass (~60-90s on a cold cache).
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0:
        return 100.0
    return 20.0 * float(np.log10(255.0 / np.sqrt(mse)))


def find_image(out_dir: Path, prefix: str) -> Path | None:
    matches = sorted(out_dir.glob(f"{prefix}_*.png"), reverse=True)
    return matches[0] if matches else None


def main() -> int:
    sha = git_sha()
    out_root = REPO_ROOT / "bench" / "results" / "smoke" / sha
    out_root.mkdir(parents=True, exist_ok=True)

    workflow = json.loads(WORKFLOW_PATH.read_text())
    # Point both checkpoint nodes at the maintainer's available SD1.5 ckpt.
    for nid in ("4", "10"):
        if nid in workflow:
            workflow[nid]["inputs"]["ckpt_name"] = CKPT_NAME
    # Fixed seed for reproducibility (Phase 1 baseline).
    seed = int(os.environ.get("SEED", "42"))
    for nid in ("3", "11"):
        if nid in workflow and "seed" in workflow[nid].get("inputs", {}):
            workflow[nid]["inputs"]["seed"] = seed
    # Phase 1 only needs a Core ML image. The reference MPS branch (nodes
    # 3/8/9) trips a known MPS f16/f32 mismatch on torch 2.0.1 + macOS 26+,
    # which would crash the whole server. Strip those nodes so the queue
    # only runs the Core ML path; SKIP_MPS=0 to opt back in.
    if os.environ.get("SKIP_MPS", "1") not in ("0", "false", "False"):
        for nid in ("3", "8", "9"):
            workflow.pop(nid, None)

    print(f"[smoke] git={sha} server={COMFY_URL} ckpt={CKPT_NAME} seed={seed}", file=sys.stderr)

    resp = http_post_json("/prompt", {"prompt": workflow})
    print(f"[smoke] queued: {resp}", file=sys.stderr)

    t0 = time.time()
    while True:
        if time.time() - t0 > TIMEOUT_S:
            print(f"[smoke] TIMEOUT after {TIMEOUT_S}s waiting for queue drain", file=sys.stderr)
            return 2
        try:
            q = http_get_json("/prompt")
            remaining = q.get("exec_info", {}).get("queue_remaining", -1)
            if remaining == 0:
                break
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            print(f"[smoke] poll error: {exc}", file=sys.stderr)
        time.sleep(2)

    comfy_out = COMFY_DIR / "output"
    coreml_png = find_image(comfy_out, "E2E-1.5-CoreML")
    if coreml_png is None:
        print(f"[smoke] missing Core ML image under {comfy_out}", file=sys.stderr)
        return 3
    coreml_dst = out_root / coreml_png.name
    shutil.copy2(coreml_png, coreml_dst)

    mps_png = find_image(comfy_out, "E2E-1.5-MPS")
    mps_dst = None
    psnr_db = None
    if mps_png is not None:
        mps_dst = out_root / mps_png.name
        shutil.copy2(mps_png, mps_dst)
        a = np.array(Image.open(coreml_dst).convert("RGB"))
        b = np.array(Image.open(mps_dst).convert("RGB"))
        if a.shape == b.shape:
            psnr_db = round(psnr(a, b), 2)

    report = {
        "git_sha": sha,
        "seed": seed,
        "ckpt_name": CKPT_NAME,
        "workflow": str(WORKFLOW_PATH.relative_to(REPO_ROOT)),
        "coreml_image": str(coreml_dst.relative_to(REPO_ROOT)),
        "mps_image": str(mps_dst.relative_to(REPO_ROOT)) if mps_dst else None,
        "psnr_db": psnr_db,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
