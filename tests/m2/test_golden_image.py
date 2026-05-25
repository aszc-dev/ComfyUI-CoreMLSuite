"""[M2-ANE] golden-image anchor.

Runs the e2e SD1.5 + CoreML workflow against a local ComfyUI server, fetches
the generated PNG, and asserts both:
  - byte-identical SHA256 against the stored golden, OR
  - PSNR >= GOLDEN_PSNR_MIN_DB against the stored golden PNG.

The hash is the strict gate (a refactor that doesn't touch the math
should hit it). PSNR is the soft gate that tolerates the drift a
toolchain bump injects through different MIL graphs / kernel selection
/ fp accumulation order — anything below the threshold is treated as a
regression.

The 20 dB default absorbs Apple Neural Engine run-to-run nondeterminism:
the same model and seed can drift several dB between runs as the 20
sampling steps amplify tiny per-step UNet differences (kernel selection /
fp accumulation order). Same-scene ANE outputs have been observed at
~23 dB, so 20 leaves margin while still catching gross regressions — a
broken image lands far lower. Bump it up for pure-refactor PRs that must
not change math; down for toolchain bumps.

Skips entirely on non-Apple-Silicon hosts or when the server / converted
model is missing, so the unit lane on Linux still passes.

The first run with no golden writes one and fails so it's reviewed before
being committed.
"""
import hashlib
import json
import os
import platform
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
COMFY_DIR = Path(os.environ.get("COMFY_DIR", REPO_ROOT.parents[1])).resolve()
COMFY_HOST = os.environ.get("COMFY_HOST", "localhost")
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"

CKPT_NAME = os.environ.get("CKPT_NAME", "v1-5-pruned-emaonly.safetensors")
WORKFLOW_PATH = (
    REPO_ROOT / "tests" / "integration" / "workflows" / "e2e-1.5-basic-conversion.json"
)
GOLDEN_DIR = Path(__file__).parent / "goldens"
GOLDEN_HASH_PATH = GOLDEN_DIR / "sd15_seed42.sha256"
GOLDEN_PNG_PATH = GOLDEN_DIR / "sd15_seed42.png"
GOLDEN_PSNR_MIN_DB = float(os.environ.get("GOLDEN_PSNR_MIN_DB", "20"))
SEED = 42


def _server_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/prompt", timeout=3) as r:
            return r.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError):
        return False


@pytest.fixture(scope="module")
def comfy_server():
    if platform.machine() != "arm64":
        pytest.skip("requires Apple Silicon")
    if not _server_reachable():
        pytest.skip(f"ComfyUI server not reachable at {COMFY_URL}")
    return COMFY_URL


def _http_post_json(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFY_URL}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def _http_get_json(path: str, timeout: int = 300) -> dict:
    """ComfyUI runs UNet inference on its single asyncio loop, so GET /prompt
    blocks while the queued prompt is executing. Use a generous timeout."""
    with urllib.request.urlopen(f"{COMFY_URL}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode())


def _drain_queue(timeout_s: int = 600) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            q = _http_get_json("/prompt")
        except (urllib.error.URLError, TimeoutError):
            # Transient block while server executes; retry until our overall
            # deadline expires.
            continue
        if q.get("exec_info", {}).get("queue_remaining", -1) == 0:
            return
        time.sleep(2)
    raise TimeoutError(f"queue did not drain within {timeout_s}s")


def _post_workflow_and_collect_png() -> bytes:
    workflow = json.loads(WORKFLOW_PATH.read_text())
    for nid in ("4", "10"):
        if nid in workflow:
            workflow[nid]["inputs"]["ckpt_name"] = CKPT_NAME
    for nid in ("3", "11"):
        if nid in workflow and "seed" in workflow[nid].get("inputs", {}):
            workflow[nid]["inputs"]["seed"] = SEED
    # Drop the MPS reference branch — only the Core ML pipeline is needed here.
    for nid in ("3", "8", "9"):
        workflow.pop(nid, None)

    _http_post_json("/prompt", {"prompt": workflow})
    _drain_queue()

    comfy_out = COMFY_DIR / "output"
    matches = sorted(comfy_out.glob("E2E-1.5-CoreML_*.png"), reverse=True)
    if not matches:
        raise FileNotFoundError(f"no Core ML image under {comfy_out}")
    return matches[0].read_bytes()


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0:
        return 100.0
    return 20.0 * float(np.log10(255.0 / np.sqrt(mse)))


def test_sd15_seed42_image_matches_golden(comfy_server):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    png_bytes = _post_workflow_and_collect_png()
    sha = hashlib.sha256(png_bytes).hexdigest()

    if not GOLDEN_HASH_PATH.exists() or not GOLDEN_PNG_PATH.exists():
        GOLDEN_HASH_PATH.write_text(sha + "\n")
        # Persist the PNG too for visual diffing + PSNR.
        tmp_path = Path(__file__).parent / "_latest_generated.png"
        tmp_path.write_bytes(png_bytes)
        shutil.copy2(tmp_path, GOLDEN_PNG_PATH)
        pytest.fail(
            f"No golden present; wrote {GOLDEN_HASH_PATH.name} and "
            f"{GOLDEN_PNG_PATH.name}. Review the image and re-run."
        )

    expected_hash = GOLDEN_HASH_PATH.read_text().strip()
    if sha == expected_hash:
        return

    # Hash drift: fall back to PSNR to distinguish a refactor-safe rounding
    # change from a real regression.
    a = np.array(Image.open(GOLDEN_PNG_PATH).convert("RGB"))
    b_path = Path(__file__).parent / "_latest_generated.png"
    b_path.write_bytes(png_bytes)
    b = np.array(Image.open(b_path).convert("RGB"))
    if a.shape != b.shape:
        pytest.fail(f"shape mismatch: golden={a.shape} actual={b.shape}")
    psnr_db = _psnr(a, b)
    assert psnr_db >= GOLDEN_PSNR_MIN_DB, (
        f"hash drifted (got {sha[:12]}.., expected {expected_hash[:12]}..) and "
        f"PSNR {psnr_db:.2f} dB < {GOLDEN_PSNR_MIN_DB} dB threshold; "
        f"diff PNG at {b_path}"
    )
