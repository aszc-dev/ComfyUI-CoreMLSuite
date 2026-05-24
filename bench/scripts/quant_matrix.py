#!/usr/bin/env python3
"""bench/scripts/quant_matrix.py — Phase 6 quantization tradeoff matrix.

For each variant in {none, 8, 6, 4}, load the corresponding .mlmodelc,
run a synthetic forward pass with a fixed seed, and record:
  - on-disk size
  - load time
  - forward-pass median latency
  - PSNR of noise_pred vs the `none` baseline (proxy for output drift)

Writes a JSON + Markdown summary into bench/results/quant_matrix_<sha>.{json,md}.

Assumes converted .mlmodelc files exist under $COMFY_DIR/models/unet/
following the convention from coreml_suite.core.naming.compose_out_name
(e.g. v1-5-pruned-emaonly_1x512x512_se_unet.mlmodelc for the baseline,
v1-5-pruned-emaonly_1x512x512_se_q4_unet.mlmodelc for 4-bit, etc.).
"""
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
COMFY_DIR = Path(os.environ.get("COMFY_DIR", REPO_ROOT.parents[1])).resolve()
MODELS_DIR = COMFY_DIR / "models" / "unet"

VARIANTS = ["none", "8", "6", "4"]
CKPT_STEM = os.environ.get("CKPT_STEM", "v1-5-pruned-emaonly")
BATCH = int(os.environ.get("BATCH", "1"))
WIDTH = int(os.environ.get("WIDTH", "512"))
HEIGHT = int(os.environ.get("HEIGHT", "512"))
ATTN_SUFFIX = os.environ.get("ATTN_SUFFIX", "se")
COMPUTE_UNIT = os.environ.get("COMPUTE_UNIT", "CPU_AND_NE")
REPEATS = int(os.environ.get("REPEATS", "20"))
INPUT_SEED = int(os.environ.get("INPUT_SEED", "0"))


def variant_path(nbits: str) -> Path:
    quant_suffix = f"_q{nbits}" if nbits != "none" else ""
    name = f"{CKPT_STEM}_{BATCH}x{WIDTH}x{HEIGHT}_{ATTN_SUFFIX}{quant_suffix}_unet.mlmodelc"
    return MODELS_DIR / name


def dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Float PSNR for noise_pred tensors normalized to a [-1, 1]-ish range."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    peak = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1.0)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return 100.0
    return 20.0 * float(np.log10(peak / np.sqrt(mse)))


def measure_variant(
    nbits: str, ref_output: "np.ndarray | None"
) -> "tuple[dict[str, Any], np.ndarray | None]":
    from python_coreml_stable_diffusion.coreml_model import CoreMLModel

    path = variant_path(nbits)
    result: dict[str, Any] = {"nbits": nbits, "path": str(path)}
    if not path.exists():
        result["error"] = f"model not found at {path}"
        return result, None
    result["size_bytes"] = dir_size_bytes(path)

    rng = np.random.default_rng(INPUT_SEED)
    t0 = time.perf_counter()
    model = CoreMLModel(str(path), COMPUTE_UNIT, "compiled")
    result["load_time_s"] = round(time.perf_counter() - t0, 4)

    expected = dict(model.expected_inputs)
    fixed_inputs = {
        name: rng.standard_normal(tuple(int(d) for d in spec["shape"])).astype(np.float16)
        for name, spec in expected.items()
    }

    # Warmup
    out = model(**fixed_inputs)
    times = []
    for _ in range(REPEATS):
        # Fresh randomness per repeat for steady-state timing, but keep
        # output capture deterministic (we run fixed_inputs at end).
        live = {
            name: rng.standard_normal(tuple(int(d) for d in spec["shape"])).astype(np.float16)
            for name, spec in expected.items()
        }
        t0 = time.perf_counter()
        model(**live)
        times.append((time.perf_counter() - t0) * 1000.0)
    result["fwd_ms_median"] = round(statistics.median(times), 3)
    result["fwd_ms_min"] = round(min(times), 3)

    # Deterministic forward for PSNR comparison across variants.
    deterministic_out = model(**fixed_inputs)["noise_pred"]
    if ref_output is None:
        result["psnr_db_vs_none"] = None
    else:
        result["psnr_db_vs_none"] = round(psnr(ref_output, deterministic_out), 2)
    return result, deterministic_out


def main() -> int:
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        sha = "nogit"

    out_dir = REPO_ROOT / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[quant-matrix] git={sha} compute_unit={COMPUTE_UNIT} repeats={REPEATS}", file=sys.stderr)

    rows: list[dict[str, Any]] = []
    ref_output: np.ndarray | None = None
    for v in VARIANTS:
        print(f"[quant-matrix] measuring nbits={v} ...", file=sys.stderr)
        try:
            row, out_tensor = measure_variant(v, ref_output)
            if v == "none":
                ref_output = out_tensor
            rows.append(row)
        except Exception as exc:
            rows.append({"nbits": v, "error": f"{type(exc).__name__}: {exc}"})

    baseline_size = next((r["size_bytes"] for r in rows if r.get("nbits") == "none" and "size_bytes" in r), None)
    for r in rows:
        if "size_bytes" in r and baseline_size:
            r["size_ratio_vs_none"] = round(r["size_bytes"] / baseline_size, 3)

    report = {
        "git_sha": sha,
        "compute_unit": COMPUTE_UNIT,
        "repeats": REPEATS,
        "input_seed": INPUT_SEED,
        "ckpt_stem": CKPT_STEM,
        "rows": rows,
    }
    (out_dir / f"quant_matrix_{sha}.json").write_text(json.dumps(report, indent=2))

    lines = [
        f"# Quantization tradeoff matrix — `{sha}`",
        "",
        f"- compute unit: {COMPUTE_UNIT}",
        f"- repeats per variant: {REPEATS}",
        f"- ckpt: {CKPT_STEM}, {BATCH}x{WIDTH}x{HEIGHT}, attn={ATTN_SUFFIX}",
        "",
        "| nbits | size (MB) | size vs none | load (s) | fwd median (ms) | fwd min (ms) | PSNR vs none (dB) | error |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "error" in r and "size_bytes" not in r:
            lines.append(f"| {r.get('nbits','?')} | - | - | - | - | - | - | {r['error']} |")
            continue
        size_mb = round(r.get("size_bytes", 0) / (1024 * 1024), 1)
        lines.append(
            f"| {r['nbits']} | {size_mb} | {r.get('size_ratio_vs_none','-')} | "
            f"{r.get('load_time_s','-')} | {r.get('fwd_ms_median','-')} | "
            f"{r.get('fwd_ms_min','-')} | {r.get('psnr_db_vs_none') if r.get('psnr_db_vs_none') is not None else '-'} | "
            f"{r.get('error','')} |"
        )
    (out_dir / f"quant_matrix_{sha}.md").write_text("\n".join(lines) + "\n")
    print(f"[quant-matrix] wrote {out_dir / f'quant_matrix_{sha}.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
