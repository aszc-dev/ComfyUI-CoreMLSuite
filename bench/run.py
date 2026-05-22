#!/usr/bin/env python3
"""
bench/run.py — Phase 1 baseline benchmark harness for ComfyUI-CoreMLSuite.

WHAT THIS MEASURES (the deterministic, easy-to-trust numbers)
-------------------------------------------------------------
For each converted Core ML UNet model in the matrix, in-process:
  * model load time (includes mlpackage compile; mlmodelc is precompiled)
  * model file size on disk
  * warmup (first forward) time
  * steady-state UNet forward latency (mean/std/min/median over N repeats)
  * an end-to-end estimate (median forward * assumed sampler steps)
  * peak process RSS during the run

It drives the Core ML model DIRECTLY with synthetic inputs shaped from the
model's own `expected_inputs`. This isolates the ANE/Core ML UNet latency —
the exact quantity that quantization (Phase 6) and the coremltools upgrade
(Phase 5) are expected to move — WITHOUT needing a live ComfyUI server, CLIP,
VAE, or a checkpoint. It is therefore the cleanest, most reproducible signal.

WHAT THIS DOES NOT MEASURE
--------------------------
  * Image quality. PSNR is a separate, decoupled concern: pass --reference-images
    and --candidate-images (e.g. the E2E-1.5-MPS / E2E-1.5-CoreML PNGs your
    existing integration workflow already produces) and PSNR is computed per
    matching filename. Without them, quality_psnr is null (and that's honest).
  * ANE / wired GPU memory. peak_rss_mb is *process* RSS only; treat as a coarse
    floor, not the true device footprint.

REQUIREMENTS
------------
  * Run on Apple Silicon (macOS) with the suite's deps installed
    (coremltools + apple/ml-stable-diffusion providing `python_coreml_stable_diffusion`).
  * numpy is required. psutil and Pillow are optional (graceful fallback).

USAGE
-----
  # Auto-discover every .mlmodelc/.mlpackage under a dir, sweep compute units:
  python bench/run.py --models-dir /path/to/ComfyUI/models/unet

  # Explicit matrix (recommended for a stable, committed baseline):
  python bench/run.py --matrix bench/matrix.json --models-dir /path/to/models/unet

  # With quality from already-produced images:
  python bench/run.py --models-dir ... \
      --reference-images /path/to/mps_pngs --candidate-images /path/to/coreml_pngs

matrix.json format (a list of configs):
  [
    {"label": "sd15-se",  "model": "dreamshaper_8_1x512x512_se.mlmodelc", "compute_unit": "CPU_AND_NE"},
    {"label": "sdxl-orig","model": "sdxl_base_1x1024x1024_orig.mlmodelc",  "compute_unit": "CPU_AND_GPU"}
  ]
Each "model" is resolved against --models-dir unless it is an absolute path.

Output: bench/results/<gitsha>.json  and  bench/results/<gitsha>.md
"""

import argparse
import dataclasses
import datetime as _dt
import glob
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ----- optional deps (graceful) ---------------------------------------------
try:
    import psutil  # type: ignore

    _PROC = psutil.Process(os.getpid())
except Exception:  # pragma: no cover - optional
    psutil = None
    _PROC = None

DEFAULT_COMPUTE_UNITS = ["CPU_AND_NE", "CPU_AND_GPU"]
COREML_EXTS = (".mlmodelc", ".mlpackage")


# ----- small helpers --------------------------------------------------------
def git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip() or "nogit"
    except Exception:
        return "nogit"


def coremltools_version() -> str:
    try:
        import coremltools as ct  # noqa: WPS433 (local import is intentional)

        return getattr(ct, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover
        return f"import-failed: {exc}"


def dir_size_bytes(path: Path) -> int:
    """Core ML models are directories (.mlmodelc / .mlpackage)."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def sample_rss_mb() -> Optional[float]:
    if _PROC is None:
        return None
    try:
        return _PROC.memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def detect_kind(expected_inputs: dict) -> str:
    """Mirror the suite's detection WITHOUT importing comfy (keeps bench light)."""
    if "time_ids" in expected_inputs and "text_embeds" in expected_inputs:
        try:
            n = expected_inputs["time_ids"]["shape"][1]
        except Exception:
            n = -1
        return "sdxl_base" if n == 6 else "sdxl_refiner" if n == 5 else "sdxl_unknown"
    if "timestep_cond" in expected_inputs:
        return "lcm"
    return "sd15"


def make_inputs(expected_inputs: dict, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Build synthetic fp16 inputs matching the model's expected shapes.

    Values are random; they do not affect latency in any meaningful way, and
    we feed fp16 to match how CoreMLModelWrapper feeds the model at runtime.
    """
    inputs: dict[str, np.ndarray] = {}
    for name, spec in expected_inputs.items():
        shape = tuple(int(d) for d in spec["shape"])
        inputs[name] = rng.standard_normal(shape).astype(np.float16)
    return inputs


def psnr(img_a: "np.ndarray", img_b: "np.ndarray") -> float:
    mse = float(np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2))
    if mse == 0:
        return 100.0
    return 20.0 * float(np.log10(255.0 / np.sqrt(mse)))


# ----- core measurement -----------------------------------------------------
@dataclasses.dataclass
class Config:
    label: str
    model_path: str
    compute_unit: str


def resolve_matrix(args) -> list[Config]:
    configs: list[Config] = []

    if args.matrix:
        entries = json.loads(Path(args.matrix).read_text())
        for e in entries:
            model = e["model"]
            if not os.path.isabs(model):
                if not args.models_dir:
                    raise SystemExit("matrix uses relative model names but --models-dir not given")
                model = os.path.join(args.models_dir, model)
            configs.append(Config(e.get("label", Path(model).stem), model, e.get("compute_unit", "CPU_AND_NE")))
        return configs

    if args.model:
        for m in args.model:
            for cu in args.compute_units:
                configs.append(Config(f"{Path(m).stem}-{cu}", m, cu))
        return configs

    if args.models_dir:
        found: list[str] = []
        for ext in COREML_EXTS:
            found += glob.glob(os.path.join(args.models_dir, f"*{ext}"))
        found = sorted(set(found))
        if not found:
            raise SystemExit(f"No {COREML_EXTS} models found under {args.models_dir}")
        for m in found:
            for cu in args.compute_units:
                configs.append(Config(f"{Path(m).stem}-{cu}", m, cu))
        return configs

    raise SystemExit("Provide one of: --matrix, --model, or --models-dir")


def measure(cfg: Config, args, rng: np.random.Generator) -> dict[str, Any]:
    from python_coreml_stable_diffusion.coreml_model import CoreMLModel  # local import: M2 only

    result: dict[str, Any] = {
        "label": cfg.label,
        "model_path": cfg.model_path,
        "compute_unit": cfg.compute_unit,
        "error": None,
    }
    path = Path(cfg.model_path)
    if not path.exists():
        result["error"] = "model path does not exist"
        return result

    sources = "compiled" if cfg.model_path.endswith(".mlmodelc") else "packages"
    peak_rss = sample_rss_mb()

    try:
        result["model_size_bytes"] = dir_size_bytes(path)

        t0 = time.perf_counter()
        model = CoreMLModel(cfg.model_path, cfg.compute_unit, sources)
        result["load_time_s"] = round(time.perf_counter() - t0, 4)
        peak_rss = max(filter(None, [peak_rss, sample_rss_mb()]), default=None)

        expected = dict(model.expected_inputs)
        result["kind"] = detect_kind(expected)
        result["expected_inputs"] = {k: list(v["shape"]) for k, v in expected.items()}

        # Pre-generate all input sets so timing excludes array allocation.
        warm_in = make_inputs(expected, rng)
        step_inputs = [make_inputs(expected, rng) for _ in range(args.repeats)]

        # Warmup (first forward — ANE prepares its graph here).
        t0 = time.perf_counter()
        out = model(**warm_in)
        result["warmup_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        if not (isinstance(out, dict) and "noise_pred" in out):
            result["error"] = "unexpected output (no 'noise_pred')"
        peak_rss = max(filter(None, [peak_rss, sample_rss_mb()]), default=None)

        # Steady-state forward latency.
        times_ms: list[float] = []
        for inp in step_inputs:
            t0 = time.perf_counter()
            model(**inp)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            rss = sample_rss_mb()
            if rss is not None:
                peak_rss = rss if peak_rss is None else max(peak_rss, rss)

        result["forward_ms"] = {
            "mean": round(statistics.fmean(times_ms), 3),
            "std": round(statistics.pstdev(times_ms), 3) if len(times_ms) > 1 else 0.0,
            "min": round(min(times_ms), 3),
            "median": round(statistics.median(times_ms), 3),
            "n": len(times_ms),
        }
        result["est_e2e_ms"] = round(result["forward_ms"]["median"] * args.assumed_steps, 1)
        result["peak_rss_mb"] = round(peak_rss, 1) if peak_rss is not None else None

    except Exception as exc:  # keep one bad model from killing the whole run
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def compute_quality(reference_dir: str, candidate_dir: str) -> dict[str, Any]:
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {"error": f"Pillow not available: {exc}"}

    ref = {p.name: p for p in Path(reference_dir).glob("*.png")}
    cand = {p.name: p for p in Path(candidate_dir).glob("*.png")}
    common = sorted(set(ref) & set(cand))
    pairs: dict[str, Any] = {}
    for name in common:
        try:
            a = np.array(Image.open(ref[name]).convert("RGB"))
            b = np.array(Image.open(cand[name]).convert("RGB"))
            if a.shape != b.shape:
                pairs[name] = {"error": f"shape mismatch {a.shape} vs {b.shape}"}
                continue
            pairs[name] = {"psnr_db": round(psnr(a, b), 2)}
        except Exception as exc:
            pairs[name] = {"error": str(exc)}
    return {
        "reference_dir": reference_dir,
        "candidate_dir": candidate_dir,
        "matched": len(common),
        "pairs": pairs,
    }


# ----- reporting ------------------------------------------------------------
def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# CoreMLSuite baseline — `{report['git_sha']}`",
        "",
        f"- timestamp: {report['timestamp']}",
        f"- host: {report['host']['machine']} / {report['host']['system']} {report['host']['release']}",
        f"- python: {report['host']['python']}",
        f"- coremltools: {report['tool_versions']['coremltools']}, numpy: {report['tool_versions']['numpy']}",
        f"- settings: repeats={report['settings']['repeats']}, "
        f"input_seed={report['settings']['input_seed']}, "
        f"assumed_steps={report['settings']['assumed_steps']}",
        "",
        "| label | kind | compute | size (MB) | load (s) | warmup (ms) | fwd median (ms) | fwd min (ms) | est e2e (ms) | peak RSS (MB) | error |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["results"]:
        if r.get("error") and "forward_ms" not in r:
            lines.append(
                f"| {r['label']} | - | {r['compute_unit']} | - | - | - | - | - | - | - | {r['error']} |"
            )
            continue
        size_mb = round(r.get("model_size_bytes", 0) / (1024 * 1024), 1)
        fwd = r.get("forward_ms", {})
        lines.append(
            f"| {r['label']} | {r.get('kind','?')} | {r['compute_unit']} | {size_mb} | "
            f"{r.get('load_time_s','-')} | {r.get('warmup_ms','-')} | {fwd.get('median','-')} | "
            f"{fwd.get('min','-')} | {r.get('est_e2e_ms','-')} | {r.get('peak_rss_mb','-')} | "
            f"{r.get('error') or ''} |"
        )

    q = report.get("quality")
    if q and q.get("pairs"):
        lines += ["", "## Quality (PSNR vs reference)", "", "| image | PSNR (dB) |", "|---|---|"]
        for name, val in q["pairs"].items():
            lines.append(f"| {name} | {val.get('psnr_db', val.get('error','?'))} |")

    if report.get("notes"):
        lines += ["", "## Notes", ""] + [f"- {n}" for n in report["notes"]]
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="CoreMLSuite Phase-1 baseline benchmark")
    ap.add_argument("--matrix", help="path to matrix.json (list of {label, model, compute_unit})")
    ap.add_argument("--models-dir", help="dir holding .mlmodelc/.mlpackage models")
    ap.add_argument("--model", action="append", help="explicit model path (repeatable)")
    ap.add_argument("--compute-units", nargs="+", default=DEFAULT_COMPUTE_UNITS,
                    help="compute units to sweep in auto/--model mode")
    ap.add_argument("--repeats", type=int, default=30, help="steady-state forward passes")
    ap.add_argument("--assumed-steps", type=int, default=20,
                    help="sampler steps used ONLY for the est_e2e_ms estimate")
    ap.add_argument("--input-seed", type=int, default=0, help="seed for synthetic input generation")
    ap.add_argument("--reference-images", help="dir of reference PNGs (e.g. MPS) for optional PSNR")
    ap.add_argument("--candidate-images", help="dir of candidate PNGs (e.g. CoreML) for optional PSNR")
    ap.add_argument("--out-dir", default="bench/results", help="where to write <gitsha>.json/.md")
    args = ap.parse_args(argv)

    sha = git_sha()
    rng = np.random.default_rng(args.input_seed)
    configs = resolve_matrix(args)

    notes = [
        "forward_ms is the latency of a single Core ML UNet forward pass with synthetic "
        "inputs; with classifier-free guidance one sampler step typically maps to one "
        "batched forward (cond+uncond) — interpret est_e2e_ms accordingly.",
        "peak_rss_mb is process RSS only and excludes ANE/wired GPU memory; treat as a floor.",
        "quality_psnr is null unless --reference-images and --candidate-images are provided.",
        "Latency varies run-to-run; 'min' is the most reproducible figure for comparisons.",
    ]
    if psutil is None:
        notes.append("psutil not installed -> peak_rss_mb is null. `pip install psutil` to capture it.")

    print(f"[bench] git={sha} coremltools={coremltools_version()} configs={len(configs)}", file=sys.stderr)

    results = []
    for cfg in configs:
        print(f"[bench] measuring {cfg.label} ({cfg.compute_unit}) ...", file=sys.stderr)
        results.append(measure(cfg, args, rng))

    report: dict[str, Any] = {
        "git_sha": sha,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "host": {
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "tool_versions": {"coremltools": coremltools_version(), "numpy": np.__version__},
        "settings": {
            "repeats": args.repeats,
            "assumed_steps": args.assumed_steps,
            "input_seed": args.input_seed,
            "compute_units": args.compute_units,
        },
        "results": results,
        "notes": notes,
    }

    if args.reference_images and args.candidate_images:
        report["quality"] = compute_quality(args.reference_images, args.candidate_images)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{sha}.json"
    md_path = out_dir / f"{sha}.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(to_markdown(report))

    print(f"[bench] wrote {json_path}", file=sys.stderr)
    print(f"[bench] wrote {md_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
