# Spike 3 — MLX interop decision

**Question (from spec):** Should the suite stay pure Core ML/ANE, or add an
interop path with DiffusionKit / ComfyUI-MLX for SD3/Flux (GPU/MLX)? Recommend,
don't implement.

**Verdict:** **Stay pure Core ML/ANE.** Document an off-ramp (point users to
existing MLX nodes for Flux); do **not** absorb MLX into the suite. The niche
MLX cannot reach — low-power, background, GPU-free — is exactly this suite's
moat. The optional on-brand build is **SD3 Medium on ANE** (upstream-supported),
plus a power benchmark to make the moat quantitative.

## The two are complements, not substitutes

MLX runs on **GPU + CPU only — it cannot target the ANE** (confirmed: MLX
devices are `mx.cpu` / `mx.gpu`; the M5 "Neural Accelerators" are inside the
GPU via Metal 4, *not* the ANE). So MLX structurally cannot serve this suite's
niche.

| Axis | Core ML / ANE (this suite) | MLX (mflux / DiffusionKit) |
|------|----------------------------|----------------------------|
| Compute target | **ANE** (or GPU) | GPU + CPU only |
| Power draw | **Very low** (~5 W GEMM-level on M4) | High (~24 W; saturates GPU) |
| Thermal / fanless / background | **Wins** — cool, doesn't block GPU | Pins the GPU |
| Concurrent with GPU work | **Yes** (game/render/train running) | No — competes for GPU |
| Raw speed | Competitive on SD1.5/SDXL | **Faster** for Flux/SDXL when GPU free |
| Model coverage | SD1.5 / SDXL / **SD3** | **Flux**, FLUX.2, Qwen-Image, Z-Image; no SD3 |
| Conversion friction | High (convert+chunk+palettize ahead) | Low (load HF weights, quant on the fly) |

The power figures are **GEMM-level, not full-pipeline diffusion** — directional,
not citable per-image. See "Unverified".

## Ecosystem reality (2024–2026)

- **DiffusionKit** (argmaxinc) — MIT. Python side runs SD3 + Flux on **MLX**;
  Swift side does SD3 on Core ML (upstreamed to apple/ml-stable-diffusion).
  **Repo archived (read-only) March 2026** — effectively end-of-life. Building
  interop on it means depending on a dead repo.
- **mflux** (filipstrand) — MIT, **active** (v0.17.x, Apr 2026, ~2k stars). The
  healthy MLX-image engine: FLUX.1/FLUX.2, Qwen-Image, Z-Image, 4/8-bit quant.
  No SD3. **GPU only.**
- **Mflux-ComfyUI** (raysers) — MIT, wraps mflux; the most-starred Comfy MLX-Flux
  node. The Flux-on-Mac-ComfyUI niche is **already occupied** here.
- The original `thoddnn/ComfyUI-MLX` that blogs cite ("70% faster") **no longer
  exists** (404, deliberately removed). Treat those claims as referencing a dead
  repo.

## Can SD3 / Flux run on ANE via Core ML?

- **apple/ml-stable-diffusion** — MIT, 17.8k stars. Not abandoned but coasting:
  **last tagged release 1.1.1 (May 2024)**, main branch commits through
  **July 2025**. Maintained at a trickle.
- **SD3 on Core ML/ANE: YES.** The README has a `--sd3-version` conversion path
  (MMDiT, 16-ch VAE, T5), upstreamed from DiffusionKit. SD3 Medium ~2B params —
  within ANE feasibility with palettization/chunking, same playbook as SDXL.
- **Flux on Core ML/ANE: effectively NO.** Flux is ~12B params (~24 GB fp16
  transformer + ~9 GB T5). ANE wants aggressive chunking (this suite already
  bisects the 1.72 GB SD1.5 UNet into sub-1 GB chunks via `bisect_model`).
  Flux would need heavy low-bit palettization **and** a dozen-plus chunks just to
  fit, with precision risk on MMDiT attention. No maintained community Core ML
  Flux exists. Flux on Mac in practice = MLX/GPU (mflux).

## Maintenance burden if we went interop (rejected)

- Two heavy new deps: **MLX** (large native, Apple-silicon-only, moves fast) +
  **mflux** (DiffusionKit is archived, so mflux is the only live option). Both
  violate the spec's "no new runtime dependencies without flagging" rule
  (`MODERNIZATION_SPEC.md` §0.7) and would coexist with the already-fragile
  `python_coreml_stable_diffusion` git dependency — doubling the moving surface.
- The interop would be a thin re-wrap of `Mflux-ComfyUI`, which already exists
  and is MIT. Zero differentiation for real maintenance cost.

## Recommendation

| Option | Effort | Payoff | Verdict |
|--------|--------|--------|---------|
| **A. Stay pure Core ML/ANE; optionally extend to SD3 Medium** | Low–Med | Med | **Do** |
| **B. Document an off-ramp** in README (Flux/SD3-on-GPU → mflux / Mflux-ComfyUI; this suite = low-power / background / GPU-free / SD1.5/SDXL/SD3 on ANE) | Trivial | High | **Do** |
| C. Build MLX interop inside the suite (mflux dep) | High | Low | **Reject** |
| D. Build Flux-on-ANE Core ML conversion | Very high (research) | Low/uncertain | **Reject** |

Net: stay pure, lean into the moat, point users elsewhere for Flux. The market
already built the MLX-Flux bridge; this suite's value is the orthogonal axis MLX
provably cannot reach. The one new build worth considering is **SD3 Medium on
ANE** (feasible, upstream-supported, on-brand) plus a **power benchmark** to
make the moat quantitative.

## Unverified

- A clean full-pipeline **diffusion** joules-per-image ANE-vs-MLX-GPU benchmark
  (only GEMM-level ~5 W vs ~24 W on M4 exists). `bench/run.py` already measures
  `CPU_AND_NE` vs `CPU_AND_GPU` — generating this would be the highest-value
  artifact to prove the moat.
- Any working community Core ML Flux conversion (none found).
- Whether Argmax has a successor to DiffusionKit (archived Mar 2026, no announced
  replacement; the live MLX-Flux torch is carried by mflux).

## Sources

- argmaxinc/DiffusionKit (archived) — https://github.com/argmaxinc/DiffusionKit
- apple/ml-stable-diffusion (+ releases) — https://github.com/apple/ml-stable-diffusion / https://github.com/apple/ml-stable-diffusion/releases
- filipstrand/mflux — https://github.com/filipstrand/mflux
- raysers/Mflux-ComfyUI — https://github.com/raysers/Mflux-ComfyUI
- MLX (unified memory / devices) — https://ml-explore.github.io/mlx/
- Apple ML research, LLMs with MLX on M5 — https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- NPU vs GPU power (GEMM-level, directional) — arXiv 2511.13450
