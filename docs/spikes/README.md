# Phase 7 — Strategic spikes (research only)

Research-only investigations per `MODERNIZATION_SPEC.md` Phase 7. None of these
are merged behavior changes. Each spike delivers a written recommendation with
effort/payoff so the maintainer can choose which (if any) become real phases.

| # | Spike | Verdict | Effort | Payoff |
|---|-------|---------|--------|--------|
| 1 | [MultiFunction models](01-multifunction-models.md) | Prototype for **controlnet-on/off only**; use enumerated shapes for batch | Medium | Medium (gated by macOS 15) |
| 2 | [Flexible / enumerated shapes](02-flexible-shapes.md) | `EnumeratedShapes` for **spatial size**; keep `chunk_batch` for batch | Medium | Medium–High |
| 3 | [MLX interop decision](03-mlx-interop.md) | **Stay pure Core ML/ANE.** Document an off-ramp; do not absorb MLX | Trivial (docs) | High (positioning) |

## Cross-cutting finding

Spikes 1 and 2 both run into the same wall: the high-value packaging features
(`MultiFunctionDescriptor`, `materialize_dynamic_shape_mlmodel`, multi-input
`EnumeratedShapes`) require **macOS 15 / iOS 18** at runtime, while the suite
today targets **macOS 13** (`coreml_suite/converter.py:110`). Adopting either
means either raising the floor (drops older-OS users — a real cost for a
community ComfyUI pack) or keeping the current per-config path as a fallback
(erodes the "one artifact" simplicity win). This OS gate, not the API, is the
dominant decision factor.

## Gate 7 report

```
## Gate 7 report
- Spike 1 (MultiFunction): recommendation written — prototype controlnet-on/off
  as 2-function package; NOT for batch size. Gated macOS 15. No code merged.
- Spike 2 (flexible shapes): recommendation written — EnumeratedShapes for
  spatial size, retains ANE per coremltools maintainer confirmation; keep
  chunk_batch for batch. No code merged.
- Spike 3 (MLX interop): recommendation written — stay pure Core ML/ANE,
  document off-ramp to mflux/Mflux-ComfyUI for Flux. Reject in-suite interop.
  No code merged.
- Open verification gaps requiring on-hardware [M2-ANE] measurement are listed
  per spike under "Unverified".
```

## Verification debt (must measure on [M2-ANE] before any promotion)

- EnumeratedShapes ANE residency for a *real* SD UNet (one unreproduced
  coremltools regression report exists — spike 2).
- Multifunction SD UNet ANE behavior + first-load compile time (no public
  report either way — spike 1).
- Full-pipeline diffusion power draw ANE vs MLX-GPU (only GEMM-level figures
  exist — spike 3). `bench/run.py` already measures `CPU_AND_NE` vs
  `CPU_AND_GPU`, so this is the highest-value artifact to make the moat
  quantitative.
