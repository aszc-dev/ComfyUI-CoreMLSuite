# CoreMLSuite baseline — `ef2a18c`

- timestamp: 2026-05-22T13:07:25+00:00
- host: arm64 / Darwin 25.1.0
- python: 3.11.14
- coremltools: 8.2, numpy: 1.23.5
- settings: repeats=30, input_seed=0, assumed_steps=20

| label | kind | compute | size (MB) | load (s) | warmup (ms) | fwd median (ms) | fwd min (ms) | est e2e (ms) | peak RSS (MB) | error |
|---|---|---|---|---|---|---|---|---|---|---|
| v1-5-pruned-emaonly_1x512x512_se_unet-CPU_AND_NE | sd15 | CPU_AND_NE | 1641.1 | 0.2841 | 225.201 | 196.969 | 196.812 | 3939.4 | 349.7 |  |
| v1-5-pruned-emaonly_1x512x512_se_unet-CPU_AND_GPU | sd15 | CPU_AND_GPU | 1641.1 | 19.959 | 558.865 | 270.259 | 267.398 | 5405.2 | 2132.9 |  |

## Notes

- forward_ms is the latency of a single Core ML UNet forward pass with synthetic inputs; with classifier-free guidance one sampler step typically maps to one batched forward (cond+uncond) — interpret est_e2e_ms accordingly.
- peak_rss_mb is process RSS only and excludes ANE/wired GPU memory; treat as a floor.
- quality_psnr is null unless --reference-images and --candidate-images are provided.
- Latency varies run-to-run; 'min' is the most reproducible figure for comparisons.
