# Quantization tradeoff matrix — `0bbd8d8`

- compute unit: CPU_AND_NE
- repeats per variant: 20
- ckpt: v1-5-pruned-emaonly, 1x512x512, attn=se

| nbits | size (MB) | size vs none | load (s) | fwd median (ms) | fwd min (ms) | PSNR vs none (dB) | error |
|---|---|---|---|---|---|---|---|
| none | 1641.1 | 1.0 | 0.1997 | 196.651 | 196.588 | - |  |
| 8 | 822.0 | 0.501 | 0.1475 | 185.223 | 183.373 | 53.54 |  |
| 6 | 617.1 | 0.376 | 0.1457 | 181.091 | 180.977 | 40.2 |  |
| 4 | 412.3 | 0.251 | 0.1427 | 178.953 | 178.753 | 27.47 |  |
