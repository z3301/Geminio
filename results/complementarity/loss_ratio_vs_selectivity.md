# Control A: head loss ratio vs selectivity at fixed backbone (ResNet-18, held-out batches)

Loss ratio = mean loss of the top-10% query-similar auxiliary images over the rest, from the language-sweep training logs. Outcomes are means over 3 seeds of the clean held-out runs; NBFU margin = target LPIPS at eps=5 minus clean. Spearman rho with two-sided p. Confound: query text varies with loss ratio (see the temperature sweep for the fixed-query test).

## medical: 12 queries, loss ratio 3.0x to 11.4x

| Outcome | rho vs loss ratio (per-query means, n=12) | p | rho (per run, n=36) | p | rho vs discriminability delta | p |
|---|---|---|---|---|---|---|
| selectivity | +0.90 | 0.000 | +0.86 | 0.000 | +0.74 | 0.006 |
| selectivity_hung | +0.78 | 0.003 | +0.74 | 0.000 | +0.57 | 0.055 |
| target_lpips | -0.59 | 0.044 | -0.61 | 0.000 | -0.61 | 0.036 |
| target_lpips_hung | -0.69 | 0.013 | -0.69 | 0.000 | -0.72 | 0.008 |
| n_unique_best | +0.35 | 0.270 | +0.27 | 0.105 | +0.33 | 0.288 |
| nbfu_margin | +0.64 | 0.024 | +0.61 | 0.000 | +0.67 | 0.017 |
| nbfu_margin_hung | +0.65 | 0.023 | +0.64 | 0.000 | +0.67 | 0.017 |

| Query | loss ratio | selectivity | selectivity (Hung.) | target LPIPS | target LPIPS (Hung.) | NBFU margin |
|---|---|---|---|---|---|---|
| pna_indirect | 3.05x | -0.018 | +0.006 | 0.930 | 1.007 | -0.114 |
| pna_synonym | 3.17x | -0.056 | +0.054 | 0.490 | 0.569 | +0.337 |
| pna_compOR | 5.12x | -0.016 | +0.002 | 0.699 | 0.792 | +0.132 |
| pna_negEX | 5.48x | -0.026 | -0.056 | 0.672 | 0.802 | +0.151 |
| pna_terse | 5.48x | -0.025 | -0.005 | 0.713 | 0.819 | +0.072 |
| pna_negPURE | 5.52x | +0.010 | +0.196 | 0.462 | 0.554 | +0.345 |
| pna_es | 6.01x | +0.025 | +0.315 | 0.526 | 0.624 | +0.295 |
| pna_base | 9.45x | +0.043 | +0.506 | 0.513 | 0.532 | +0.311 |
| pna_xlong | 10.67x | +0.062 | +0.394 | 0.485 | 0.595 | +0.300 |
| pna_verbose | 11.21x | +0.039 | +0.345 | 0.573 | 0.656 | +0.207 |
| pna_compAND | 11.36x | +0.065 | +0.392 | 0.497 | 0.501 | +0.374 |
| pna_typo | 11.44x | +0.087 | +0.399 | 0.398 | 0.403 | +0.426 |

## uav: 12 queries, loss ratio 31.0x to 165.4x

| Outcome | rho vs loss ratio (per-query means, n=12) | p | rho (per run, n=36) | p | rho vs discriminability delta | p |
|---|---|---|---|---|---|---|
| selectivity | +0.37 | 0.236 | +0.15 | 0.384 | -0.10 | 0.746 |
| selectivity_hung | +0.47 | 0.124 | +0.36 | 0.031 | +0.08 | 0.812 |
| target_lpips | -0.27 | 0.391 | -0.13 | 0.439 | +0.23 | 0.471 |
| target_lpips_hung | -0.20 | 0.527 | -0.18 | 0.299 | +0.15 | 0.649 |
| n_unique_best | -0.11 | 0.743 | -0.14 | 0.409 | -0.06 | 0.853 |
| nbfu_margin | +0.13 | 0.681 | -0.02 | 0.886 | -0.34 | 0.286 |
| nbfu_margin_hung | +0.05 | 0.880 | +0.08 | 0.663 | -0.14 | 0.665 |

| Query | loss ratio | selectivity | selectivity (Hung.) | target LPIPS | target LPIPS (Hung.) | NBFU margin |
|---|---|---|---|---|---|---|
| sol_typo | 31.03x | +0.096 | +0.162 | 0.493 | 0.547 | +0.189 |
| sol_es | 36.63x | +0.039 | +0.095 | 0.533 | 0.541 | +0.163 |
| sol_compOR | 40.27x | +0.033 | +0.043 | 0.544 | 0.667 | +0.147 |
| sol_negPURE | 44.81x | +0.128 | +0.209 | 0.441 | 0.442 | +0.251 |
| sol_synonym | 52.84x | +0.042 | +0.064 | 0.541 | 0.631 | +0.144 |
| sol_compAND | 92.27x | +0.065 | +0.137 | 0.527 | 0.582 | +0.161 |
| sol_negEX | 93.15x | +0.067 | +0.147 | 0.510 | 0.583 | +0.173 |
| sol_xlong | 106.26x | +0.117 | +0.196 | 0.462 | 0.463 | +0.216 |
| sol_verbose | 114.65x | +0.039 | +0.165 | 0.537 | 0.550 | +0.139 |
| sol_base | 130.54x | +0.081 | +0.200 | 0.509 | 0.536 | +0.176 |
| sol_indirect | 161.54x | +0.078 | +0.119 | 0.510 | 0.569 | +0.166 |
| sol_terse | 165.36x | +0.134 | +0.251 | 0.457 | 0.472 | +0.229 |

