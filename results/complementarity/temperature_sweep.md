# Control C: temperature sweep (fixed query and backbone, loss ratio varied through T)

ResNet-18 client, held-out batch, attack seeds 42/123/256. One head per (T, replicate); rep-1 heads share init seed 0, rep-2 heads share init seed 1, and the T = 100 rep-1 head is the unseeded language-sweep base head. Its second attack run (ce_resnet18 tag) enters only the repeatability table. Primary statistics are over heads (loss ratio is a head property); per-run statistics are pseudo-replicated and secondary. Metrics re-scored from saved tensors.

## medical

### Run manifest: 10 heads, 30 attack runs (+3 same-seed re-attacks of the base head, repeatability only)

| head | T | rep | init seed | loss ratio | n runs | target_lpips | selectivity | target_lpips_hung | selectivity_hung | nontarget_lpips | nontarget_lpips_hung | n_unique_best | lpips |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T10_r1 | 10 | 1 | 0 | 9.32x | 3 | 0.605 | -0.001 | 0.655 | 0.239 | 0.604 | 0.894 | 2.667 | 0.842 |
| T10_r2 | 10 | 2 | 1 | 9.50x | 3 | 0.689 | 0.027 | 0.698 | 0.322 | 0.715 | 1.020 | 2.000 | 0.942 |
| T30_r1 | 30 | 1 | 0 | 9.35x | 3 | 1.008 | -0.019 | 1.072 | -0.006 | 0.989 | 1.066 | 1.667 | 1.070 |
| T30_r2 | 30 | 2 | 1 | 10.13x | 3 | 0.494 | 0.047 | 0.510 | 0.566 | 0.542 | 1.075 | 2.000 | 0.935 |
| T100_r1 | 100 | 1 | unseeded | 9.45x | 3 | 0.513 | 0.043 | 0.532 | 0.506 | 0.556 | 1.037 | 2.000 | 0.915 |
| T100_r2 | 100 | 2 | 1 | 16.79x | 3 | 0.446 | 0.096 | 0.453 | 0.398 | 0.542 | 0.851 | 2.000 | 0.756 |
| T300_r1 | 300 | 1 | 0 | 12.44x | 3 | 0.660 | 0.024 | 0.677 | 0.462 | 0.684 | 1.139 | 1.333 | 1.029 |
| T300_r2 | 300 | 2 | 1 | 11.03x | 3 | 0.429 | 0.072 | 0.504 | 0.487 | 0.501 | 0.992 | 1.000 | 0.876 |
| T1000_r1 | 1000 | 1 | 0 | 13.34x | 3 | 0.687 | 0.007 | 0.765 | 0.335 | 0.694 | 1.100 | 1.333 | 1.018 |
| T1000_r2 | 1000 | 2 | 1 | 13.90x | 3 | 0.467 | 0.084 | 0.482 | 0.450 | 0.552 | 0.932 | 2.333 | 0.825 |

### Per-T means over heads and attack seeds (duplicates excluded)

| T | heads | mean loss ratio | n runs | target_lpips | selectivity | target_lpips_hung | selectivity_hung | nontarget_lpips | nontarget_lpips_hung | n_unique_best | lpips |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 2 | 9.41x | 6 | 0.647 | 0.013 | 0.676 | 0.280 | 0.660 | 0.957 | 2.333 | 0.892 |
| 30 | 2 | 9.74x | 6 | 0.751 | 0.014 | 0.791 | 0.280 | 0.765 | 1.071 | 1.833 | 1.002 |
| 100 | 2 | 13.12x | 6 | 0.480 | 0.069 | 0.492 | 0.452 | 0.549 | 0.944 | 2.000 | 0.836 |
| 300 | 2 | 11.73x | 6 | 0.545 | 0.048 | 0.591 | 0.475 | 0.593 | 1.065 | 1.167 | 0.952 |
| 1000 | 2 | 13.62x | 6 | 0.577 | 0.046 | 0.623 | 0.393 | 0.623 | 1.016 | 1.833 | 0.921 |

### Head-level Spearman (PRIMARY; n = 10 heads, one mean per head)

| metric | rho vs loss ratio | p | leave-one-head-out rho range | p range | rho vs log T | p |
|---|---|---|---|---|---|---|
| target_lpips | -0.48 | 0.162 | -0.62 to -0.35 | 0.077 to 0.356 | -0.34 | 0.329 |
| selectivity | +0.67 | 0.033 | +0.55 to +0.80 | 0.010 to 0.125 | +0.34 | 0.329 |
| target_lpips_hung | -0.52 | 0.128 | -0.70 to -0.33 | 0.036 to 0.381 | -0.25 | 0.493 |
| selectivity_hung | +0.28 | 0.425 | +0.03 to +0.45 | 0.224 to 0.932 | +0.34 | 0.329 |
| nontarget_lpips | -0.37 | 0.293 | -0.50 to -0.20 | 0.170 to 0.606 | -0.25 | 0.493 |
| nontarget_lpips_hung | -0.07 | 0.855 | -0.32 to +0.28 | 0.406 to 0.932 | +0.22 | 0.538 |
| n_unique_best | -0.21 | 0.554 | -0.49 to +0.10 | 0.181 to 0.806 | -0.43 | 0.211 |
| lpips | -0.33 | 0.347 | -0.55 to -0.08 | 0.125 to 0.831 | -0.07 | 0.839 |

### T effect within each replicate (per-T head means, Spearman vs log T, n = 5 heads each)

| metric | rep 1 rho (p) | rep 2 rho (p) |
|---|---|---|
| target_lpips | +0.20 (0.747) | -0.70 (0.188) |
| selectivity | +0.50 (0.391) | +0.70 (0.188) |
| target_lpips_hung | +0.20 (0.747) | -0.70 (0.188) |
| selectivity_hung | +0.50 (0.391) | +0.30 (0.624) |
| nontarget_lpips | +0.20 (0.747) | -0.30 (0.624) |
| nontarget_lpips_hung | +0.80 (0.104) | -0.60 (0.285) |
| n_unique_best | -0.87 (0.054) | +0.22 (0.718) |
| lpips | +0.30 (0.624) | -0.70 (0.188) |

Loss ratio by T within replicate: rep 1: 9.32 / 9.35 / 9.45 / 12.44 / 13.34; rep 2: 9.50 / 10.13 / 16.79 / 11.03 / 13.90.

### Per-run Spearman (SECONDARY, pseudo-replicated; n = 30 runs, duplicates excluded)

| metric | rho vs loss ratio | p | rho vs log T | p |
|---|---|---|---|---|
| target_lpips | -0.48 | 0.008 | -0.28 | 0.129 |
| selectivity | +0.62 | 0.000 | +0.36 | 0.051 |
| target_lpips_hung | -0.49 | 0.006 | -0.21 | 0.260 |
| selectivity_hung | +0.27 | 0.149 | +0.31 | 0.095 |
| nontarget_lpips | -0.32 | 0.089 | -0.20 | 0.292 |
| nontarget_lpips_hung | -0.09 | 0.629 | +0.17 | 0.365 |
| n_unique_best | -0.23 | 0.218 | -0.31 | 0.096 |
| lpips | -0.29 | 0.118 | -0.01 | 0.966 |

### Repeatability of one head: base head attacked twice with the same attack seed (3 pairs; a description of run-to-run variation, not a detection threshold)

| metric | seed 42 | seed 123 | seed 256 | mean abs | max abs |
|---|---|---|---|---|---|
| target_lpips | +0.024 | -0.108 | -0.099 | 0.077 | 0.108 |
| selectivity | +0.014 | +0.034 | +0.019 | 0.022 | 0.034 |
| target_lpips_hung | +0.043 | -0.515 | -0.128 | 0.229 | 0.515 |
| selectivity_hung | -0.107 | +0.504 | +0.032 | 0.214 | 0.504 |
| nontarget_lpips | +0.038 | -0.074 | -0.080 | 0.064 | 0.080 |
| nontarget_lpips_hung | -0.064 | -0.011 | -0.097 | 0.057 | 0.097 |
| n_unique_best | +0.000 | +1.000 | +1.000 | 0.667 | 1.000 |
| lpips | -0.035 | -0.135 | -0.106 | 0.092 | 0.135 |

### Between-replicate difference at fixed T (rep 1 − rep 2 head means; the two heads share T but differ in init seed and training trajectory) vs the between-T range of pooled means

| metric | T=10 | T=30 | T=100 | T=300 | T=1000 | between-T range |
|---|---|---|---|---|---|---|
| target_lpips | -0.084 | +0.514 | +0.067 | +0.231 | +0.219 | 0.271 |
| selectivity | -0.027 | -0.067 | -0.053 | -0.049 | -0.077 | 0.056 |
| target_lpips_hung | -0.043 | +0.563 | +0.079 | +0.173 | +0.283 | 0.299 |
| selectivity_hung | -0.083 | -0.572 | +0.108 | -0.025 | -0.115 | 0.195 |
| nontarget_lpips | -0.111 | +0.447 | +0.014 | +0.183 | +0.142 | 0.216 |
| nontarget_lpips_hung | -0.126 | -0.009 | +0.186 | +0.148 | +0.168 | 0.126 |
| n_unique_best | +0.667 | -0.333 | +0.000 | +0.333 | -1.000 | 1.167 |
| lpips | -0.100 | +0.135 | +0.159 | +0.153 | +0.193 | 0.167 |

## uav

### Run manifest: 10 heads, 30 attack runs (+3 same-seed re-attacks of the base head, repeatability only)

| head | T | rep | init seed | loss ratio | n runs | target_lpips | selectivity | target_lpips_hung | selectivity_hung | nontarget_lpips | nontarget_lpips_hung | n_unique_best | lpips |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T10_r1 | 10 | 1 | 0 | 18.92x | 3 | 0.578 | 0.005 | 0.584 | 0.055 | 0.583 | 0.639 | 3.333 | 0.627 |
| T10_r2 | 10 | 2 | 1 | 98.18x | 3 | 0.401 | 0.190 | 0.500 | 0.252 | 0.591 | 0.752 | 1.000 | 0.692 |
| T30_r1 | 30 | 1 | 0 | 66.56x | 3 | 0.512 | 0.069 | 0.634 | 0.097 | 0.581 | 0.730 | 1.000 | 0.708 |
| T30_r2 | 30 | 2 | 1 | 134.10x | 3 | 0.481 | 0.108 | 0.539 | 0.168 | 0.589 | 0.707 | 1.333 | 0.667 |
| T100_r1 | 100 | 1 | unseeded | 130.54x | 3 | 0.509 | 0.081 | 0.536 | 0.200 | 0.589 | 0.736 | 1.333 | 0.690 |
| T100_r2 | 100 | 2 | 1 | 111.18x | 3 | 0.535 | 0.049 | 0.612 | 0.157 | 0.584 | 0.769 | 1.000 | 0.734 |
| T300_r1 | 300 | 1 | 0 | 107.70x | 3 | 0.507 | 0.093 | 0.592 | 0.124 | 0.600 | 0.716 | 1.000 | 0.688 |
| T300_r2 | 300 | 2 | 1 | 111.50x | 3 | 0.538 | 0.073 | 0.553 | 0.173 | 0.612 | 0.726 | 1.667 | 0.684 |
| T1000_r1 | 1000 | 1 | 0 | 60.53x | 3 | 0.561 | 0.017 | 0.600 | 0.071 | 0.578 | 0.670 | 2.333 | 0.654 |
| T1000_r2 | 1000 | 2 | 1 | 104.33x | 3 | 0.485 | 0.095 | 0.546 | 0.173 | 0.581 | 0.719 | 1.333 | 0.679 |

### Per-T means over heads and attack seeds (duplicates excluded)

| T | heads | mean loss ratio | n runs | target_lpips | selectivity | target_lpips_hung | selectivity_hung | nontarget_lpips | nontarget_lpips_hung | n_unique_best | lpips |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 2 | 58.55x | 6 | 0.489 | 0.097 | 0.542 | 0.153 | 0.587 | 0.695 | 2.167 | 0.660 |
| 30 | 2 | 100.33x | 6 | 0.496 | 0.089 | 0.586 | 0.132 | 0.585 | 0.719 | 1.167 | 0.688 |
| 100 | 2 | 120.86x | 6 | 0.522 | 0.065 | 0.574 | 0.179 | 0.587 | 0.753 | 1.167 | 0.712 |
| 300 | 2 | 109.60x | 6 | 0.523 | 0.083 | 0.573 | 0.149 | 0.606 | 0.721 | 1.333 | 0.686 |
| 1000 | 2 | 82.43x | 6 | 0.523 | 0.056 | 0.573 | 0.122 | 0.579 | 0.695 | 1.833 | 0.666 |

### Head-level Spearman (PRIMARY; n = 10 heads, one mean per head)

| metric | rho vs loss ratio | p | leave-one-head-out rho range | p range | rho vs log T | p |
|---|---|---|---|---|---|---|
| target_lpips | -0.43 | 0.214 | -0.60 to -0.22 | 0.088 to 0.576 | +0.15 | 0.684 |
| selectivity | +0.50 | 0.138 | +0.32 to +0.68 | 0.042 to 0.406 | -0.07 | 0.839 |
| target_lpips_hung | -0.41 | 0.244 | -0.58 to -0.28 | 0.099 to 0.460 | +0.22 | 0.538 |
| selectivity_hung | +0.61 | 0.060 | +0.47 to +0.82 | 0.007 to 0.205 | -0.02 | 0.946 |
| nontarget_lpips | +0.56 | 0.090 | +0.45 to +0.68 | 0.042 to 0.224 | -0.20 | 0.586 |
| nontarget_lpips_hung | +0.35 | 0.328 | +0.10 to +0.57 | 0.112 to 0.798 | -0.15 | 0.684 |
| n_unique_best | -0.21 | 0.562 | -0.32 to +0.11 | 0.402 to 0.785 | +0.18 | 0.618 |
| lpips | +0.25 | 0.489 | -0.03 to +0.43 | 0.244 to 0.932 | -0.17 | 0.634 |

### T effect within each replicate (per-T head means, Spearman vs log T, n = 5 heads each)

| metric | rep 1 rho (p) | rep 2 rho (p) |
|---|---|---|
| target_lpips | -0.40 (0.505) | +0.70 (0.188) |
| selectivity | +0.40 (0.505) | -0.60 (0.285) |
| target_lpips_hung | +0.20 (0.747) | +0.60 (0.285) |
| selectivity_hung | +0.30 (0.624) | -0.20 (0.747) |
| nontarget_lpips | -0.10 (0.873) | -0.40 (0.505) |
| nontarget_lpips_hung | +0.10 (0.873) | -0.20 (0.747) |
| n_unique_best | -0.21 (0.741) | +0.58 (0.306) |
| lpips | +0.00 (1.000) | -0.20 (0.747) |

Loss ratio by T within replicate: rep 1: 18.92 / 66.56 / 130.54 / 107.70 / 60.53; rep 2: 98.18 / 134.10 / 111.18 / 111.50 / 104.33.

### Per-run Spearman (SECONDARY, pseudo-replicated; n = 30 runs, duplicates excluded)

| metric | rho vs loss ratio | p | rho vs log T | p |
|---|---|---|---|---|
| target_lpips | -0.32 | 0.085 | +0.15 | 0.430 |
| selectivity | +0.44 | 0.015 | -0.11 | 0.557 |
| target_lpips_hung | -0.25 | 0.185 | +0.01 | 0.954 |
| selectivity_hung | +0.52 | 0.003 | +0.13 | 0.510 |
| nontarget_lpips | +0.49 | 0.006 | -0.01 | 0.977 |
| nontarget_lpips_hung | +0.33 | 0.077 | -0.02 | 0.932 |
| n_unique_best | -0.30 | 0.105 | -0.01 | 0.959 |
| lpips | +0.30 | 0.103 | +0.04 | 0.841 |

### Repeatability of one head: base head attacked twice with the same attack seed (3 pairs; a description of run-to-run variation, not a detection threshold)

| metric | seed 42 | seed 123 | seed 256 | mean abs | max abs |
|---|---|---|---|---|---|
| target_lpips | -0.003 | +0.096 | +0.004 | 0.034 | 0.096 |
| selectivity | +0.011 | -0.102 | -0.008 | 0.040 | 0.102 |
| target_lpips_hung | +0.021 | +0.036 | +0.006 | 0.021 | 0.036 |
| selectivity_hung | -0.017 | +0.025 | +0.008 | 0.017 | 0.025 |
| nontarget_lpips | +0.008 | -0.007 | -0.004 | 0.006 | 0.008 |
| nontarget_lpips_hung | +0.004 | +0.061 | +0.014 | 0.026 | 0.061 |
| n_unique_best | +0.000 | +1.000 | -1.000 | 0.667 | 1.000 |
| lpips | +0.006 | +0.062 | +0.015 | 0.028 | 0.062 |

### Between-replicate difference at fixed T (rep 1 − rep 2 head means; the two heads share T but differ in init seed and training trajectory) vs the between-T range of pooled means

| metric | T=10 | T=30 | T=100 | T=300 | T=1000 | between-T range |
|---|---|---|---|---|---|---|
| target_lpips | +0.178 | +0.031 | -0.026 | -0.031 | +0.076 | 0.034 |
| selectivity | -0.185 | -0.039 | +0.032 | +0.019 | -0.078 | 0.041 |
| target_lpips_hung | +0.084 | +0.094 | -0.075 | +0.039 | +0.053 | 0.044 |
| selectivity_hung | -0.197 | -0.071 | +0.043 | -0.049 | -0.102 | 0.057 |
| nontarget_lpips | -0.008 | -0.008 | +0.006 | -0.012 | -0.002 | 0.026 |
| nontarget_lpips_hung | -0.113 | +0.023 | -0.033 | -0.010 | -0.049 | 0.058 |
| n_unique_best | +2.333 | -0.333 | +0.333 | -0.667 | +1.000 | 1.000 |
| lpips | -0.065 | +0.042 | -0.044 | +0.004 | -0.024 | 0.052 |

