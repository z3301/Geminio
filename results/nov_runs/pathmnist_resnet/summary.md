# Held-out reruns: mean ± 95% CI over seeds

Target LPIPS/PSNR use the best-matching reconstruction for each target image. Hung. = Hungarian one-to-one matching (minimum total LPIPS); uniq = distinct reconstructions best-match used (mean over seeds; 8 = one-to-one, 1 = collapsed). Δ is the paired (same-seed) difference in best-match target LPIPS against the reference condition; positive Δ means worse reconstruction, i.e. less leakage.

## 1. Headline: NBFU eps=5 across attacks

| Condition | n | Target LPIPS | Target PSNR | Selectivity | Target LPIPS (Hung.) | Selectivity (Hung.) | uniq | Attack F1 | Reference | Δ target LPIPS | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pathmnist_tum_base_ho | 3 | 0.534 ± 0.034 | 14.67 ± 1.97 | 0.017 ± 0.008 | 0.747 ± 0.231 | 0.062 ± 0.195 | 1.0 ± 0.0 | 0.167 ± 0.179 | — | — | — |
| pathmnist_tum_base_dpsgd_eps5.0_ho | 3 | 0.802 ± 0.114 | 13.52 ± 0.69 | -0.006 ± 0.021 | 0.827 ± 0.094 | 0.007 ± 0.125 | 2.0 ± 2.5 | 0.083 ± 0.179 | pathmnist_tum_base_ho | 0.268 ± 0.132 | 0.013 |
| pathmnist_tum_base_ig_ho | 3 | 0.498 ± 0.088 | 15.57 ± 1.84 | 0.004 ± 0.045 | 0.677 ± 0.100 | 0.042 ± 0.149 | 1.3 ± 1.4 | 0.250 ± 0.311 | — | — | — |
| pathmnist_tum_base_ig_dpsgd_eps5.0_ho | 3 | 0.831 ± 0.016 | 13.50 ± 0.30 | -0.009 ± 0.021 | 0.859 ± 0.076 | -0.013 ± 0.108 | 2.3 ± 2.9 | 0.042 ± 0.179 | pathmnist_tum_base_ig_ho | 0.333 ± 0.104 | 0.005 |
## 2. Robust VLM as the attacker's guide

| Condition | n | Target LPIPS | Target PSNR | Selectivity | Target LPIPS (Hung.) | Selectivity (Hung.) | uniq | Attack F1 | Reference | Δ target LPIPS | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pathmnist_vlmclip_ho | 3 | 0.439 ± 0.021 | 16.66 ± 0.24 | -0.065 ± 0.004 | 0.550 ± 0.277 | -0.024 ± 0.350 | 3.7 ± 1.4 | 0.208 ± 0.474 | — | — | — |
| pathmnist_vlmsimclip2_ho | 3 | 0.440 ± 0.021 | 16.32 ± 0.74 | -0.050 ± 0.006 | 0.538 ± 0.372 | -0.021 ± 0.454 | 3.7 ± 1.4 | 0.375 ± 0.311 | pathmnist_vlmclip_ho | 0.001 ± 0.042 | 0.933 |
| pathmnist_vlmsimclip4_ho | 3 | 0.423 ± 0.022 | 16.21 ± 0.60 | -0.043 ± 0.022 | 0.604 ± 0.272 | -0.089 ± 0.372 | 3.0 ± 0.0 | 0.417 ± 0.179 | pathmnist_vlmclip_ho | -0.016 ± 0.044 | 0.249 |
| pathmnist_vlmfare2_ho | 3 | 0.450 ± 0.013 | 16.41 ± 1.28 | -0.063 ± 0.012 | 0.637 ± 0.021 | -0.075 ± 0.069 | 2.7 ± 1.4 | 0.250 ± 0.311 | pathmnist_vlmclip_ho | 0.011 ± 0.026 | 0.202 |
| pathmnist_vlmfare4_ho | 3 | 0.404 ± 0.037 | 16.07 ± 1.01 | -0.000 ± 0.034 | 0.652 ± 0.427 | -0.119 ± 0.491 | 3.0 ± 0.0 | 0.250 ± 0.000 | pathmnist_vlmclip_ho | -0.034 ± 0.048 | 0.092 |
| pathmnist_vlmtecoa2_ho | 3 | 0.439 ± 0.044 | 16.62 ± 0.67 | -0.035 ± 0.058 | 0.638 ± 0.353 | -0.059 ± 0.436 | 2.3 ± 1.4 | 0.333 ± 0.474 | pathmnist_vlmclip_ho | 0.000 ± 0.046 | 0.974 |
| pathmnist_vlmtecoa4_ho | 3 | 0.426 ± 0.061 | 16.48 ± 0.90 | -0.029 ± 0.012 | 0.580 ± 0.185 | 0.012 ± 0.193 | 2.7 ± 1.4 | 0.125 ± 0.000 | pathmnist_vlmclip_ho | -0.013 ± 0.039 | 0.288 |
