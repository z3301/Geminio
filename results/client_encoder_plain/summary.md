# client_encoder_plain: plain head vs rigged head, same encoder and seed

All image metrics are re-scored from the saved tensors (best-match and Hungarian matching). Per-encoder Δ is paired over seeds (t-test, df = n−1); the aggregate line reports the mean of the per-encoder Δs with a Wilcoxon signed-rank test over encoders and over all run pairs. Sign convention: plain minus rigged (positive target-LPIPS Δ = the plain head leaks less).

## medical: plain − rigged

### medical: Target LPIPS, best-match (lower = more leakage)

| Encoder | rigged | plain | Δ (plain − rigged) ± 95% CI | paired p |
|---|---|---|---|---|
| clip | 0.867 ± 0.096 | 0.868 ± 0.048 | 0.001 ± 0.104 | 0.961 |
| simclip2 | 0.762 ± 0.037 | 0.750 ± 0.067 | -0.013 ± 0.042 | 0.322 |
| simclip4 | 0.768 ± 0.046 | 0.796 ± 0.169 | 0.027 ± 0.215 | 0.637 |
| fare2 | 0.795 ± 0.071 | 0.763 ± 0.156 | -0.032 ± 0.090 | 0.263 |
| fare4 | 0.771 ± 0.015 | 0.738 ± 0.163 | -0.032 ± 0.148 | 0.446 |
| tecoa2 | 0.790 ± 0.031 | 0.669 ± 0.132 | -0.122 ± 0.101 | 0.035 |
| tecoa4 | 0.775 ± 0.024 | 0.751 ± 0.041 | -0.024 ± 0.036 | 0.101 |
| **all** | | | **-0.028** (2/7 encoders > 0, Wilcoxon p = 0.156) | 6/21 run pairs > 0, Wilcoxon p = 0.019 |

### medical: Selectivity, best-match (non-target minus target LPIPS)

| Encoder | rigged | plain | Δ (plain − rigged) ± 95% CI | paired p |
|---|---|---|---|---|
| clip | -0.029 ± 0.005 | -0.027 ± 0.003 | 0.001 ± 0.005 | 0.349 |
| simclip2 | -0.039 ± 0.009 | -0.025 ± 0.005 | 0.014 ± 0.010 | 0.026 |
| simclip4 | -0.033 ± 0.019 | -0.014 ± 0.016 | 0.019 ± 0.015 | 0.030 |
| fare2 | -0.048 ± 0.017 | -0.008 ± 0.021 | 0.040 ± 0.011 | 0.004 |
| fare4 | -0.041 ± 0.020 | -0.011 ± 0.024 | 0.029 ± 0.030 | 0.053 |
| tecoa2 | -0.050 ± 0.006 | -0.014 ± 0.019 | 0.036 ± 0.013 | 0.007 |
| tecoa4 | -0.042 ± 0.017 | -0.018 ± 0.013 | 0.024 ± 0.029 | 0.072 |
| **all** | | | **+0.023** (7/7 encoders > 0, Wilcoxon p = 0.016) | 21/21 run pairs > 0, Wilcoxon p = 0.000 |

### medical: Target LPIPS, Hungarian one-to-one

| Encoder | rigged | plain | Δ (plain − rigged) ± 95% CI | paired p |
|---|---|---|---|---|
| clip | 1.074 ± 0.272 | 1.070 ± 0.368 | -0.004 ± 0.597 | 0.981 |
| simclip2 | 0.863 ± 0.373 | 0.887 ± 0.072 | 0.024 ± 0.441 | 0.835 |
| simclip4 | 0.824 ± 0.099 | 1.063 ± 0.286 | 0.239 ± 0.275 | 0.065 |
| fare2 | 0.832 ± 0.132 | 0.926 ± 0.111 | 0.094 ± 0.128 | 0.088 |
| fare4 | 0.808 ± 0.095 | 1.005 ± 0.460 | 0.198 ± 0.495 | 0.228 |
| tecoa2 | 0.822 ± 0.048 | 0.882 ± 0.382 | 0.061 ± 0.417 | 0.595 |
| tecoa4 | 0.831 ± 0.079 | 0.990 ± 0.099 | 0.159 ± 0.101 | 0.021 |
| **all** | | | **+0.110** (6/7 encoders > 0, Wilcoxon p = 0.031) | 18/21 run pairs > 0, Wilcoxon p = 0.009 |

### medical: Selectivity, Hungarian one-to-one

| Encoder | rigged | plain | Δ (plain − rigged) ± 95% CI | paired p |
|---|---|---|---|---|
| clip | -0.062 ± 0.402 | -0.026 ± 0.397 | 0.036 ± 0.766 | 0.859 |
| simclip2 | -0.034 ± 0.492 | 0.009 ± 0.082 | 0.042 ± 0.561 | 0.777 |
| simclip4 | 0.067 ± 0.071 | -0.136 ± 0.424 | -0.202 ± 0.414 | 0.170 |
| fare2 | 0.056 ± 0.182 | 0.003 ± 0.126 | -0.053 ± 0.222 | 0.410 |
| fare4 | 0.070 ± 0.121 | -0.109 ± 0.524 | -0.180 ± 0.642 | 0.351 |
| tecoa2 | 0.033 ± 0.004 | -0.094 ± 0.505 | -0.127 ± 0.503 | 0.390 |
| tecoa4 | 0.055 ± 0.188 | -0.155 ± 0.123 | -0.211 ± 0.113 | 0.015 |
| **all** | | | **-0.099** (2/7 encoders > 0, Wilcoxon p = 0.078) | 5/21 run pairs > 0, Wilcoxon p = 0.038 |

### medical: Non-target LPIPS, best-match

| Encoder | rigged | plain | Δ (plain − rigged) ± 95% CI | paired p |
|---|---|---|---|---|
| clip | 0.838 ± 0.096 | 0.841 ± 0.047 | 0.003 ± 0.109 | 0.922 |
| simclip2 | 0.723 ± 0.036 | 0.725 ± 0.070 | 0.001 ± 0.051 | 0.927 |
| simclip4 | 0.735 ± 0.040 | 0.782 ± 0.184 | 0.047 ± 0.222 | 0.460 |
| fare2 | 0.748 ± 0.062 | 0.755 ± 0.139 | 0.007 ± 0.079 | 0.731 |
| fare4 | 0.730 ± 0.027 | 0.727 ± 0.140 | -0.003 ± 0.125 | 0.922 |
| tecoa2 | 0.740 ± 0.027 | 0.654 ± 0.116 | -0.086 ± 0.090 | 0.055 |
| tecoa4 | 0.733 ± 0.025 | 0.733 ± 0.051 | -0.000 ± 0.064 | 0.993 |
| **all** | | | **-0.004** (4/7 encoders > 0, Wilcoxon p = 0.812) | 9/21 run pairs > 0, Wilcoxon p = 0.683 |

### medical: Non-target LPIPS, Hungarian

| Encoder | rigged | plain | Δ (plain − rigged) ± 95% CI | paired p |
|---|---|---|---|---|
| clip | 1.012 ± 0.186 | 1.045 ± 0.092 | 0.032 ± 0.242 | 0.626 |
| simclip2 | 0.829 ± 0.160 | 0.896 ± 0.010 | 0.066 ± 0.163 | 0.222 |
| simclip4 | 0.890 ± 0.167 | 0.927 ± 0.146 | 0.037 ± 0.188 | 0.487 |
| fare2 | 0.888 ± 0.074 | 0.929 ± 0.097 | 0.041 ± 0.141 | 0.339 |
| fare4 | 0.878 ± 0.100 | 0.896 ± 0.082 | 0.018 ± 0.148 | 0.656 |
| tecoa2 | 0.855 ± 0.051 | 0.788 ± 0.144 | -0.067 ± 0.094 | 0.092 |
| tecoa4 | 0.887 ± 0.110 | 0.835 ± 0.054 | -0.052 ± 0.062 | 0.071 |
| **all** | | | **+0.011** (5/7 encoders > 0, Wilcoxon p = 0.812) | 10/21 run pairs > 0, Wilcoxon p = 0.517 |

### medical: Distinct reconstructions used by best-match (8 = one-to-one)

| Encoder | rigged | plain | Δ (plain − rigged) ± 95% CI | paired p |
|---|---|---|---|---|
| clip | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | n/a |
| simclip2 | 2.667 ± 2.868 | 1.000 ± 0.000 | -1.667 ± 2.868 | 0.130 |
| simclip4 | 1.333 ± 1.434 | 1.333 ± 1.434 | 0.000 ± 2.484 | 1.000 |
| fare2 | 1.667 ± 1.434 | 1.000 ± 0.000 | -0.667 ± 1.434 | 0.184 |
| fare4 | 2.333 ± 1.434 | 1.000 ± 0.000 | -1.333 ± 1.434 | 0.057 |
| tecoa2 | 2.000 ± 0.000 | 1.333 ± 1.434 | -0.667 ± 1.434 | 0.184 |
| tecoa4 | 2.333 ± 2.868 | 1.000 ± 0.000 | -1.333 ± 2.868 | 0.184 |
| **all** | | | **-0.810** (0/7 encoders > 0, Wilcoxon p = 0.041) | 1/21 run pairs > 0, Wilcoxon p = 0.002 |

### medical: Batch LPIPS (metrics.json)

| Encoder | rigged | plain | Δ (plain − rigged) ± 95% CI | paired p |
|---|---|---|---|---|
| clip | 1.029 ± 0.119 | 1.054 ± 0.105 | 0.025 ± 0.154 | 0.558 |
| simclip2 | 0.839 ± 0.093 | 0.896 ± 0.014 | 0.056 ± 0.101 | 0.139 |
| simclip4 | 0.875 ± 0.150 | 0.963 ± 0.055 | 0.088 ± 0.116 | 0.083 |
| fare2 | 0.876 ± 0.045 | 0.930 ± 0.084 | 0.054 ± 0.093 | 0.129 |
| fare4 | 0.863 ± 0.082 | 0.925 ± 0.082 | 0.062 ± 0.023 | 0.007 |
| tecoa2 | 0.850 ± 0.048 | 0.814 ± 0.065 | -0.035 ± 0.045 | 0.077 |
| tecoa4 | 0.877 ± 0.058 | 0.877 ± 0.039 | -0.000 ± 0.047 | 0.973 |
| **all** | | | **+0.036** (5/7 encoders > 0, Wilcoxon p = 0.109) | 14/21 run pairs > 0, Wilcoxon p = 0.009 |

## medical: same-seed floor (duplicate rigged run minus original)

| Encoder | seed | target_lpips | selectivity | target_lpips_hung | selectivity_hung | nontarget_lpips | nontarget_lpips_hung | n_unique_best | lpips |
|---|---|---|---|---|---|---|---|---|---|
| clip | 42 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## medical: head loss ratios (top 10% by query similarity vs rest)

| Encoder | rigged (base query) | plain head, own query | plain head, base query |
|---|---|---|---|
| clip | 8.21x | 1.20x | 1.20x |
| simclip2 | 10.23x | 1.21x | 1.21x |
| simclip4 | 10.66x | 1.20x | 1.20x |
| fare2 | 7.61x | 1.20x | 1.20x |
| fare4 | 11.82x | 1.21x | 1.21x |
| tecoa2 | 14.50x | 1.21x | 1.21x |
| tecoa4 | 11.67x | 1.21x | 1.21x |

## uav: plain − rigged

### uav: Target LPIPS, best-match (lower = more leakage)

| Encoder | rigged | plain | Δ (plain − rigged) ± 95% CI | paired p |
|---|---|---|---|---|
| clip | 0.683 ± 0.021 | 0.726 ± 0.074 | 0.042 ± 0.058 | 0.089 |
| simclip2 | 0.618 ± 0.046 | 0.698 ± 0.023 | 0.080 ± 0.067 | 0.036 |
| simclip4 | 0.589 ± 0.034 | 0.571 ± 0.038 | -0.018 ± 0.040 | 0.194 |
| fare2 | 0.604 ± 0.038 | 0.572 ± 0.044 | -0.032 ± 0.038 | 0.069 |
| fare4 | 0.530 ± 0.075 | 0.578 ± 0.020 | 0.048 ± 0.080 | 0.123 |
| tecoa2 | 0.489 ± 0.050 | 0.541 ± 0.041 | 0.052 ± 0.050 | 0.046 |
| tecoa4 | 0.480 ± 0.043 | 0.542 ± 0.030 | 0.062 ± 0.067 | 0.057 |
| **all** | | | **+0.034** (5/7 encoders > 0, Wilcoxon p = 0.078) | 15/21 run pairs > 0, Wilcoxon p = 0.004 |

### uav: Selectivity, best-match (non-target minus target LPIPS)

| Encoder | rigged | plain | Δ (plain − rigged) ± 95% CI | paired p |
|---|---|---|---|---|
| clip | -0.007 ± 0.015 | -0.000 ± 0.008 | 0.007 ± 0.011 | 0.114 |
| simclip2 | 0.036 ± 0.026 | 0.001 ± 0.016 | -0.036 ± 0.010 | 0.004 |
| simclip4 | 0.025 ± 0.025 | 0.004 ± 0.040 | -0.021 ± 0.015 | 0.028 |
| fare2 | 0.032 ± 0.042 | 0.019 ± 0.035 | -0.013 ± 0.036 | 0.255 |
| fare4 | 0.067 ± 0.072 | 0.020 ± 0.039 | -0.047 ± 0.037 | 0.033 |
| tecoa2 | 0.092 ± 0.036 | -0.005 ± 0.067 | -0.097 ± 0.055 | 0.017 |
| tecoa4 | 0.122 ± 0.031 | -0.013 ± 0.019 | -0.135 ± 0.050 | 0.007 |
| **all** | | | **-0.049** (1/7 encoders > 0, Wilcoxon p = 0.031) | 4/21 run pairs > 0, Wilcoxon p = 0.000 |

### uav: Target LPIPS, Hungarian one-to-one

| Encoder | rigged | plain | Δ (plain − rigged) ± 95% CI | paired p |
|---|---|---|---|---|
| clip | 0.794 ± 0.027 | 0.904 ± 0.244 | 0.110 ± 0.251 | 0.199 |
| simclip2 | 0.631 ± 0.081 | 0.743 ± 0.060 | 0.112 ± 0.089 | 0.033 |
| simclip4 | 0.663 ± 0.129 | 0.639 ± 0.048 | -0.024 ± 0.084 | 0.345 |
| fare2 | 0.653 ± 0.067 | 0.584 ± 0.032 | -0.069 ± 0.082 | 0.069 |
| fare4 | 0.547 ± 0.049 | 0.602 ± 0.062 | 0.055 ± 0.073 | 0.084 |
| tecoa2 | 0.518 ± 0.104 | 0.575 ± 0.157 | 0.058 ± 0.236 | 0.403 |
| tecoa4 | 0.484 ± 0.053 | 0.562 ± 0.012 | 0.078 ± 0.041 | 0.014 |
| **all** | | | **+0.046** (5/7 encoders > 0, Wilcoxon p = 0.156) | 14/21 run pairs > 0, Wilcoxon p = 0.026 |

### uav: Selectivity, Hungarian one-to-one

| Encoder | rigged | plain | Δ (plain − rigged) ± 95% CI | paired p |
|---|---|---|---|---|
| clip | -0.021 ± 0.163 | -0.012 ± 0.228 | 0.009 ± 0.066 | 0.606 |
| simclip2 | 0.170 ± 0.133 | 0.038 ± 0.122 | -0.132 ± 0.172 | 0.081 |
| simclip4 | 0.020 ± 0.161 | -0.015 ± 0.022 | -0.036 ± 0.152 | 0.418 |
| fare2 | 0.061 ± 0.047 | 0.033 ± 0.043 | -0.028 ± 0.065 | 0.212 |
| fare4 | 0.136 ± 0.042 | 0.046 ± 0.089 | -0.090 ± 0.058 | 0.022 |
| tecoa2 | 0.139 ± 0.115 | -0.024 ± 0.193 | -0.163 ± 0.264 | 0.117 |
| tecoa4 | 0.301 ± 0.089 | -0.020 ± 0.029 | -0.321 ± 0.075 | 0.003 |
| **all** | | | **-0.109** (1/7 encoders > 0, Wilcoxon p = 0.031) | 3/21 run pairs > 0, Wilcoxon p = 0.000 |

### uav: Non-target LPIPS, best-match

| Encoder | rigged | plain | Δ (plain − rigged) ± 95% CI | paired p |
|---|---|---|---|---|
| clip | 0.677 ± 0.010 | 0.726 ± 0.067 | 0.049 ± 0.057 | 0.066 |
| simclip2 | 0.654 ± 0.041 | 0.699 ± 0.037 | 0.044 ± 0.063 | 0.095 |
| simclip4 | 0.614 ± 0.014 | 0.575 ± 0.035 | -0.039 ± 0.049 | 0.077 |
| fare2 | 0.636 ± 0.041 | 0.591 ± 0.040 | -0.045 ± 0.046 | 0.052 |
| fare4 | 0.596 ± 0.018 | 0.597 ± 0.035 | 0.001 ± 0.043 | 0.932 |
| tecoa2 | 0.581 ± 0.014 | 0.536 ± 0.027 | -0.045 ± 0.035 | 0.031 |
| tecoa4 | 0.602 ± 0.012 | 0.529 ± 0.027 | -0.073 ± 0.032 | 0.010 |
| **all** | | | **-0.015** (3/7 encoders > 0, Wilcoxon p = 0.578) | 7/21 run pairs > 0, Wilcoxon p = 0.179 |

### uav: Non-target LPIPS, Hungarian

| Encoder | rigged | plain | Δ (plain − rigged) ± 95% CI | paired p |
|---|---|---|---|---|
| clip | 0.773 ± 0.170 | 0.892 ± 0.016 | 0.119 ± 0.185 | 0.110 |
| simclip2 | 0.801 ± 0.058 | 0.781 ± 0.063 | -0.020 ± 0.092 | 0.445 |
| simclip4 | 0.683 ± 0.038 | 0.623 ± 0.033 | -0.060 ± 0.068 | 0.064 |
| fare2 | 0.714 ± 0.026 | 0.617 ± 0.025 | -0.096 ± 0.050 | 0.014 |
| fare4 | 0.683 ± 0.016 | 0.648 ± 0.046 | -0.035 ± 0.055 | 0.111 |
| tecoa2 | 0.657 ± 0.017 | 0.551 ± 0.041 | -0.106 ± 0.032 | 0.005 |
| tecoa4 | 0.785 ± 0.045 | 0.542 ± 0.029 | -0.243 ± 0.034 | 0.001 |
| **all** | | | **-0.063** (1/7 encoders > 0, Wilcoxon p = 0.219) | 4/21 run pairs > 0, Wilcoxon p = 0.008 |

### uav: Distinct reconstructions used by best-match (8 = one-to-one)

| Encoder | rigged | plain | Δ (plain − rigged) ± 95% CI | paired p |
|---|---|---|---|---|
| clip | 2.667 ± 1.434 | 1.000 ± 0.000 | -1.667 ± 1.434 | 0.038 |
| simclip2 | 1.667 ± 1.434 | 2.333 ± 1.434 | 0.667 ± 1.434 | 0.184 |
| simclip4 | 2.000 ± 0.000 | 2.333 ± 1.434 | 0.333 ± 1.434 | 0.423 |
| fare2 | 2.000 ± 4.303 | 5.000 ± 0.000 | 3.000 ± 4.303 | 0.095 |
| fare4 | 2.000 ± 0.000 | 2.667 ± 1.434 | 0.667 ± 1.434 | 0.184 |
| tecoa2 | 3.000 ± 2.484 | 5.667 ± 1.434 | 2.667 ± 3.795 | 0.094 |
| tecoa4 | 2.000 ± 0.000 | 6.000 ± 0.000 | 4.000 ± 0.000 | n/a |
| **all** | | | **+1.381** (6/7 encoders > 0, Wilcoxon p = 0.109) | 14/21 run pairs > 0, Wilcoxon p = 0.011 |

### uav: Batch LPIPS (metrics.json)

| Encoder | rigged | plain | Δ (plain − rigged) ± 95% CI | paired p |
|---|---|---|---|---|
| clip | 0.779 ± 0.130 | 0.898 ± 0.072 | 0.118 ± 0.202 | 0.128 |
| simclip2 | 0.760 ± 0.031 | 0.772 ± 0.032 | 0.012 ± 0.051 | 0.424 |
| simclip4 | 0.682 ± 0.021 | 0.630 ± 0.040 | -0.052 ± 0.030 | 0.017 |
| fare2 | 0.700 ± 0.030 | 0.609 ± 0.019 | -0.091 ± 0.049 | 0.015 |
| fare4 | 0.653 ± 0.021 | 0.636 ± 0.033 | -0.017 ± 0.053 | 0.306 |
| tecoa2 | 0.623 ± 0.023 | 0.558 ± 0.019 | -0.065 ± 0.041 | 0.021 |
| tecoa4 | 0.711 ± 0.027 | 0.547 ± 0.022 | -0.163 ± 0.015 | 0.000 |
| **all** | | | **-0.037** (2/7 encoders > 0, Wilcoxon p = 0.297) | 6/21 run pairs > 0, Wilcoxon p = 0.024 |

## uav: same-seed floor (duplicate rigged run minus original)

| Encoder | seed | target_lpips | selectivity | target_lpips_hung | selectivity_hung | nontarget_lpips | nontarget_lpips_hung | n_unique_best | lpips |
|---|---|---|---|---|---|---|---|---|---|
| clip | 42 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## uav: head loss ratios (top 10% by query similarity vs rest)

| Encoder | rigged (base query) | plain head, own query | plain head, base query |
|---|---|---|---|
| clip | 98.46x | 0.47x | 0.47x |
| simclip2 | 95.58x | 0.54x | 0.54x |
| simclip4 | 32.77x | 0.56x | 0.56x |
| fare2 | 84.75x | 0.50x | 0.50x |
| fare4 | 21.61x | 0.65x | 0.65x |
| tecoa2 | 56.14x | 0.57x | 0.57x |
| tecoa4 | 56.06x | 0.43x | 0.43x |

