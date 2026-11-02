# Reference values for uninformative reconstruction sets

Same matching code as the real runs. `dataset sample` draws 8 images from the head-training half (mean over draws, with the 5th and 95th percentile). A reported target LPIPS is informative only below these values.

## medical (2 targets of 8)

| set | target_lpips | selectivity | target_lpips_hung | selectivity_hung | nontarget_lpips | n_unique_best |
|---|---|---|---|---|---|---|
| dataset sample | 0.327 [0.294, 0.361] | -0.016 [-0.048, 0.025] | 0.346 [0.299, 0.397] | -0.014 [-0.081, 0.044] | 0.311 [0.291, 0.329] | 4.660 [3.000, 6.000] |
| other batch images | 0.295 | 0.029 | 0.312 | 0.015 | 0.324 | 5.000 |
| uniform grey | 0.756 | -0.036 | 0.756 | -0.036 | 0.720 | 2.000 |
| uniform noise | 1.605 | -0.023 | 1.612 | -0.014 | 1.582 | 1.000 |
| batch mean | 0.322 | 0.002 | 0.322 | 0.002 | 0.324 | 3.000 |

## pathmnist (2 targets of 8)

| set | target_lpips | selectivity | target_lpips_hung | selectivity_hung | nontarget_lpips | n_unique_best |
|---|---|---|---|---|---|---|
| dataset sample | 0.420 [0.383, 0.451] | -0.027 [-0.069, 0.032] | 0.518 [0.405, 0.600] | -0.067 [-0.168, 0.057] | 0.394 [0.355, 0.448] | 3.500 [2.000, 5.000] |
| other batch images | 0.398 | 0.016 | 0.400 | 0.070 | 0.415 | 3.000 |
| uniform grey | 0.983 | -0.052 | 0.983 | -0.052 | 0.931 | 2.000 |
| uniform noise | 1.501 | -0.027 | 1.524 | -0.042 | 1.474 | 3.000 |
| batch mean | 0.374 | -0.001 | 0.374 | -0.001 | 0.373 | 2.000 |

## uav (2 targets of 8)

| set | target_lpips | selectivity | target_lpips_hung | selectivity_hung | nontarget_lpips | n_unique_best |
|---|---|---|---|---|---|---|
| dataset sample | 0.535 [0.485, 0.571] | 0.035 [-0.003, 0.086] | 0.543 [0.492, 0.579] | 0.037 [-0.005, 0.088] | 0.570 [0.558, 0.580] | 4.560 [3.000, 6.000] |
| other batch images | 0.560 | 0.014 | 0.560 | 0.026 | 0.574 | 6.000 |
| uniform grey | 0.980 | -0.035 | 0.980 | -0.035 | 0.944 | 3.000 |
| uniform noise | 1.398 | -0.043 | 1.402 | -0.036 | 1.355 | 2.000 |
| batch mean | 0.632 | -0.026 | 0.632 | -0.026 | 0.606 | 2.000 |

