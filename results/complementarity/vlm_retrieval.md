# Control B: retrieval precision of the attacker VLM on the auxiliary set

Query similarity of each guide VLM ranks the auxiliary half of the data; AUROC and precision in the top 10% (the slice Geminio's loss ratio weights) for the target class, against the base rate. Attack strength is the held-out target LPIPS of the ResNet-18 attack guided by that VLM (mean of 3 seeds, lower = stronger). Spearman rho over the CLIP-family guides.

## medical

| Guide | AUROC | prec@10% | base rate | positives in top 10% | AUROC (any-label) | prec@10% (any) | base rate (any) | target LPIPS (best) | target LPIPS (Hung.) |
|---|---|---|---|---|---|---|---|---|---|
| biomedclip | 0.639 | 0.0089 | 0.0038 | 10/1122 | 0.628 | 0.0285 | 0.0123 | n/a | n/a |
| clip | 0.480 | 0.0036 | 0.0038 | 4/1122 | 0.491 | 0.0098 | 0.0123 | 0.687 | 0.864 |
| simclip2 | 0.386 | 0.0036 | 0.0038 | 4/1122 | 0.424 | 0.0089 | 0.0123 | 0.448 | 0.574 |
| simclip4 | 0.405 | 0.0027 | 0.0038 | 3/1122 | 0.419 | 0.0098 | 0.0123 | 0.444 | 0.615 |
| fare2 | 0.381 | 0.0027 | 0.0038 | 3/1123 | 0.420 | 0.0098 | 0.0123 | 0.377 | 0.602 |
| fare4 | 0.397 | 0.0018 | 0.0038 | 2/1122 | 0.417 | 0.0089 | 0.0123 | 0.491 | 0.669 |
| tecoa2 | 0.426 | 0.0036 | 0.0038 | 4/1122 | 0.442 | 0.0116 | 0.0123 | 0.411 | 0.685 |
| tecoa4 | 0.463 | 0.0036 | 0.0038 | 4/1122 | 0.476 | 0.0125 | 0.0123 | 0.419 | 0.450 |

Spearman auroc vs target_lpips over 7 CLIP-family guides: rho = +0.36, p = 0.432

Spearman prec_top10 vs target_lpips over 7 CLIP-family guides: rho = +0.06, p = 0.900

Spearman any_auroc vs target_lpips over 7 CLIP-family guides: rho = +0.07, p = 0.879

Spearman any_prec_top10 vs target_lpips over 7 CLIP-family guides: rho = -0.36, p = 0.423

Spearman auroc vs target_lpips_hung over 7 CLIP-family guides: rho = +0.39, p = 0.383

Spearman prec_top10 vs target_lpips_hung over 7 CLIP-family guides: rho = -0.04, p = 0.933

Spearman any_auroc vs target_lpips_hung over 7 CLIP-family guides: rho = +0.11, p = 0.819

Spearman any_prec_top10 vs target_lpips_hung over 7 CLIP-family guides: rho = +0.00, p = 1.000

## uav

| Guide | AUROC | prec@10% | base rate | positives in top 10% | target LPIPS (best) | target LPIPS (Hung.) |
|---|---|---|---|---|---|---|
| clip | 0.916 | 0.5594 | 0.0765 | 113/202 | 0.470 | 0.530 |
| simclip2 | 0.966 | 0.5792 | 0.0765 | 117/202 | 0.473 | 0.512 |
| simclip4 | 0.868 | 0.4010 | 0.0765 | 81/202 | 0.565 | 0.629 |
| fare2 | 0.971 | 0.6337 | 0.0765 | 128/202 | 0.515 | 0.534 |
| fare4 | 0.909 | 0.4356 | 0.0765 | 88/202 | 0.562 | 0.636 |
| tecoa2 | 0.922 | 0.5446 | 0.0765 | 110/202 | 0.496 | 0.539 |
| tecoa4 | 0.694 | 0.0594 | 0.0765 | 12/202 | 0.591 | 0.742 |

Spearman auroc vs target_lpips over 7 CLIP-family guides: rho = -0.68, p = 0.094

Spearman prec_top10 vs target_lpips over 7 CLIP-family guides: rho = -0.75, p = 0.052

Spearman auroc vs target_lpips_hung over 7 CLIP-family guides: rho = -0.79, p = 0.036

Spearman prec_top10 vs target_lpips_hung over 7 CLIP-family guides: rho = -0.86, p = 0.014

