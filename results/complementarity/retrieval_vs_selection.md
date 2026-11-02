# Guide retrieval vs. target selection on the private batch (ResNet-18 client)

One point per (dataset, guide head). x = AUROC with which the guide's query similarity ranks the target class on the auxiliary set; y = the two target images' share of the summed per-example last-layer weight-gradient norms on the private batch (equal contribution = 0.250). Spearman rank correlation. The heads are NOT independent replicates (shared guide families, one private batch per dataset), the pooled p-value below is reported next to a permutation null that shuffles AUROC within each dataset, and the 0.30 column is descriptive, not a threshold: 0.253 (ChestMNIST BiomedCLIP) and 0.266 (PathMNIST CLIP) are within one-batch noise of equal contribution, so categorical "selects / does not select" language is avoided. The paired contrast against an honest head (see `private_batch_profile.md`) is the sharper quantity.

| scope | n heads | rho | p | p (AUROC shuffled within dataset) | heads >= 0.30 | AUROC of those |
|---|---|---|---|---|---|---|
| all | 25 | +0.67 | 0.0002 | 0.0010 | 7/25 | 0.97, 0.97, 0.92, 0.92, 0.92, 0.91, 0.87 |
| medical | 9 | +0.53 | 0.1447 | n/a | 0/9 | - |
| pathmnist | 8 | +0.50 | 0.2070 | n/a | 1/8 | 0.87 |
| uav | 8 | +0.90 | 0.0024 | n/a | 6/8 | 0.97, 0.97, 0.92, 0.92, 0.92, 0.91 |

Median share, guides with AUROC < 0.75: 0.180 (n = 16); AUROC >= 0.75: 0.342 (n = 9). The 0.75 split is post hoc and has exceptions on both sides, so it is reported as a description of these 25 heads, not as a criterion.

| dataset | guide | AUROC | target share | on-batch target/other loss |
|---|---|---|---|---|
| medical | biomedclip (pneumonia_descriptive) | 0.64 | 0.253 | 1.93x |
| medical | biomedclip (pna_base) | 0.64 | 0.249 | 2.41x |
| medical | clip | 0.48 | 0.216 | 0.78x |
| medical | tecoa4 | 0.46 | 0.169 | 0.59x |
| medical | tecoa2 | 0.43 | 0.127 | 0.45x |
| medical | simclip4 | 0.40 | 0.191 | 0.76x |
| medical | fare4 | 0.40 | 0.165 | 0.61x |
| medical | simclip2 | 0.39 | 0.187 | 0.62x |
| medical | fare2 | 0.38 | 0.214 | 0.89x |
| pathmnist | biomedclip | 0.87 | 0.362 | 1.86x |
| pathmnist | tecoa4 | 0.78 | 0.205 | 0.24x |
| pathmnist | tecoa2 | 0.69 | 0.150 | 0.19x |
| pathmnist | fare4 | 0.69 | 0.208 | 0.36x |
| pathmnist | fare2 | 0.67 | 0.173 | 0.28x |
| pathmnist | clip | 0.60 | 0.266 | 0.44x |
| pathmnist | simclip2 | 0.56 | 0.135 | 0.18x |
| pathmnist | simclip4 | 0.54 | 0.154 | 0.17x |
| uav | fare2 | 0.97 | 0.406 | 9.31x |
| uav | simclip2 | 0.97 | 0.464 | 9.45x |
| uav | tecoa2 | 0.92 | 0.398 | 6.40x |
| uav | clip (sol_base) | 0.92 | 0.326 | 1.64x |
| uav | clip | 0.92 | 0.338 | 2.14x |
| uav | fare4 | 0.91 | 0.342 | 4.33x |
| uav | simclip4 | 0.87 | 0.281 | 1.68x |
| uav | tecoa4 | 0.69 | 0.087 | 0.02x |
