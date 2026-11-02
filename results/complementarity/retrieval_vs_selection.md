# Guide retrieval vs. target selection on the private batch (ResNet-18 client)

One point per (dataset, guide head). x = AUROC with which the guide's query similarity ranks the target class on the auxiliary set; y = share of the last-layer weight-gradient norm the rigged head puts on the two target images of the private batch (uniform = 0.250). Spearman rank correlation; the per-dataset n is small, so read the pooled line and the sign agreement together.

| scope | n heads | rho | p | heads above uniform | AUROC of those |
|---|---|---|---|---|---|
| all | 25 | +0.67 | 0.0002 | 10/25 | 0.97, 0.97, 0.92, 0.92, 0.92, 0.91, 0.87, 0.87, 0.64, 0.60 |
| medical | 9 | +0.53 | 0.1447 | 1/9 | 0.64 |
| pathmnist | 8 | +0.50 | 0.2070 | 2/8 | 0.87, 0.60 |
| uav | 8 | +0.90 | 0.0024 | 7/8 | 0.97, 0.97, 0.92, 0.92, 0.92, 0.91, 0.87 |

Heads whose guide has AUROC < 0.75: 2/16 above uniform (median share 0.180). AUROC >= 0.75: 8/9 above uniform (median 0.342).

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
