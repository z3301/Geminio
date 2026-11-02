# Per-sample statistics of the private batch, ResNet-18 client

Same quantities as private_batch_profile.md for the ResNet-18 client and the guide heads of the held-out tables. Uniform gradient share for the two targets is 0.250. `AUROC` is the guide's retrieval of the target class on the auxiliary set (control B / retrieval scan).

## medical

| guide head | AUROC | target share | target own prob | other own prob | target loss | other loss | on-batch loss ratio |
|---|---|---|---|---|---|---|---|
| biomedclip (pneumonia_descriptive) | 0.64 | 0.253 | 0.0038 | 0.0898 | 5.575 | 2.881 | 1.93x |
| biomedclip (pna_base) | 0.64 | 0.249 | 0.0002 | 0.0476 | 8.615 | 3.569 | 2.41x |
| clip | 0.48 | 0.216 | 0.0032 | 0.0295 | 5.751 | 7.401 | 0.78x |
| simclip2 | 0.39 | 0.187 | 0.0031 | 0.0187 | 355.784 | 570.849 | 0.62x |
| simclip4 | 0.40 | 0.191 | 0.0000 | 0.1281 | 658.004 | 860.425 | 0.76x |
| fare2 | 0.38 | 0.214 | 0.0006 | 0.0353 | 512.761 | 576.872 | 0.89x |
| fare4 | 0.40 | 0.165 | 0.0000 | 0.1667 | 341.344 | 559.225 | 0.61x |
| tecoa2 | 0.43 | 0.127 | 0.0064 | 0.1667 | 252.191 | 556.257 | 0.45x |
| tecoa4 | 0.46 | 0.169 | 0.0000 | 0.0000 | 329.900 | 560.854 | 0.59x |

## pathmnist

| guide head | AUROC | target share | target own prob | other own prob | target loss | other loss | on-batch loss ratio |
|---|---|---|---|---|---|---|---|
| biomedclip | 0.87 | 0.362 | 0.0396 | 0.3487 | 3.240 | 1.746 | 1.86x |
| clip | 0.60 | 0.266 | 0.0002 | 0.3553 | 8.753 | 20.098 | 0.44x |
| simclip2 | 0.56 | 0.135 | 0.0082 | 0.4061 | 6.257 | 34.239 | 0.18x |
| simclip4 | 0.54 | 0.154 | 0.0044 | 0.4051 | 6.621 | 39.961 | 0.17x |
| fare2 | 0.67 | 0.173 | 0.0837 | 0.2576 | 2.488 | 8.959 | 0.28x |
| fare4 | 0.69 | 0.208 | 0.0109 | 0.4268 | 5.395 | 14.898 | 0.36x |
| tecoa2 | 0.69 | 0.150 | 0.0056 | 0.3277 | 5.352 | 28.534 | 0.19x |
| tecoa4 | 0.78 | 0.205 | 0.0234 | 0.2684 | 3.794 | 15.906 | 0.24x |

## uav

| guide head | AUROC | target share | target own prob | other own prob | target loss | other loss | on-batch loss ratio |
|---|---|---|---|---|---|---|---|
| clip (sol_base) | 0.92 | 0.326 | 0.3682 | 0.7863 | 34.511 | 21.100 | 1.64x |
| clip | 0.92 | 0.338 | 0.3899 | 0.8023 | 47.016 | 21.942 | 2.14x |
| simclip2 | 0.97 | 0.464 | 0.3573 | 0.7233 | 344.897 | 36.509 | 9.45x |
| simclip4 | 0.87 | 0.281 | 0.2574 | 0.3231 | 389.814 | 232.192 | 1.68x |
| fare2 | 0.97 | 0.406 | 0.3482 | 0.7141 | 280.877 | 30.159 | 9.31x |
| fare4 | 0.91 | 0.342 | 0.2248 | 0.4822 | 308.818 | 71.324 | 4.33x |
| tecoa2 | 0.92 | 0.398 | 0.2681 | 0.7504 | 119.687 | 18.690 | 6.40x |
| tecoa4 | 0.69 | 0.087 | 0.5234 | 0.3064 | 10.266 | 558.563 | 0.02x |

