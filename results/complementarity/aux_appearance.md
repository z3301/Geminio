# Control B follow-up: appearance of each guide's top-ranked auxiliary images

Top-16 auxiliary images by query similarity per guide; mean LPIPS (AlexNet, 224 px) to the two held-out pneumonia targets vs 16 random auxiliary images (seed 0). Attack strength = held-out target LPIPS of the ResNet-18 attack guided by that VLM.

| Guide | top-16 LPIPS to targets | random LPIPS | pneumonia in top-16 | label mix | attack target LPIPS (Hung.) |
|---|---|---|---|---|---|
| biomedclip | 0.452 | 0.415 | 0 | atelectasis 2, cardiomegaly 3, infiltration 6, edema 2, normal 3 | n/a |
| clip | 0.374 | 0.415 | 0 | atelectasis 3, effusion 2, infiltration 3, consolidation 1, normal 7 | 0.864 |
| simclip2 | 0.392 | 0.415 | 0 | atelectasis 3, infiltration 1, consolidation 1, edema 1, normal 10 | 0.574 |
| simclip4 | 0.430 | 0.415 | 0 | atelectasis 1, infiltration 3, consolidation 1, normal 11 | 0.615 |
| fare2 | 0.387 | 0.415 | 0 | effusion 2, infiltration 1, consolidation 1, normal 12 | 0.602 |
| fare4 | 0.422 | 0.415 | 0 | atelectasis 1, cardiomegaly 1, effusion 2, infiltration 1, pneumothorax 1, edema 1, normal 9 | 0.669 |
| tecoa2 | 0.435 | 0.415 | 0 | atelectasis 1, cardiomegaly 1, infiltration 2, normal 12 | 0.685 |
| tecoa4 | 0.465 | 0.415 | 0 | atelectasis 1, infiltration 2, consolidation 2, normal 11 | 0.450 |

Spearman top-16 LPIPS vs target_lpips over 7 CLIP-family guides: rho = -0.39, p = 0.383

Spearman top-16 LPIPS vs target_lpips_hung over 7 CLIP-family guides: rho = -0.36, p = 0.432
