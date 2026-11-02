# Retrieval scan: pathmnist auxiliary half (3587 images), query template "Any colorectal histopathology image showing {name}"

AUROC of the query similarity for each class under each guide (precision in the top 10% in parentheses; base rate in the last column). Computed before any attack run.

| class | biomedclip | clip | simclip2 | simclip4 | fare2 | fare4 | tecoa2 | tecoa4 | base rate |
|---|---|---|---|---|---|---|---|---|---|
| 0 adipose | 0.68 (0.25) | 0.24 (0.00) | 0.40 (0.04) | 0.49 (0.11) | 0.41 (0.05) | 0.41 (0.05) | 0.39 (0.08) | 0.62 (0.26) | 0.187 |
| 1 background | 0.33 (0.00) | 0.11 (0.00) | 0.10 (0.00) | 0.13 (0.00) | 0.12 (0.00) | 0.20 (0.00) | 0.35 (0.00) | 0.48 (0.01) | 0.118 |
| 2 debris | 0.64 (0.02) | 0.44 (0.01) | 0.52 (0.04) | 0.47 (0.02) | 0.53 (0.03) | 0.56 (0.03) | 0.62 (0.05) | 0.39 (0.01) | 0.047 |
| 3 lymphocytes | 0.90 (0.28) | 0.79 (0.41) | 0.52 (0.07) | 0.60 (0.06) | 0.59 (0.12) | 0.62 (0.10) | 0.71 (0.15) | 0.65 (0.06) | 0.088 |
| 4 mucus | 0.74 (0.20) | 0.53 (0.06) | 0.54 (0.12) | 0.45 (0.10) | 0.48 (0.08) | 0.35 (0.06) | 0.54 (0.05) | 0.40 (0.01) | 0.144 |
| 5 smooth muscle | 0.85 (0.28) | 0.85 (0.33) | 0.94 (0.57) | 0.91 (0.54) | 0.94 (0.62) | 0.91 (0.58) | 0.94 (0.59) | 0.89 (0.47) | 0.083 |
| 6 normal colon mucosa | 0.92 (0.67) | 0.94 (0.68) | 0.72 (0.17) | 0.72 (0.15) | 0.82 (0.26) | 0.78 (0.22) | 0.67 (0.16) | 0.67 (0.21) | 0.103 |
| 7 cancer-associated stroma | 0.72 (0.02) | 0.75 (0.05) | 0.79 (0.24) | 0.64 (0.11) | 0.76 (0.20) | 0.61 (0.11) | 0.74 (0.18) | 0.64 (0.08) | 0.059 |
| 8 colorectal adenocarcinoma epithelium | 0.87 (0.57) | 0.60 (0.22) | 0.56 (0.17) | 0.54 (0.16) | 0.67 (0.29) | 0.69 (0.28) | 0.69 (0.30) | 0.78 (0.38) | 0.172 |

