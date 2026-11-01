# Reported results against the uninformative-set reference

Reference = 8 images drawn from the head-training half, scored with the same matching code (`floor_baselines.md`). Target LPIPS below the 5th percentile is the only region where the metric itself evidences recovery; selectivity above the 95th percentile likewise. The reference is a property of the batch and is identical across arms, so paired contrasts (rigged vs plain vs swapped, guide vs guide) are unaffected by it.

## medical

Reference target LPIPS 0.327 [0.294, 0.361]; reference selectivity -0.016 [-0.048, +0.025].

| condition | target LPIPS | vs reference | selectivity | vs reference |
|---|---|---|---|---|
| ResNet-18 · medical_pna_base_ho | 0.513 | above p95 | +0.043 | above p95 |
| ResNet-18 · medical_pna_compAND_ho | 0.497 | above p95 | +0.065 | above p95 |
| ResNet-18 · medical_pna_compOR_ho | 0.699 | above p95 | -0.016 | inside |
| ResNet-18 · medical_pna_es_ho | 0.526 | above p95 | +0.025 | inside |
| ResNet-18 · medical_pna_indirect_ho | 0.930 | above p95 | -0.018 | inside |
| ResNet-18 · medical_pna_negEX_ho | 0.672 | above p95 | -0.026 | inside |
| ResNet-18 · medical_pna_negPURE_ho | 0.462 | above p95 | +0.010 | inside |
| ResNet-18 · medical_pna_synonym_ho | 0.490 | above p95 | -0.056 | below p5 |
| ResNet-18 · medical_pna_terse_ho | 0.713 | above p95 | -0.025 | inside |
| ResNet-18 · medical_pna_typo_ho | 0.398 | above p95 | +0.087 | above p95 |
| ResNet-18 · medical_pna_verbose_ho | 0.573 | above p95 | +0.039 | above p95 |
| ResNet-18 · medical_pna_xlong_ho | 0.485 | above p95 | +0.062 | above p95 |
| ResNet-18 · medical_pneumonia_descriptive_gradinversion_ho | 0.918 | above p95 | -0.010 | inside |
| ResNet-18 · medical_pneumonia_descriptive_ho | 0.590 | above p95 | +0.030 | above p95 |
| ResNet-18 · medical_pneumonia_descriptive_hoB16 | 0.542 | above p95 | -0.005 | inside |
| ResNet-18 · medical_pneumonia_descriptive_hoB32 | 0.737 | above p95 | -0.018 | inside |
| ResNet-18 · medical_pneumonia_descriptive_ig_ho | 0.592 | above p95 | +0.011 | inside |
| ResNet-18 · medical_pneumonia_descriptive_reclvl_ho | 0.802 | above p95 | -0.050 | below p5 |
| ResNet-18 · medical_vlmclip_ho | 0.687 | above p95 | -0.030 | inside |
| ResNet-18 · medical_vlmfare2_ho | 0.377 | above p95 | +0.036 | above p95 |
| ResNet-18 · medical_vlmfare4_ho | 0.491 | above p95 | +0.032 | above p95 |
| ResNet-18 · medical_vlmsimclip2_ho | 0.448 | above p95 | +0.034 | above p95 |
| ResNet-18 · medical_vlmsimclip4_ho | 0.444 | above p95 | +0.067 | above p95 |
| ResNet-18 · medical_vlmtecoa2_ho | 0.411 | above p95 | -0.018 | inside |
| ResNet-18 · medical_vlmtecoa4_ho | 0.419 | above p95 | +0.023 | inside |
| ViT-L clip · plain head | 0.868 | above p95 | -0.027 | inside |
| ViT-L fare2 · plain head | 0.763 | above p95 | -0.008 | inside |
| ViT-L fare4 · plain head | 0.738 | above p95 | -0.011 | inside |
| ViT-L simclip2 · plain head | 0.750 | above p95 | -0.025 | inside |
| ViT-L simclip4 · plain head | 0.796 | above p95 | -0.014 | inside |
| ViT-L tecoa2 · plain head | 0.669 | above p95 | -0.014 | inside |
| ViT-L tecoa4 · plain head | 0.751 | above p95 | -0.018 | inside |
| ViT-L clip · rigged head | 0.867 | above p95 | -0.029 | inside |
| ViT-L fare2 · rigged head | 0.795 | above p95 | -0.048 | inside |
| ViT-L fare4 · rigged head | 0.771 | above p95 | -0.041 | inside |
| ViT-L simclip2 · rigged head | 0.762 | above p95 | -0.039 | inside |
| ViT-L simclip4 · rigged head | 0.768 | above p95 | -0.033 | inside |
| ViT-L tecoa2 · rigged head | 0.790 | above p95 | -0.050 | below p5 |
| ViT-L tecoa4 · rigged head | 0.775 | above p95 | -0.042 | inside |

## uav

Reference target LPIPS 0.535 [0.485, 0.571]; reference selectivity +0.035 [-0.003, +0.086].

| condition | target LPIPS | vs reference | selectivity | vs reference |
|---|---|---|---|---|
| ResNet-18 · uav_sol_base_ho | 0.509 | inside | +0.081 | inside |
| ResNet-18 · uav_sol_compAND_ho | 0.527 | inside | +0.065 | inside |
| ResNet-18 · uav_sol_compOR_ho | 0.544 | inside | +0.033 | inside |
| ResNet-18 · uav_sol_es_ho | 0.533 | inside | +0.039 | inside |
| ResNet-18 · uav_sol_indirect_ho | 0.510 | inside | +0.078 | inside |
| ResNet-18 · uav_sol_negEX_ho | 0.510 | inside | +0.067 | inside |
| ResNet-18 · uav_sol_negPURE_ho | 0.441 | below p5 | +0.128 | above p95 |
| ResNet-18 · uav_sol_synonym_ho | 0.541 | inside | +0.042 | inside |
| ResNet-18 · uav_sol_terse_ho | 0.457 | below p5 | +0.134 | above p95 |
| ResNet-18 · uav_sol_typo_ho | 0.493 | inside | +0.096 | above p95 |
| ResNet-18 · uav_sol_verbose_ho | 0.537 | inside | +0.039 | inside |
| ResNet-18 · uav_sol_xlong_ho | 0.462 | below p5 | +0.117 | above p95 |
| ResNet-18 · uav_solar_panels_gradinversion_ho | 0.576 | above p95 | +0.058 | inside |
| ResNet-18 · uav_solar_panels_ho | 0.511 | inside | +0.098 | above p95 |
| ResNet-18 · uav_solar_panels_hoB16 | 0.555 | inside | +0.044 | inside |
| ResNet-18 · uav_solar_panels_hoB32 | 0.465 | below p5 | +0.114 | above p95 |
| ResNet-18 · uav_solar_panels_ig_ho | 0.453 | below p5 | +0.138 | above p95 |
| ResNet-18 · uav_solar_panels_reclvl_ho | 0.677 | above p95 | -0.017 | below p5 |
| ResNet-18 · uav_vlmclip_ho | 0.470 | below p5 | +0.120 | above p95 |
| ResNet-18 · uav_vlmfare2_ho | 0.515 | inside | +0.076 | inside |
| ResNet-18 · uav_vlmfare4_ho | 0.562 | inside | +0.017 | inside |
| ResNet-18 · uav_vlmsimclip2_ho | 0.473 | below p5 | +0.111 | above p95 |
| ResNet-18 · uav_vlmsimclip4_ho | 0.565 | inside | +0.001 | inside |
| ResNet-18 · uav_vlmtecoa2_ho | 0.496 | inside | +0.104 | above p95 |
| ResNet-18 · uav_vlmtecoa4_ho | 0.591 | above p95 | -0.063 | below p5 |
| ViT-L clip · plain head | 0.726 | above p95 | -0.000 | inside |
| ViT-L fare2 · plain head | 0.572 | above p95 | +0.019 | inside |
| ViT-L fare4 · plain head | 0.578 | above p95 | +0.020 | inside |
| ViT-L simclip2 · plain head | 0.698 | above p95 | +0.001 | inside |
| ViT-L simclip4 · plain head | 0.571 | inside | +0.004 | inside |
| ViT-L tecoa2 · plain head | 0.541 | inside | -0.005 | below p5 |
| ViT-L tecoa4 · plain head | 0.542 | inside | -0.013 | below p5 |
| ViT-L clip · rigged head | 0.683 | above p95 | -0.007 | below p5 |
| ViT-L fare2 · rigged head | 0.604 | above p95 | +0.032 | inside |
| ViT-L fare4 · rigged head | 0.530 | inside | +0.067 | inside |
| ViT-L simclip2 · rigged head | 0.618 | above p95 | +0.036 | inside |
| ViT-L simclip4 · rigged head | 0.589 | above p95 | +0.025 | inside |
| ViT-L tecoa2 · rigged head | 0.489 | inside | +0.092 | above p95 |
| ViT-L tecoa4 · rigged head | 0.480 | below p5 | +0.122 | above p95 |

