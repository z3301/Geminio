# Reference values for uninformative reconstruction sets

Same matching code as the real runs. `dataset sample` draws 8 images from the head-training half (mean over draws, with the 5th and 95th percentile). A reported target LPIPS is informative only below these values.

## medical (2 targets of 8)

| set | target_lpips | selectivity | target_lpips_hung | selectivity_hung | nontarget_lpips | n_unique_best |
|---|---|---|---|---|---|---|
| dataset sample | 0.327 [0.294, 0.361] | -0.016 [-0.048, 0.025] | 0.346 [0.299, 0.397] | -0.014 [-0.081, 0.044] | 0.311 [0.291, 0.329] | 4.660 [3.000, 6.000] |
| other batch images | 0.295 | 0.029 | 0.312 | 0.015 | 0.324 | 5.000 |
| uniform grey | 0.756 | -0.036 | 0.756 | -0.036 | 0.720 | 2.000 |
| uniform noise | 1.607 | -0.022 | 1.628 | -0.037 | 1.585 | 3.000 |
| batch mean | 0.322 | 0.002 | 0.322 | 0.002 | 0.324 | 3.000 |

## uav (2 targets of 8)

| set | target_lpips | selectivity | target_lpips_hung | selectivity_hung | nontarget_lpips | n_unique_best |
|---|---|---|---|---|---|---|
| dataset sample | 0.535 [0.485, 0.571] | 0.035 [-0.003, 0.086] | 0.543 [0.492, 0.579] | 0.037 [-0.005, 0.088] | 0.570 [0.558, 0.580] | 4.560 [3.000, 6.000] |
| other batch images | 0.560 | 0.014 | 0.560 | 0.026 | 0.574 | 6.000 |
| uniform grey | 0.980 | -0.035 | 0.980 | -0.035 | 0.944 | 3.000 |
| uniform noise | 1.403 | -0.042 | 1.418 | -0.051 | 1.361 | 3.000 |
| batch mean | 0.632 | -0.026 | 0.632 | -0.026 | 0.606 | 2.000 |


## What this implies (2026-10-31)

Per-condition placement is in `baseline_comparison.md` (`prototype/compare_to_baselines.py`).

**The reference is identical across arms**, since it depends only on the batch. Every paired contrast
in this project therefore stands: rigged vs plain vs swapped head, guide vs guide, defended vs
undefended, encoder vs encoder. What the reference bounds is the *absolute* reading of target LPIPS.

**Medical: target LPIPS never evidences recovery.** Every medical condition, across both client
architectures, all 24 queries and all guides, scores above the 95th percentile of the dataset-sample
reference (0.327 [0.294, 0.361]). The best medical number anywhere is 0.377 (FARE-2 guide, ResNet-18).
So on this metric no medical configuration beats drawing eight random chest X-rays. Part of this is the
metric: at 64x64 upscaled to 224, ChestMNIST images are mutually similar in LPIPS, which makes the
reference low and leaves little room above it. Either way, medical target LPIPS cannot be used to claim
reconstruction. Medical *selectivity* does clear the reference for several ResNet-18 queries
(pna_base +0.043, compAND +0.065 against a reference of -0.016 [-0.048, +0.025]), so the selection
signal survives where the recovery claim does not.

**UAV: a subset of conditions clears the reference.** Target LPIPS below the 5th percentile (0.485):
the IG attack (0.453), the negPURE, terse and xlong queries (0.441-0.462), batch 32 (0.465), the
vanilla-CLIP and SimCLIP-2 guides (0.470, 0.473), and the TeCoA-4 ViT-L client (0.480). The base query
at 0.509 sits inside the reference range. Selectivity clears the reference for the strong robust
encoders (TeCoA-4 +0.122, TeCoA-2 +0.092 against +0.035 [-0.003, +0.086]) but not for SimCLIP-2/4 or
FARE-2 (+0.025 to +0.036), which fall inside it.

**Consequence for the defense section.** "NBFU at eps=5 defeats IG in both domains" needs splitting.
On UAV the undefended IG attack is at 0.453, below the reference, and the defense moves it to 0.727:
the defense removes a signal that was demonstrably there. On medical the undefended attack is at 0.592,
already above the reference, so the same contrast shows a defense suppressing something the metric never
showed. Report the medical arm as a paired reduction without an absolute-recovery premise.

**Consequence for wording.** Phrases like "faithful reconstructions of the targeted chest X-rays" are
not supportable. The defensible medical statements are comparative.
