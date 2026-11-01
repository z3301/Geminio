# Message to Zarif, 2026-10-24: both sweeps verified, two tables, and a reframing to settle before the theory section

(Supersedes the unsent 2026-10-15 draft. Tables 1 and 2 are read from `results/heldout_rerun/summary.md` and `results/client_encoder/summary.md`; the "early checks" section is from the Oct 17-19 analysis notes.)

Subject: Geminio/NBFU: two result tables and a reframing to settle before you write the theory

Hi Zarif,

Both GPU sweeps are finished and have been through an independent check, so here are the two tables that matter for the theory section, plus a framing problem I think we should settle before you write it. CVPR is 23 days out.

## Status

348 held-out reruns on the ResNet-18 client (private batches with 2 target images out of 8, 3 seeds) and 48 runs where the FL client's backbone is a CLIP-family ViT-L/14. No failures. Every number below is target-image LPIPS (lower = more leakage) with paired same-seed tests and 95% CIs over 3 seeds.

Two corrections to what you saw in May. First, I dropped Geminio's attack F1: it scores near 1.0 even for pure-noise reconstructions, so it cannot separate attacks. Second, the medical numbers in the May draft came from a private batch that contained no pneumonia image at all (a class-index bug). Everything below is on corrected held-out batches, and the medical direction changed.

## Table 1. Robust VLM as the attacker's guide

ResNet-18 client. Medical query "pneumonia_descriptive", UAV query "solar_panels". Δ is the paired difference vs vanilla CLIP as the guide; negative Δ means a stronger attack.

| Guide VLM | Medical target LPIPS | Δ vs CLIP | p | UAV target LPIPS | Δ vs CLIP | p |
|---|---|---|---|---|---|---|
| CLIP (vanilla) | 0.687 ± 0.200 | — | — | 0.470 ± 0.180 | — | — |
| SimCLIP-2 | 0.448 ± 0.007 | −0.239 | .037 | 0.473 ± 0.182 | +0.002 | .962 |
| SimCLIP-4 | 0.444 ± 0.044 | −0.244 | .041 | 0.565 ± 0.050 | +0.095 | .205 |
| FARE-2 | 0.377 ± 0.126 | −0.311 | .003 | 0.515 ± 0.043 | +0.044 | .326 |
| FARE-4 | 0.491 ± 0.086 | −0.196 | .033 | 0.562 ± 0.013 | +0.091 | .168 |
| TeCoA-2 | 0.411 ± 0.048 | −0.277 | .016 | 0.496 ± 0.070 | +0.025 | .694 |
| TeCoA-4 | 0.419 ± 0.051 | −0.269 | .037 | 0.591 ± 0.022 | +0.121 | .103 |

On medical, every robust guide makes the attack stronger: 6 of 6 encoders, 18 of 18 seed pairs. That survives Holm correction as a block (sign test p ≈ 8e-6) and holds under Hungarian one-to-one matching (18 of 18). For reference, BiomedCLIP as the guide gives 0.590 ± 0.178 on the same batch, so the ImageNet-robust CLIPs beat the domain-specific guide too. On UAV there is no effect (6 of 18 seed pairs). This reverses the May draft's medical direction, which came from the broken batch and the F1 metric.

## Table 2. Robust encoder as the FL client's backbone

ViT-L/14 visual tower plus the Geminio head. The attacker's guide is fixed: BiomedCLIP for medical, vanilla CLIP for UAV. Δ target and Δ non-target are paired differences vs the vanilla-CLIP client. Selectivity is non-target LPIPS minus target LPIPS, given as best-match / Hungarian one-to-one matching.

UAV (solar-panel targets):

| Client encoder | Target LPIPS | Δ target (p) | Δ non-target | Selectivity (best / Hungarian) |
|---|---|---|---|---|
| CLIP (vanilla) | 0.683 ± 0.021 | — | — | −0.007 / −0.021 |
| SimCLIP-2 | 0.618 ± 0.046 | −0.065 (.008) | −0.022 | +0.036 / +0.170 |
| SimCLIP-4 | 0.589 ± 0.034 | −0.095 (.002) | −0.063 | +0.025 / +0.020 |
| FARE-2 | 0.604 ± 0.038 | −0.079 (.011) | −0.040 | +0.032 / +0.061 |
| FARE-4 | 0.530 ± 0.075 | −0.154 (.019) | −0.080 | +0.067 / +0.136 |
| TeCoA-2 | 0.489 ± 0.050 | −0.195 (.003) | −0.095 | +0.092 / +0.139 |
| TeCoA-4 | 0.480 ± 0.043 | −0.204 (.001) | −0.075 | +0.122 / +0.301 |
| ResNet-18 (reference) | 0.476 ± 0.155 | −0.207 (.023) | −0.086 | +0.114 / +0.195 |

Medical (pneumonia targets):

| Client encoder | Target LPIPS | Δ target (p) | Δ non-target | Selectivity (best / Hungarian) |
|---|---|---|---|---|
| CLIP (vanilla) | 0.867 ± 0.096 | — | — | −0.029 / −0.062 |
| SimCLIP-2 | 0.762 ± 0.037 | −0.104 (.024) | −0.115 | −0.039 / −0.034 |
| SimCLIP-4 | 0.768 ± 0.046 | −0.098 (.090) | −0.103 | −0.033 / +0.067 |
| FARE-2 | 0.795 ± 0.071 | −0.071 (.171) | −0.090 | −0.048 / +0.056 |
| FARE-4 | 0.771 ± 0.015 | −0.096 (.065) | −0.108 | −0.041 / +0.070 |
| TeCoA-2 | 0.790 ± 0.031 | −0.076 (.121) | −0.098 | −0.050 / +0.033 |
| TeCoA-4 | 0.775 ± 0.024 | −0.092 (.042) | −0.105 | −0.042 / +0.055 |
| ResNet-18 (reference) | 0.574 ± 0.135 | −0.292 (.004) | −0.243 | +0.021 / +0.363 |

On UAV, all six robust backbones leak more than vanilla CLIP (6 of 6 significant), the effect is dose-ordered (the ε=4 variant leaks more than the ε=2 variant in all three families), and the gain is 1.5 to 3 times larger on the query-matched images than on the rest. Selectivity goes from about zero with vanilla CLIP to +0.12 with TeCoA-4, and +0.30 under one-to-one matching.

On medical, every ViT-L client barely reconstructs anything (target LPIPS 0.76 to 0.87). The non-target gain is as large as or larger than the target gain for all six, and selectivity is near zero either way. The attack produces a generic chest X-ray template. The ResNet-18 client leaks far more than any ViT-L client on medical.

## Reframing

I do not think "robust VLMs amplify language-guided gradient inversion" can be the headline, for three reasons.

1. The attacker-side effect is one domain only. Medical is clear; UAV is null.
2. The client-side direction is already predicted. Zhang, Chen and Li (CVPR Workshops 2022) showed that federated clients doing adversarial training leak more under gradient inversion. LeakyCLIP uses adversarial fine-tuning to make CLIP embeddings easier to invert. Torp, Smith and Breuer (ICLR 2026) argue that robust features drive reconstruction leakage. A reviewer will read Table 2 as confirmation.
3. "Robustness is not a privacy defense" argues against a claim no one made.

When I checked this framing against the literature on Oct 13, I could not find a version of it that a CVPR reviewer would not read as incremental.

The claim no prior paper has made is complementarity. Zhang et al. attribute the extra leakage of adversarially trained clients to suppressed confidence on the true label (p_y → 0), which turns the output-layer gradient (p − y) ⊗ h into a clean copy of each sample's feature vector h. Geminio's rigged head produces exactly that condition on clean inputs, and only for the query-matched samples. That is what the loss amplification is. So the malicious head supplies the selection and the gradient condition, and the robust backbone supplies features that invert well. The two are complementary, not alternatives, and nobody has put them together.

The prediction that follows: the leakage gain from a robust backbone should concentrate on query-matched samples and grow with the robustness level. The UAV half of Table 2 already shows both. Medical does not, most plausibly because ImageNet-robust ViT-L features do not align with chest X-rays; the attack fails for every ViT-L client there regardless of robustness.

## Early checks of the prediction (preliminary: 3 seeds, no multiple-comparison correction)

1. Plain-head control. I retrained an ordinary, unweighted head on each of the seven frozen client encoders and attacked it with the same pipeline. On UAV, the plain head gives selectivity ≈ 0 on all seven encoders, while the rigged head is positive on all six robust ones (paired difference significant on 5 of 7; vanilla CLIP is not selective under either head). So selectivity is a head property, and the robust backbone only turns it into reconstructions. The encoder effect exists without rigging too: plain-head target LPIPS is 0.73 on CLIP vs 0.54 to 0.58 on the robust encoders. That is the two-factor decomposition in one experiment.
2. Selectivity tracks the head's loss ratio at a fixed backbone. Across the 24-query medical language sweep (loss ratios 3x to 11x), Spearman correlation between loss ratio and selectivity is +0.90 best-match / +0.78 Hungarian (p ≤ .003). UAV has the same sign but is saturated (31x to 165x), so it is not separable.
3. Temperature. Varying Geminio's softmax temperature T in {10, 30, 100, 300, 1000} at a fixed query moves the medical loss ratio from 7.6x to 13.9x and pooled target LPIPS is monotone in T (per-run Spearman −0.58, n = 33, p < .001). Only the T = 10 vs T ≥ 100 contrast is clean; neighbouring temperatures sit inside the same-seed noise floor.
4. The medical attacker-side effect in Table 1 is not retrieval. Vanilla CLIP and all six robust CLIPs rank pneumonia at or below chance on the auxiliary set (AUROC 0.43 to 0.49; BiomedCLIP 0.60), and the top-ranked auxiliary images are no closer in LPIPS to the held-out targets than random ones. So robust guides amplify some guide-specific, non-pneumonia subset of the auxiliary set, and that still strengthens the attack on pneumonia targets. On UAV, by contrast, retrieval precision varies by guide and tracks attack strength (ρ = −0.86, p = .014), which explains the null UAV average.
5. A query-swap control (does UAV selectivity follow the query when I rig the head for airport runways instead of solar panels?) has run but is not scored yet.

## What I would propose you formalize

The two-factor decomposition: leakage on a sample = (how far the head drives p_y toward 0 on that sample) × (how invertible the backbone's feature h is), with Geminio's loss amplification as the mechanism for the first factor and robust features as the mechanism for the second. Checks 1 to 3 above are the empirical anchors. The open item the theory has to accommodate is Table 1: when the robust VLM is only the attacker's scoring tool and the inverted gradient comes from ResNet-18, the effect has to run through which auxiliary images get high loss, and neither retrieval precision nor appearance explains it. My remaining candidates are the feature-space geometry of the amplified subset or head-training variance; I would value your read.

## NBFU

The noisy clipped-update defense stays as the defense section, with one change I want to flag.

I added a reference I should have had from the start: eight images drawn at random from the same dataset, scored through the identical matching code, as the value an uninformative reconstruction produces. That gives target LPIPS 0.327 (5th to 95th percentile 0.294 to 0.361) on medical and 0.535 (0.485 to 0.571) on UAV. It changes how several numbers read.

On UAV the picture holds. The undefended Inverting Gradients attack sits at 0.453, below the 5th percentile, and ε = 5 moves it to 0.727: the defense removes a signal that was demonstrably present. Eight UAV conditions clear the reference, including three queries, batch 32, two attacker guides and the TeCoA-4 client.

On medical it does not. The undefended attack is at 0.592, already above the medical 95th percentile, and so is every other medical configuration in the project, across both client architectures, all 24 queries and all seven guides. The best medical number anywhere is 0.377. So on target LPIPS no medical setup beats a random draw from the dataset, and "the defense stops the attack on chest X-rays" is not a claim I can make. Part of this is the metric: ChestMNIST at 64 by 64 upscaled to 224 makes any two images mutually similar in LPIPS, which leaves little room above the reference. Medical selectivity does clear its reference for several queries, so the selection signal survives where the recovery claim does not.

Everything paired is unaffected, since the reference depends only on the batch and is identical across arms. So the defended-versus-undefended, guide-versus-guide and head-versus-head comparisons all stand; what goes is the absolute reading.

The rest of the defense section is unchanged. NBFU at ε = 5 weakens the language-guided attack across the 24-query sweep as a whole (medical 11 of 12 queries, UAV 12 of 12). Against HF-GradInv at batch 8 the sign is right in both domains but not separable at 3 seeds (p = .056 / .108). Clip-only (σ = 0) gives no measurable protection anywhere, so the noise is the mechanism, not clipping; this corrects the May draft. Batch-16 and batch-32 claims depend on which matching is used, so I will drop them.

## Asks

1. SLADE checkpoints. The public weights on Hugging Face (mdzhossain/SLADE, slade2.pt and slade4.pt) are md5-identical to SimCLIP's simclip2.pt and simclip4.pt, so I cannot include SLADE until I have the real checkpoints. The pipeline accepts them as drop-ins in pretrained_vlm/.
2. Your view on the complementarity framing, or your objections to it, before you write the theory.
3. A second medical dataset we could run quickly. ChestMNIST is the only one so far.
4. A short call this week. Given the 23 days left, I would like to lock the framing by Oct 28 so the theory and the experiments section can be written in parallel.

I can send the full summary files and the per-run CSVs whenever useful.

Dan
