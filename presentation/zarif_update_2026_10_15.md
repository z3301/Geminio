# Message to Zarif, 2026-10-15: results tables and proposed reframing

Subject: Both sweeps done; two tables and a proposed reframing before you write the theory

Hi Zarif,

Both GPU sweeps are finished and have been through an independent check, so I want to get the two tables that matter to you in front of you before you start the theory section. There is also a framing problem I think we should settle first.

## Status

348 held-out reruns on the ResNet-18 client (held-out private batches with 2 target images of 8, 3 seeds, finished today) and 48 runs where the FL client's backbone is a CLIP-family ViT-L/14 (finished yesterday). No failures. Every number below is target-image LPIPS (lower = more leakage) with paired same-seed tests and 95% CIs over 3 seeds.

Two corrections to what you saw in May. First, I dropped Geminio's attack F1: it scores near 1.0 even for pure-noise reconstructions, so it is not usable. Second, the medical numbers in the May draft came from a private batch that contained no pneumonia image at all (a class-index bug). Everything below is on corrected held-out batches, and the medical direction changed.

## Table 1. Robust VLM as the attacker's guide

ResNet-18 client. Medical query "pneumonia_descriptive", UAV query "solar_panels". Δ is the paired difference vs vanilla CLIP; negative Δ means a stronger attack.

| Guide VLM | Medical target LPIPS | Δ vs CLIP | p | UAV target LPIPS | Δ vs CLIP | p |
|---|---|---|---|---|---|---|
| CLIP (vanilla) | 0.687 ± 0.200 | — | — | 0.470 ± 0.180 | — | — |
| SimCLIP-2 | 0.448 ± 0.007 | −0.239 | .037 | 0.473 ± 0.182 | +0.002 | .962 |
| SimCLIP-4 | 0.444 ± 0.044 | −0.244 | .041 | 0.565 ± 0.050 | +0.095 | .205 |
| FARE-2 | 0.377 ± 0.126 | −0.311 | .003 | 0.515 ± 0.043 | +0.044 | .326 |
| FARE-4 | 0.491 ± 0.086 | −0.196 | .033 | 0.562 ± 0.013 | +0.091 | .168 |
| TeCoA-2 | 0.411 ± 0.048 | −0.277 | .016 | 0.496 ± 0.070 | +0.025 | .694 |
| TeCoA-4 | 0.419 ± 0.051 | −0.269 | .037 | 0.591 ± 0.022 | +0.121 | .103 |

On medical, every robust guide makes the attack stronger: 6 of 6 encoders, 18 of 18 seed pairs. That survives Holm correction as a block (sign test p ≈ 8e-6) and holds under Hungarian one-to-one matching (18 of 18). On UAV there is no effect (6 of 18 seed pairs). This reverses the May draft's medical direction, which came from the broken batch and the F1 metric.

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

On UAV, all six robust backbones leak more than vanilla CLIP, the effect is dose-ordered (the ε=4 variant beats the ε=2 variant for both FARE and TeCoA), and the gain is 1.5 to 3 times larger on the query-matched images than on the rest. Selectivity goes from about zero with vanilla CLIP to +0.12 with TeCoA-4, and +0.30 under one-to-one matching.

On medical, every ViT-L client barely reconstructs anything (target LPIPS 0.76 to 0.87). The non-target gain is as large as or larger than the target gain for all six, and selectivity is near zero either way. The attack produces a generic chest X-ray template. The ResNet-18 client leaks far more than any ViT-L client on medical.

## Reframing

I do not think "robust VLMs amplify language-guided gradient inversion" can be the headline, for three reasons.

1. The attacker-side effect is one domain only. Medical is clear; UAV is null.
2. The client-side direction is already predicted. Zhang, Chen and Li (CVPR Workshops 2022) showed that federated clients doing adversarial training leak more under gradient inversion. LeakyCLIP uses adversarial fine-tuning to make CLIP embeddings easier to invert. Torp, Smith and Breuer (ICLR 2026) argue that robust features drive reconstruction leakage. A reviewer will read Table 2 as confirmation.
3. "Robustness is not a privacy defense" argues against a claim no one made.

An independent novelty check on Oct 13 rated that framing borderline-reject at CVPR.

The claim no prior paper has made is complementarity. Zhang et al. attribute the extra leakage of adversarially trained clients to suppressed confidence on the true label (p_y → 0), which turns the output-layer gradient (p − y) ⊗ h into a clean copy of each sample's feature vector h. Geminio's rigged head produces exactly that condition on clean inputs, and only for the query-matched samples. That is what the loss amplification is. So the malicious head supplies the selection and the gradient condition, and the robust backbone supplies features that invert well. The two are complementary, not alternatives.

The prediction that follows: the leakage gain from a robust backbone should concentrate on query-matched samples and grow with the robustness level. The UAV half of Table 2 already shows both. Medical does not, most plausibly because ImageNet-robust ViT-L features do not align with chest X-rays; the attack fails for every ViT-L client there regardless of robustness.

## Theory section

What I would propose you formalize: the two-factor decomposition (head amplification × feature invertibility), and two falsifiable corollaries we can test before the deadline.

1. With an honest head, robust backbones should raise leakage uniformly rather than selectively. About 2 to 3 GPU-days to run.
2. At a fixed backbone, selectivity should scale with the head's loss ratio.

One open mechanism question for Table 1. The attacker-side medical effect needs its own explanation, because the VLM only scores the auxiliary images when the head is trained, so the mechanism has to run through which images get high loss. Loss ratios do not explain it: the medical loss ratio rose for 5 of 6 robust guides and the UAV loss ratio fell for 5 of 6, yet only medical got stronger. My candidate is retrieval precision of robust CLIP on X-rays versus vanilla CLIP. The embeddings exist, so this is cheap to test.

## NBFU

The noisy clipped-update defense stays as the defense section. After Holm correction, the defensible claims are that NBFU at ε=5 defeats Inverting Gradients in both domains, and defeats the language-guided attack across the 24-query sweep taken as a whole. Clip-only (σ=0) gives no measurable protection, so the noise is the mechanism, not clipping; this corrects the May draft. Batch-16 and batch-32 claims depend on which matching is used, so I will drop them. Four conditions where NBFU appears not to help (medical batch 32, BatchNorm train mode, medical GradInversion, the "indirect" query) are floor effects: the undefended attack already fails there, so I will not report them as the defense failing.

## Asks

1. SLADE checkpoints. The public weights on Hugging Face (mdzhossain/SLADE, slade2.pt and slade4.pt) are md5-identical to SimCLIP's simclip2.pt and simclip4.pt, so I cannot include SLADE until I have the real checkpoints. The pipeline already accepts them as drop-ins in pretrained_vlm/.
2. Your view on the complementarity framing before you write the theory, or your objections to it.
3. Do you know a second medical dataset we could run quickly? ChestMNIST is the only one so far.
4. A short call this week, given the Nov 16 deadline. I have you down as free weekdays except Thursday after 2 PM and Friday.

I can send the full summary files (results/heldout_rerun/summary.md and results/client_encoder/summary.md) and the per-run CSVs whenever useful.

Dan
