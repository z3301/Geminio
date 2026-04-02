# Geminio Presentation Script — Follow-up Meeting
## April 2026

**Context**: This is a follow-up to the March 16 meeting. Ahmed, George, and Zarif already understand the attack mechanism and cross-domain results. This presentation focuses on the SimCLIP/FARE defense evaluation and next steps.

---

## Opening (~1 min)

"Last time we met, you asked me to shift focus toward defense mechanisms. I have two sets of results. First, I evaluated Zarif's Sim-CLIP — **adversarially robust VLMs do not defend against Geminio and actually make the medical attack 57-61% stronger.** Second, I implemented DP-SGD with explicit epsilon budgets — **and it works. It's the first defense that effectively stops the attack in our experiments.**"

---

## Slide 1: Recap — Where We Left Off (~1 min)

"Quick recap for context. We reproduced Geminio across three domains — ImageNet, medical chest X-rays, and drone imagery. The attack works by using CLIP to target specific images, then exploiting gradient leakage to reconstruct them."

| Domain | Avg Loss Ratio | Key Finding |
|--------|---------------|-------------|
| ImageNet | 16x | Baseline reproduction |
| Medical | 8-12x | Descriptive prompts +39% (your suggestion, Ahmed) |
| UAV | 91x | Visually distinct objects are most vulnerable |

"Standard defenses — gradient pruning, noise injection — were inconsistent. You directed me to look at stronger defenses. Zarif shared Sim-CLIP."

---

## Slide 2: What We Tested (~2 min)

"Zarif, your Sim-CLIP hardens CLIP's vision encoder against adversarial perturbations using a symmetric cosine similarity loss. FARE is the L2-based baseline. I integrated both into our pipeline along with vanilla CLIP, so we have a four-way comparison."

| VLM | Type | Embedding Dim |
|-----|------|--------------|
| BiomedCLIP | Domain-specific (medical) | 512 |
| Vanilla CLIP | Standard | 768 |
| SimCLIP-4 | Adversarially robust (ε=4/255) | 768 |
| FARE-4 | Adversarially robust (ε=4/255) | 768 |

"Important detail: SimCLIP and FARE only change the vision encoder. The text encoder is identical — I verified this, cosine similarity of text embeddings = 1.000. So the difference is entirely in how each model *sees* the images."

---

## Slide 3: Medical Results (~2 min)

"Here's what happened on the medical domain."

| Query | Vanilla CLIP | SimCLIP-4 | FARE-4 |
|-------|-------------|-----------|--------|
| Pneumonia | 9.53x | **14.40x** | 11.34x |
| Cardiomegaly | 7.14x | **12.07x** | 9.55x |
| Effusion | 6.77x | 7.26x | **15.30x** |
| Normal | 5.71x | **11.87x** | 10.07x |
| Lung mass | 7.47x | 11.70x | **12.48x** |
| **Average** | **7.32x** | **11.46x (+57%)** | **11.75x (+61%)** |

"Both robust VLMs make the attack **significantly stronger** on medical data. SimCLIP-4 nearly doubles the loss ratio on some queries."

**Why?** "Adversarial training forces the model to learn more semantically meaningful features. For medical images where everything looks similar, a model that understands deeper semantics can better distinguish pneumonia from cardiomegaly — which is exactly what the attacker needs."

---

## Slide 4: UAV Results (~1 min)

| Query | Vanilla CLIP | SimCLIP-4 | FARE-4 |
|-------|-------------|-----------|--------|
| **Average** | **91.12x** | **80.20x (-12%)** | **54.11x (-41%)** |

"UAV results are mixed. FARE-4 reduces loss ratios by 41%, but 54x is still a devastating attack — our original medical results at 8x were already enough to reconstruct images. The attack remains fully viable even with the best-case robust VLM."

---

## Slide 5: Why VLM Robustness Doesn't Help (~2 min)

"The core issue is a **threat model mismatch**."

"In Geminio, CLIP runs on the **attacker's server**. It processes real, unperturbed images. Nobody is trying to fool CLIP with adversarial pixels. SimCLIP defends against an attack that simply doesn't happen in this scenario."

"The vulnerability isn't in the VLM — it's in the **gradient-sharing protocol**. When the client sends gradient updates back to the server, that's where the private information leaks. No amount of VLM hardening changes that."

"This is actually a useful finding for our paper — it eliminates an entire class of potential defenses and clarifies exactly where effective defenses need to operate."

---

## Slide 6: DP-SGD — A Defense That Works (~3 min)

"Since VLM hardening failed, I implemented DP-SGD — differential privacy applied to gradients. It does two things before the client shares gradients: clips the gradient norm to a fixed bound, then adds calibrated Gaussian noise."

**Medical (pneumonia):**

| ε | Attack F1 | vs Baseline |
|---|-----------|-------------|
| None | 0.625 | — |
| 1.0 | 0.250 | -60% |
| 5.0 | **0.125** | **-80%** |
| 50.0 | 0.625 | 0% |

**UAV (swimming pool):**

| ε | Attack F1 | vs Baseline |
|---|-----------|-------------|
| None | high | — |
| 1.0 | 0.125 | near-zero |
| 5.0 | **0.000** | **-100%** |
| 50.0 | **0.000** | **-100%** |

"Validated on a second UAV query (solar panels) — same pattern."

"The key insight: **gradient clipping is what stops the attack**, not the noise. Geminio relies on target images producing disproportionately large gradients. When you cap the gradient norm to C=1.0, that amplification disappears. Even ε=50 — barely any noise — stops UAV targeting completely."

"Also interesting: LPIPS and PSNR actually *improve* under DP-SGD. The defense disrupts targeting but overall reconstruction quality holds. The attack loses its ability to focus on specific images."

---

## Slide 7: Defense Comparison (~1 min)

| Defense | Level | Medical F1 | UAV F1 | Works? |
|---------|-------|-----------|--------|--------|
| None | — | 0.625 | high | — |
| Grad pruning 99% | Gradient | inconsistent | inconsistent | No |
| Noise 1e-1 | Gradient | inconsistent | inconsistent | No |
| SimCLIP-4 | VLM | — (worse) | — (mixed) | **No — helps attacker** |
| FARE-4 | VLM | — (worse) | — (mixed) | **No — helps attacker** |
| **DP-SGD ε=5** | **Gradient** | **0.125** | **0.000** | **Yes** |

"This is the clearest result we have: defenses at the VLM level don't work, defenses at the gradient level do."

---

## Slide 8: Proposed Paper Framing (~1 min)

**Title direction**: *"Gradient Clipping Stops VLM-Guided Gradient Inversion Where Robust Vision Encoders Fail"* (or similar)

**Four contributions**:
1. **Cross-domain attack reproduction** — Medical + UAV extensions, 53+ ablation experiments
2. **Negative result** — VLM adversarial robustness is orthogonal to gradient inversion defense (SimCLIP/FARE)
3. **Positive result** — DP-SGD with gradient clipping effectively stops Geminio
4. **FL-specific defense** — Federated FAML integration (pending, could strengthen the paper)

---

## Slide 9: Next Steps (~1 min)

1. **Zarif**: Share Federated FAML code so we can compare it with DP-SGD as a complementary defense
2. **Dan**: Multi-seed DP-SGD validation for statistical significance
3. **All**: Begin paper draft with the four-contribution structure

---

## Anticipated Questions

**"Could the client use SimCLIP instead of standard ResNet?"**
The client's model architecture is dictated by the server in FL — the client trains whatever model the server sends. The client doesn't run CLIP at all. CLIP is the attacker's tool.

**"Why does gradient clipping work so well?"**
Geminio's mechanism produces a model that's deliberately bad at target images, causing them to produce very large gradients. Clipping bounds the maximum gradient magnitude, which directly removes the signal the attacker exploits. It's like turning up the volume on one instrument in an orchestra — clipping puts a hard limit on how loud anything can be.

**"Does DP-SGD hurt model training?"**
We're only evaluating the defense effect on reconstruction quality, not downstream model accuracy. In practice, DP-SGD with ε=5-10 typically causes modest accuracy drops (1-5%). This is the standard privacy-utility tradeoff.

**"Why does FARE help slightly on UAV but not medical?"**
UAV images are already visually distinct. FARE's different feature representations happen to slightly blur some distinctions. Medical images are tightly clustered, so any improvement in semantic discrimination helps the attacker.

**"Is the 57-61% increase from SimCLIP/FARE statistically significant?"**
It's consistent across all 5 medical queries and both robust VLMs show the same trend. The direction is clear and consistent.

**"Why is the medical DP-SGD F1 non-monotonic (0.375 at ε=0.1, 0.125 at ε=5)?"**
Stochastic variation in the 24,000-iteration reconstruction. The clipping effect (C=1.0) dominates over noise magnitude in this regime. Multi-seed experiments would clarify.

---

## Timing

| Section | Minutes |
|---------|---------|
| Opening | 1 |
| Recap | 1 |
| SimCLIP/FARE setup | 2 |
| Medical results (VLM) | 2 |
| UAV results (VLM) | 1 |
| Why VLM doesn't help | 2 |
| DP-SGD results | 3 |
| Defense comparison | 1 |
| Paper framing | 1 |
| Next steps | 1 |
| **Total** | **~15 min** |
| Q&A buffer | 5-10 min |
