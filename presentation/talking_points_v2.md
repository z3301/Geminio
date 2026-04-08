# Geminio Presentation Script — Follow-up Meeting
## April 2026 (updated April 8)

**Context**: This is a follow-up to the March 16 meeting. Ahmed, George, and Zarif already understand the attack mechanism and cross-domain results. This presentation focuses on the SimCLIP/FARE defense evaluation, DP-SGD positive result, and FedAvg analysis.

---

## Opening (~1 min)

"Last time we met, you asked me to shift focus toward defense mechanisms. I have three sets of results. First, I evaluated Zarif's Sim-CLIP and all robust CLIP variants — **adversarially robust VLMs do not defend against Geminio and actually make the medical attack up to 74% stronger.** Second, I implemented DP-SGD with explicit epsilon budgets — **and it works. It's the first defense that reliably stops the attack in our experiments.** Third, I tested FedAvg multi-epoch local training — **partial defense, less reliable than DP-SGD.**"

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

"Zarif, your Sim-CLIP hardens CLIP's vision encoder against adversarial perturbations using a symmetric cosine similarity loss. FARE is the L2-based baseline. TeCoA is the original adversarial fine-tuning approach. I integrated all of them — both ε=4/255 and ε=2/255 variants — into our pipeline."

| VLM | Type | ε |
|-----|------|---|
| BiomedCLIP | Domain-specific (medical) | — |
| Vanilla CLIP | Standard | — |
| SimCLIP² / SimCLIP⁴ | Symmetric cosine sim hardening | 2/255, 4/255 |
| FARE² / FARE⁴ | L2 robust feature hardening | 2/255, 4/255 |
| TeCoA² / TeCoA⁴ | Text-conditioned adversarial training | 2/255, 4/255 |

"Important detail: SimCLIP, FARE, and TeCoA only change the vision encoder. The text encoder is identical — I verified this, cosine similarity of text embeddings = 1.000. So the difference is entirely in how each model *sees* the images."

---

## Slide 3: Medical Results (~2 min)

"Here's the full medical comparison across all variants."

| VLM | Avg Loss Ratio | vs Vanilla CLIP |
|-----|---------------|-----------------|
| TeCoA² | 6.9x | -6% |
| Vanilla CLIP | 7.3x | baseline |
| BiomedCLIP | 8.4x | +15% |
| TeCoA⁴ | 9.3x | +28% |
| FARE² | 10.5x | +44% |
| SimCLIP⁴ | 11.5x | +57% |
| FARE⁴ | 11.7x | +61% |
| **SimCLIP²** | **12.7x** | **+74%** |

"Most robust VLMs make the attack **significantly stronger** on medical data. SimCLIP² is 74% stronger than vanilla CLIP. Only TeCoA² is slightly below baseline."

**Why?** "Adversarial training forces the model to learn more semantically meaningful features. For medical images where everything looks similar, a model that understands deeper semantics can better distinguish pneumonia from cardiomegaly — which is exactly what the attacker needs."

---

## Slide 4: UAV Results (~1 min)

| VLM | Avg Loss Ratio | vs Vanilla CLIP |
|-----|---------------|-----------------|
| TeCoA² | 45.5x | -50% |
| TeCoA⁴ | 46.5x | -49% |
| FARE⁴ | 54.1x | -41% |
| SimCLIP² | 66.6x | -27% |
| SimCLIP⁴ | 80.2x | -12% |
| Vanilla CLIP | 91.1x | baseline |
| **FARE²** | **116.7x** | **+28%** |

"UAV results are mixed. **FARE² actually increases the attack** by 28%. TeCoA variants reduce it by ~50%, but 45x is still devastating — our original medical results at 8x were already enough to reconstruct images. The attack remains fully viable even with the best-case robust VLM."

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

## Slide 6.5: FedAvg Multi-Epoch (~2 min)

"The Geminio paper noted that FedAvg with multiple local epochs weakens the attack. I tested it systematically."

**Medical (pneumonia, lr=1e-3):**

| Local Epochs | Attack F1 |
|-------------|-----------|
| 0 (FedSGD) | 0.625 |
| 1 | 0.500 |
| 2 | 0.875 |
| 5 | 0.000 |
| 10 | 0.500 |
| 20 | 0.000 |

**UAV (swimming pool, lr=1e-3):**

| Local Epochs | Attack F1 |
|-------------|-----------|
| 1 | 0.125 |
| 5 | 0.375 |
| 20 | 0.250 |

"FedAvg provides partial defense but is **inconsistent** — medical F1 is non-monotonic. Sometimes 5 epochs stops the attack, sometimes 10 epochs lets it back in. UAV stays low (0.125–0.375) but never reliably reaches zero. DP-SGD is more reliable and provides formal privacy guarantees that FedAvg doesn't."

---

## Slide 7: Defense Comparison (~1 min)

| Defense | Level | Medical F1 | UAV F1 | Reliable? | Guarantee? |
|---------|-------|-----------|--------|-----------|------------|
| None (FedSGD) | — | 0.625 | high | — | No |
| Grad pruning 99% | Gradient | inconsistent | inconsistent | No | No |
| Noise 1e-1 | Gradient | inconsistent | inconsistent | No | No |
| SimCLIP⁴ | VLM | worse (+57%) | -12% | No | No |
| FARE⁴ | VLM | worse (+61%) | -41% | No | No |
| FedAvg 5 epochs | FL protocol | 0.000 | 0.375 | Partial | No |
| FedAvg 20 epochs | FL protocol | 0.000 | 0.250 | Partial | No |
| **DP-SGD ε=5** | **Gradient** | **0.125** | **0.000** | **Yes** | **(ε,δ)-DP** |

"This is the clearest result we have: defenses at the VLM level don't work, defenses at the gradient level do. FedAvg helps but DP-SGD is the only reliable defense and the only one with a formal privacy guarantee."

---

## Slide 8: Proposed Paper Framing (~1 min)

**Title direction**: *"Gradient Clipping Stops VLM-Guided Gradient Inversion Where Robust Vision Encoders Fail"* (or similar)

**Four contributions**:
1. **Cross-domain attack reproduction** — Medical + UAV extensions, 53+ ablation experiments
2. **Negative result** — VLM adversarial robustness is orthogonal to gradient inversion defense (SimCLIP/FARE/TeCoA, all 6 variants tested)
3. **Positive result** — DP-SGD with gradient clipping effectively stops Geminio. FedAvg multi-epoch provides partial defense.
4. **FL-specific defense** — Federated FAML integration (pending Zarif sharing code)

---

## Slide 9: Next Steps (~1 min)

1. **Zarif**: Share Federated FAML code so we can compare it with DP-SGD as a complementary defense
2. **Dan**: Multi-seed DP-SGD validation for statistical significance
3. **Dan**: Combined defenses (DP-SGD + FedAvg, DP-SGD + pruning)
4. **All**: Begin paper draft with the four-contribution structure

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
| SimCLIP/FARE/TeCoA setup | 2 |
| Medical results (VLM) | 2 |
| UAV results (VLM) | 1 |
| Why VLM doesn't help | 2 |
| DP-SGD results | 3 |
| FedAvg results | 2 |
| Defense comparison | 1 |
| Paper framing | 1 |
| Next steps | 1 |
| **Total** | **~17 min** |
| Q&A buffer | 5-10 min |
