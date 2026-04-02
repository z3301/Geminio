# Geminio Presentation Script — Follow-up Meeting
## April 2026

**Context**: This is a follow-up to the March 16 meeting. Ahmed, George, and Zarif already understand the attack mechanism and cross-domain results. This presentation focuses on the SimCLIP/FARE defense evaluation and next steps.

---

## Opening (~1 min)

"Last time we met, you asked me to shift focus toward defense mechanisms. Zarif shared the Sim-CLIP repository on March 24, and I've completed the evaluation. The short answer: **adversarially robust VLMs do not defend against Geminio — and in the medical domain, they actually make the attack 57-61% stronger.**"

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

## Slide 6: What Would Actually Work (~2 min)

"Defenses need to operate at the gradient level. Here's where I think we should focus."

| Approach | Level | Status |
|----------|-------|--------|
| Gradient pruning | Gradient | Tested — inconsistent |
| Noise injection | Gradient | Tested — inconsistent |
| DP-SGD | Gradient | Not yet tested — has mathematical guarantees |
| **Federated FAML** | **FL protocol** | **Not yet tested — most promising** |
| Secure aggregation | Cryptographic | Theoretical baseline |

"Ahmed, Zarif — Federated FAML operates at the right level. It was designed for FL-specific adversarial threats. If it can reduce reconstruction quality against Geminio, that's the paper."

---

## Slide 7: Proposed Paper Framing (~1 min)

**Title direction**: *"Why Robust VLMs Don't Stop Gradient Inversion: Cross-Domain Evaluation of Defenses Against Geminio"* (or similar)

**Three contributions**:
1. **Cross-domain attack reproduction** — Medical + UAV extensions, 53 ablation experiments
2. **Negative result** — VLM adversarial robustness is orthogonal to gradient inversion defense (SimCLIP/FARE evaluation)
3. **FL-specific defense** — Federated FAML (or another gradient-level mechanism) as the actual solution

"The negative result is publishable on its own — it's a common misconception that hardening model components provides general robustness. We show that's not the case for gradient inversion."

---

## Slide 8: Next Steps (~1 min)

1. **Zarif**: Share Federated FAML code/paper so I can integrate it with the Geminio pipeline
2. **Dan**: Implement DP-SGD evaluation with explicit ε budgets as a defense baseline
3. **Dan**: Run Phase 3 reconstruction on SimCLIP/FARE models to get perceptual quality metrics
4. **All**: Begin paper draft outlining the three-contribution structure

---

## Anticipated Questions

**"Could the client use SimCLIP instead of standard ResNet?"**
The client's model architecture is dictated by the server in FL — the client trains whatever model the server sends. The client doesn't run CLIP at all. CLIP is the attacker's tool.

**"What if the client adds SimCLIP as a preprocessing step?"**
This would change the feature distribution the model sees, but wouldn't affect gradient leakage. The gradients still contain per-sample information regardless of how the input was preprocessed.

**"Why does FARE help slightly on UAV but not medical?"**
UAV images are already visually distinct, so the embedding space is already well-separated. FARE's different feature representations happen to slightly blur some of those distinctions. Medical images are the opposite — they're tightly clustered, so any improvement in semantic discrimination helps the attacker.

**"Is the 57-61% increase statistically significant?"**
It's consistent across all 5 medical queries and both robust VLMs show the same trend. We'd need to run multi-seed experiments with the SimCLIP/FARE variants for formal significance testing, but the direction is clear and consistent.

---

## Timing

| Section | Minutes |
|---------|---------|
| Opening | 1 |
| Recap | 1 |
| What we tested | 2 |
| Medical results | 2 |
| UAV results | 1 |
| Why it doesn't help | 2 |
| What would work | 2 |
| Paper framing | 1 |
| Next steps | 1 |
| **Total** | **~13 min** |
| Q&A buffer | 5-10 min |
