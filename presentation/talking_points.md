# Geminio Presentation Script — Dr. Imteaj Meeting
## March 16, 2026

---

## Opening (~1 min)

"I've been working on reproducing and extending the Geminio gradient inversion attack from ICCV. The core question is: **can a malicious federated learning server steal specific private images just by describing what it wants in plain English?** The answer is yes — and I've now shown it works across three very different image domains."

---

## Slide 1: What is Geminio? (~2 min)

"Geminio exploits federated learning. The attacker — the server — sends a rigged model to clients. This model has been deliberately trained to be bad at classifying certain images, the ones matching the attacker's text query. When the client trains on it and sends back gradients, the query-matching images produce the biggest gradient signals, and the attacker can reconstruct them."

**Key points to hit:**
- Three phases: VLM embedding, malicious model training, gradient inversion
- Uses CLIP to bridge text queries → image targeting
- Loss function: `mean(per_sample_loss * (1 - softmax(clip_sim * T)))`
- We use the paper's own codebase as our foundation — the `breaching` library handles gradient inversion, their `core/` has model definitions. We built `prototype/` on top for new domains.

---

## Slide 2: Three Domains (~3 min)

"The paper only tests on ImageNet. We extended to two new domains: medical chest X-rays and aerial drone imagery."

| Domain | Dataset | VLM | Backbone | Avg Loss Ratio |
|--------|---------|-----|----------|---------------|
| ImageNet | 50K images, 1000 classes | CLIP ViT-L/14 | ResNet34 | **16x** |
| Medical | ChestMNIST, 11K images, 15 classes | BiomedCLIP | ResNet18 | **12x** |
| UAV | UAVScenes, 4K drone images, 18 classes | CLIP ViT-L/14 | ResNet18 | **91x** |

"The UAV domain is dramatically more vulnerable — 91x loss ratio — because aerial objects like solar panels and bridges are visually distinctive. Medical images all look similar, so the VLM has less to work with."

**Show:** `presentation/figures/demo_cross_domain.png`

---

## Slide 3: Reconstruction Results (~2 min)

"Here's what the attack actually recovers from gradients."

| Domain | LPIPS (lower=better) | PSNR (higher=better) |
|--------|---------------------|---------------------|
| Medical baseline | 0.79 | 11.3 dB |
| Medical (pneumonia query) | 1.02 | 10.9 dB |
| UAV baseline | 0.60 | 12.3 dB |
| UAV (solar panels query) | 0.66 | 12.4 dB |

"UAV reconstructions are visibly better — you can make out roofs, roads, structural features. Medical X-rays are harder because they're low-resolution and visually uniform."

**Show:** `presentation/figures/demo_medical.png` and `demo_uav.png`

---

## Slide 4: Ablation Studies (~5 min)

"I ran 53 experiments to address potential reviewer concerns. Four main experiments:"

### 4a. Batch Composition — Does the target need to be present?

"YES. When pneumonia images are in the batch, Attack F1 = 1.0. When they're absent, it drops to 0.125 — essentially random. This proves the attack is genuinely targeted, not just general gradient leakage."

**Show:** `presentation/figures/demo_batch_composition.png`

### 4b. Temperature Scaling — Why T=100?

"Temperature controls how sharply the loss function distinguishes targets from non-targets. At T=1, there's no targeting — it acts like a normal model. T=10 is too weak. The sweet spot is T=50 to T=200. T=100 is consistent with what's used in contrastive learning literature."

| T | Behavior | Attack F1 |
|---|----------|-----------|
| 1 | No targeting (baseline-equivalent) | 1.00 (but untargeted) |
| 10 | Too weak | 0.38 |
| 50-200 | Effective targeting | 0.63-0.88 |

### 4c. Pseudo-Labels — Does the attacker need true labels?

"No. We generated pseudo-labels using BiomedCLIP zero-shot classification — only 29.6% accurate, basically guessing. The attack *still works*, actually slightly better than with true labels. This is important because it removes the assumption that the attacker knows the ground-truth labels."

| Labels | Accuracy | Attack F1 |
|--------|----------|-----------|
| True | 100% | 0.625 |
| Pseudo (BiomedCLIP) | 29.6% | **0.750** |

### 4d. Reproducibility

"Five random seeds: medical results are very stable, UAV has more variance but conclusions hold. Larger batch sizes surprisingly *improve* medical Attack F1 but degrade UAV."

---

## Slide 5: Defenses Don't Work (~1 min)

"We tested gradient pruning at 70/90/99% and noise injection at three scales. Neither reliably stops the attack. Aggressive pruning at 99% paradoxically *improves* some medical metrics — it may be removing noise rather than the attack signal. The only defense with mathematical guarantees would be differential privacy (DP-SGD), which we haven't tested yet."

---

## Slide 6: Key Takeaways (~1 min)

1. **Geminio extends to medical and aerial domains** with varying effectiveness
2. **Visual distinctiveness drives vulnerability**: UAV (91x) >> ImageNet (16x) >> Medical (12x)
3. **The attack is truly targeted**: it only works when the target is present in the batch
4. **No insider knowledge needed**: pseudo-labels with 29.6% accuracy still work
5. **Standard defenses fail**: pruning and noise don't reliably protect
6. **Descriptive prompts help**: +39% improvement in medical domain with better prompt engineering

---

## Slide 7: Next Steps (~1 min)

1. **Compare against other attacks** — FEDLEAK, GUIDE — are they stronger?
2. **DP-SGD evaluation** — the only defense with formal privacy guarantees
3. **Per-target-image analysis** — does the attack preferentially recover the *specific* query-matching images within a batch?
4. **Paper draft** — the cross-domain comparison is our primary contribution

---

## Anticipated Questions

**"How does this compare to the paper's reported numbers?"**
We can't directly compare reconstruction quality because they use batch size 64/128 on ImageNet with ResNet34, and we use batch size 8 on smaller datasets with ResNet18. Our loss ratios are comparable (theirs: ~15-20x on ImageNet, ours: ~16x). The extension to new domains is the contribution.

**"Why ResNet18 instead of ResNet34?"**
Smaller datasets don't benefit from larger models, and it trains 6x faster. The classifier head architecture is identical.

**"Why BiomedCLIP instead of regular CLIP for medical?"**
CLIP ViT-L/14 was trained on internet images — it doesn't understand "pleural effusion" or "cardiomegaly." BiomedCLIP was trained on PubMed literature and medical images, so it produces better text-image alignment for medical queries.

**"What's the practical threat here?"**
A hospital participating in federated learning could have specific patient X-rays stolen by a malicious server. The attacker doesn't need to know what's in the data — they just describe what they want.

---

## Timing

| Section | Minutes |
|---------|---------|
| Opening | 1 |
| What is Geminio | 2 |
| Three Domains | 3 |
| Reconstruction | 2 |
| Ablations | 5 |
| Defenses | 1 |
| Takeaways | 1 |
| Next Steps | 1 |
| **Total** | **~16 min** |
| Q&A buffer | 5-10 min |
