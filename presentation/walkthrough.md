# Walkthrough Script — Detailed Report v2
## For presenting `reports/detailed_report_v2.md`
### Last updated: April 8, 2026

---

### Section 1: Project Overview (~30 sec)

"We're reproducing and extending the Geminio attack — a gradient inversion attack in federated learning that lets a malicious server steal specific private images by describing what it wants in plain English. The original paper only tested on ImageNet. We extended it to medical chest X-rays and aerial drone imagery, then pivoted to evaluating defenses."

---

### Section 2: Attack Mechanism (~2 min)

"The attack has three phases."

"First, the attacker uses a vision-language model — CLIP or BiomedCLIP — to compute embeddings for every training image and for the text query, like 'any chest X-ray showing pneumonia.'"

"Second, the attacker trains a rigged model. The key is this loss function —" *(point to the code block)* "— it down-weights the loss on images that match the query. So the model learns everything except the targets. When a victim trains on this model, the targets produce the biggest gradients because the model is worst at them."

"Third, the attacker runs gradient inversion on those gradients — 24,000 iterations of optimization — and recovers the target images."

"We measure success with the loss ratio: how much worse is the model on targets vs everything else. Higher means the attack amplifies the target signal more. The whole attack happens entirely during FL training — round 1 the server sends the rigged model, round 2 it receives gradients and runs inversion. No inference involved."

---

### Section 3: Visual Demonstration (~3 min)

"Let me show you what this actually looks like."

*(Scroll to Figure 3.1)* "This is ImageNet — 128 private images on top, baseline reconstruction in the middle, and the Geminio reconstruction targeting 'any human faces' on the bottom. You can see the face-like images come through more clearly."

*(Scroll to Figure 3.2)* "Medical domain — chest X-rays. You can see reconstruction quality is lower. All X-rays look similar, so the VLM has less to distinguish."

*(Scroll to Figure 3.3)* "UAV domain — drone imagery. This works much better. Solar panels and buildings have distinctive visual features, so the reconstructions are more recognizable."

*(Scroll to Figure 3.4)* "Here's the cross-domain comparison side by side. Visual distinctiveness is the key factor."

*(Scroll to Figure 3.5)* "And this is the batch composition result — when pneumonia images are in the batch, perfect identification. When they're absent, it drops to near-random. The attack is genuinely targeted."

---

### Section 4: Loss Surface Reshaping Results (~2 min)

"Here are the numbers for phase 2 — training the malicious models."

"ImageNet averages about 16x loss ratio across 5 queries. That reproduces the paper."

"Medical domain — we started with generic prompts like 'any chest X-ray showing pneumonia' and got about 8x. Ahmed suggested trying more descriptive prompts — radiological language like 'bilateral patchy opacities' — and that improved it to about 12x. A 39% improvement just from better prompt engineering."

"UAV domain is dramatically higher — 91x average. Solar panels alone hit 174x. The visual distinctiveness of aerial objects makes this domain much more vulnerable."

"One important note — the swimming pool query got 127x but there are zero pool images in UAVScenes. That ratio is meaningless. This is why we ran the controlled batch composition experiments."

---

### Section 5: Reconstruction Results (~2 min)

"Phase 3 — actually recovering images from gradients."

*(Point to the metrics table)* "Note that ImageNet metrics are dashes — we only saved grid JPEGs, not per-image tensors, so we can't compute LPIPS or PSNR. The original paper doesn't report these for ImageNet either. For our two extension domains, UAV consistently produces better perceptual quality — lower LPIPS, higher PSNR. Medical is harder because of the low resolution and visual uniformity."

---

### Section 6: Methodology Deviations (~1 min)

"We made several engineering decisions that differ from the paper, all documented with justification."

"The main ones: we added temperature scaling — the paper uses raw cosine similarity but that produces near-uniform softmax weights. T=100 is standard in contrastive learning. We only train the classifier head, not the full network. We use BiomedCLIP instead of regular CLIP for medical. And we use BCEWithLogitsLoss for UAV because it's multi-label. Each of these has an ablation or citation backing it up."

---

### Section 7: Cross-Domain Analysis (~1 min)

"The headline finding for the attack: visual distinctiveness drives vulnerability. UAV at 91x loss ratio, ImageNet at 16x, medical at 12x. The VLM embedding spread predicts attack strength — CLIP on UAV images has much wider similarity distributions than BiomedCLIP on chest X-rays."

---

### Section 8: Ablation Studies (~5 min)

"We ran 53 experiments to address five potential reviewer concerns."

**8.1 Batch Composition:** "Does the target need to be in the batch? Yes. Medical Attack F1 drops from 1.0 to 0.125 without pneumonia images. The baseline model stays at 1.0 regardless — that's the control showing it's not just reconstruction quality, it's specifically the targeting."

**8.2 Temperature:** "T=1 is basically a normal model — best perceptual metrics but no targeting. T=10 is too weak. T=50 to 200 is the effective zone. T=100 is a reasonable default."

**8.3 Pseudo-Labels:** "The attacker doesn't need true labels. BiomedCLIP zero-shot classification is only 29.6% accurate but the attack still works — actually outperforms true labels on Attack F1: 0.75 vs 0.625."

**8.4 Seeds:** "Five random seeds. Medical is rock-solid — identical Attack F1 across all seeds. UAV has more variance, which makes sense given the visual diversity."

**8.5 Batch Sizes:** "Interesting split — medical Attack F1 actually improves with larger batches, while UAV degrades."

---

### Section 9: VLM Robustness Defense Evaluation (~3 min)

"After the March 16 meeting, we pivoted from attack reproduction to defenses. Zarif suggested testing his SimCLIP — adversarially robust CLIP variants. We tested all 6 variants: SimCLIP², SimCLIP⁴, FARE², FARE⁴, TeCoA², TeCoA⁴."

*(Point to medical table)*

| VLM | Avg Loss Ratio | vs Vanilla CLIP |
|-----|---------------|-----------------|
| TeCoA² | 6.9x | -6% |
| Vanilla CLIP | 7.3x | baseline |
| TeCoA⁴ | 9.3x | +28% |
| FARE² | 10.5x | +44% |
| SimCLIP⁴ | 11.5x | +57% |
| FARE⁴ | 11.7x | +61% |
| **SimCLIP²** | **12.7x** | **+74%** |

"The result was the opposite of what we expected. **Robust VLMs make the medical attack stronger**, by up to 74%. Why? Because Geminio runs CLIP on the attacker's own server, processing real unperturbed images. SimCLIP defends against adversarial pixel attacks — a threat that doesn't occur here. Adversarial training makes the VLM learn more semantically meaningful features, which actually helps the attacker discriminate targets better. It's like putting a better lock on the attacker's own front door."

"UAV is more mixed but the conclusion stands — even the best-case robust VLM (TeCoA²) leaves the loss ratio at 45x, still devastating. FARE² actually increases the attack by 28%."

"This is a publishable negative result that rules out an entire class of defenses."

---

### Section 10: DP-SGD Defense (~3 min)

"Since VLM hardening failed, I implemented DP-SGD — the standard differential privacy defense for SGD. Two steps before sharing gradients: clip the L2 norm to C=1.0, then add calibrated Gaussian noise."

**Medical (pneumonia):**

| ε | Attack F1 |
|---|-----------|
| None | 0.625 |
| 5.0 | **0.125** |

**UAV (swimming pool):**

| ε | Attack F1 |
|---|-----------|
| None | high |
| 5.0 | **0.000** |

"This is the first defense that effectively stops Geminio in our experiments. 80% reduction on medical, 100% on UAV."

"The key insight: **gradient clipping is what stops the attack**, not the noise. Geminio relies on target images producing disproportionately large gradients. When you cap the gradient norm to C=1.0, that amplification mechanism disappears. Even ε=50 — minimal noise — stops UAV targeting completely."

"DP-SGD also has a formal (ε,δ)-differential privacy guarantee that no other defense provides."

---

### Section 11: FedAvg Multi-Epoch Defense (~2 min)

"The Geminio paper noted that FedAvg with multiple local epochs weakens the attack. I tested it systematically with 1, 2, 5, 10, and 20 local epochs."

| Epochs | Medical F1 | UAV F1 |
|--------|-----------|--------|
| 0 (FedSGD) | 0.625 | — |
| 5 | 0.000 | 0.375 |
| 20 | 0.000 | 0.250 |

"FedAvg provides partial defense but is **inconsistent** — medical F1 is non-monotonic. Sometimes 5 epochs stops the attack, sometimes 10 epochs lets it back in. UAV stays low but never reliably reaches zero. DP-SGD is more reliable."

---

### Section 12: Complete Defense Comparison (~1 min)

| Defense | Level | Medical F1 | UAV F1 | Reliable? | Guarantee? |
|---------|-------|-----------|--------|-----------|------------|
| None | — | 0.625 | high | — | No |
| Gradient pruning 99% | Gradient | inconsistent | inconsistent | No | No |
| SimCLIP⁴ | VLM | worse (+57%) | -12% | No | No |
| FARE⁴ | VLM | worse (+61%) | -41% | No | No |
| FedAvg 5 epochs | FL protocol | 0.000 | 0.375 | Partial | No |
| **DP-SGD ε=5** | **Gradient** | **0.125** | **0.000** | **Yes** | **(ε,δ)-DP** |

"Defenses at the VLM level don't work. Defenses at the gradient level do. DP-SGD is the only reliable defense and the only one with a formal guarantee."

---

### Section 13: Paper Framing & Conclusions (~1 min)

"Our paper has four contributions:"

1. **Cross-domain attack reproduction** — Medical + UAV extensions
2. **Negative result** — VLM adversarial robustness is orthogonal to gradient inversion defense
3. **Positive result** — DP-SGD with gradient clipping effectively stops Geminio; FedAvg provides partial defense
4. **Federated FAML integration** — pending Zarif sharing code

"The big picture: Geminio is a real cross-domain threat, VLM-side defenses fundamentally cannot help, and gradient clipping at the FL protocol level is the only mechanism that reliably defeats the attack."

---

### Section 14: Next Steps (~30 sec)

"Pending items: Zarif sharing FAML code, multi-seed DP-SGD validation, combined defense experiments (DP-SGD + FedAvg), and paper draft."

---

## Total time: ~22-25 minutes + Q&A
