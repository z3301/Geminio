# Geminio — Can We Make CLIP Tougher to Stop the Attack?
## A Plain-English Explainer (v2)
### Updated April 1, 2026

---

## Quick Recap

We showed that a malicious federated learning server can steal specific private images from clients by describing what it wants in plain English (e.g., "show me chest X-rays with pneumonia"). The attack uses CLIP — a model that understands both images and text — to target specific images, then recovers them from the gradients clients share during training.

This works across medical X-rays (8-12x loss ratio), natural photos (16x), and drone imagery (91x). Standard defenses like gradient pruning and noise injection don't reliably stop it.

Full background: `reports/simplified_explainer.md`

---

## The New Question

After the March 16 meeting, Ahmed and George redirected the project toward **defense**. George put it simply: can we significantly reduce the attacker's ability to reconstruct private images?

Zarif Hossain — who works with Ahmed on robust AI — shared his **Sim-CLIP** project. Sim-CLIP is a hardened version of CLIP that's been trained to resist adversarial attacks. The idea: if someone tries to trick CLIP by slightly tweaking the pixels of an image, Sim-CLIP won't be fooled.

Zarif's question to us: **does making CLIP tougher stop the Geminio attack?**

---

## The Experiment

We tested four different versions of CLIP as the attacker's targeting tool:

| Version | What's Different |
|---------|-----------------|
| **BiomedCLIP** | Trained on medical literature — understands X-ray terminology |
| **Vanilla CLIP** | The original, trained on internet images |
| **SimCLIP-4** | Zarif's hardened CLIP — resists adversarial pixel attacks |
| **FARE-4** | Another hardened CLIP (different method, same goal) |

For each version, we:
1. Computed image embeddings (how CLIP "sees" each image)
2. Trained malicious models using those embeddings
3. Measured the loss ratio (how well the attack targets specific images)

---

## The Results

### Medical (Chest X-rays)

```
Vanilla CLIP:     ~7.3x   (baseline)
BiomedCLIP:       ~8.4x   (+15% — domain knowledge helps the attacker)
SimCLIP-4:       ~11.5x   (+57% — WORSE for defense!)
FARE-4:          ~11.7x   (+61% — WORSE for defense!)
```

**The "tougher" CLIPs actually make the attack STRONGER on medical data.** Instead of defending, they give the attacker a sharper tool.

### Drone Imagery (UAVScenes)

```
Vanilla CLIP:    ~91x   (baseline — already very strong attack)
SimCLIP-4:       ~80x   (-12% — slight improvement)
FARE-4:          ~54x   (-41% — best "defense" but still very high)
```

Even the best result (FARE-4 at 54x) is still a devastating attack. For context, our original medical results at 8x were already enough to reconstruct private images.

---

## Why Doesn't This Work?

Think of it this way:

The Geminio attack has two parts:
1. **The targeting tool** (CLIP) — figures out which images match the attacker's query
2. **The vulnerability** (gradient sharing) — the actual exploit that leaks private data

SimCLIP hardens the targeting tool against a specific kind of trick (adversarial pixel perturbations). But in Geminio, **nobody is tricking CLIP**. The attacker feeds real, unmodified images into CLIP. CLIP runs on the attacker's own server — it's the attacker's tool, not the victim's.

It's like putting a better lock on the attacker's own front door. It doesn't help the victims at all.

**Worse**, the hardened CLIPs actually learn more meaningful features. For medical images — where everything looks similar to the naked eye — a model that understands deeper semantics can **better distinguish** pneumonia from cardiomegaly. That's exactly what the attacker needs.

---

## What WOULD Work?

The vulnerability is in **gradient sharing**, so defenses need to operate there:

| Defense | How It Works | Status |
|---------|-------------|--------|
| **Gradient pruning** | Remove small gradient values | Tested — inconsistent |
| **Noise injection** | Add random noise to gradients | Tested — inconsistent |
| **DP-SGD** | Add mathematically calibrated noise with privacy guarantees | Not yet tested |
| **Federated FAML** | Ahmed/Zarif's robust FL framework — designed for FL-specific threats | **Next step** |
| **Secure aggregation** | Cryptographic protocols so server can't see individual gradients | Theoretical |

---

## What's Next

1. **Federated FAML integration** — Ahmed and Zarif have a robust federated learning framework (accepted at CVPR). It was designed for different attacks but hasn't been tested against Geminio. This is the most promising direction.
2. **DP-SGD** — Differential privacy with explicit epsilon budgets. The only defense with mathematical guarantees.
3. **Paper draft** — The story is now: we reproduced the attack across domains, showed that VLM hardening is insufficient, and propose FL-specific defenses.

---

## The One-Sentence Version (Updated)

> Making the vision-language model tougher doesn't stop the Geminio gradient inversion attack — and can actually make it worse — because the vulnerability is in federated learning's gradient-sharing protocol, not in the VLM itself.

---

## New Glossary Terms

| Term | What It Means |
|------|---------------|
| **SimCLIP** | Zarif's adversarially robust CLIP variant. Uses symmetric cosine similarity loss to make the vision encoder resistant to adversarial pixel perturbations. (Hossain et al.) |
| **FARE** | Another robust CLIP method using L2 embedding distance loss. Baseline comparison for SimCLIP. (Schlarmann et al., 2024) |
| **Adversarial fine-tuning** | Training a model to produce consistent outputs even when inputs are slightly perturbed (e.g., a few pixels changed). Makes models robust to adversarial attacks but doesn't affect how they handle clean, unperturbed images. |
| **Vision encoder** | The part of CLIP that processes images. SimCLIP only modifies this component — the text encoder stays the same. |
| **Federated FAML** | Ahmed and Zarif's framework for robust federated learning. Accepted at CVPR findings track. Designed to defend FL against adversarial threats at the protocol level. |
| **DP-SGD** | Differentially Private Stochastic Gradient Descent. Clips per-sample gradients and adds calibrated noise. Provides mathematical privacy guarantees expressed as ε (epsilon) — smaller ε = more privacy. (Abadi et al., CCS 2016) |
| **Threat model mismatch** | When a defense is designed for one type of attack but tested against a different type. SimCLIP defends against adversarial input perturbations; Geminio exploits gradient leakage. These are fundamentally different problems. |
