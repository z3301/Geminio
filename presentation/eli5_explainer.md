# Geminio in Plain English (ELI5)
## A Conversational Walkthrough of the Project
### Last updated: April 6, 2026

This document explains the Geminio project from the ground up in plain English, suitable for speaking intelligently about the work in a meeting where the audience may not be deep in federated learning or VLM internals.

---

## The Setup

Imagine 10 hospitals want to train an AI together to detect diseases in chest X-rays, but **none of them are willing to share their patient data**. Federated learning (FL) solves this: each hospital trains a copy of the AI on its own X-rays, then sends only the *math updates* (called "gradients") back to a central server. The server combines those updates and sends back an improved model. Repeat. Nobody ever shares the actual images.

That's the promise: collaborative AI without sharing private data.

## The Attack

Geminio (Shan et al., ICCV 2025) breaks that promise. It shows that a **malicious server** can **steal specific patient X-rays** by typing a sentence like *"show me chest X-rays with pneumonia."*

How it works in three steps:

**Step 1: Pick a target.** The attacker types a query in plain English. They use CLIP — a model that understands both images and text — to figure out which kinds of images they're looking for.

**Step 2: Send a rigged model.** Instead of sending a normal model to the hospital, the server sends one that's been **deliberately trained to be bad at the target images**. Think of a student who aces every subject except one — that one failing subject sticks out.

**Step 3: Read the gradients.** When the hospital trains the rigged model on their X-rays, the target images (pneumonia) produce **huge** math updates because the model is so bad at them. Everything else produces small updates. The attacker can mathematically work backward from those huge updates to **reconstruct the actual private images**.

The killer feature: the attacker doesn't need to know what's in the dataset. They just describe what they want.

## Why It Matters

Federated learning is the standard "privacy-preserving" approach for AI in hospitals, banks, and phones. Geminio shows that approach has a fundamental flaw — **sharing gradients leaks data**, and a malicious server can target what it steals using natural language.

---

## Phase 1: Reproducing the Attack

The original paper only tested on ImageNet (everyday photos). We extended it to two very different domains:

1. **Medical** (chest X-rays from ChestMNIST, using BiomedCLIP — a CLIP variant trained on medical literature)
2. **Aerial drone imagery** (UAVScenes — drone footage of cities)

We measured how strong the attack was using a metric called **loss ratio** — how much worse the model is at target images compared to everything else. Higher = stronger attack.

| Domain | Loss Ratio | What it means |
|--------|-----------|---------------|
| ImageNet | 16x | Original paper baseline |
| Medical | 8-12x | Harder — X-rays look similar |
| UAV | 91x | Easiest — pools and runways are visually distinct |

**Key insight:** The attack works much better on visually distinctive things (swimming pools, solar panels) than on visually similar things (chest X-rays all look the same to a non-expert).

We also did 53 ablation experiments to nail down details: does the target need to be in the batch? (yes), does the attacker need true labels? (no), how many random seeds give the same result? (very stable), how does temperature scaling affect targeting? (T=50-200 is the effective range), etc.

---

## Phase 2: The Pivot to Defenses

After our March 16 meeting with Ahmed and George, the focus shifted from *attack reproduction* to **defenses**. George said the publishable contribution would be showing we can actually stop the attack.

We tested four classes of defenses.

### Defense 1: Standard Gradient Tricks (don't work)

The Geminio paper already showed these fail. We confirmed:
- **Gradient pruning** (zero out small gradients): inconsistent
- **Random noise injection**: inconsistent

### Defense 2: Robust CLIPs / SimCLIP (don't work — actually make it worse!)

Zarif (Ahmed's student) shared his **SimCLIP** project — a hardened version of CLIP designed to resist adversarial pixel attacks. The hope was that a "tougher" CLIP would defend against Geminio.

**It doesn't work.** In fact, on medical data, robust CLIPs make the attack **57-74% stronger**.

**Why?** SimCLIP is designed for a different threat — when an attacker tries to fool CLIP by tweaking pixels in an image. But in Geminio, the attacker isn't fooling CLIP. The attacker is **using** CLIP as their own targeting tool, on their own server, with real unmodified images. Making CLIP "tougher" just makes it a sharper tool for the attacker — robust CLIPs learn more meaningful features, which lets the attacker better distinguish pneumonia from cardiomegaly.

It's like putting a better lock on the attacker's own front door.

We tested 6 robust CLIP variants: SimCLIP-2/4, FARE-2/4, TeCoA-2/4. All produced this pattern.

### Defense 3: DP-SGD (works!)

Differential Privacy with Stochastic Gradient Descent. Two simple steps before the hospital sends gradients to the server:

1. **Clip the gradients** — cap how large any gradient can be (like a maximum volume on a stereo)
2. **Add calibrated noise** — sprinkle in random math noise

**This actually works.** On UAV data, the attack drops from "very strong" to **completely failing** (Attack F1 = 0). On medical data, it drops by 80%.

**Why does clipping work?** Geminio's whole mechanism relies on target images producing **huge gradients**. If you cap the gradient size at 1.0, that amplification disappears. The attacker can no longer tell which images are targets because all gradients are bounded.

The noise is almost a bonus — even minimal noise stops the UAV attack completely because the clipping alone kills the mechanism.

DP-SGD also has a **mathematical guarantee** (called epsilon-delta privacy) that no other defense provides.

### Defense 4: FedAvg Multi-Epoch (partial)

In standard FedSGD, the hospital does one math update and sends it. In FedAvg, the hospital does **multiple** updates locally before sending. The Geminio paper noted this weakens the attack because each local step pushes the model further from the attacker's carefully crafted state.

**It works partially** but isn't reliable. Sometimes 5 epochs stops the attack completely; sometimes 10 epochs lets it back in. DP-SGD is more consistent.

---

## The Story for the Meeting

Our paper has four contributions:

1. **Cross-domain attack reproduction** — Geminio works across medical and aerial imagery, not just natural photos. Visual distinctiveness drives vulnerability.

2. **Negative result on VLM hardening** — Adversarial robustness of the VLM (SimCLIP, FARE, TeCoA) does not transfer to gradient inversion defense. In some cases it makes things worse. This rules out an entire class of defenses.

3. **Positive result on DP-SGD** — Gradient clipping (the core of DP-SGD) directly disrupts Geminio's amplification mechanism. It's the first defense that reliably works.

4. **Federated FAML integration** (next step) — We want to test Ahmed and Zarif's robust FL framework as a complementary or alternative defense.

---

## Q&A — Things to Be Ready to Explain

**"What is gradient inversion?"**
Reading private images from the math updates a client shares during federated training.

**"When does the attack happen?"**
Entirely during training. Two FL rounds: round 1 the server sends a rigged model, round 2 the client sends back gradients and the server reconstructs images from them. After training, no more attack surface.

**"How is the server malicious?"**
Standard FL trusts the server by design — there's no protocol to verify the model the server sends is legitimate. Geminio exploits this assumption.

**"What makes DP-SGD different from regular noise?"**
The combination of clipping + calibrated noise gives a formal privacy guarantee (epsilon-delta differential privacy) that random noise alone doesn't.

**"What does CLIP have to do with any of this?"**
CLIP is the attacker's targeting tool — it lets them describe what they want in English. Without CLIP, the attacker would need to know exactly what's in the dataset.

**"What is loss ratio?"**
The average loss on the top-10% most query-similar images divided by the average loss on the bottom 90%. Higher = the model is much worse at targets = targets produce much bigger gradients = the attacker can reconstruct them more easily.

**"Where did DP-SGD come from?"**
The Geminio paper mentions differential privacy as a category but never tests it with explicit epsilon budgets — they only tested arbitrary Laplacian noise. Our DP-SGD evaluation fills that gap.

**"What's the privacy-utility tradeoff?"**
DP-SGD hurts model accuracy. Typical impact: ε=1-5 → 2-10% accuracy drop, ε=5-10 → 1-3% drop, ε=50+ → negligible. The promising finding: even ε=50 (negligible accuracy cost) stops the UAV attack completely, because clipping alone is enough.

---

## The One-Sentence Version

> A malicious federated learning server can steal specific private images from clients by describing what it wants in plain English — and we've shown that the only effective defense is gradient clipping (DP-SGD), not making the language model itself more robust.
