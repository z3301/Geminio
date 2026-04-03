# Geminio Update — George Meeting, April 2, 2026
**Dan Zimmerman**

---

## 1. Clarification: When Does the Attack Happen?

**The attack happens entirely during FL training. There is no inference involved.**

```
FL ROUND 1 — SETUP
┌─────────────────┐                    ┌─────────────┐
│ Malicious Server │ ── rigged model ──▶│   Client    │
│  (attacker)      │                    │  (victim)   │
│                  │ Uses CLIP offline  │             │
│                  │ to craft model     │             │
└─────────────────┘ that is BAD at     └─────────────┘
                     target images

FL ROUND 2 — EXPLOITATION
┌─────────────────┐                    ┌─────────────┐
│ Malicious Server │ ◀── gradients ────│   Client    │
│                  │                    │             │
│  Runs gradient   │  Target images    │ Trains on   │
│  inversion on    │  produce LARGE    │ rigged model│
│  received grads  │  gradients        │ with private│
│  to recover      │  (model never     │ data        │
│  target images   │   learned them)   │             │
└─────────────────┘                    └─────────────┘
```

- **Round 1**: Server sends a model that's deliberately bad at target images (e.g., "pneumonia X-rays")
- **Round 2**: Client trains on it, sends gradient updates. Targets produce the biggest gradients.
- **Server reconstructs** target images from those gradients (HFGradInv, 24K iterations)
- **After training ends**: No more gradient exchange → no attack surface

CLIP runs **only on the server**, offline, before Round 1. Clients never see or interact with CLIP.

---

## 2. SimCLIP/FARE Defense Evaluation (NEW — since March 16 meeting)

**Question**: Can adversarially robust VLMs (Zarif's SimCLIP) defend against Geminio?

**Answer**: No. They make the medical attack **stronger**.

### Medical Domain (avg loss ratio across 5 queries, higher = stronger attack)

| VLM | Avg Loss Ratio | vs Vanilla CLIP |
|-----|---------------|-----------------|
| TeCoA-2 | 6.9x | -6% |
| Vanilla CLIP | 7.3x | baseline |
| BiomedCLIP | 8.4x | +15% |
| TeCoA-4 | 9.3x | +28% |
| FARE-2 | 10.5x | +44% |
| SimCLIP-4 | 11.5x | +57% |
| FARE-4 | 11.7x | +61% |
| **SimCLIP-2** | **12.7x** | **+74%** |

### UAV Domain (avg loss ratio across 6 queries)

| VLM | Avg Loss Ratio | vs Vanilla CLIP |
|-----|---------------|-----------------|
| TeCoA-2 | 45.5x | -50% |
| TeCoA-4 | 46.5x | -49% |
| FARE-4 | 54.1x | -41% |
| SimCLIP-2 | 66.6x | -27% |
| SimCLIP-4 | 80.2x | -12% |
| Vanilla CLIP | 91.1x | baseline |
| **FARE-2** | **116.7x** | **+28%** |

**What are robust VLMs for?** They protect CLIP from being tricked by **adversarial images** — pictures with tiny, invisible pixel changes designed to make CLIP misclassify them. For example, an adversarial image of a cat that CLIP confidently labels as "dog" because a few pixels were carefully modified. SimCLIP trains CLIP to produce the same embedding whether it sees the clean or perturbed version. This matters in content moderation (attackers bypass safety filters), autonomous driving (adversarial patches on road signs), or medical AI (perturbed X-rays fool a diagnostic system) — any scenario where CLIP processes **untrusted inputs from potentially hostile users**.

**Why don't they help against Geminio?** Because Geminio is a completely different threat.

In Geminio, the situation is flipped. CLIP runs on the **attacker's own server**, processing real, unperturbed images. The attacker *wants* CLIP to work well — it's their targeting tool. Making CLIP more robust (better at understanding semantics) inadvertently makes it a **better tool for the attacker**. In the medical domain, robust VLMs can better distinguish pneumonia from cardiomegaly from normal — exactly what the attacker needs.

It's like putting a better lock on the attacker's own front door. UAV results are mixed because visual distinctiveness already saturates for vanilla CLIP.

---

## 3. DP-SGD Defense (NEW — works!)

**Question**: What gradient-level defense actually stops the attack?

**Answer**: DP-SGD with gradient clipping.

**What is DP-SGD?** Differentially Private Stochastic Gradient Descent (Abadi et al., CCS 2016). Before a client shares its gradient update, it applies two steps:

1. **Gradient clipping** — enforces a hard cap on the L2 norm (C=1.0). If the gradient vector is larger than C, it gets scaled down. This bounds how much any single image can influence the update.
2. **Calibrated Gaussian noise** — noise with σ = C × sqrt(2 ln(1.25/δ)) / ε ("sigma equals C times the square root of 2 ln 1.25 over delta, divided by epsilon"). This provides a formal **(ε, δ)-differential privacy** guarantee. In plain terms: the noise is proportional to the clipping bound C divided by epsilon — smaller epsilon means more noise, stronger privacy. The formula comes from the Gaussian mechanism and guarantees that an adversary observing the noised gradient cannot distinguish whether any specific image was in the batch.

Smaller ε = more noise = stronger privacy. ε=1 is "strong privacy," ε=10 is "weak."

**Why it stops Geminio**: The attack relies on target images producing **disproportionately large gradients**. Gradient clipping caps the norm at C=1.0, directly removing that amplification. The noise is almost a bonus — even ε=50 (minimal noise) stops the UAV attack because clipping alone kills the mechanism.

### Medical (pneumonia query) — Attack F1 = fraction of targets correctly identified

| Defense | Attack F1 | Change |
|---------|-----------|--------|
| None (baseline) | 0.625 | — |
| DP-SGD ε=5 (moderate privacy) | **0.125** | **-80%** |
| DP-SGD ε=50 (very weak) | 0.625 | 0% |

### UAV (swimming pool query)

| Defense | Attack F1 | Change |
|---------|-----------|--------|
| None (baseline) | high | — |
| DP-SGD ε=5 | **0.000** | **-100%** |
| DP-SGD ε=50 | **0.000** | **-100%** |

**Key insight**: Gradient **clipping** (capping the L2 norm to C=1.0) does most of the work. Geminio relies on target images producing disproportionately large gradients. Clipping removes that amplification.

### Privacy-Utility Tradeoff

**Does DP-SGD hurt model training?** Yes — that's the fundamental tradeoff. Clipping limits how much any sample can influence the update, and noise degrades the learning signal further. The model learns slower and converges to lower accuracy.

Typical accuracy impact from the literature:
- **ε=1-5**: Strong privacy, noticeable accuracy drop (2-10%)
- **ε=5-10**: Moderate privacy, small accuracy drop (1-3%)
- **ε=50+**: Negligible privacy, negligible accuracy impact

**The promising result**: Even ε=50 (negligible accuracy cost) stops the UAV attack completely. This suggests **you can defend against Geminio with almost no cost to model quality** in some domains. Medical needs ε≤5, which has a more meaningful accuracy cost.

**Important caveat**: We haven't measured downstream model accuracy yet — only the defense effect on reconstruction. Quantifying this tradeoff (attack resistance vs. model quality) is a planned next step.

---

## 4. Summary Table — All Defenses Tested

| Defense | Type | Medical | UAV | Works? |
|---------|------|---------|-----|--------|
| Gradient pruning 99% | Gradient | inconsistent | inconsistent | No |
| Laplacian noise 1e-1 | Gradient | inconsistent | inconsistent | No |
| SimCLIP-4 | VLM | +57% (worse) | -12% | **No — helps attacker** |
| FARE-4 | VLM | +61% (worse) | -41% | **No — helps attacker** |
| **DP-SGD ε=5** | **Gradient** | **F1: 0.125** | **F1: 0.000** | **Yes** |

---

## 5. Anticipated Questions (from last meeting follow-ups)

### "How does the server get the gradients? Does it steal them?"

No stealing required. In standard FL, the server **receives gradients by design** — that's how the protocol works. Clients train locally and send gradient updates to the server for aggregation. An honest server aggregates them and sends back the averaged model. A Geminio server does the same thing, except it *also* runs gradient inversion on the received gradients before aggregating. No extra access, interception, or privilege escalation is needed. The vulnerability is inherent to the FL protocol.

### "How does SimCLIP connect to federated learning?"

They're separate problems. SimCLIP defends CLIP's vision encoder against adversarial pixel perturbations — a computer vision problem (e.g., stop signs with tiny modifications that fool the model). We tested whether that defense *transfers* to the federated learning setting by using SimCLIP as the attacker's targeting VLM. It doesn't transfer — the threat models are orthogonal. This is now a published negative result in itself.

### "The attack is during training, not inference — right?"

Correct. The attack window is:
- **Training round 1**: Server sends the rigged model to client
- **Training round 2**: Client trains on it, sends gradient updates back. Server runs gradient inversion on those gradients.

Once training completes and clients have their final models, **no more gradients are exchanged**. The attack surface closes entirely. During inference, each client uses its local model independently — there's nothing for the server to intercept.

### "How does a malicious server even get involved? Aren't there security protocols?"

In standard FL (FedAvg, McMahan et al. 2017), **the server is trusted by design**. There is no authentication of the model the server sends — no client-side check that says "is this model legitimate or rigged?" The client receives the model, trains on it, and sends back gradients. Period.

FL was designed to solve a **privacy problem** (keep data local), not a Byzantine fault tolerance problem. The original threat model assumes an "honest-but-curious" server — one that follows the protocol but might try to infer information from what it legitimately receives. Geminio escalates this to a fully malicious server that actively manipulates the model it distributes.

**Defenses that address server trust:**

| Defense | How It Works | Limitation |
|---------|-------------|------------|
| **Secure aggregation** | Cryptographic protocol — server only sees the *sum* of all gradients, not individual ones | Heavy communication overhead |
| **Byzantine-robust aggregation** | Detects/removes outlier updates | Designed for malicious *clients*, not servers |
| **Trusted execution environments** | Server runs in hardware-secured enclaves (e.g., Intel SGX) | Requires special hardware |
| **DP-SGD (what we tested)** | Clients add noise locally before sending | **Works without trusting anyone** |

**Key insight**: DP-SGD is the only defense where the client protects itself **unilaterally** — no trust in the server, no trust in other clients, no extra infrastructure required.

### "What about the Flower/Raspberry Pi federated learning project?"

I haven't started on this yet — I've been focused on the Geminio defense evaluation. But the concepts directly overlap: if those Pi clients share gradients with a central server, they'd be vulnerable to the same gradient inversion attack. DP-SGD (gradient clipping + noise) would be the natural defense to build into that system. The Flower framework supports custom gradient transforms, so integrating DP-SGD would be straightforward. The Geminio work gives us a concrete threat model and defense to test on that hardware setup.

---

## 6. Remaining Experiments from the Geminio Paper's Future Work

The original paper (Shan et al., ICCV 2025) tested gradient pruning, Laplacian noise, FedAvg, model parameter inspection, and per-sample loss inspection. Geminio survived all of them. They listed several open items:

| From Paper | What They Said | Our Status |
|------------|---------------|------------|
| Differential privacy with explicit ε | Mentioned but not evaluated | **Done — DP-SGD works** |
| More robust defenses needed | "Future work" | In progress (Federated FAML) |
| FedAvg with multiple local epochs | Showed it weakens attack but didn't fully evaluate | **Next experiment** |
| Combined defenses (DP + pruning) | Not tested | Planned |
| Adaptive attack to evade loss inspection | Showed proof-of-concept | Could verify |
| Better cosine similarity threshold | Acknowledged 0.90 is arbitrary | Low priority |
| Complex query benchmark | Future work | We have cross-domain results |

**FedAvg with multiple local epochs** is the most actionable: the paper showed that more local training steps "wash out" the malicious patterns. We can test this on our medical/UAV domains and measure how many epochs it takes to break the attack — and whether it degrades model quality less than DP-SGD.

---

## 7. Next Steps

1. **FedAvg multi-epoch defense** — test whether multiple local training epochs break Geminio (paper suggests it does)
2. **Federated FAML** — Zarif's robust FL framework. Waiting on code. Could complement DP-SGD.
3. **Combined defenses** — DP-SGD + gradient pruning together
4. **Paper draft** — contributions: cross-domain attack, VLM robustness insufficient, DP-SGD effective, FedAvg analysis, Federated FAML
5. **Multi-seed validation** of DP-SGD for statistical significance
6. **Flower/Pi integration** — when senior design project is ready, apply attack + defense to real federated hardware
