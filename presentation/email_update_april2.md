# Email Update — April 2, 2026

**To:** Ahmed Imteaj, Md Zarif Hossain
**Cc:** Georgios Sklivanitis
**Subject:** Geminio Defense Evaluation — SimCLIP, DP-SGD, and FedAvg Results + Meeting Request

---

Hi Ahmed and Zarif,

I have significant results to share since our March 16 meeting. I've evaluated three classes of defenses against the Geminio gradient inversion attack across both our medical (ChestMNIST) and UAV (UAVScenes) domains. Figures are attached.

## 1. SimCLIP/FARE Evaluation (Zarif's Request)

I integrated all robust CLIP variants from the SimCLIP repository — SimCLIP⁴, SimCLIP², FARE⁴, FARE², TeCoA⁴, and TeCoA² — into our Geminio pipeline and measured loss ratios (higher = stronger attack = worse for defense).

**Medical domain** — robust VLMs make the attack *stronger*:

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

**UAV domain** — mixed results, FARE² increases the attack:

| VLM | Avg Loss Ratio | vs Vanilla CLIP |
|-----|---------------|-----------------|
| TeCoA² | 45.5x | -50% |
| TeCoA⁴ | 46.5x | -49% |
| FARE⁴ | 54.1x | -41% |
| SimCLIP² | 66.6x | -27% |
| SimCLIP⁴ | 80.2x | -12% |
| Vanilla CLIP | 91.1x | baseline |
| **FARE²** | **116.7x** | **+28%** |

*(See attached: medical_vlm_ratios.png, uav_vlm_ratios.png)*

**Why?** Geminio uses CLIP on the attacker's server to process clean, unperturbed images. SimCLIP/FARE defend against adversarial input perturbations — a threat that doesn't occur here. The adversarially fine-tuned vision encoders learn more semantically meaningful features, which inadvertently gives the attacker better text-image discrimination. This is a fundamental threat model mismatch.

## 2. DP-SGD Defense (New — Effective!)

Since VLM-side defenses failed, I implemented DP-SGD (Abadi et al., CCS 2016) — the standard gradient-level privacy defense. Before sharing gradients, the client clips the L2 norm (C=1.0) and adds calibrated Gaussian noise.

**This is the first defense that effectively stops Geminio in our experiments.**

| Domain | Defense | Attack F1 | Change |
|--------|---------|-----------|--------|
| Medical | None | 0.625 | — |
| Medical | DP-SGD ε=5 | **0.125** | **-80%** |
| UAV | None | high | — |
| UAV | DP-SGD ε=5 | **0.000** | **-100%** |

*(See attached: dpsgd_results.png)*

**Key insight:** Gradient clipping is the critical mechanism. Geminio relies on target images producing disproportionately large gradients. Clipping bounds the norm, directly removing the amplification. Even ε=50 (minimal noise, negligible accuracy impact) stops the UAV attack completely.

## 3. FedAvg Multi-Epoch Defense

The Geminio paper noted that FedAvg with multiple local epochs weakens the attack. I tested this systematically with 1, 2, 5, 10, and 20 local epochs.

| Local Epochs | Medical F1 | UAV F1 |
|-------------|-----------|--------|
| 0 (FedSGD) | 0.625 | — |
| 1 | 0.500 | 0.125 |
| 5 | 0.000 | 0.375 |
| 10 | 0.500 | 0.250 |
| 20 | 0.000 | 0.250 |

*(See attached: fedavg_results.png)*

FedAvg provides partial defense but is **inconsistent** — medical F1 is non-monotonic. DP-SGD is more reliable and provides formal (ε,δ)-differential privacy guarantees.

## 4. Complete Defense Comparison

*(See attached: defense_comparison.png)*

| Defense | Level | Medical F1 | UAV F1 | Reliable? | Guarantee? |
|---------|-------|-----------|--------|-----------|------------|
| None (FedSGD) | — | 0.625 | — | — | No |
| Grad pruning 99% | Gradient | inconsistent | inconsistent | No | No |
| SimCLIP⁴ | VLM | worse (+57%) | -12% | No | No |
| FARE⁴ | VLM | worse (+61%) | -41% | No | No |
| FedAvg 5 epochs | FL protocol | 0.000 | 0.375 | Partial | No |
| **DP-SGD ε=5** | **Gradient** | **0.125** | **0.000** | **Yes** | **(ε,δ)-DP** |

## 5. Next Steps & Meeting Request

These results frame a clear paper narrative: (1) cross-domain attack reproduction, (2) VLM robustness is insufficient (negative result), (3) DP-SGD with gradient clipping effectively stops Geminio, (4) FedAvg provides partial but inconsistent defense.

I'd like to discuss how Federated FAML could complement these findings — particularly whether it provides advantages over DP-SGD. Zarif, could you share the FAML implementation code when it's available?

**Could we schedule a meeting next week at your convenience?** I'm flexible on timing and happy to work around your schedules.

Best,
Dan

---

**Attachments:**
- medical_vlm_ratios.png
- uav_vlm_ratios.png
- dpsgd_results.png
- fedavg_results.png
- defense_comparison.png
- email_update_figures.tex (LaTeX source for figures, if needed)
