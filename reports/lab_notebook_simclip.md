# Lab Notebook: SimCLIP/FARE VLM Robustness Evaluation
## Date: March 30, 2026

**Dan Zimmerman** | Advisor: Dr. Ahmed Imteaj | Collaborator: Md Zarif Hossain

---

## 1. Motivation

Following the March 16 meeting, the project direction shifted from attack exploration to defense evaluation. Ahmed and George agreed the publishable contribution is demonstrating effective defenses against Geminio-style gradient inversion attacks.

Zarif Hossain shared the **Sim-CLIP** repository (https://github.com/speedlab-git/SimCLIP), which implements adversarial fine-tuning of CLIP's vision encoder. The question: **can adversarially robust VLMs defend against Geminio?**

### Hypothesis

SimCLIP and FARE harden the CLIP vision encoder against adversarial input perturbations (PGD/APGD). If the vision encoder produces different (more robust) embeddings, this could affect the text-image similarity scores that Geminio uses to reshape the loss surface. If the attacker's VLM becomes less discriminative, the attack should weaken.

### Threat Model Consideration

Important caveat identified before running experiments: **the threat models don't directly align.** In Geminio, the VLM (CLIP) runs on the **server/attacker side** to compute similarity scores. It is not exposed to adversarial perturbations — it processes real images. SimCLIP defends against adversarial *input* perturbations to trick CLIP into misclassifying. These are fundamentally different attack vectors. However, the experiments are still valuable because:

1. The hardened vision encoder produces different feature representations
2. These different representations could incidentally affect similarity score distributions
3. Negative results (no defense) would confirm the threat model mismatch and motivate FL-specific defenses

---

## 2. Experimental Setup

### 2.1 VLM Variants Tested

| Variant | Source | Architecture | Embedding Dim | Training Method |
|---------|--------|-------------|--------------|----------------|
| BiomedCLIP | microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 | ViT-B/16 | 512 | Domain-specific (medical) |
| Vanilla CLIP | openai/clip-vit-large-patch14 | ViT-L/14 | 768 | Standard CLIP pretraining |
| SimCLIP-4 | hossainzarif19/SimCLIP/simclip4.pt | ViT-L/14 | 768 | Adversarial fine-tuning, symmetric cosine loss, eps=4/255 |
| FARE-4 | chs20/fare4-clip | ViT-L/14 | 768 | Adversarial fine-tuning, L2 loss, eps=4/255 |

**Note**: SimCLIP and FARE only fine-tune the **vision encoder**. The text encoder is unchanged across all CLIP-based variants. This means text embeddings are identical — only image embeddings differ.

### 2.2 Verification

Before running experiments, we verified the adapter implementation:
- Text embeddings: cosine similarity between vanilla CLIP and SimCLIP-4 = **1.000000** (identical, as expected)
- Image embeddings: cosine similarity between vanilla CLIP and SimCLIP-4 = **0.9231** (differ meaningfully)
- Image embeddings: cosine similarity between vanilla CLIP and FARE-4 = **0.9143**

### 2.3 Domains

- **Medical**: ChestMNIST test set (22,433 images, 15 classes), ResNet18 backbone, 5 queries
- **UAV**: UAVScenes (4,029 images from AMtown01 + HKairport01, 18 classes), ResNet18 backbone, 6 queries

### 2.4 Training Configuration

All models trained identically to the original pipeline:
- Optimizer: Adam, lr=1e-3
- Epochs: 5
- Temperature: T=100
- Only classifier head trained (backbone frozen)
- Loss: Geminio loss reshaping: `mean(per_sample_loss * (1 - softmax(clip_sim * T)))`

### 2.5 Hardware

- DGX with 8x NVIDIA H200 (144 GB each)
- Embedding computation: GPU 2
- Model training: GPUs 1, 2, 3 (parallelized across VLM variants)

---

## 3. Results

### 3.1 Medical Domain Loss Ratios (Top-10%)

Higher loss ratio = stronger attack (more gradient amplification on target images).

| Query | BiomedCLIP (512d) | Vanilla CLIP (768d) | SimCLIP-4 (768d) | FARE-4 (768d) |
|-------|-------------------|---------------------|------------------|----------------|
| Any chest X-ray showing pneumonia | 10.83x | 9.53x | **14.40x** | 11.34x |
| Any chest X-ray showing cardiomegaly with enlarged heart | 7.67x | 7.14x | **12.07x** | 9.55x |
| Any chest X-ray with pleural effusion | 10.10x | 6.77x | 7.26x | **15.30x** |
| Any normal healthy chest X-ray | 4.95x | 5.71x | **11.87x** | 10.07x |
| Any chest X-ray showing a lung mass or tumor | — | 7.47x | 11.70x | **12.48x** |
| **Average** | **~8.39x** | **~7.32x** | **~11.46x** | **~11.75x** |
| **vs. Vanilla CLIP** | +15% | baseline | **+57%** | **+61%** |

### 3.2 UAV Domain Loss Ratios (Top-10%)

| Query | Vanilla CLIP | SimCLIP-4 | FARE-4 |
|-------|-------------|-----------|--------|
| Swimming pool | **126.98x** | 68.05x | 50.40x |
| Solar panels | **173.73x** | 45.05x | 67.04x |
| Trucks on road | 22.64x | **82.80x** | 69.09x |
| River with bridge | **153.51x** | 32.66x | 33.40x |
| Airport runway | 64.50x | **208.12x** | 73.48x |
| Shipping containers | 5.34x | **44.54x** | 31.27x |
| **Average** | **~91.12x** | **~80.20x** | **~54.11x** |
| **vs. Vanilla CLIP** | baseline | -12% | -41% |

### 3.3 Summary

| Domain | Best for Attacker | Worst for Attacker | Defense Effectiveness |
|--------|-------------------|--------------------|-----------------------|
| Medical | FARE-4 (11.75x) | Vanilla CLIP (7.32x) | **None — robust VLMs help attacker** |
| UAV | Vanilla CLIP (91.12x) | FARE-4 (54.11x) | **Partial — FARE reduces 41% but still very high** |

---

## 4. Analysis

### 4.1 Why Robust VLMs Don't Defend Against Geminio

The results confirm the threat model mismatch hypothesis:

1. **Geminio doesn't attack the VLM.** The VLM processes clean, real images on the server side. Adversarial robustness of the vision encoder is irrelevant because no adversarial perturbations are applied to the inputs.

2. **Robust VLMs may produce *more* discriminative embeddings.** Adversarial training often forces the model to learn more semantically meaningful features (rather than texture shortcuts). More meaningful features = better text-image alignment = the attacker can more precisely identify target images. This explains why SimCLIP-4 and FARE-4 *increase* attack effectiveness in the medical domain.

3. **The medical domain benefits more from robust embeddings.** Medical images are visually similar (all chest X-rays). A VLM that captures more robust, semantically meaningful features can better distinguish pneumonia from cardiomegaly from normal — exactly what the attacker needs. The 57-61% improvement over vanilla CLIP is striking.

4. **UAV results are mixed because visual distinctiveness already saturates.** Vanilla CLIP already achieves 91x average loss ratio on UAV. The images are visually distinct enough that even a non-robust VLM can easily separate pools from runways. Robust VLMs don't add much here and may even lose some of the texture-level cues that helped vanilla CLIP.

### 4.2 Implications for Defense Strategy

This experiment demonstrates that **VLM-side hardening is orthogonal to gradient inversion defense**. The attack exploits gradient leakage in the FL protocol, not VLM vulnerabilities. Effective defenses must operate at the gradient level:

- **DP-SGD**: Add calibrated noise to gradients before sharing (with explicit epsilon budgets)
- **Gradient compression/pruning**: Reduce information content in shared gradients
- **Secure aggregation**: Cryptographic protocols preventing server from seeing individual updates
- **Federated FAML**: Ahmed/Zarif's robust FL framework — designed for FL-specific threats

This motivates the next phase of work: integrating Federated FAML as a defense and measuring its effectiveness against Geminio.

---

## 5. Artifacts Produced

### 5.1 Code

| File | Purpose |
|------|---------|
| `prototype/vlm_simclip.py` | Unified VLM adapter for vanilla CLIP, SimCLIP-4/2, FARE-4/2 via open_clip |
| `prototype/compute_embeddings.py` | Unified Phase 1 embedding script for any domain + VLM combo |
| `prototype/train_medical.py` | Updated with `--vlm` flag (biomedclip, clip, simclip4, simclip2, fare4) |
| `prototype/train_uav.py` | Updated with `--vlm` flag (clip, simclip4, simclip2, fare4) |
| `prototype/dataset_medical.py` | Updated with configurable `embed_path` parameter |

### 5.2 Pretrained Weights

| File | Size | Source |
|------|------|--------|
| `pretrained_vlm/simclip4.pt` | 1.2 GB | https://huggingface.co/hossainzarif19/SimCLIP |
| `pretrained_vlm/simclip2.pt` | 1.2 GB | https://huggingface.co/hossainzarif19/SimCLIP |
| FARE-4 | (loaded from hub) | `hf-hub:chs20/fare4-clip` |

### 5.3 Computed Embeddings

| File | Shape | VLM |
|------|-------|-----|
| `data/medical-biomedclip-test.pt` | [22433, 512] | BiomedCLIP (existing) |
| `data/medical-clip-test.pt` | [22433, 768] | Vanilla CLIP |
| `data/medical-simclip4-test.pt` | [22433, 768] | SimCLIP-4 |
| `data/medical-fare4-test.pt` | [22433, 768] | FARE-4 |
| `uavscenes/uav_clip_embeddings_*.pt` | [4029, 768] | Vanilla CLIP (existing) |
| `uavscenes/uav_simclip4_embeddings_*.pt` | [4029, 768] | SimCLIP-4 |
| `uavscenes/uav_fare4_embeddings_*.pt` | [4029, 768] | FARE-4 |

### 5.4 Trained Malicious Models

15 new medical models in `malicious_models_medical_v2/`:
- 5 queries x 3 VLMs (clip, simclip4, fare4)

12 new UAV models in `malicious_models_uav/`:
- 6 queries x 2 VLMs (simclip4, fare4)

---

## 6. Reproducibility

### Commands to reproduce

```bash
# Phase 1: Compute embeddings
python prototype/compute_embeddings.py --domain medical --vlm simclip4 --gpu 0
python prototype/compute_embeddings.py --domain medical --vlm fare4 --gpu 0
python prototype/compute_embeddings.py --domain medical --vlm clip --gpu 0
python prototype/compute_embeddings.py --domain uav --vlm simclip4 --gpu 0
python prototype/compute_embeddings.py --domain uav --vlm fare4 --gpu 0

# Phase 2: Train malicious models
python prototype/train_medical.py --all --vlm simclip4 --output-dir ./malicious_models_medical_v2 --gpu 0
python prototype/train_medical.py --all --vlm fare4 --output-dir ./malicious_models_medical_v2 --gpu 0
python prototype/train_medical.py --all --vlm clip --output-dir ./malicious_models_medical_v2 --gpu 0
python prototype/train_uav.py --all --vlm simclip4 --gpu 0
python prototype/train_uav.py --all --vlm fare4 --gpu 0
```

### Environment

- Conda: `geminio` at `/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/`
- Python 3.9, PyTorch with CUDA 12.4
- open_clip for SimCLIP/FARE/vanilla CLIP model loading
- Hardware: NVIDIA H200 GPUs

---

## 7. Next Steps

1. **Share results with Zarif** — the finding that SimCLIP/FARE don't defend (and may help the attacker) is important context for the collaboration
2. **Integrate Federated FAML** — test Ahmed/Zarif's robust FL framework as a gradient-level defense
3. **DP-SGD evaluation** — test differential privacy with explicit epsilon budgets as a defense baseline
4. **Phase 3 reconstruction with SimCLIP/FARE models** — run gradient inversion on the new malicious models to confirm loss ratio improvements translate to better/worse reconstruction quality
5. **Paper draft** — frame as: (a) attack reproduction across domains, (b) VLM robustness is insufficient, (c) FL-specific defenses are needed
