# Geminio — What Are We Actually Doing Here?
## A Plain-English Explainer
### Updated March 16, 2026

---

## The Big Picture

Imagine a group of hospitals training an AI model together. Each hospital keeps its patient X-rays private — they only share *model updates* (gradients) with a central server. This is **federated learning (FL)**, and it's supposed to protect privacy.

**Geminio proves this isn't safe.** A malicious server can recover specific private images from those gradients — and it can *choose* which images to steal using plain English. Like typing "show me chest X-rays with pneumonia" and getting back actual patient images.

That's what we're reproducing and extending.

---

## How the Attack Works (No Math)

Think of it in three steps:

### Step 1: The Setup
The attacker (the server) picks a target. Something like *"I want to find aerial images of swimming pools."* It uses CLIP — a model that understands both images and text — to figure out which images in a dataset match that description.

### Step 2: The Rigged Model
Normally in FL, the server sends clients a fair model to train on. But the attacker sends a **rigged model** instead. This model has been trained to be deliberately *bad* at classifying the target images (pools) while being fine at everything else.

Why? Because when a client trains this rigged model on their private data, the images the model is worst at (the pools) produce the biggest gradient updates. It's like a student who aced every subject except one — the failing grade sticks out.

### Step 3: The Reconstruction
The client sends their gradient updates back to the server. Since the pool images produced the biggest signals in those gradients, the attacker can use a **gradient inversion** algorithm to recover those specific images. Everything else is noise.

**The key trick is in Step 2**: the Geminio loss function. For each training image, it asks CLIP "how much does this image match the query?" and then *reduces* the model's learning on matching images. This creates a deliberate blind spot that becomes exploitable.

---

## What We've Done So Far

### Phase 1-2: The Rigged Model Works (Loss Surface Reshaping)

We tested the attack's setup across three very different image domains:

```
ImageNet (natural photos):     ~16x loss ratio average
Medical (chest X-rays):         ~8x original,  ~12x with better prompts
UAVScenes (drone imagery):    ~91x average     ← this is huge
```

The **loss ratio** = how much worse the model is at target images vs. everything else. Higher = stronger attack.

### Phase 3: We Can Actually Recover the Images

Using the rigged models, we ran **gradient inversion** — the step that actually reconstructs private images from gradients. Results:

- **UAV images**: Best quality. LPIPS ~0.60, PSNR ~12-13 dB. Drone images of roofs, roads, and fields are reconstructed with recognizable visual features.
- **Medical images**: Lower quality. LPIPS ~0.80-1.0, PSNR ~10-11 dB. X-rays are harder to recover due to their visual similarity.
- **Standard defenses don't stop it**: We tested gradient pruning (removing small gradient values) and noise injection (adding random noise to gradients). Neither reliably blocked the attack.

### Phase 4: Ablation Studies (Addressing Reviewer Feedback)

We ran 53 total experiments to answer reviewer questions:

**1. Does the attack need the target in the batch?** YES.
- When pneumonia images are present, Attack F1 = 1.0 (perfect identification)
- When pneumonia images are absent, Attack F1 drops to 0.125 (nearly random)
- This means the attack is targeted — it specifically identifies query-matching images

**2. How sensitive is the temperature parameter?**
- T=1 (no scaling): acts like a normal model, no targeting
- T=10: too weak — can't separate targets from non-targets
- T=50-200: effective targeting zone
- T=100 (our default): good balance, consistent with contrastive learning literature

**3. Does the attacker need real labels?** NO.
- We tested BiomedCLIP zero-shot labels (only 29.6% accurate — basically guessing)
- The attack *still works* — actually slightly *better* than with real labels
- This means the attack doesn't need insider knowledge about the data

**4. Are results reproducible?** YES.
- 5 different random seeds: medical results nearly identical, UAV has more variance
- The main conclusions hold across all seeds

**5. What about bigger batches?** Mixed results.
- Medical: larger batches (16, 32) actually *improve* Attack F1
- UAV: larger batches degrade quality as expected
- The attack scales differently across domains

---

## The Key Numbers

### Reconstruction Quality

| Domain | LPIPS (↓ better) | PSNR (↑ better) |
|--------|-----------------|-----------------|
| Medical (baseline) | 0.79 | 11.3 dB |
| Medical (query-guided) | 1.02 | 10.9 dB |
| UAV (baseline) | 0.60 | 12.3 dB |
| UAV (query-guided) | 0.66 | 12.4 dB |

### Key Finding: Batch Composition Matters

| Condition | Attack F1 |
|-----------|-----------|
| Pneumonia present + Geminio model | **1.00** |
| Pneumonia absent + Geminio model | **0.13** |
| Either condition + Normal model | 1.00 |

This proves the attack specifically *targets* query-matching images — it's not just general gradient leakage.

---

## What's CLIP and Why Does It Matter?

CLIP is a model trained by OpenAI on 400 million image-text pairs from the internet. It learned to connect images with text descriptions — you can give it a photo and a sentence, and it tells you how well they match (a "similarity score" from 0 to 1).

Geminio uses CLIP to bridge the gap between what the attacker *wants* (a text query) and what's in the training data (images). Without CLIP, the attacker would need to know exactly which images are in the dataset. With CLIP, they just describe what they're looking for in English.

**BiomedCLIP** is the same idea but trained on medical literature and images. It understands terms like "pleural effusion" and "cardiomegaly" — but because medical images are more visually uniform, its similarity scores are more clustered together, which weakens the attack.

---

## The Files We Built

```
Our code
├── core/                          (from original Geminio)
│   ├── vlm.py                     CLIP text embeddings
│   └── models.py                  ResNet models with custom classifier
├── train_single_query.py          Train one ImageNet attack model
│
├── prototype/                     (our extensions)
│   ├── vlm_medical.py             BiomedCLIP text embeddings
│   ├── dataset_medical.py         ChestMNIST + embeddings
│   ├── train_medical.py           Train medical attack models (with temperature + pseudo-label support)
│   ├── dataset_uav.py             UAVScenes + segmentation→classification
│   ├── train_uav.py               Train UAV attack models
│   ├── reconstruct_medical.py     Gradient inversion + metrics + defenses (medical)
│   ├── reconstruct_uav.py         Gradient inversion + metrics + defenses (UAV)
│   ├── prepare_controlled_batches.py  Generate controlled batch compositions
│   ├── generate_pseudo_labels.py  BiomedCLIP zero-shot pseudo-labels
│   └── evaluate_all.py            Aggregate metrics across all runs
│
├── malicious_models/              5 ImageNet attack models
├── malicious_models_medical_v2/   7 medical attack models
├── malicious_models_uav/          6 UAV attack models
├── malicious_models_ablation/     5 temperature + pseudo-label variants
└── results/                       46 experiment result directories
```

---

## What's Next

1. **Compare to other attacks** — Are other gradient inversion methods (FEDLEAK, GUIDE) stronger or weaker than Geminio?
2. **Stronger defenses** — Test differential privacy (DP-SGD) with explicit privacy budgets (ε), which provides mathematical guarantees
3. **Per-image analysis** — Which specific images in a batch get reconstructed best? Does the attack actually preferentially recover the target images?
4. **Paper draft** — The cross-domain comparison is the main contribution

---

## The One-Sentence Version

> We're showing that federated learning leaks private data to a malicious server that uses AI language understanding (CLIP) to steal specific images described in plain English — and it works across medical, natural, and aerial image domains, but only when the target images are actually present in the victim's data batch.

---

## Glossary of Key Terms

| Term | What It Means |
|------|---------------|
| **Federated Learning (FL)** | Training a shared model across multiple parties without sharing raw data — only model updates (gradients). Proposed by McMahan et al. (2017). |
| **Gradient Inversion Attack (GIA)** | Recovering private training images from shared gradient updates. First demonstrated by Zhu et al. (NeurIPS 2019). |
| **Geminio** | The specific attack we're studying (Shan et al., 2411.14937) — adds language-guided targeting to GIA using VLMs. |
| **HFGradInv** | The specific gradient inversion algorithm we use (Jeon et al., NeurIPS 2021). Uses 24,000 optimization iterations to match observed gradients. |
| **CLIP** | Contrastive Language-Image Pre-training. OpenAI's model (Radford et al., ICML 2021) that connects images and text in a shared embedding space. |
| **BiomedCLIP** | CLIP variant trained on biomedical data and PubMed literature (Zhang et al., NeurIPS 2023). Better at understanding medical terminology. |
| **VLM** | Vision-Language Model — any model that understands both images and text (CLIP, BiomedCLIP, etc.) |
| **Loss Ratio** | Our metric: avg loss on target images ÷ avg loss on non-target images. Higher = attack works better at amplifying target gradients. |
| **Gradient Pruning** | A defense technique that zeros out the smallest gradient values by magnitude. For example, 90% pruning removes 90% of gradient entries that have the smallest absolute values. This is meant to remove fine-grained information the attacker needs for reconstruction while preserving the most important learning signals. (Zhu et al., NeurIPS 2019) |
| **Gradient Noise Injection** | A defense that adds random noise (Laplacian or Gaussian) to gradient updates before sharing them. This obscures the exact gradient values the attacker needs. Related to differential privacy (Dwork et al., 2006). |
| **Differential Privacy (DP-SGD)** | A mathematically rigorous defense that clips per-sample gradients and adds calibrated noise, providing a formal privacy guarantee expressed as an ε budget. Smaller ε = more privacy. (Abadi et al., CCS 2016) |
| **LPIPS** | Learned Perceptual Image Patch Similarity (Zhang et al., CVPR 2018). Measures how similar two images look to a neural network (AlexNet). Lower = more similar = better reconstruction. Range [0, 1+]. |
| **PSNR** | Peak Signal-to-Noise Ratio. Classic image quality metric in dB. Computed as 10·log10(1/MSE). Higher = better. Typical range: 10-30 dB for our experiments. |
| **CW-SSIM** | Complex Wavelet Structural Similarity (Sampat et al., 2009). Measures structural similarity using wavelet decomposition. Higher = better. Range [0, 1]. More robust to small spatial shifts than regular SSIM. |
| **Attack F1** | Geminio's custom metric: for each reconstructed image, compute the cosine similarity of its output-layer gradients with the ground truth image's gradients. If similarity ≥ 0.90, count as "identified." Attack F1 = fraction identified. Measures whether reconstructions are functionally equivalent to the originals. |
| **Temperature Scaling** | A parameter (T) that controls how sharply the Geminio loss function distinguishes between target and non-target images. `softmax(similarity * T)`: T=1 → uniform weighting (no targeting), T=100+ → sharp targeting. Analogous to temperature in softmax used in contrastive learning (Radford et al., 2021). |
| **Pseudo-Labels** | Labels generated by a model (BiomedCLIP zero-shot classification) instead of human annotations. In our case, BiomedCLIP assigns each chest X-ray a disease label by comparing its embedding to text descriptions of each disease. Only 29.6% accurate, but the attack still works. |
| **Malicious Model** | The rigged model the attacker sends to victims, designed to amplify gradients from target images |
| **ResNet** | Residual Network — the image classification backbone we use (He et al., CVPR 2016). ResNet18 for medical/UAV prototypes, ResNet34 for ImageNet. |
| **Classifier Head** | The small 3-layer network on top of ResNet that we actually train (the rest stays frozen). 512→256→64→num_classes. |
| **BatchNorm** | Batch Normalization layers in ResNet that track running statistics during training. Whether the model is in training or evaluation mode affects these statistics and can impact attack success (Valadi et al., 2508.19819). |
| **Cosine Similarity** | A measure of how similar two vectors are, regardless of magnitude. Range [-1, 1]. Used by CLIP to compare image and text embeddings, and by Attack F1 to compare gradient vectors. |
| **UAVScenes** | An aerial drone imagery dataset (ICCV 2025) with semantic segmentation labels (roofs, roads, pools, etc.) |
| **ChestMNIST** | A chest X-ray dataset derived from NIH ChestX-ray14 (Wang et al., 2017). 14 disease labels + 1 normal class. We use the 64×64 version from MedMNIST (Yang et al., Nature Scientific Data 2023). |
| **BCEWithLogitsLoss** | Binary Cross-Entropy with Logits — the loss function used for multi-label classification (UAVScenes) where each image can have multiple labels simultaneously, unlike CrossEntropyLoss which assumes exactly one label per image. |
| **Hungarian Matching** | An algorithm that finds the optimal one-to-one pairing between reconstructed and ground truth images in a batch, used to align images before computing quality metrics. (Kuhn, 1955) |
| **Secure Aggregation** | A protocol where gradient updates are cryptographically aggregated so the server sees only the sum, not individual updates. Defeats gradient inversion but adds communication overhead. (Bonawitz et al., CCS 2017) |
