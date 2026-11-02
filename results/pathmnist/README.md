# PathMNIST: the second medical dataset

Added 2026-11-02 to meet the 2026-10-07 plan of two medical datasets plus one surveillance dataset, and to test the
explanation of the ChestMNIST null: the medical attack never engaged because no guide could retrieve pneumonia on
ChestMNIST (control B, `results/complementarity/vlm_retrieval.md`), so the rigged head amplified a non-pneumonia
subset and the private targets received less than a uniform share of the gradient
(`results/complementarity/private_batch_profile.md`). A medical dataset on which the guide *can* retrieve the target
class is the direct test: the attack should select its targets there.

## Dataset

PathMNIST (MedMNIST v3, 64 px, `data/medmnist/pathmnist_64.npz`): 9 colorectal histopathology tissue classes
(H&E, NCT-CRC-HE-100K / CRC-VAL-HE-7K). The test split (7,180 images, the external CRC-VAL-HE-7K center) plays the
role ChestMNIST's test split plays: the first half of each class is the attacker's auxiliary set (3,587 images), the
second half is held out and the private batch is drawn from it (`prototype/prepare_target_batch.py --domain pathmnist`).
Images are RGB and visually heterogeneous across classes, so the uninformative-set reference (`floor_baselines.md`)
should leave more headroom than the mutually similar 64 px chest X-rays did.

Pipeline: `prototype/domains.py` registers the dataset; `prototype/dataset_medical.GeminioMedMNIST` loads it;
`compute_embeddings.py --domain pathmnist --vlm {guide}` writes `data/pathmnist-{guide}-test.pt`;
`train_medical.py --dataset pathmnist` trains ResNet-18 heads into `malicious_models_pathmnist/`;
`client_encoder.py --domain pathmnist` trains ViT-L heads (`malicious_models_client_encoder/pathmnist_*`);
`reconstruct_medical.py --dataset pathmnist` runs the attack (result dirs `results/pathmnist_*`).

## Target class: fixed before any attack run

`prototype/scan_medmnist_retrieval.py` scored every class under every guide on the auxiliary half with the template
"Any colorectal histopathology image showing {class name}" (`retrieval_scan.md`, 2026-11-02 14:02). Rule set in
advance: take the clinically sensitive class if at least one guide retrieves it clearly above chance; report the
whole scan either way.

Class 8, **colorectal adenocarcinoma epithelium** (tumor tissue; base rate 0.172), is both the sensitive class and a
retrievable one: BiomedCLIP AUROC 0.87 (precision in the top 10% 0.57), CLIP 0.60, SimCLIP-2 0.56, SimCLIP-4 0.54,
FARE-2 0.67, FARE-4 0.69, TeCoA-2 0.69, TeCoA-4 0.78. That spread (0.54 to 0.87) is what a within-dataset test of
"retrieval predicts selection" needs. For comparison, on ChestMNIST the same guides ranked pneumonia at 0.42-0.49
(BiomedCLIP 0.63).

Attacker query (all heads): "Any colorectal histopathology image showing colorectal adenocarcinoma epithelium",
tag `tum_base`. Private batch `assets/pathmnist_heldout/` (manifest alongside): 2 target images of 8, one image from
each of 6 other classes, seed 0, all from the held-out half.

## Design (pre-registered 2026-11-02, see `scripts/make_nov_jobs.py`)

| group | client | conditions | seeds | runs |
|---|---|---|---|---|
| A | ResNet-18 | 8 guide heads (BiomedCLIP base + 7 CLIP-family); base head with NBFU eps=5; IG; IG + NBFU | 42/123/256 | 33 |
| C | ViT-L, 7 CLIP-family encoders | rigged head | 3 | 21 |
| D (part) | ViT-L, 3 non-robust comparators | rigged head | 3 | 9 |
| E (part) | ViT-L, all 10 encoders | plain head | 3 | 30 |

Endpoints, same as the other domains: target LPIPS and selectivity (best-match and Hungarian), read against the
uninformative-set reference for this batch; per-sample gradient share on the private batch
(`private_batch_profile{,_resnet18}.md`); guide retrieval AUROC against attack strength across the 8 guides.

Expectations for the attack runs, recorded here with an honest note on what is and is not timestamped.
**Substantiated by file order:** the target class and query were fixed from the retrieval scan (13:53) before any head
was trained (13:55-13:58) and before any attack ran; the job design (13:52) predates both. **NOT substantiated:** this
README was written at 14:21, after the per-sample profile (14:00) and the correlation table (14:01), so the first item
below must not be described in the paper as a pre-registered prediction. Only the attack-side items (2)-(4) were
written before their data existed.
(1) The BiomedCLIP-guided ResNet-18 head places more of the last-layer gradient norm on the two tumor images than an
equal share, unlike every ChestMNIST head. *(Written after the profile; descriptive only.)*
(2) Target selectivity is positive and clears the reference.
(3) Across guides, retrieval tracks attack strength (as on UAV, where top-decile precision vs Hungarian target LPIPS
gives rho -0.86), which on this dataset would make TeCoA-4 the strongest CLIP-family guide and SimCLIP-4 the weakest.
(4) Whether target LPIPS beats the random-draw reference is open; the ChestMNIST failure to do so was partly a property
of the metric on 64 px X-rays.

## Results

**Reference for this batch** (`prototype/floor_baselines.py`, 50 random draws of 8 images from the auxiliary half, 2026-11-02):
target LPIPS 0.420 [5th-95th percentile 0.383, 0.451], selectivity -0.027 [-0.069, +0.032]; Hungarian target LPIPS
0.518 [0.405, 0.600]. The other seven batch images score 0.398 against the targets, the batch mean 0.374. So PathMNIST
leaves more headroom than ChestMNIST (0.327) and less than UAV (0.535).

**Group A, ResNet-18 client, 33 runs, 3 seeds, finished 15:13 (`results/nov_runs/pathmnist_resnet/summary.md`).
Three of the four expectations above are NOT met, and the direction of (3) is reversed.**

| guide | AUROC | target share | target LPIPS | vs reference | selectivity | distinct recons |
|---|---|---|---|---|---|---|
| BiomedCLIP | 0.87 | 0.362 | 0.534 | above p95 | +0.017 | 1.0 |
| TeCoA-4 | 0.78 | 0.205 | 0.426 | inside | -0.029 | 2.7 |
| FARE-4 | 0.69 | 0.208 | **0.404** | inside | -0.000 | 3.0 |
| TeCoA-2 | 0.69 | 0.150 | 0.439 | inside | -0.035 | 2.3 |
| FARE-2 | 0.67 | 0.173 | 0.450 | inside | -0.063 | 2.7 |
| CLIP | 0.60 | 0.266 | 0.439 | inside | -0.065 | 3.7 |
| SimCLIP-2 | 0.56 | 0.135 | 0.440 | inside | -0.050 | 3.7 |
| SimCLIP-4 | 0.54 | 0.154 | 0.423 | inside | -0.043 | 3.0 |

1. **(4) No recovery.** Nothing reaches the 5th percentile of the random-draw reference (0.383). Seven guides sit
   inside the reference range and BiomedCLIP is above its 95th percentile, i.e. worse than an uninformative draw.
   PathMNIST therefore joins ChestMNIST: on a ResNet-18 client, target LPIPS never evidences recovery, even though the
   reference leaves more headroom here (0.420) than on chest X-rays (0.327).
2. **(2) Selectivity fails.** Only BiomedCLIP is positive (+0.017) and that value is inside the reference range
   (-0.027 [-0.069, +0.032]); all seven CLIP-family guides are negative.
3. **(3) Reversed.** Retrieval does NOT predict attack strength here: Spearman(AUROC, target LPIPS) = +0.29
   (p = .49), and the sign means better retrieval goes with *worse* reconstruction; across the CLIP family alone it is
   flat (-0.07, p = .88). Under Hungarian matching the reverse trend is stronger (+0.62, p = .10).
4. **The decisive observation: the head that selects most reconstructs worst.** BiomedCLIP is the only medical
   ResNet-18 head above equal contribution (0.362 of the last-layer gradient norm on the two tumor images) and it gives
   the worst target LPIPS of the eight, with best-match collapsing onto a single reconstruction in all three seeds
   (distinct recons = 1.0). So on this client, concentrating the gradient on the targets is not sufficient for
   recovering them; compare the UAV client, where the same concentration does yield recovery.
   This separates the two factors more sharply than any of the earlier controls, and it is the reason the ViT-L runs
   (group C, in flight) matter: those heads place 0.72-0.95 on the targets.
5. **NBFU.** eps = 5 raises target LPIPS from 0.534 to 0.802 against HF-GradInv (paired d +0.268, p = .013) and from
   0.498 to 0.831 against IG (+0.333, p = .005). As on ChestMNIST, the undefended arm was already at or above the
   reference, so this shows the defended arm recovers nothing, not that a demonstrated recovery was removed.

**Caveats.** One private batch, three seeds; best-match collapses in several arms, so the Hungarian columns of
`summary.md` should be read alongside; the AUROC-vs-strength test has n = 8 guides.

