# Control C: temperature sweep (re-run of the lost 2026-10-17/18 experiment)

**Question.** Corollary (ii) of the complementarity framing: at a fixed query and backbone, does the
rigged head's loss ratio predict how selectively and how well the query-matched images are
reconstructed? Control A (results/complementarity/loss_ratio_vs_selectivity.md) shows the correlation
across the 24 language-sweep queries, but there the query text varies with the loss ratio. Here the
query is fixed ("Any chest X-ray showing pneumonia", BiomedCLIP; "aerial drone image showing solar
panels on rooftops", CLIP) and the loss ratio is moved through Geminio's softmax temperature T in
(1 − softmax(sim · T)).

**Design.** ResNet-18 client, held-out private batch (2 targets of 8), HF-GradInv 24k iterations,
seeds 42/123/256. T ∈ {10, 30, 100, 300, 1000}. T = 100 is the paper default and the existing
language-sweep base head. Two heads per T: replicate 1 (seed 0; T = 100 is the unseeded base head)
and replicate 2 (seed 1, all five T), so head-to-head variation at fixed T can be compared with the
between-T effect. The base head's second attack run (`results/{domain}_{tag}_ce_resnet18_s{seed}`, same
head and seed under another batch tag) is the same-seed noise floor, because the ResNet-18 attack is
not bit-reproducible (cudnn benchmark).

**Heads** (`prototype/run_temperature_train.py`, `malicious_models_tempsweep/`, loss ratios in `manifest.csv`).
UAV heads could not be trained on 2026-10-25: the UAVScenes root `/raid/scratch/data` became
root-owned with no group read bit in the Oct 23 storage restore. Add them with
`python prototype/run_temperature_train.py --domain uav --rep {1,2} --gpu N` once access is restored, then
re-run `scripts/make_rerun_jobs.py` and the queue.

**Runs.** `results/{domain}_{tag}_T{T}_s{seed}` (rep 1) and `results/{domain}_{tag}_T{T}r2_s{seed}` (rep 2);
job list `jobs.tsv` (from `scripts/make_rerun_jobs.py`), queue state under `queue/`. Resume:
`GPUS="4 5 7" bash scripts/run_job_queue.sh results/temp_sweep/jobs.tsv`.

**Scoring.** `python prototype/analyze_temperature_sweep.py --gpu N` → `results/complementarity/temperature_sweep.{md,csv}`:
per-T means (pooled and per head), per-run Spearman vs log T and vs loss ratio, the same-seed floor and
the head-to-head difference at fixed T. Read the floor before interpreting neighbouring temperatures.

**Caveats.** One head per (T, replicate); T is the manipulated variable and the loss ratio is
descriptive; 3 seeds; no multiple-comparison correction. The Oct 17/18 result (files lost) was: pooled
target LPIPS monotone in T (rho −1.00 over 5 per-T means; −0.58 per run, n = 33, p < .001), only the
T = 10 vs T ≥ 100 contrast clean against the noise floor.

## Result of the 2026-10-25 re-run (medical: 10 heads, 30 attack runs + 3 same-seed re-attacks)

Scored with `prototype/analyze_temperature_sweep.py` → `results/complementarity/temperature_sweep.md`.
Wording below follows an independent review (OpenAI validator, 2026-10-25 01:40) whose main points were
verified against the code and the per-run CSV: the loss ratio is a head property, so head-level statistics
(n = 10) are primary and per-run statistics (n = 30) are pseudo-replicated; the three same-seed re-attacks of
the base head are excluded from all correlations; heads within a replicate share the init seed, so
replicate differences are training-trajectory differences, not initialization variance.

- **Head-level association with the loss ratio (primary, n = 10):** best-match selectivity rho +0.67
  (p = .033; leave-one-head-out +0.55 to +0.80, p .010–.125); target LPIPS −0.48 (p = .16; LOO −0.62 to
  −0.35); Hungarian target LPIPS −0.52 (p = .13); Hungarian selectivity +0.28 (n.s.). Direction is stable
  under leave-one-head-out, significance is not, and nothing survives a Bonferroni correction over the
  metrics. The association depends on the two extreme heads (the failing T = 30 rep-1 head at 1.01 LPIPS
  and the 16.8x T = 100 rep-2 head); dropping either moves target-LPIPS rho to about −0.35. Non-target
  quality also improves mildly with the loss ratio (−0.37), so part of the signal is "everything improves".
  State this as an association consistent with corollary (ii), not as a prediction confirmed.
- **Per-run (secondary, n = 30):** loss ratio vs selectivity +0.62, vs target LPIPS −0.48 / −0.49 Hung.
  (p ≤ .008), but these treat runs sharing a head as independent and overstate the evidence.
- **T itself:** the two replicates disagree on the sign of the T effect (target LPIPS vs log T: rep 1
  +0.20, rep 2 −0.70; n = 5 each); head-level rho vs log T is −0.34 (n.s.). Loss ratio rises
  monotonically with T within rep 1 (9.3 → 13.3x) but not rep 2 (16.8x spike at T = 100), and moves only
  ~9.4x → ~13.6x over two decades of T: with Adam normalizing per-parameter gradient scale, T is a weak
  lever on the loss ratio here. Honest statement: no reliable T effect at this sample size, not "T has no
  effect".
- **Repeatability (3 pairs, one head; a description, not a threshold):** best-match target LPIPS differs
  by 0.02–0.11 between two attacks of the same head and seed; Hungarian by 0.04–0.52, the large value
  being a single assignment flip.
- **Relation to the lost Oct 17/18 run:** that run reported per-T means monotone in T (rho −1.00) with
  unseeded heads whose loss ratios (7.6/10.0/9.45/13.9/13.9x) were themselves not ordered by T, and it
  did not report a loss-ratio correlation that can be compared. Treat it as unverifiable exploratory
  history; the current run does not reproduce a T effect.
- Open items from the review: why the T = 30 rep-1 head fails for the whole batch (gradient/weight-norm
  profile vs the other heads would test a third-variable explanation); whether the global top-10% loss
  ratio captures what the per-minibatch softmax weight emphasizes (also measure it on held-out images);
  more init seeds per T are the only way to settle the T question (the design is 2 blocks × 5 T, paired).
- Diagnostic for the failing T = 30 rep-1 head (2026-10-25 01:50): its layer weight norms (|W1| 20.7, |W2| 13.9,
  |W3| 5.0, |b3| 0.36, largest output row 3.9) sit inside the range of the nine working heads (|W1| 18.2–21.2,
  |W2| 10.5–14.3, |W3| 3.6–5.1), so weight scale does not explain the whole-batch failure. The attack itself
  (optimization from the patterned init) fails on all three seeds for this head; the per-class loss profile on
  the private batch and the gradient norm it produces remain to be checked.

## UAV result (2026-10-26; 10 heads, 30 attack runs + 3 same-seed re-attacks; access restored that morning)

- **Loss ratios are not a function of T.** Rep 1: 18.9 / 66.6 / 130.5 (base) / 107.7 / 60.5x; rep 2: 98.2 / 134.1 /
  111.2 / 111.5 / 104.3x. The seeded rep-1 T = 10 head is the only low-ratio head (18.9x); everything else sits at
  60–134x. Head-level rho vs log T is ≈ 0 for every metric, and the within-replicate T trends disagree in sign.
- **Loss ratio vs outcome (head level, n = 10):** Hungarian selectivity +0.61 (p = .060; LOO +0.47 to +0.82),
  best-match selectivity +0.50 (p = .14), target LPIPS −0.43 (p = .21), Hungarian target LPIPS −0.41 (p = .24),
  non-target LPIPS +0.56 (p = .09). So on UAV a higher loss ratio goes with better *separation* of the targets,
  achieved as much by worse non-target reconstruction as by better target reconstruction; the low-ratio T = 10
  rep-1 head (selectivity 0.005, target LPIPS 0.578) anchors the association. Per-run (n = 30, pseudo-replicated):
  Hungarian selectivity +0.52 (p = .003), non-target LPIPS +0.49 (p = .006), target LPIPS −0.32 (p = .085).
- **Variance structure:** between-replicate differences at fixed T (up to 0.18 target LPIPS at T = 10) dwarf the
  between-T range of pooled means (0.03). Repeatability of one head: target LPIPS 0.00–0.10 best-match, 0.01–0.04
  Hungarian across the 3 pairs.
- Consistent with control A on UAV (same signs, not separable across the 24 queries) and with the plain-head
  control's reading that selectivity is a head property: the ResNet-18 UAV attack reconstructs targets at
  0.48–0.58 LPIPS for every head, and what the loss ratio moves is which images in the batch come out best.

## Replicate structure (independent review 2026-10-27; checks in `../complementarity/validation_2026_10_27_checks.md`)

The pooled head-level correlations mix two head replicates whose loss-ratio ranges differ. Centring the
loss ratio and the outcome within each replicate before correlating (n = 10 heads) gives:

| domain | outcome | rep 1 rho | rep 2 rho | replicate-centred rho | p |
|---|---|---|---|---|---|
| medical | selectivity, best-match | +0.50 | +1.00 | **+0.70** | .025 |
| medical | selectivity, Hungarian | +0.50 | 0.00 | +0.26 | .467 |
| medical | target LPIPS | +0.20 | −0.70 | −0.19 | .603 |
| uav | selectivity, best-match | +0.90 | −0.30 | +0.38 | .276 |
| uav | selectivity, Hungarian | +1.00 | −0.50 | +0.61 | .060 |
| uav | target LPIPS | −0.90 | +0.40 | −0.24 | .511 |

So the medical best-match selectivity association survives replicate-centring, while the UAV one rests
on replicate 1, whose heads span 19–131x; replicate 2 spans only 98–134x and reverses the sign. Report
the UAV result as present in the replicate with a wide loss-ratio range, not as a general association.
The target-LPIPS associations do not survive centring in either domain.

Two further limits: the ResNet-18 attack is not repeatable at a fixed seed (same-seed Hungarian
differences up to 0.52 on medical), so head means carry attack noise; and across the ViT-L panel of
control D the rigged-head loss ratio does not track selectivity at all (vanilla CLIP 98x gives −0.007,
FARE-4 22x gives +0.067), so the loss ratio is at best a within-backbone predictor.
