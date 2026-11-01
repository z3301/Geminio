# Control D: plain (honest) head on each client encoder (re-run of the lost 2026-10-18/19 experiment)

**Question.** The client-encoder sweep (results/client_encoder/summary.md) showed that robust ViT-L
backbones leak more under Geminio and, on UAV, selectively (query-matched images gain 1.5–3× more
than the rest). The complementarity framing says that two factors do this: the rigged head supplies
the selection (it drives p_y → 0 on query-matched samples), the robust backbone supplies invertible
features. This control removes the head factor: the same frozen encoders get an ordinary head trained
with the mean loss (no query weighting, no per-batch loss normalization) and are attacked identically.

**Pre-registered hypotheses (sign convention: plain minus rigged).**
- H1 (selectivity is a head property): with the plain head, selectivity ≈ 0 on every encoder, and the
  paired Δ selectivity (plain − rigged) is negative on the robust encoders. Primary endpoint: UAV
  Hungarian and best-match selectivity, Wilcoxon over the 7 encoders.
- H2 (the head adds target leakage): Δ target LPIPS (plain − rigged) > 0, i.e. the rigged head
  reconstructs the targets better than the plain head on the same encoder.
- Encoder effect without rigging: plain-head target LPIPS is lower on robust encoders than on vanilla
  CLIP (robust features invert better even with an honest head).

**Design.** 7 encoders (clip, simclip2/4, fare2/4, tecoa2/4) × 2 domains × seeds 42/123/256 = 42 runs,
plus 2 duplicate-seed rigged runs (medical/clip and uav/clip, seed 42, tag `ce_dup_s42`) to measure the
same-seed floor of the ViT-L attack. Heads: `prototype/client_encoder.py --plain` →
`malicious_models_client_encoder/{domain}_{enc}_plain.pt` (seed 0, same as the rigged heads; features
from `results/client_encoder/cache`). Attack: HF-GradInv, 24k iterations, held-out batches, identical to
`scripts/run_client_encoder.sh`. Runs: `results/{domain}_{query}_enc{enc}_ce_plain_s{seed}`.

**Scoring.** `python scripts/summarize_ci.py --experiment client_encoder_plain --gpu N` →
`results/client_encoder_plain/{summary.md,runs.csv}`. Rigged dirs are re-scored live. Compare the paired
Δs against the dup floor the summary prints, not only against p-values.

**Caveats.** The plain loss removes both batch normalization and query weighting (a third arm with
uniform weights would isolate the weighting; not run). Attack F1 is head-dependent and must not be
compared across heads. Plain-head loss ratios are ~1.2× (medical) / <1× (UAV) against the base query,
rigged heads 8–12× / 22–98×. 3 seeds, no Holm correction. The Oct 19 result (files lost): UAV H1
supported (plain selectivity ~0 on all 7, rigged positive on all robust, Δ significant 5/7; vanilla CLIP
not selective under either head), encoder effect present without rigging (plain target LPIPS 0.73 CLIP vs
0.54–0.58 robust), H2 weak (+0.03); medical at floor under best-match, rigged leaks targets more on 6/7
under Hungarian (Δ +0.11); ViT-L attack bitwise deterministic at fixed seed.

Queue: `jobs.tsv` here lists the D jobs; the shared D+E queue is `results/rerun_oct24/jobs.tsv`
(`GPUS="2 3 6" bash scripts/run_job_queue.sh results/rerun_oct24/jobs.tsv`).

## Result of the 2026-10-25/27 re-run (44 runs, 0 failed; `results/client_encoder_plain/summary.md`)

Identical to the Oct 19 numbers (the ViT-L attack is bitwise deterministic at a fixed seed: the duplicate rigged
runs differ from the originals by 0.0000 on every metric, and the retrained seed-0 heads reproduce the lost ones).
Sign convention: plain minus rigged.

- **H1 supported on UAV.** Plain-head selectivity is ≈ 0 on all seven encoders (best-match −0.013 to +0.020;
  Hungarian −0.024 to +0.046). The rigged head is positive on all six robust encoders, and the paired Δ is negative
  on 6/7 (Wilcoxon over encoders p = .031; 17/21 run pairs, p < .001), significant per encoder on 5/7 (SimCLIP-2,
  SimCLIP-4, FARE-4, TeCoA-2, TeCoA-4). Vanilla CLIP is not selective under either head (−0.007 vs −0.000), so the
  selectivity needs both the rigged head and a robust backbone. Hungarian selectivity: Δ −0.109, 6/7, p = .031.
- **Encoder effect exists without rigging.** Plain-head target LPIPS: CLIP 0.726 vs robust 0.541–0.698 (TeCoA-2/4
  0.541/0.542, FARE-2/4 0.572/0.578, SimCLIP-4 0.571, SimCLIP-2 0.698). Robust features invert better with an
  honest head too.
- **H2 weak.** The rigged head improves target reconstruction by only +0.034 LPIPS on average (5/7 encoders,
  Wilcoxon p = .078; 15/21 run pairs, p = .004); SimCLIP-4 and FARE-2 go the other way. Non-target LPIPS is
  unchanged on average (−0.015, n.s.), with suppression only on the strongest robust encoders (TeCoA-4 −0.073,
  TeCoA-2 −0.045, p ≤ .03).
- **Medical.** Best-match metrics sit at the floor (n_unique = 1 for every plain run). Under Hungarian matching the
  rigged head leaks the pneumonia targets more than the plain head on 6/7 encoders (Δ +0.110, p = .031), but
  absolute target LPIPS stays 0.81–1.07, so the attack is still failing there. Best-match selectivity Δ is
  +0.023 (7/7, p = .016) but both arms are negative (−0.05 to −0.01), i.e. less negative under the plain head.

Reading: on UAV the rigged head supplies the *selection* (which images come out best) and the robust backbone
supplies the *invertibility* (how well anything comes out); the two multiply rather than substitute. Caveats as
above: 3 seeds, 7 encoders, no Holm; the plain loss removes both batch normalization and query weighting.

Figure: `results/client_encoder_plain/plain_vs_rigged.png` (`prototype/plot_head_contrast.py`): per encoder, rigged vs plain vs
swapped-query head for UAV selectivity (both matchings), UAV target LPIPS and medical Hungarian target LPIPS.

## Limits of this control (independent review 2026-10-27; checks in `../complementarity/validation_2026_10_27_checks.md`)

These bound what control D can support. Numbers re-derived from `results/client_encoder_plain/runs.csv`
by `prototype/verify_controls_critique.py`.

1. **The interaction is descriptive, not generalisable.** The per-encoder interaction contrast
   (rigged − plain on encoder *e*, minus the same difference on vanilla CLIP) is positive on all three
   seeds for 6/6 robust encoders under best-match selectivity and 4/6 under Hungarian, and 0/6 on
   medical. But every contrast shares one non-robust comparator, so the non-robust side has n = 1.
   Nothing here separates "robust" from "this particular OpenAI CLIP checkpoint". Testing the
   interaction needs other non-robust ViT-L encoders (OpenCLIP-LAION, SigLIP, DINOv2, MAE) and a
   head × encoder-class model with seeds as replicates.
2. **"Multiply" is not measured.** Across the 7 encoders the head's selectivity effect correlates with
   plain-head invertibility at rho = −0.68 (p = .094); among the 6 robust encoders alone, −0.49 (n.s.).
   The whole association comes from vanilla CLIP being extreme on both axes. State complementarity as a
   hypothesis consistent with the pattern, not as a demonstrated product of two factors.
3. **One private batch per domain.** All 44 runs use a single held-out 8-image batch per domain
   (`assets/{medical_heldout_pna,uav_heldout_solar}_manifest.json`, seed 0). The three attack seeds vary
   the reconstruction optimiser, not the data, so no claim extends to other batches. Repeating D on 3–5
   further 8-image batches, with batch as a random factor, is the cheapest way to fix this.
4. **This re-run is a recomputation, not a replication.** The ViT-L attack is bitwise deterministic at a
   fixed seed (duplicate runs differ by 0.0000), and the heads retrain from seed 0, so reproducing the
   Oct 19 numbers carries no extra evidential weight. The H1/H2 statements also live in this file
   alongside the results, so treat them as design notes rather than a timestamped pre-registration.
5. **Medical best-match selectivity should not be quoted.** 19 of 21 medical plain-head runs map every
   ground-truth image to a single reconstruction (the exceptions are SimCLIP-4 seed 123 and TeCoA-2
   seed 256, at 2). With one reconstruction, selectivity measures how much that image happens to
   resemble the pneumonia targets relative to the others, which is a property of the batch. The
   "+0.023, 7/7, p = .016" line is uninterpretable and is withdrawn.
6. **"Floor" is uncalibrated.** Calling the medical attack a floor effect compares it to nothing. A
   gradient-free baseline (random auxiliary images, the batch mean, noise, scored through the same
   matching code) is needed to say what "recovered nothing" scores. Not yet run.
7. **The encoder effect is 5 of 6, not 6 of 6.** Paired against vanilla CLIP under the plain head on UAV,
   target LPIPS is lower on SimCLIP-4, FARE-2, FARE-4, TeCoA-2 and TeCoA-4 (p .003–.020); SimCLIP-2 is
   −0.028, p = .34. Batch LPIPS is lower on all 6 (p ≤ .009). The claim is reconstructability under
   HF-GradInv on this batch, not intrinsic invertibility.
8. **The mechanism is inferred, never measured here.** Loss ratios are computed on auxiliary images. No
   measurement exists of per-sample confidence or gradient share on the 8 private images under the three
   heads, so selection by low true-class probability is not distinguished from selection by gradient
   norm. Logging loss, p_y and last-layer gradient share for the private batch under each head would
   settle it cheaply. Not yet run.
9. **Statistics.** Wilcoxon over 7 encoders has a floor of p = .0156, so p = .031 means 6/7 with the
   dissenting encoder smallest in magnitude: effectively a sign test. Per-encoder paired t has df = 2.
   Run-level p-values over 21 pairs are pseudo-replicated. No correction is applied across the ~40
   contrasts reported here and in control E.
