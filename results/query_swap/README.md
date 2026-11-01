# Control E: query swap (re-run of the lost 2026-10-19/20 experiment; never scored before the loss)

**Question.** If the rigged head supplies the selection, selectivity should follow the query: a head
rigged for a different class present in the same batch should stop favouring the solar / pneumonia
targets and start favouring the swap class, on the same encoder, batch and seed.

**Pre-registered endpoints (sign convention: swap minus rigged; swap minus plain as secondary).**
- Primary: UAV solar-target selectivity, swap − rigged, negative (Wilcoxon over 7 encoders); swap − plain ≈ 0.
- Positive control: the swap class's own selectivity (alt_* metrics) is positive under the swap head.
  On UAV this is exploratory: the auxiliary half has no airstrip images, so the head cannot learn the
  airstrip concept from aux labels, only from the CLIP similarity of the query (see
  `swap_query_retrieval.md`).
- Medical: the attack is near floor for every ViT-L client, so report Hungarian metrics and treat the
  domain as secondary.

**Queries** (same attacker VLM as the rigged heads): medical "Any chest X-ray showing pleural effusion"
(tag `qeff`; the held-out batch has one effusion image, 6-2.png); UAV "aerial drone image showing an
airport runway" (tag `qair`; the batch has two airstrip images, 0-… and 5-…). Query-choice evidence:
`swap_query_retrieval.md` (`prototype/analyze_swap_query_retrieval.py`). Heads:
`prototype/client_encoder.py --query ... --query-tag ...` → `malicious_models_client_encoder/{domain}_{enc}_{tag}.pt`;
each head's JSON reports its loss ratio for the swap query and the residual loss ratio for the base query.

**Design.** 7 encoders × 2 domains × seeds 42/123/256 = 42 ViT-L runs, HF-GradInv 24k iterations,
held-out solar / pneumonia batches. Runs: `results/{domain}_{query}_enc{enc}_ce_{tag}_s{seed}`.

**Scoring.** `python scripts/summarize_ci.py --experiment client_encoder_swap --tag medical=qeff,uav=qair --alt-target medical=effusion,uav=airstrip --gpu N`
→ `results/client_encoder_swap_qair_qeff/{summary.md,runs.csv}` (swap − rigged and swap − plain blocks,
alt-target metrics, dup floor, head loss ratios).

Queue: `jobs.tsv` here lists the E jobs; they run after D in `results/rerun_oct24/jobs.tsv`.

## Result (2026-10-27; 42 swap runs, 0 failed; `results/client_encoder_swap_qair_qeff/summary.md`)

Pre-registered endpoints, sign convention swap minus rigged (then swap minus plain):

- **Primary endpoint supported on UAV.** Solar-target selectivity falls when the head is rigged for airport
  runways instead: best-match Δ −0.054, negative on 6/7 encoders (Wilcoxon over encoders p = .031; 18/21 run
  pairs, p < .001); Hungarian Δ −0.094, 5/7 (p = .156 over encoders; 17/21 run pairs, p = .008). Against the plain
  head the swap head is indistinguishable on the solar targets (selectivity Δ −0.005 best-match / +0.014 Hungarian,
  target LPIPS Δ +0.016, all n.s.). So the rigged head's solar selectivity is a property of *its query*, not of
  the backbone or of rigging in general.
- **The rigged head's target advantage is query-specific.** Solar target LPIPS rises under the swap head on
  7/7 encoders (Δ +0.050, p = .016; Hungarian +0.064, 5/7) while non-target LPIPS does not move (−0.004, n.s.).
- **Positive control (airstrip targets) weak.** Under the swap head, airstrip selectivity rises by +0.020
  best-match (6/7 encoders, p = .078; 19/21 run pairs, p = .001) but not under Hungarian matching (−0.016, n.s.),
  and airstrip target LPIPS itself is unchanged (+0.005 Hungarian). As pre-registered, this was exploratory: the
  auxiliary half has no airstrip images, so the head's 25–52x airstrip loss ratio comes only from CLIP similarity.
- **Medical (secondary; attack at floor).** The swap (effusion) head reconstructs *everything* slightly better
  than the rigged head (target −0.057, non-target −0.049, batch −0.051, each 0/7 encoders > 0, p = .016), so
  pneumonia selectivity is unchanged (+0.008, n.s.); effusion selectivity rises +0.017 best-match (6/7, p = .031)
  but not under Hungarian. Against the plain head, the swap head still leaks pneumonia targets more under
  Hungarian (−0.142, 7/7, p = .016) with higher pneumonia selectivity (+0.073, 7/7, p = .016); the swap heads keep a
  1.7–2.3x residual loss ratio on the pneumonia query, which is the likely reason. Absolute target LPIPS remains
  0.64–0.84, so nothing recognisable is recovered.

Caveats: 3 seeds, 7 encoders, no multiple-comparison correction; one swap query per domain; best-match
selectivity collapses onto one or two reconstructions in most runs (n_unique 1–3), so prefer the Hungarian
columns where they disagree.

## Limits of this control (independent review 2026-10-27; checks in `../complementarity/validation_2026_10_27_checks.md`)

1. **"The swap head behaves like the plain head" holds only for the aggregate solar metrics.** Per
   encoder, swap minus plain differs significantly in both directions: batch LPIPS is worse under the
   swap head on SimCLIP-4 (+0.059, p = .035), FARE-2 (+0.092, p = .007), TeCoA-2 (+0.071, p = .013) and
   TeCoA-4 (+0.092, p = .010), and better on SimCLIP-2 (−0.086, p = .013). Solar target LPIPS likewise
   splits by sign (SimCLIP-2 −0.066, p = .011; TeCoA-2 +0.051, p = .032; TeCoA-4 +0.046, p = .014).
   Word the result as "no detected difference in the aggregate", not as equivalence; the plain-head CIs
   (e.g. TeCoA-2 selectivity −0.005 ± 0.067) are wide next to the rigged effect (+0.092).
2. **Rigging for any query concentrates the reconstructions.** Mean distinct reconstructions used by
   best-match on UAV: plain 3.57, rigged 2.19, swap 2.24. Per encoder the drop from plain to swap is
   large on the encoders where the plain head spreads out (FARE-2 5.00 → 1.67, TeCoA-2 5.67 → 3.00,
   TeCoA-4 6.00 → 3.00). So one effect of a rigged head is to collapse the batch onto fewer
   reconstructions, and the query decides which ones. That is an alternative reading of "selection"
   worth reporting beside the selectivity numbers.
3. **One private batch, one swap query per domain.** Same single-batch limit as control D, plus a single
   alternative query, so "selectivity follows the query" rests on one query pair per domain.
4. **The positive control is weak by construction.** The UAV auxiliary half contains no airstrip images,
   so the swap head can only learn the concept through CLIP similarity. Airstrip selectivity rises
   under best-match (+0.020, 6/7, p = .078) but not under Hungarian (−0.016, n.s.), and airstrip target
   LPIPS does not move. Do not present the positive control as passed.
5. **Medical is confounded.** The effusion-rigged head reconstructs the whole batch slightly better than
   the solar-equivalent head (target −0.057, non-target −0.049, batch −0.051, each 0/7 encoders above
   zero), so unchanged pneumonia selectivity cannot be read as query-specificity. The swap heads also
   keep a 1.7–2.3x residual loss ratio on the pneumonia query.
