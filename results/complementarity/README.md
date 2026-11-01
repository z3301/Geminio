# Complementarity checks (re-created 2026-10-25 after the Oct 23 storage restore)

Framing under test: Geminio's malicious head supplies the condition Zhang, Chen & Li (CVPRW 2022)
identified (confidence on the true label driven to zero, so the output-layer gradient carries a clean
copy of each sample's features), on clean inputs and only for query-matched samples; an adversarially
robust backbone supplies features that invert well. Head and backbone are complementary.

| Check | Script | Output | Status 2026-10-25 |
|---|---|---|---|
| A. Loss ratio vs selectivity across the 24-query sweep (fixed ResNet-18) | `prototype/analyze_loss_ratio_selectivity.py` | `loss_ratio_vs_selectivity.{md,csv,png}` | re-run; medical rho +0.90 / +0.78 (Hung.), NBFU margin +0.64; UAV saturated, n.s. — identical to Oct 17 |
| B. Retrieval precision of each guide VLM on the aux set | `prototype/analyze_vlm_retrieval.py` | `vlm_retrieval.{md,csv}` | medical re-run: all 7 CLIP-family guides rank pneumonia at/below chance (any-label AUROC 0.42–0.49, base rate 1.2%; BiomedCLIP 0.63), no correlation with attack strength (|rho| ≤ 0.39) — as on Oct 17; UAV re-run 2026-10-26: retrieval varies by guide (FARE-2 AUROC 0.97, SimCLIP-2 0.97 … TeCoA-4 collapses to 0.69, prec@10% 0.06 vs base 0.08) and tracks attack strength (prec@10% vs Hungarian target LPIPS rho −0.86, p = .014; vs best-match −0.75, p = .052) — explains the null UAV average in the attacker-guide table, as on Oct 17 |
| B'. Appearance of top-ranked aux images | `prototype/analyze_aux_appearance.py` | `aux_appearance.{md,csv}` | medical re-run: top-16 of every guide is mostly 'normal', 0 pneumonia; LPIPS to targets 0.37–0.47 vs random 0.42; no correlation with attack strength (|rho| ≤ 0.39) — as on Oct 18 |
| C. Temperature sweep (fixed query) | `prototype/run_temperature_train.py`, `prototype/analyze_temperature_sweep.py` | `temperature_sweep.{md,csv}` | medical DONE 01:26 (27/27). Head-level (n = 10, primary): loss ratio vs best-match selectivity +0.67 (p .033, LOO .010–.125), vs target LPIPS −0.48 (p .16), Hung. −0.52 (p .13); direction stable, significance fragile to single heads; no reliable T effect (replicates disagree in sign). Independent OpenAI review completed 01:40 and folded in (results/temp_sweep/README.md); UAV DONE 2026-10-26 13:45 (27/27): loss ratio 19–134x unrelated to T; head-level Hungarian selectivity vs loss ratio +0.61 (p .06), non-target LPIPS +0.56, target LPIPS −0.43 (n.s.); between-replicate variance ≫ between-T; selectivity, not target quality, is what the loss ratio moves on UAV |
| D. Plain-head control | `prototype/client_encoder.py --plain`, `scripts/summarize_ci.py --experiment client_encoder_plain` | `../client_encoder_plain/summary.md` | DONE 2026-10-27 (44/44, identical to Oct 19; attack bitwise deterministic): UAV selectivity is a head property (plain ≈ 0 on 7/7, rigged > 0 on all robust, Δ 6/7 p = .031, 5/7 per-encoder), invertibility an encoder property (plain-head target LPIPS 0.73 CLIP vs 0.54–0.70 robust), H2 weak (+0.034, p = .078); medical Hungarian rigged leaks targets more 6/7 (+0.110, p = .031) but at floor (results/plain_head/README.md) |
| E. Query-swap control | `prototype/client_encoder.py --query`, `scripts/summarize_ci.py --experiment client_encoder_swap` | `../client_encoder_swap_qair_qeff/summary.md` | DONE 2026-10-27 (42/42): UAV solar selectivity disappears under the runway head (Δ −0.054, 6/7, p = .031; ≈ plain head) and solar target LPIPS rises on 7/7 (+0.050, p = .016) with non-targets unchanged: the rigged head's selection is query-specific; airstrip positive control weak (best-match +0.020, p = .078; Hungarian null); medical at floor with a generic-improvement confound (results/query_swap/README.md) |

Job lists: `scripts/make_rerun_jobs.py`. Queues: `results/temp_sweep/jobs.tsv` (C) and
`results/rerun_oct24/jobs.tsv` (D then E). Logs under each experiment's `queue/logs/`.

## Independent review, 2026-10-27

Controls C, D and E were sent as one evidence packet to an external model for adversarial review; its
checkable claims were then re-derived locally by `prototype/verify_controls_critique.py` →
`validation_2026_10_27_checks.md`. Verdict: partially supported. Every number in the packet matched the
saved tables. What changed in the write-ups:

- the head × backbone interaction is descriptive (one non-robust comparator), and "multiply" is not measured
- all of D and E rests on one private batch per domain; the seeds vary the optimiser, not the data
- the UAV loss-ratio association in C holds in one head replicate only; medical survives replicate-centring
- swap vs plain agrees only in aggregate; rigging for any query collapses reconstruction diversity
- medical best-match selectivity is withdrawn (19/21 runs map everything to one reconstruction)
- D is a recomputation of a deterministic attack, not a replication
- the Zhang et al. mechanism is approximate proportionality with factor −(1 − p_y), not a clean copy
  (verified against the primary source; `paper/cvpr/related_work.tex` corrected)

Two cheap follow-ups are specified but not run: gradient-free floor baselines, and per-sample confidence
and gradient share on the private batch under the rigged, plain and swapped heads.
