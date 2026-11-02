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
| F. Uninformative-set reference | `prototype/floor_baselines.py`, `prototype/compare_to_baselines.py` | `floor_baselines.{md,csv}`, `baseline_comparison.md` | DONE 2026-10-31: medical reference 0.327 [0.294, 0.361], UAV 0.535 [0.485, 0.571]; all medical conditions above the medical p95, 8 UAV conditions below the UAV p5 |
| G. Per-sample profile of the private batch | `prototype/private_batch_profile.py` | `private_batch_profile.{md,csv}` | DONE 2026-10-31: UAV rigged head puts 0.519 of the last-layer gradient on the 2 targets (uniform 0.250, 7/7 encoders) with own-label probability 0.14–0.60 against 0.68–0.78 elsewhere; plain 0.150, swap 0.097. Medical rigged head puts 0.149, *below* uniform and below its own plain head (0.287), so the mechanism never engages there |
| E. Query-swap control | `prototype/client_encoder.py --query`, `scripts/summarize_ci.py --experiment client_encoder_swap` | `../client_encoder_swap_qair_qeff/summary.md` | DONE 2026-10-27 (42/42): UAV solar selectivity disappears under the runway head (Δ −0.054, 6/7, p = .031; ≈ plain head) and solar target LPIPS rises on 7/7 (+0.050, p = .016) with non-targets unchanged: the rigged head's selection is query-specific; airstrip positive control weak (best-match +0.020, p = .078; Hungarian null); medical at floor with a generic-improvement confound (results/query_swap/README.md) |
| H. Per-sample profile, ResNet-18 guide heads | `prototype/private_batch_profile_resnet.py` | `private_batch_profile_resnet18.{md,csv}` | DONE 2026-11-02: 25 guide heads (9 ChestMNIST, 8 PathMNIST, 8 UAV). ChestMNIST: NO head exceeds the uniform 0.250 target share (BiomedCLIP 0.249/0.253, CLIP-family 0.13–0.22; robust-guide heads at per-image loss 250–860 on every image), so the 18/18 robust-guide effect is whole-batch reconstruction quality, not selection. UAV: 7/8 heads above uniform (0.28–0.46); TeCoA-4, the one guide that fails retrieval (AUROC 0.69), puts 0.087 on the targets. PathMNIST: BiomedCLIP head (AUROC 0.87) 0.362; CLIP-family (0.54–0.78) 0.13–0.27 |
| I. Guide retrieval vs target selection | `prototype/analyze_retrieval_selection.py`, `prototype/plot_retrieval_selection.py` | `retrieval_vs_selection.md`, `paper/cvpr/fig/retrieval_vs_share.pdf` | DONE 2026-11-02: Spearman(AUROC, target share) = +0.67, p = .0002 over 25 heads (UAV alone +0.90, p = .002); AUROC ≥ 0.75: 8/9 heads above uniform (median 0.34); < 0.75: 2/16 (median 0.18). The selection factor of the two-factor account is bounded by retrieval |

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

One of the two follow-ups is now done: `floor_baselines.md` and `baseline_comparison.md`
(`prototype/floor_baselines.py`, `prototype/compare_to_baselines.py`, run 2026-10-31) score
uninformative image sets through the same matching code, and place every reported condition against
them. Headline: no medical condition anywhere beats eight images drawn at random from the dataset on
target LPIPS, while a subset of UAV conditions does. Paired contrasts are unaffected, since the
reference is a property of the batch. The second is now done too: `private_batch_profile.md` measures per-image loss, own-label probability
and last-layer gradient share under all three heads. On UAV it confirms the assumed mechanism directly.
On medical it shows the mechanism never engages, which explains the domain null end to end when read
with control B's retrieval result.
