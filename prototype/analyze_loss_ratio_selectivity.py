"""Control A: does the rigged head's loss ratio predict selectivity at a fixed backbone?

Corollary (ii) of the complementarity framing: with the backbone fixed (ResNet-18), the
leakage advantage of the query-matched images should scale with how strongly the head
amplifies their loss. The 24-query language sweep varies the query text, and with it the
head's loss ratio (results/lang_sweep_logs/manifest.csv), while the private batch, attack
and seeds stay fixed (results/heldout_rerun/runs.csv, conditions {domain}_{tag}_ho and
{domain}_{tag}_dpsgd_eps5.0_ho). Confound: the query text itself varies with the loss ratio;
the temperature sweep (control C) removes it.

Per domain this reports Spearman correlations between the head's loss ratio (and the VLM
discriminability delta from discriminability.csv) and the per-query means of selectivity,
target LPIPS (best-match and Hungarian) and the NBFU margin (eps=5 minus clean target LPIPS),
plus the same correlations over individual runs (n = 12 queries x 3 seeds).

Outputs: results/complementarity/loss_ratio_vs_selectivity.{md,csv,png}
Usage:   python prototype/analyze_loss_ratio_selectivity.py
"""
import csv
import os
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'results', 'heldout_rerun', 'runs.csv')
MANIFEST = os.path.join(ROOT, 'results', 'lang_sweep_logs', 'manifest.csv')
DISCRIM = os.path.join(ROOT, 'results', 'lang_sweep_logs', 'discriminability.csv')
OUT = os.path.join(ROOT, 'results', 'complementarity', 'loss_ratio_vs_selectivity')
METRICS = ['selectivity', 'selectivity_hung', 'target_lpips', 'target_lpips_hung', 'n_unique_best']


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    manifest = {(r['domain'], r['tag']): float(r['loss_ratio']) for r in load_rows(MANIFEST) if r['ok'] == 'True'}
    discrim = {(r['domain'], r['tag']): float(r['delta']) for r in load_rows(DISCRIM)} if os.path.exists(DISCRIM) else {}
    runs = load_rows(RUNS)
    per_run = []   # one row per (domain, tag, seed)
    for (domain, tag), lr in sorted(manifest.items()):
        clean = {int(r['seed']): r for r in runs if r['condition'] == f'{domain}_{tag}_ho'}
        nbfu = {int(r['seed']): r for r in runs if r['condition'] == f'{domain}_{tag}_dpsgd_eps5.0_ho'}
        for seed, r in clean.items():
            row = dict(domain=domain, tag=tag, seed=seed, loss_ratio=lr, discrim_delta=discrim.get((domain, tag), float('nan')))
            for m in METRICS:
                row[m] = float(r[m])
            row['nbfu_margin'] = (float(nbfu[seed]['target_lpips']) - float(r['target_lpips'])) if seed in nbfu else float('nan')
            row['nbfu_margin_hung'] = (float(nbfu[seed]['target_lpips_hung']) - float(r['target_lpips_hung'])) if seed in nbfu else float('nan')
            per_run.append(row)
    outcomes = METRICS + ['nbfu_margin', 'nbfu_margin_hung']
    # per-query means
    per_query = []
    for (domain, tag) in sorted({(r['domain'], r['tag']) for r in per_run}):
        rr = [r for r in per_run if r['domain'] == domain and r['tag'] == tag]
        row = dict(domain=domain, tag=tag, n_seeds=len(rr), loss_ratio=rr[0]['loss_ratio'], discrim_delta=rr[0]['discrim_delta'])
        for m in outcomes:
            row[m] = float(np.nanmean([r[m] for r in rr]))
        per_query.append(row)

    lines = ['# Control A: head loss ratio vs selectivity at fixed backbone (ResNet-18, held-out batches)', '',
             'Loss ratio = mean loss of the top-10% query-similar auxiliary images over the rest, from the '
             'language-sweep training logs. Outcomes are means over 3 seeds of the clean held-out runs; '
             'NBFU margin = target LPIPS at eps=5 minus clean. Spearman rho with two-sided p. '
             'Confound: query text varies with loss ratio (see the temperature sweep for the fixed-query test).', '']
    summary = []
    for domain in ['medical', 'uav']:
        q = [r for r in per_query if r['domain'] == domain]
        rr = [r for r in per_run if r['domain'] == domain]
        if not q:
            continue
        lrs = [r['loss_ratio'] for r in q]
        lines += [f'## {domain}: {len(q)} queries, loss ratio {min(lrs):.1f}x to {max(lrs):.1f}x', '',
                  '| Outcome | rho vs loss ratio (per-query means, n={}) | p | rho (per run, n={}) | p | rho vs discriminability delta | p |'.format(len(q), len(rr)),
                  '|---|---|---|---|---|---|---|']
        for m in outcomes:
            x = np.array([r['loss_ratio'] for r in q]); y = np.array([r[m] for r in q])
            ok = ~np.isnan(y)
            rho, p = stats.spearmanr(x[ok], y[ok]) if ok.sum() > 2 else (float('nan'), float('nan'))
            xr = np.array([r['loss_ratio'] for r in rr]); yr = np.array([r[m] for r in rr]); okr = ~np.isnan(yr)
            rho_r, p_r = stats.spearmanr(xr[okr], yr[okr]) if okr.sum() > 2 else (float('nan'), float('nan'))
            xd = np.array([r['discrim_delta'] for r in q]); okd = ok & ~np.isnan(xd)
            rho_d, p_d = stats.spearmanr(xd[okd], y[okd]) if okd.sum() > 2 else (float('nan'), float('nan'))
            lines.append(f'| {m} | {rho:+.2f} | {p:.3f} | {rho_r:+.2f} | {p_r:.3f} | {rho_d:+.2f} | {p_d:.3f} |')
            summary.append(dict(domain=domain, outcome=m, rho_query=rho, p_query=p, n_query=int(ok.sum()),
                                rho_run=rho_r, p_run=p_r, n_run=int(okr.sum()), rho_discrim=rho_d, p_discrim=p_d))
        lines += ['', '| Query | loss ratio | selectivity | selectivity (Hung.) | target LPIPS | target LPIPS (Hung.) | NBFU margin |',
                  '|---|---|---|---|---|---|---|']
        for r in sorted(q, key=lambda r: r['loss_ratio']):
            lines.append(f"| {r['tag']} | {r['loss_ratio']:.2f}x | {r['selectivity']:+.3f} | {r['selectivity_hung']:+.3f} | "
                         f"{r['target_lpips']:.3f} | {r['target_lpips_hung']:.3f} | {r['nbfu_margin']:+.3f} |")
        lines.append('')

    with open(OUT + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_query[0].keys()))
        w.writeheader(); w.writerows(per_query)
    with open(OUT + '_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    with open(OUT + '.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        for i, domain in enumerate(['medical', 'uav']):
            q = [r for r in per_query if r['domain'] == domain]
            for j, (m, lab) in enumerate([('selectivity_hung', 'Selectivity (Hungarian)'), ('target_lpips_hung', 'Target LPIPS (Hungarian)')]):
                ax = axes[i, j]
                ax.scatter([r['loss_ratio'] for r in q], [r[m] for r in q], c='#0E6E66')
                for r in q:
                    ax.annotate(r['tag'].split('_', 1)[1], (r['loss_ratio'], r[m]), fontsize=7, xytext=(3, 3), textcoords='offset points')
                ax.set_xscale('log'); ax.set_xlabel('head loss ratio (log)'); ax.set_ylabel(lab); ax.set_title(domain)
        fig.tight_layout(); fig.savefig(OUT + '.png', dpi=150)
    except Exception as e:  # plotting is optional
        print('plot skipped:', e)
    print('\n'.join(lines))
    print(f'\nWrote {OUT}.md/.csv/_summary.csv/.png')


if __name__ == '__main__':
    main()
