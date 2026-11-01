"""Score the temperature sweep (control C): selectivity and target leakage vs Geminio's temperature T.

Runs (held-out batch, ResNet-18 client, HF-GradInv, attack seeds 42/123/256):
  rep 1  T != 100  results/{domain}_{tag}_T{T}_s{seed}        heads malicious_models_tempsweep/{domain}/{tag}_T{T}.pt
                                                              (head init seed 0, shared by all rep-1 heads)
         T == 100  results/{domain}_{tag}_ho_s{seed}          the language-sweep base head (unseeded; loss ratio from
                                                              results/lang_sweep_logs/manifest.csv), plus
                   results/{domain}_{tag}_ce_resnet18_s{seed} the same head attacked again under another batch
                                                              tag: same-seed re-attacks, used ONLY for repeatability
  rep 2  all T     results/{domain}_{tag}_T{T}r2_s{seed}      heads with init seed 1 (shared by all rep-2 heads)

Unit of inference. The loss ratio is a property of the head, so the primary statistics are over
heads (one value per head = mean over its attack seeds; n = 10 per domain), with a leave-one-head-out
range. Per-run correlations (one value per attack run, duplicates excluded) are reported as secondary
and are pseudo-replicated (runs sharing a head are not independent). Within a replicate the heads share
the init seed and batch order, so differences between replicates at fixed T are training-trajectory
differences, not initialization variance in the usual sense; the T effect is therefore also reported
within each replicate (paired by T).

Image metrics are re-scored from the saved tensors with scripts/summarize_ci.py (best-match and
Hungarian matching).

Outputs: results/complementarity/temperature_sweep.{md,csv}
Usage:   python prototype/analyze_temperature_sweep.py --gpu 0
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from summarize_ci import image_metrics  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'complementarity', 'temperature_sweep')
TAG = {'medical': 'pna_base', 'uav': 'sol_base'}
TEMPS = [10, 30, 100, 300, 1000]
SEEDS = [42, 123, 256]
METRICS = ['target_lpips', 'selectivity', 'target_lpips_hung', 'selectivity_hung', 'nontarget_lpips', 'nontarget_lpips_hung', 'n_unique_best', 'lpips']


def loss_ratios():
    out = {}
    p = os.path.join(ROOT, 'results', 'lang_sweep_logs', 'manifest.csv')
    with open(p) as f:
        for r in csv.DictReader(f):
            if r['tag'] in TAG.values() and r['ok'] == 'True':
                out[(r['domain'], 100, 1)] = float(r['loss_ratio'])
    p = os.path.join(ROOT, 'malicious_models_tempsweep', 'manifest.csv')
    if os.path.exists(p):
        with open(p) as f:
            for r in csv.DictReader(f):
                out[(r['domain'], int(r['T']), int(r['rep']))] = float(r['loss_ratio'])
    return out


def runs_for(domain):
    tag = TAG[domain]
    for seed in SEEDS:
        yield dict(T=100, rep=1, seed=seed, kind='ho', run_dir=f'results/{domain}_{tag}_ho_s{seed}')
        yield dict(T=100, rep=1, seed=seed, kind='dup', run_dir=f'results/{domain}_{tag}_ce_resnet18_s{seed}')
        for T in TEMPS:
            if T != 100:
                yield dict(T=T, rep=1, seed=seed, kind='sweep', run_dir=f'results/{domain}_{tag}_T{T}_s{seed}')
            yield dict(T=T, rep=2, seed=seed, kind='sweep', run_dir=f'results/{domain}_{tag}_T{T}r2_s{seed}')


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 4 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float('nan'), float('nan')
    rho, p = stats.spearmanr(x[ok], y[ok])
    return float(rho), float(p)


def f2(x):
    return 'n/a' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:+.2f}'


def f3(x):
    return 'n/a' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=None)
    args = ap.parse_args()
    import lpips
    device = f'cuda:{args.gpu}' if args.gpu is not None and torch.cuda.is_available() else 'cpu'
    lp = lpips.LPIPS(net='alex', verbose=False).to(device)
    lrs = loss_ratios()
    rows = []
    for domain in ['medical', 'uav']:
        for spec in runs_for(domain):
            d = os.path.join(ROOT, spec['run_dir'])
            if not os.path.exists(os.path.join(d, 'metrics.json')):
                continue
            with open(os.path.join(d, 'metrics.json')) as f:
                m = json.load(f)
            row = dict(domain=domain, **spec, head=f"T{spec['T']}_r{spec['rep']}",
                       init_seed={1: 0, 2: 1}[spec['rep']] if not (spec['T'] == 100 and spec['rep'] == 1) else None,
                       lpips=float(m.get('lpips', float('nan'))), loss_ratio=lrs.get((domain, spec['T'], spec['rep']), float('nan')))
            row.update(image_metrics(domain, d, lp, device))
            rows.append(row)
    if not rows:
        sys.exit('no runs found')
    with open(OUT + '.csv', 'w', newline='') as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('domain', 'head', 'T', 'rep', 'init_seed', 'seed', 'kind'), k))
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    lines = ['# Control C: temperature sweep (fixed query and backbone, loss ratio varied through T)', '',
             'ResNet-18 client, held-out batch, attack seeds 42/123/256. One head per (T, replicate); rep-1 heads share init '
             'seed 0, rep-2 heads share init seed 1, and the T = 100 rep-1 head is the unseeded language-sweep base head. '
             'Its second attack run (ce_resnet18 tag) enters only the repeatability table. Primary statistics are over '
             'heads (loss ratio is a head property); per-run statistics are pseudo-replicated and secondary. Metrics '
             're-scored from saved tensors.', '']
    for domain in ['medical', 'uav']:
        dr = [r for r in rows if r['domain'] == domain]
        if not dr:
            continue
        if len({r['T'] for r in dr}) < 2:
            lines += [f'## {domain}', '', f'_Only T = 100 runs present ({len(dr)} runs); the sweep heads for this domain '
                      'have not been trained (UAVScenes root unreadable since the Oct 23 restore)._', '']
            continue
        corr = [r for r in dr if r['kind'] != 'dup']          # one row per (head, attack seed)
        heads = sorted({(r['T'], r['rep']) for r in corr})
        head_rows = []
        for T, rep in heads:
            hr = [r for r in corr if r['T'] == T and r['rep'] == rep]
            h = dict(T=T, rep=rep, head=f'T{T}_r{rep}', n_runs=len(hr), loss_ratio=hr[0]['loss_ratio'])
            for m in METRICS:
                h[m] = float(np.nanmean([r[m] for r in hr]))
            head_rows.append(h)
        lines += [f'## {domain}', '', f'### Run manifest: {len(heads)} heads, {len(corr)} attack runs (+{len(dr) - len(corr)} same-seed re-attacks of the base head, repeatability only)', '',
                  '| head | T | rep | init seed | loss ratio | n runs | ' + ' | '.join(METRICS) + ' |',
                  '|---|---|---|---|---|---|' + '---|' * len(METRICS)]
        for h in head_rows:
            init = 'unseeded' if (h['T'] == 100 and h['rep'] == 1) else {1: 0, 2: 1}[h['rep']]
            lines.append(f"| {h['head']} | {h['T']} | {h['rep']} | {init} | {h['loss_ratio']:.2f}x | {h['n_runs']} | "
                         + ' | '.join(f3(h[m]) for m in METRICS) + ' |')
        # per-T pooled means (ho runs only for T = 100 rep 1)
        lines += ['', '### Per-T means over heads and attack seeds (duplicates excluded)', '',
                  '| T | heads | mean loss ratio | n runs | ' + ' | '.join(METRICS) + ' |', '|---|---|---|---|' + '---|' * len(METRICS)]
        means = {}
        for T in TEMPS:
            sel = [r for r in corr if r['T'] == T]
            if not sel:
                continue
            means[T] = {m: float(np.nanmean([r[m] for r in sel])) for m in METRICS}
            lines.append(f"| {T} | {len({r['rep'] for r in sel})} | {np.nanmean([r['loss_ratio'] for r in sel]):.2f}x | {len(sel)} | "
                         + ' | '.join(f3(means[T][m]) for m in METRICS) + ' |')
        # PRIMARY: head-level correlations with leave-one-head-out range
        lines += ['', f'### Head-level Spearman (PRIMARY; n = {len(head_rows)} heads, one mean per head)', '',
                  '| metric | rho vs loss ratio | p | leave-one-head-out rho range | p range | rho vs log T | p |', '|---|---|---|---|---|---|---|']
        for m in METRICS:
            x = [h['loss_ratio'] for h in head_rows]; y = [h[m] for h in head_rows]
            rho, p = spearman(x, y)
            loo = [spearman([v for j, v in enumerate(x) if j != i], [v for j, v in enumerate(y) if j != i]) for i in range(len(x))]
            loo_r = [a for a, _ in loo if not math.isnan(a)]; loo_p = [b for _, b in loo if not math.isnan(b)]
            rT, pT = spearman([math.log(h['T']) for h in head_rows], y)
            lines.append(f"| {m} | {f2(rho)} | {f3(p)} | {f2(min(loo_r)) if loo_r else 'n/a'} to {f2(max(loo_r)) if loo_r else 'n/a'} | "
                         f"{f3(min(loo_p)) if loo_p else 'n/a'} to {f3(max(loo_p)) if loo_p else 'n/a'} | {f2(rT)} | {f3(pT)} |")
        # within-replicate T trend (paired by T)
        lines += ['', '### T effect within each replicate (per-T head means, Spearman vs log T, n = 5 heads each)', '',
                  '| metric | ' + ' | '.join(f'rep {rep} rho (p)' for rep in (1, 2)) + ' |', '|---|---|---|']
        for m in METRICS:
            cells = []
            for rep in (1, 2):
                hr = [h for h in head_rows if h['rep'] == rep]
                rho, p = spearman([math.log(h['T']) for h in hr], [h[m] for h in hr])
                cells.append(f'{f2(rho)} ({f3(p)})')
            lines.append(f'| {m} | ' + ' | '.join(cells) + ' |')
        lines += ['', 'Loss ratio by T within replicate: ' + '; '.join(
            f"rep {rep}: " + ' / '.join(f"{h['loss_ratio']:.2f}" for h in head_rows if h['rep'] == rep) for rep in (1, 2)) + '.']
        # SECONDARY: per-run correlations (pseudo-replicated)
        lines += ['', f'### Per-run Spearman (SECONDARY, pseudo-replicated; n = {len(corr)} runs, duplicates excluded)', '',
                  '| metric | rho vs loss ratio | p | rho vs log T | p |', '|---|---|---|---|---|']
        for m in METRICS:
            r1, p1 = spearman([r['loss_ratio'] for r in corr], [r[m] for r in corr])
            r2, p2 = spearman([math.log(r['T']) for r in corr], [r[m] for r in corr])
            lines.append(f'| {m} | {f2(r1)} | {f3(p1)} | {f2(r2)} | {f3(p2)} |')
        # repeatability of the same head
        lines += ['', '### Repeatability of one head: base head attacked twice with the same attack seed (3 pairs; a description of run-to-run variation, not a detection threshold)', '',
                  '| metric | ' + ' | '.join(f'seed {s}' for s in SEEDS) + ' | mean abs | max abs |', '|---|' + '---|' * (len(SEEDS) + 2)]
        for m in METRICS:
            cells, diffs = [], []
            for seed in SEEDS:
                a = [r for r in dr if r['kind'] == 'ho' and r['seed'] == seed]
                b = [r for r in dr if r['kind'] == 'dup' and r['seed'] == seed]
                if a and b:
                    d = a[0][m] - b[0][m]; diffs.append(abs(d)); cells.append(f'{d:+.3f}')
                else:
                    cells.append('n/a')
            lines.append(f'| {m} | ' + ' | '.join(cells) + (f' | {np.mean(diffs):.3f} | {np.max(diffs):.3f} |' if diffs else ' | n/a | n/a |'))
        # between-replicate difference at fixed T
        lines += ['', '### Between-replicate difference at fixed T (rep 1 − rep 2 head means; the two heads share T but differ in init seed and training trajectory) vs the between-T range of pooled means', '',
                  '| metric | ' + ' | '.join(f'T={T}' for T in TEMPS) + ' | between-T range |', '|---|' + '---|' * (len(TEMPS) + 1)]
        for m in METRICS:
            cells = []
            for T in TEMPS:
                a = [h for h in head_rows if h['T'] == T and h['rep'] == 1]
                b = [h for h in head_rows if h['T'] == T and h['rep'] == 2]
                cells.append(f"{a[0][m] - b[0][m]:+.3f}" if a and b else 'n/a')
            rng = (max(v[m] for v in means.values()) - min(v[m] for v in means.values())) if means else float('nan')
            lines.append(f'| {m} | ' + ' | '.join(cells) + f' | {rng:.3f} |')
        lines.append('')
    with open(OUT + '.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines)); print(f'\nWrote {OUT}.md/.csv ({len(rows)} runs)')


if __name__ == '__main__':
    main()
