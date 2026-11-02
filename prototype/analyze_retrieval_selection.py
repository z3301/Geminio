"""Does a guide's retrieval of the target class predict how much gradient its rigged head puts on the targets?

Joins the per-head rows of private_batch_profile_resnet18.csv (ResNet-18 client, one head per guide and
dataset) and, when present, the ViT-L rigged heads of private_batch_profile.csv, and reports Spearman
correlations between the guide's retrieval AUROC on the auxiliary set and the target share of the last-layer
gradient on the private batch. Uniform share for 2 targets of 8 is 0.250.

Output: results/complementarity/retrieval_vs_selection.md
Usage:  python prototype/analyze_retrieval_selection.py
"""
import csv
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROF = os.path.join(ROOT, 'results', 'complementarity', 'private_batch_profile_resnet18.csv')
OUT = os.path.join(ROOT, 'results', 'complementarity', 'retrieval_vs_selection.md')


def heads():
    by = {}
    for r in csv.DictReader(open(PROF)):
        k = (r['domain'], r['guide'])
        d = by.setdefault(k, dict(domain=r['domain'], guide=r['guide'], auroc=float(r['retrieval_auroc']),
                                  share=0.0, t_loss=[], o_loss=[]))
        if r['is_target'] == 'True':
            d['share'] += float(r['grad_share'])
            d['t_loss'].append(float(r['loss']))
        else:
            d['o_loss'].append(float(r['loss']))
    rows = list(by.values())
    for d in rows:
        d['ratio'] = np.mean(d['t_loss']) / np.mean(d['o_loss'])
    return [d for d in rows if not np.isnan(d['auroc'])]


def main():
    rows = heads()
    L = ['# Guide retrieval vs. target selection on the private batch (ResNet-18 client)', '',
         'One point per (dataset, guide head). x = AUROC with which the guide\'s query similarity ranks the target '
         'class on the auxiliary set; y = share of the last-layer weight-gradient norm the rigged head puts on the two '
         'target images of the private batch (uniform = 0.250). Spearman rank correlation; the per-dataset n is small, '
         'so read the pooled line and the sign agreement together.', '',
         '| scope | n heads | rho | p | heads above uniform | AUROC of those |', '|---|---|---|---|---|---|']
    for scope in ['all', 'medical', 'pathmnist', 'uav']:
        rr = rows if scope == 'all' else [r for r in rows if r['domain'] == scope]
        if len(rr) < 4:
            continue
        rho, p = stats.spearmanr([r['auroc'] for r in rr], [r['share'] for r in rr])
        above = [r for r in rr if r['share'] > 0.25]
        au = ', '.join(f"{r['auroc']:.2f}" for r in sorted(above, key=lambda r: -r['auroc'])) or '-'
        L.append(f'| {scope} | {len(rr)} | {rho:+.2f} | {p:.4f} | {len(above)}/{len(rr)} | {au} |')
    lo = [r for r in rows if r['auroc'] < 0.75]
    hi = [r for r in rows if r['auroc'] >= 0.75]
    L += ['', f'Heads whose guide has AUROC < 0.75: {sum(r["share"] > 0.25 for r in lo)}/{len(lo)} above uniform '
          f'(median share {np.median([r["share"] for r in lo]):.3f}). AUROC >= 0.75: '
          f'{sum(r["share"] > 0.25 for r in hi)}/{len(hi)} above uniform (median {np.median([r["share"] for r in hi]):.3f}).',
          '', '| dataset | guide | AUROC | target share | on-batch target/other loss |', '|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: (r['domain'], -r['auroc'])):
        L.append(f"| {r['domain']} | {r['guide']} | {r['auroc']:.2f} | {r['share']:.3f} | {r['ratio']:.2f}x |")
    with open(OUT, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
