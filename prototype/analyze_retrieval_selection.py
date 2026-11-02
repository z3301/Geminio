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


def stratified_p(rows, n_perm=20000, seed=0):
    """Upper-tail p for the pooled Spearman rho when AUROC is shuffled within each dataset.

    Removes the between-dataset component: if the pooled association were only a reflection of the
    datasets differing in both retrieval and share, this null would reproduce it.
    """
    rng = np.random.RandomState(seed)
    x = np.array([r['auroc'] for r in rows])
    y = np.array([r['share'] for r in rows])
    dom = np.array([r['domain'] for r in rows])
    obs = stats.spearmanr(x, y).correlation
    hits = 0
    for _ in range(n_perm):
        xp = x.copy()
        for d in set(dom):
            m = dom == d
            xp[m] = rng.permutation(x[m])
        if stats.spearmanr(xp, y).correlation >= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def main():
    rows = heads()
    L = ['# Guide retrieval vs. target selection on the private batch (ResNet-18 client)', '',
         'One point per (dataset, guide head). x = AUROC with which the guide\'s query similarity ranks the target '
         'class on the auxiliary set; y = the two target images\' share of the summed per-example last-layer '
         'weight-gradient norms on the private batch (equal contribution = 0.250). Spearman rank correlation. '
         'The heads are NOT independent replicates (shared guide families, one private batch per dataset), the '
         'pooled p-value below is reported next to a permutation null that shuffles AUROC within each dataset, and '
         'the 0.30 column is descriptive, not a threshold: 0.253 (ChestMNIST BiomedCLIP) and 0.266 (PathMNIST CLIP) '
         'are within one-batch noise of equal contribution, so categorical "selects / does not select" language is '
         'avoided. The paired contrast against an honest head (see `private_batch_profile.md`) is the sharper '
         'quantity.', '',
         '| scope | n heads | rho | p | p (AUROC shuffled within dataset) | heads >= 0.30 | AUROC of those |',
         '|---|---|---|---|---|---|---|']
    for scope in ['all', 'medical', 'pathmnist', 'uav']:
        rr = rows if scope == 'all' else [r for r in rows if r['domain'] == scope]
        if len(rr) < 4:
            continue
        rho, p = stats.spearmanr([r['auroc'] for r in rr], [r['share'] for r in rr])
        pp = f'{stratified_p(rr):.4f}' if scope == 'all' else 'n/a'
        above = [r for r in rr if r['share'] >= 0.30]
        au = ', '.join(f"{r['auroc']:.2f}" for r in sorted(above, key=lambda r: -r['auroc'])) or '-'
        L.append(f'| {scope} | {len(rr)} | {rho:+.2f} | {p:.4f} | {pp} | {len(above)}/{len(rr)} | {au} |')
    lo = [r for r in rows if r['auroc'] < 0.75]
    hi = [r for r in rows if r['auroc'] >= 0.75]
    L += ['', f'Median share, guides with AUROC < 0.75: {np.median([r["share"] for r in lo]):.3f} (n = {len(lo)}); '
          f'AUROC >= 0.75: {np.median([r["share"] for r in hi]):.3f} (n = {len(hi)}). The 0.75 split is post hoc and '
          'has exceptions on both sides, so it is reported as a description of these 25 heads, not as a criterion.',
          '', '| dataset | guide | AUROC | target share | on-batch target/other loss |', '|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: (r['domain'], -r['auroc'])):
        L.append(f"| {r['domain']} | {r['guide']} | {r['auroc']:.2f} | {r['share']:.3f} | {r['ratio']:.2f}x |")
    with open(OUT, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
