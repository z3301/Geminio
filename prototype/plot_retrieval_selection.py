"""Figure: guide retrieval AUROC vs. target gradient share, one point per guide head.

Reads results/complementarity/private_batch_profile_resnet18.csv (ResNet-18 client; AUROC joined there)
and writes paper/cvpr/fig/retrieval_vs_share.pdf and .png.
Usage: python prototype/plot_retrieval_selection.py
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROF = os.path.join(ROOT, 'results', 'complementarity', 'private_batch_profile_resnet18.csv')
OUT = os.path.join(ROOT, 'paper', 'cvpr', 'fig', 'retrieval_vs_share')
LABEL = {'medical': 'ChestMNIST (pneumonia)', 'pathmnist': 'PathMNIST (tumor)', 'uav': 'UAVScenes (solar panels)'}
MARK = {'medical': 'o', 'pathmnist': 's', 'uav': '^'}
COL = {'medical': '#1f77b4', 'pathmnist': '#d62728', 'uav': '#2ca02c'}


def main():
    pts = {}
    for r in csv.DictReader(open(PROF)):
        k = (r['domain'], r['guide'])
        d = pts.setdefault(k, dict(domain=r['domain'], guide=r['guide'], auroc=float(r['retrieval_auroc']), share=0.0))
        if r['is_target'] == 'True':
            d['share'] += float(r['grad_share'])
    pts = [p for p in pts.values() if not np.isnan(p['auroc'])]
    x = np.array([p['auroc'] for p in pts])
    y = np.array([p['share'] for p in pts])
    rho, pval = stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.axhline(0.25, color='0.5', lw=0.8, ls='--')
    ax.text(0.305, 0.258, 'uniform share', fontsize=7, color='0.35')
    ax.axhline(0.0, color='0.8', lw=0.5)
    for dom in ['medical', 'pathmnist', 'uav']:
        sel = [p for p in pts if p['domain'] == dom]
        ax.scatter([p['auroc'] for p in sel], [p['share'] for p in sel], marker=MARK[dom], s=28, color=COL[dom],
                   label=LABEL[dom], edgecolor='k', linewidth=0.4, zorder=3)
    for p in pts:
        if p['guide'].startswith('biomedclip') and p['domain'] == 'pathmnist':
            ax.annotate('BiomedCLIP', (p['auroc'], p['share']), xytext=(-46, 6), textcoords='offset points', fontsize=7)
        if p['guide'] == 'tecoa4' and p['domain'] == 'uav':
            ax.annotate('TeCoA-4', (p['auroc'], p['share']), xytext=(4, -10), textcoords='offset points', fontsize=7)
    ax.set_xlabel('Guide retrieval AUROC (target class, auxiliary set)', fontsize=8)
    ax.set_ylabel('Target share of last-layer gradient', fontsize=8)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.0, 0.55)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc='upper left', frameon=False)
    ax.set_title(f'Spearman $\\rho$ = {rho:+.2f} over {len(pts)} heads ($p$ = {pval:.4f})', fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + '.pdf')
    fig.savefig(OUT + '.png', dpi=200)
    print(f'wrote {OUT}.pdf/.png  rho={rho:+.2f} p={pval:.4f} n={len(pts)}')


if __name__ == '__main__':
    main()
