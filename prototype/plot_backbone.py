"""Figure: client backbone x head type on UAV. Target LPIPS and selectivity per encoder, rigged vs honest head.

Reads results/client_encoder_plain/runs.csv (arms 'rigged' and 'plain'; regenerate it with
`scripts/summarize_ci.py --experiment client_encoder_plain --domains uav medical pathmnist` so the comparator
encoders are included once their runs exist). Writes paper/cvpr/fig/backbone_uav.pdf/.png.
Usage: python prototype/plot_backbone.py [--domain uav]
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'results', 'client_encoder_plain', 'runs.csv')
ORDER = ['clip', 'laion2b', 'dinov2', 'in21k', 'simclip2', 'simclip4', 'fare2', 'fare4', 'tecoa2', 'tecoa4']
NAME = {'clip': 'CLIP', 'laion2b': 'CLIP\nLAION-2B', 'dinov2': 'DINOv2', 'in21k': 'ViT-L\nIN-21k',
        'simclip2': 'SimCLIP-2', 'simclip4': 'SimCLIP-4', 'fare2': 'FARE-2', 'fare4': 'FARE-4',
        'tecoa2': 'TeCoA-2', 'tecoa4': 'TeCoA-4'}
ROBUST = {'simclip2', 'simclip4', 'fare2', 'fare4', 'tecoa2', 'tecoa4'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='uav')
    ap.add_argument('--ref', type=float, nargs=2, default=(0.485, 0.571), help='reference p5 p95 for target LPIPS')
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(RUNS)) if r['domain'] == args.domain and r['arm'] in ('rigged', 'plain')]
    encs = [e for e in ORDER if any(r['encoder'] == e for r in rows)]

    def stat(enc, arm, key):
        v = [float(r[key]) for r in rows if r['encoder'] == enc and r['arm'] == arm]
        return (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0) if v else (np.nan, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    x = np.arange(len(encs))
    w = 0.38
    for ax, key, ylabel in [(axes[0], 'target_lpips', 'Target LPIPS (lower = more leakage)'),
                            (axes[1], 'selectivity_hung', 'Selectivity, Hungarian matching')]:
        for k, (arm, col, lab) in enumerate([('plain', '0.75', 'honest head'), ('rigged', '#d62728', 'rigged head')]):
            m = [stat(e, arm, key) for e in encs]
            ax.bar(x + (k - 0.5) * w, [a for a, _ in m], w, yerr=[b for _, b in m], color=col, label=lab,
                   edgecolor='k', linewidth=0.4, capsize=2, error_kw=dict(lw=0.6))
        if key == 'target_lpips':
            ax.axhspan(args.ref[0], args.ref[1], color='#1f77b4', alpha=0.12, lw=0)
            ax.text(len(encs) - 0.5, args.ref[1] + 0.01, 'random-draw reference', fontsize=6.5, ha='right', color='#1f77b4')
            ax.set_ylim(0.3, 0.85)
        else:
            ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([NAME[e].replace('\n', ' ') for e in encs], fontsize=6.5, rotation=30, ha='right')
        for t, e in zip(ax.get_xticklabels(), encs):
            t.set_color('#8c1515' if e in ROBUST else 'k')
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.tick_params(axis='y', labelsize=7)
    axes[0].legend(fontsize=7, frameon=False, loc='upper right')
    fig.tight_layout()
    out = os.path.join(ROOT, 'paper', 'cvpr', 'fig', f'backbone_{args.domain}')
    fig.savefig(out + '.pdf')
    fig.savefig(out + '.png', dpi=200)
    print(f'wrote {out}.pdf/.png ({len(encs)} encoders: {encs})')


if __name__ == '__main__':
    main()
