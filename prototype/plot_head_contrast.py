"""Figure for controls D and E: per-encoder selectivity and target LPIPS under the rigged, plain and
query-swapped heads (means over 3 attack seeds, error bars = 95% t-CI), from the summarizer runs.csv files.

Usage: python prototype/plot_head_contrast.py
Output: results/client_encoder_plain/plain_vs_rigged.png (UAV best-match + Hungarian selectivity, target LPIPS;
        medical Hungarian), results/client_encoder_plain/plain_vs_rigged.csv
"""
import csv
import math
import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENC = ['clip', 'simclip2', 'simclip4', 'fare2', 'fare4', 'tecoa2', 'tecoa4']
LABEL = {'clip': 'CLIP', 'simclip2': 'SimCLIP-2', 'simclip4': 'SimCLIP-4', 'fare2': 'FARE-2', 'fare4': 'FARE-4', 'tecoa2': 'TeCoA-2', 'tecoa4': 'TeCoA-4'}
ARMS = [('rigged', '#1B2422', 'rigged head (solar / pneumonia query)'), ('plain', '#8A6A1F', 'plain head'), ('swap', '#0E6E66', 'swapped-query head')]


def load():
    rows = []
    for path, keep in [('results/client_encoder_plain/runs.csv', {'rigged', 'plain'}),
                       ('results/client_encoder_swap_qair_qeff/runs.csv', {'swap'})]:
        with open(os.path.join(ROOT, path)) as f:
            rows += [r for r in csv.DictReader(f) if r['arm'] in keep]
    return rows


def agg(rows, domain, arm, enc, key):
    v = np.array([float(r[key]) for r in rows if r['domain'] == domain and r['arm'] == arm and r['encoder'] == enc], float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float('nan'), float('nan')
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float('nan')
    return float(v.mean()), float(half)


def main():
    rows = load()
    panels = [('uav', 'selectivity', 'UAV: selectivity, best-match'), ('uav', 'selectivity_hung', 'UAV: selectivity, Hungarian'),
              ('uav', 'target_lpips', 'UAV: target LPIPS, best-match (lower = more leakage)'), ('medical', 'target_lpips_hung', 'Medical: target LPIPS, Hungarian')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    out = []
    x = np.arange(len(ENC)); w = 0.26
    for ax, (domain, key, title) in zip(axes.ravel(), panels):
        for i, (arm, color, label) in enumerate(ARMS):
            means, halves = zip(*[agg(rows, domain, arm, e, key) for e in ENC])
            ax.bar(x + (i - 1) * w, means, w, yerr=[0 if math.isnan(h) else h for h in halves], color=color, alpha=0.9, capsize=2, label=label)
            for e, m, h in zip(ENC, means, halves):
                out.append(dict(domain=domain, metric=key, arm=arm, encoder=e, mean=m, ci95_half=h))
        ax.set_xticks(x); ax.set_xticklabels([LABEL[e] for e in ENC], rotation=30, ha='right')
        ax.set_title(title, fontsize=10); ax.axhline(0, color='#999', lw=0.6)
        if 'selectivity' in key:
            ax.set_ylabel('non-target − target LPIPS')
        else:
            ax.set_ylabel('LPIPS')
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle('Controls D and E: same encoder and attack seed, three heads (mean ± 95% CI over 3 seeds)', fontsize=11)
    fig.tight_layout()
    path = os.path.join(ROOT, 'results', 'client_encoder_plain', 'plain_vs_rigged')
    fig.savefig(path + '.png', dpi=150)
    with open(path + '.csv', 'w', newline='') as f:
        wri = csv.DictWriter(f, fieldnames=list(out[0].keys())); wri.writeheader(); wri.writerows(out)
    print('wrote', path + '.png/.csv')


if __name__ == '__main__':
    main()
