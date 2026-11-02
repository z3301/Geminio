"""Figure: ground truth vs reconstruction grids for selected runs.

Each run directory holds a_truth.jpg (8 ground-truth images, 224 px each in one row) and final_rec.jpg
(reconstructions on top of the truth). This script tiles the truth row once and one reconstruction row per
listed run, marks the two target columns, and writes paper/cvpr/fig/recon_{name}.pdf/.png.
Usage: python prototype/plot_reconstructions.py --name uav_backbone \
           --runs "CLIP, honest:results/uav_sol_base_encclip_ce_plain_s42" "TeCoA-4, rigged:results/uav_sol_base_enctecoa4_ce_s42"
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rows(run_dir):
    """(truth row, reconstruction row) as uint8 arrays [224, 8*224, 3]."""
    rec = np.asarray(Image.open(os.path.join(ROOT, run_dir, 'final_rec.jpg')).convert('RGB'))
    h = rec.shape[0] // 2
    return rec[h:], rec[:h]


def targets(run_dir):
    m = json.load(open(os.path.join(ROOT, run_dir, 'metrics.json')))
    data_dir = m['provenance']['data_dir'].rstrip('/')
    man = json.load(open(os.path.join(ROOT, data_dir + '_manifest.json')))['samples']
    return [i for i, s in enumerate(man) if s['is_target']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--runs', nargs='+', required=True, help='"label:run_dir" entries')
    args = ap.parse_args()
    items = [r.split(':', 1) for r in args.runs]
    truth, _ = rows(items[0][1])
    tg = targets(items[0][1])
    n = 1 + len(items)
    fig, axes = plt.subplots(n, 1, figsize=(7.0, 0.95 * n))
    panels = [('private batch', truth)] + [(lab, rows(d)[1]) for lab, d in items]
    for ax, (lab, img) in zip(np.atleast_1d(axes), panels):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_ylabel(lab, fontsize=7, rotation=0, ha='right', va='center', labelpad=4)
        w = img.shape[1] // 8
        for t in tg:
            ax.add_patch(plt.Rectangle((t * w + 2, 2), w - 5, img.shape[0] - 5, fill=False, ec='#d62728', lw=1.2))
    fig.subplots_adjust(left=0.16, right=0.995, top=0.995, bottom=0.005, hspace=0.06)
    out = os.path.join(ROOT, 'paper', 'cvpr', 'fig', f'recon_{args.name}')
    fig.savefig(out + '.pdf')
    fig.savefig(out + '.png', dpi=170)
    print(f'wrote {out}.pdf/.png; target columns {tg}')


if __name__ == '__main__':
    main()
