"""LaTeX table of the per-head private-batch profile (supplement).

Reads results/complementarity/private_batch_profile_resnet18.csv and writes paper/cvpr/tables/profile_full.tex
(a tabular body only; the supplement wraps it).
Usage: python prototype/make_profile_table.py
"""
import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROF = os.path.join(ROOT, 'results', 'complementarity', 'private_batch_profile_resnet18.csv')
OUT = os.path.join(ROOT, 'paper', 'cvpr', 'tables', 'profile_full.tex')
DNAME = {'medical': 'ChestMNIST', 'pathmnist': 'PathMNIST', 'uav': 'UAVScenes'}
GNAME = {'clip': 'CLIP', 'simclip2': 'SimCLIP-2', 'simclip4': 'SimCLIP-4', 'fare2': 'FARE-2', 'fare4': 'FARE-4',
         'tecoa2': 'TeCoA-2', 'tecoa4': 'TeCoA-4', 'biomedclip': 'BiomedCLIP',
         'biomedclip (pneumonia_descriptive)': 'BiomedCLIP (descriptive query)',
         'biomedclip (pna_base)': 'BiomedCLIP (base query)', 'clip (sol_base)': 'CLIP (base query head)'}


def main():
    by = {}
    for r in csv.DictReader(open(PROF)):
        d = by.setdefault((r['domain'], r['guide']), dict(auroc=float(r['retrieval_auroc']), share=0.0,
                                                           t_own=[], o_own=[], t_loss=[], o_loss=[]))
        if r['is_target'] == 'True':
            d['share'] += float(r['grad_share']); d['t_own'].append(float(r['own_prob'])); d['t_loss'].append(float(r['loss']))
        else:
            d['o_own'].append(float(r['own_prob'])); d['o_loss'].append(float(r['loss']))
    L = ['\\begin{tabular}{llcccccc}', '\\toprule',
         'Dataset & Guide & AUROC & Target share & Own prob (targets) & Own prob (others) & Loss (targets) & Loss (others) \\\\',
         '\\midrule']
    for dom in ['medical', 'pathmnist', 'uav']:
        rows = sorted([(k[1], v) for k, v in by.items() if k[0] == dom], key=lambda kv: -kv[1]['auroc'])
        for i, (g, v) in enumerate(rows):
            name = f'\\multirow{{{len(rows)}}}{{*}}{{{DNAME[dom]}}}' if i == 0 else ''
            bold = v['share'] > 0.25
            share = f"\\textbf{{{v['share']:.3f}}}" if bold else f"{v['share']:.3f}"
            L.append(f"{name} & {GNAME.get(g, g)} & {v['auroc']:.2f} & {share} & {np.mean(v['t_own']):.3f} & "
                     f"{np.mean(v['o_own']):.3f} & {np.mean(v['t_loss']):.1f} & {np.mean(v['o_loss']):.1f} \\\\")
        L.append('\\midrule')
    L[-1] = '\\bottomrule'
    L.append('\\end{tabular}')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(L) + '\n')
    print(f'wrote {OUT} ({len(by)} heads)')


if __name__ == '__main__':
    main()
