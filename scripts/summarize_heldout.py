"""
Summarize the held-out reruns (scripts/make_heldout_jobs.py) with 95% confidence intervals.

Each condition is a result dir name minus its _s{seed} suffix. Image metrics on the
target images come from scripts/summarize_ci.py: best-match (best reconstruction per
target image: min LPIPS, max PSNR; selectivity = non-target minus target LPIPS) and
Hungarian one-to-one matching (`_hung` columns; the matching that minimizes total LPIPS).
`uniq` is the number of distinct reconstructions best-match used out of the batch; when it
is 1 every image mapped to the same reconstruction and best-match selectivity means little.
Each condition is compared, seed by seed, with its reference:
  - defended runs (NBFU, clip-only, record-level, any clipping bound) vs the same run
    without the defense (same attack, query, batch, BatchNorm mode, backbone setting);
  - robust attacker VLMs vs vanilla CLIP.

Usage:
    python scripts/summarize_heldout.py --gpu 0
"""
import argparse
import csv
import json
import math
import os
import re
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from summarize_ci import ci, fmt, image_metrics, paired  # noqa: E402

JOBS = os.path.join(REPO, 'results', 'heldout_rerun', 'jobs.tsv')
METRIC_KEYS = ['target_lpips', 'target_psnr', 'selectivity', 'target_lpips_hung', 'selectivity_hung',
               'n_unique_best', 'lpips', 'attack_f1', 'avg_cos_sim']


def reference(cond):
    ref = re.sub(r'_dpsgd_(eps[\d.]+|cliponly)|_reclvl', '', cond)
    ref = re.sub(r'_hoC[\d.]+$', '_ho', ref)
    ref = re.sub(r'_vlm[a-z0-9]+_ho', '_vlmclip_ho', ref)
    ref = re.sub(r'_enc([a-z0-9]+)_ce_plain$', r'_enc\1_ce', ref)   # plain head vs rigged head, same encoder
    return None if ref == cond else ref


def block(cond):
    if '_enc' in cond:
        return '5. ViT-L client encoders'
    if '_vlm' in cond:
        return '2. Robust VLM as the attacker\'s guide'
    if re.search(r'_(pna|sol)_', cond):
        return '3. Language sweep'
    if re.search(r'_(pneumonia_descriptive|solar_panels|tum_base)(_ig|_gradinversion)?(_dpsgd_eps5\.0)?_ho$', cond):
        return '1. Headline: NBFU eps=5 across attacks'
    return '4. Robustness checks'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--jobs', default=JOBS, help='jobs.tsv to summarize (default: the held-out rerun list)')
    parser.add_argument('--out-dir', default=None, help='Where summary.md / runs.csv go (default: next to --jobs)')
    args = parser.parse_args()

    import lpips
    device = f'cuda:{args.gpu}' if args.gpu is not None and torch.cuda.is_available() else 'cpu'
    lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device)

    order, runs = [], {}
    for line in open(args.jobs):
        out = line.split('\t')[0]
        name = os.path.basename(out)
        cond, seed = re.match(r'(.+)_s(\d+)$', name).groups()
        if cond not in order:
            order.append(cond)
        mpath = os.path.join(REPO, out, 'metrics.json')
        if not os.path.exists(mpath):
            continue
        with open(mpath) as f:
            m = json.load(f)
        domain = name.split('_')[0]
        row = dict(condition=cond, seed=int(seed), domain=domain,
                   lpips=float(m['lpips']), attack_f1=float(m['attack_f1']), avg_cos_sim=float(m['avg_cos_sim']))
        row.update(image_metrics(domain, os.path.join(REPO, out), lpips_fn, device))
        runs.setdefault(cond, {})[int(seed)] = row

    if not runs:
        sys.exit('No finished runs yet.')
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.jobs))
    with open(os.path.join(out_dir, 'runs.csv'), 'w', newline='') as f:
        rows = [r for c in order for r in runs.get(c, {}).values()]
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = ['# Held-out reruns: mean ± 95% CI over seeds', '',
             'Target LPIPS/PSNR use the best-matching reconstruction for each target image. '
             'Hung. = Hungarian one-to-one matching (minimum total LPIPS); uniq = distinct '
             'reconstructions best-match used (mean over seeds; 8 = one-to-one, 1 = collapsed). '
             'Δ is the paired (same-seed) difference in best-match target LPIPS against the '
             'reference condition; positive Δ means worse reconstruction, i.e. less leakage.', '']
    current = None
    for cond in sorted(order, key=lambda c: (block(c), order.index(c))):
        if cond not in runs:
            continue
        if block(cond) != current:
            current = block(cond)
            lines += [f'## {current}', '',
                      '| Condition | n | Target LPIPS | Target PSNR | Selectivity | Target LPIPS (Hung.) | Selectivity (Hung.) | uniq | Attack F1 | Reference | Δ target LPIPS | p |',
                      '|---|---|---|---|---|---|---|---|---|---|---|---|']
        seeds = runs[cond]
        cells = [fmt(ci([r[k] for r in seeds.values()])['mean'], ci([r[k] for r in seeds.values()])['half'],
                     2 if k == 'target_psnr' else (1 if k == 'n_unique_best' else 3))
                 for k in ['target_lpips', 'target_psnr', 'selectivity', 'target_lpips_hung',
                           'selectivity_hung', 'n_unique_best', 'attack_f1']]
        ref = reference(cond)
        if ref and ref in runs:
            pd = paired({s: r['target_lpips'] for s, r in seeds.items()},
                        {s: r['target_lpips'] for s, r in runs[ref].items()})
            ref_txt, d_txt = ref, fmt(pd['diff'], pd['half'])
            p_txt = 'n/a' if math.isnan(pd['p']) else f"{pd['p']:.3f}"
        else:
            ref_txt, d_txt, p_txt = '—', '—', '—'
        lines.append(f"| {cond} | {len(seeds)} | {' | '.join(cells)} | {ref_txt} | {d_txt} | {p_txt} |")
    lines.append('')
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print(f'\n{sum(len(v) for v in runs.values())} runs summarized -> {out_dir}/summary.md, runs.csv')


if __name__ == '__main__':
    main()
