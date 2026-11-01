"""Place reported target LPIPS and selectivity against the reference values.

Reads results/complementarity/floor_baselines.csv plus the run tables, and
marks each condition below the 5th percentile, inside, or above the 95th
percentile of the dataset-sample reference.

Output: results/complementarity/baseline_comparison.md
Usage:  python prototype/compare_to_baselines.py
"""
import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results', 'complementarity', 'baseline_comparison.md')


def read(p):
    with open(os.path.join(ROOT, p)) as f:
        return list(csv.DictReader(f))


def verdict(v, lo, hi, lower_is_better=True):
    if lower_is_better:
        return 'below p5' if v < lo else ('inside' if v <= hi else 'above p95')
    return 'above p95' if v > hi else ('inside' if v >= lo else 'below p5')


def main():
    ref = {(r['domain'], r['metric']): (float(r['mean']), float(r['p5']), float(r['p95']))
           for r in read('results/complementarity/floor_baselines.csv') if r['set'] == 'dataset sample'}
    ho, ce = {}, {}
    for r in read('results/heldout_rerun/runs.csv'):
        ho.setdefault(r['condition'], []).append(r)
    for r in read('results/client_encoder_plain/runs.csv'):
        ce.setdefault((r['domain'], r['arm'], r['encoder']), []).append(r)

    L = ['# Reported results against the uninformative-set reference', '',
         'Reference = 8 images drawn from the head-training half, scored with the same matching code '
         '(`floor_baselines.md`). Target LPIPS below the 5th percentile is the only region where the '
         'metric itself evidences recovery; selectivity above the 95th percentile likewise. '
         'The reference is a property of the batch and is identical across arms, so paired contrasts '
         '(rigged vs plain vs swapped, guide vs guide) are unaffected by it.', '']
    for dom in ['medical', 'uav']:
        m, lo, hi = ref[(dom, 'target_lpips')]
        sm, slo, shi = ref[(dom, 'selectivity')]
        L += [f'## {dom}', '',
              f'Reference target LPIPS {m:.3f} [{lo:.3f}, {hi:.3f}]; '
              f'reference selectivity {sm:+.3f} [{slo:+.3f}, {shi:+.3f}].', '',
              '| condition | target LPIPS | vs reference | selectivity | vs reference |',
              '|---|---|---|---|---|']
        rows = []
        for cond, rr in sorted(ho.items()):
            if rr[0]['domain'] != dom or '_dpsgd' in cond or 'bntrain' in cond or 'frozen' in cond:
                continue
            rows.append((f'ResNet-18 · {cond}', rr))
        for (d, arm, enc), rr in sorted(ce.items()):
            if d == dom and arm in ('rigged', 'plain'):
                rows.append((f'ViT-L {enc} · {arm} head', rr))
        for label, rr in rows:
            t = np.mean([float(x['target_lpips']) for x in rr])
            s = np.mean([float(x['selectivity']) for x in rr])
            L.append(f'| {label} | {t:.3f} | {verdict(t, lo, hi)} | {s:+.3f} | '
                     f'{verdict(s, slo, shi, lower_is_better=False)} |')
        L.append('')
    with open(OUT, 'w') as f:
        f.write('\n'.join(L) + '\n')
    n = sum(1 for x in L if x.startswith('| ') and 'below p5' in x)
    print('\n'.join(L[:12]))
    print(f'... wrote {OUT}; {n} rows below the 5th percentile on target LPIPS or above on selectivity')


if __name__ == '__main__':
    main()
