"""Per-class retrieval scan of a MedMNIST dataset for every guide VLM.

Run before choosing the attack target of a new medical dataset: for each class and each
guide, how well the query similarity ranks that class on the auxiliary half (AUROC,
precision in the top 10%, base rate). Written so the target class is fixed on retrieval
alone, before any attack result exists (same logic as control B, which found that no
CLIP-family guide retrieves pneumonia on ChestMNIST).

Output: results/{domain}/retrieval_scan.{md,csv}
Usage:  python prototype/scan_medmnist_retrieval.py --domain pathmnist --gpu 0
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from prototype.analyze_vlm_retrieval import retrieval  # noqa: E402
from prototype.domains import GUIDES, MEDICAL, embed_path  # noqa: E402
from prototype.train_medical import _get_text_features  # noqa: E402

TEMPLATE = {'pathmnist': 'Any colorectal histopathology image showing {name}'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='pathmnist', choices=[d for d in MEDICAL if d != 'medical'])
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    spec = MEDICAL[args.domain]

    from prototype.dataset_medical import GeminioMedMNIST
    ds = GeminioMedMNIST(root=os.path.join(ROOT, 'data'), split='test', train=True, transform=None, size=64,
                         embed_path=embed_path(args.domain, spec['attacker_vlm'], os.path.join(ROOT, 'data')),
                         flag=spec['flag'])
    aux = np.asarray(ds.samples)
    targets = np.asarray(ds.targets)[aux]
    names = {int(k): v for k, v in ds.label_names.items()}

    rows = []
    for g in GUIDES:
        path = embed_path(args.domain, g, os.path.join(ROOT, 'data'))
        if not os.path.exists(path):
            print(f'no embeddings for {g}: {path}')
            continue
        emb = torch.load(path).float()[aux]
        for c, name in names.items():
            txt = _get_text_features(TEMPLATE[args.domain].format(name=name), g, device).float().cpu()
            sims = (emb @ txt.t()).squeeze(-1).numpy()
            rows.append(dict(domain=args.domain, guide=g, cls=c, name=name, **retrieval(sims, targets == c)))

    out_dir = os.path.join(ROOT, 'results', args.domain)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'retrieval_scan.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    guides = [g for g in GUIDES if any(r['guide'] == g for r in rows)]
    L = [f'# Retrieval scan: {spec["flag"]} auxiliary half ({len(aux)} images), query template '
         f'"{TEMPLATE[args.domain]}"', '',
         'AUROC of the query similarity for each class under each guide (precision in the top 10% in parentheses; '
         'base rate in the last column). Computed before any attack run.', '',
         '| class | ' + ' | '.join(guides) + ' | base rate |', '|---|' + '---|' * (len(guides) + 1)]
    for c, name in names.items():
        cells = []
        for g in guides:
            r = next(x for x in rows if x['guide'] == g and x['cls'] == c)
            cells.append(f"{r['auroc']:.2f} ({r['prec_top10']:.2f})")
        base = next(x for x in rows if x['cls'] == c)['base_rate']
        L.append(f'| {c} {name} | ' + ' | '.join(cells) + f' | {base:.3f} |')
    L.append('')
    with open(os.path.join(out_dir, 'retrieval_scan.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
