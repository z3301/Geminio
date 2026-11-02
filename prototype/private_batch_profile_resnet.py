"""Per-sample statistics of the private batch under the ResNet-18 guide heads.

Companion to private_batch_profile.py (ViT-L client): the same per-image loss,
own-label probability and last-layer gradient share, for the ResNet-18 client
whose head was rigged with each guide VLM. Joins each guide's retrieval AUROC
on the auxiliary set where a retrieval table exists.

Output: results/complementarity/private_batch_profile_resnet18.{md,csv}
Usage:  python prototype/private_batch_profile_resnet.py --gpu 0
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.models import GeminioResNet18  # noqa: E402
from prototype.domains import MEDICAL  # noqa: E402
from prototype.private_batch_profile import DOM, load_batch, per_sample  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'complementarity', 'private_batch_profile_resnet18')
GUIDES = ['clip', 'simclip2', 'simclip4', 'fare2', 'fare4', 'tecoa2', 'tecoa4']
TAG = MEDICAL['pathmnist']['tag']


def heads(domain):
    """{label: head path} for the guide heads the held-out tables used."""
    if domain == 'medical':
        h = {'biomedclip (pneumonia_descriptive)': 'malicious_models_medical_v2/pneumonia_descriptive.pt',
             'biomedclip (pna_base)': 'malicious_models_lang_sweep/medical/pna_base.pt'}
        h.update({g: f'malicious_models_medical_v2/Any_chest_X-ray_showing_pneumonia_{g}.pt' for g in GUIDES})
    elif domain == 'uav':
        stem = 'malicious_models_uav/aerial_drone_image_showing_solar_panels_on_rooftops'
        h = {'clip (sol_base)': 'malicious_models_lang_sweep/uav/sol_base.pt', 'clip': stem + '.pt'}
        h.update({g: f'{stem}_{g}.pt' for g in GUIDES if g != 'clip'})
    else:
        h = {'biomedclip': f'malicious_models_pathmnist/{TAG}.pt'}
        h.update({g: f'malicious_models_pathmnist/{TAG}_{g}.pt' for g in GUIDES})
    return {k: v for k, v in h.items() if os.path.exists(os.path.join(ROOT, v))}


def retrieval_auroc(domain):
    """Guide -> AUROC of the target class on the auxiliary set, if measured."""
    out = {}
    p = os.path.join(ROOT, 'results', 'complementarity', 'vlm_retrieval.csv')
    if domain in ('medical', 'uav') and os.path.exists(p):
        for r in csv.DictReader(open(p)):
            if r['domain'] == domain:
                out[r['guide']] = float(r['auroc'])
    p = os.path.join(ROOT, 'results', domain, 'retrieval_scan.csv')
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            if int(r['cls']) == MEDICAL[domain]['target']:
                out[r['guide']] = float(r['auroc'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))

    rows, L = [], ['# Per-sample statistics of the private batch, ResNet-18 client', '',
                   'Same quantities as private_batch_profile.md for the ResNet-18 client and the guide heads of the '
                   'held-out tables. Uniform gradient share for the two targets is 0.250. `AUROC` is the guide\'s '
                   'retrieval of the target class on the auxiliary set (control B / retrieval scan).', '']
    for domain, spec in DOM.items():
        hp = heads(domain)
        if not hp or spec['target'] is None or not os.path.isdir(os.path.join(ROOT, spec['batch'])):
            continue
        x, y, files, names = load_batch(domain)
        tgt = np.array([spec['target'] in n for n in names])
        au = retrieval_auroc(domain)
        model = GeminioResNet18(num_classes=spec['n_classes']).to(device).eval()
        with torch.no_grad():
            h = model.extractor(x.to(device))
        L += [f'## {domain}', '', '| guide head | AUROC | target share | target own prob | other own prob | '
              'target loss | other loss | on-batch loss ratio |', '|---|---|---|---|---|---|---|---|']
        for label, rel in hp.items():
            model.clf.load_state_dict(torch.load(os.path.join(ROOT, rel), map_location=device, weights_only=True))
            with torch.no_grad():
                a1 = torch.relu(model.clf[0](h))
                feat = torch.relu(model.clf[2](a1))
                logits = model.clf[4](feat)
            loss, own, gnorm = per_sample(domain, logits.float(), y.to(device).float() if domain == 'uav'
                                          else y.to(device), feat.float())
            share = gnorm / gnorm.sum()
            g = label.split(' ')[0]
            for i, f in enumerate(files):
                rows.append(dict(domain=domain, guide=label, image=f, is_target=bool(tgt[i]), loss=float(loss[i]),
                                 own_prob=float(own[i]), grad_norm=float(gnorm[i]), grad_share=float(share[i]),
                                 retrieval_auroc=au.get(g, float('nan'))))
            t, o = tgt, ~tgt
            L.append(f"| {label} | {au.get(g, float('nan')):.2f} | {share[t].sum():.3f} | {own[t].mean():.4f} | "
                     f"{own[o].mean():.4f} | {loss[t].mean():.3f} | {loss[o].mean():.3f} | "
                     f"{loss[t].mean() / loss[o].mean():.2f}x |")
        L.append('')
    with open(OUT + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUT + '.md', 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
