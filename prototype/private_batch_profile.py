"""Per-sample statistics of the private batch under each trained head.

For every (encoder, head) pair this records, for each of the 8 held-out
images: its loss, the probability the head assigns to its own label, and the
norm of its contribution to the last-layer weight gradient. Lets the
per-image quantities be read directly instead of inferred from auxiliary-set
summaries.

Output: results/complementarity/private_batch_profile.{md,csv}
Usage:  python prototype/private_batch_profile.py --gpu 0
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from prototype.client_encoder import ClipEncoderClassifier, make_head  # noqa: E402

from prototype.domains import MEDICAL  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'complementarity', 'private_batch_profile')
HEAD_DIR = os.path.join(ROOT, 'malicious_models_client_encoder')
ENC = ['clip', 'simclip2', 'simclip4', 'fare2', 'fare4', 'tecoa2', 'tecoa4', 'laion2b', 'dinov2', 'in21k']
DOM = {
    'medical': dict(batch='assets/medical_heldout_pna', n_classes=15, swap='qeff',
                    target='pneumonia', alt='effusion'),
    'pathmnist': dict(batch=MEDICAL['pathmnist']['batch'], n_classes=9, swap=None,
                      target=MEDICAL['pathmnist']['target_name'], alt=None),
    'uav': dict(batch='assets/uav_heldout_solar', n_classes=18, swap='qair',
                target='solar_board', alt='airstrip'),
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_batch(domain):
    """The 8 held-out images, their labels and their manifest label names."""
    spec = DOM[domain]
    d = os.path.join(ROOT, spec['batch'])
    with open(d.rstrip('/') + '_manifest.json') as f:
        manifest = json.load(f)['samples']
    by_file = {m['file']: m for m in manifest}
    files = sorted(glob.glob(os.path.join(d, '*.png')),
                   key=lambda f: int(os.path.basename(f).split('-')[0]))
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    x = torch.stack([tf(Image.open(f).convert('RGB')) for f in files])
    names = [by_file[os.path.basename(f)]['labels'] for f in files]
    if domain in MEDICAL:
        from scripts.summarize_ci import class_index
        y = torch.tensor([class_index(domain, n[0]) for n in names])
    else:
        from prototype.dataset_uav import IDX_TO_CLASS_NAME
        idx = {v: k for k, v in IDX_TO_CLASS_NAME.items()}
        y = torch.zeros(len(files), DOM['uav']['n_classes'])
        for i, n in enumerate(names):
            for lab in n:
                y[i, idx[lab]] = 1.0
    return x, y, [os.path.basename(f) for f in files], names


def head_paths(domain, enc):
    s = DOM[domain]['swap']
    out = {'rigged': f'{domain}_{enc}.pt', 'plain': f'{domain}_{enc}_plain.pt'}
    if s:
        out['swap'] = f'{domain}_{enc}_{s}.pt'
    return out


def per_sample(domain, logits, y, feat):
    """loss, own-label probability, and last-layer weight-gradient norm per image.

    The last-layer weight gradient for image i is the outer product of its
    output residual with its penultimate activation, so its Frobenius norm is
    the product of the two norms.
    """
    if domain in MEDICAL:
        yl = y.long()
        p = torch.softmax(logits, dim=1)
        loss = F.cross_entropy(logits, yl, reduction='none')
        own = p[torch.arange(len(yl)), yl]
        resid = p.clone()
        resid[torch.arange(len(yl)), yl] -= 1.0
    else:
        p = torch.sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction='none').sum(1)
        own = torch.stack([p[i][y[i] > 0].mean() if (y[i] > 0).any() else p[i].mean()
                           for i in range(len(y))])
        resid = p - y
    gnorm = resid.norm(dim=1) * feat.norm(dim=1)
    return (loss.detach().cpu().numpy(), own.detach().cpu().numpy(),
            gnorm.detach().cpu().numpy())


def write(rows):
    with open(OUT + '.csv', 'w', newline='') as f:
        c = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        c.writeheader()
        c.writerows(rows)
    L = ['# Per-sample statistics of the private batch', '',
         'Uniform gradient share would be 1/8 = 0.125; the two query-matched images together '
         'would hold 0.250. `own prob` is the probability the head assigns to the image own label '
         '(mean over its positive labels for the multi-label UAV case).', '']
    for domain in DOM:
        dr = [r for r in rows if r['domain'] == domain]
        if not dr:
            continue
        L += [f'## {domain}', '',
              '| encoder | head | target share | alt-class share | target own prob | other own prob | target loss | other loss |',
              '|---|---|---|---|---|---|---|---|']
        for enc in ENC:
            for arm in ['rigged', 'plain', 'swap']:
                rr = [r for r in dr if r['encoder'] == enc and r['head'] == arm]
                if not rr:
                    continue
                t = [r for r in rr if r['is_target']]
                o = [r for r in rr if not r['is_target']]
                a = [r for r in rr if r['is_alt']]
                import numpy as _np
                L.append(f"| {enc} | {arm} | {sum(r['grad_share'] for r in t):.3f} | "
                         f"{sum(r['grad_share'] for r in a):.3f} | "
                         f"{_np.mean([r['own_prob'] for r in t]):.4f} | "
                         f"{_np.mean([r['own_prob'] for r in o]):.4f} | "
                         f"{_np.mean([r['loss'] for r in t]):.3f} | "
                         f"{_np.mean([r['loss'] for r in o]):.3f} |")
        L.append('')
    with open(OUT + '.md', 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))

    rows = []
    for domain, spec in DOM.items():
        if spec['target'] is None or not os.path.isdir(os.path.join(ROOT, spec['batch'])):
            print(f'skip {domain}: no target class or batch yet')
            continue
        x, y, files, names = load_batch(domain)
        tgt = np.array([spec['target'] in n for n in names])
        alt = np.array([spec['alt'] in n for n in names]) if spec['alt'] else np.zeros(len(names), bool)
        for enc in ENC:
            if not any(os.path.exists(os.path.join(HEAD_DIR, f)) for f in head_paths(domain, enc).values()):
                continue
            model = ClipEncoderClassifier(enc, num_classes=spec['n_classes']).to(device).eval()
            with torch.no_grad():
                h = model.encode(x.to(device))
            for arm, fname in head_paths(domain, enc).items():
                path = os.path.join(HEAD_DIR, fname)
                if not os.path.exists(path):
                    continue
                head = make_head(h.shape[1], spec['n_classes']).to(device).eval()
                head.load_state_dict(torch.load(path, map_location=device))
                with torch.no_grad():
                    a1 = torch.relu(head[0](h))
                    feat = torch.relu(head[2](a1))
                    logits = head[4](feat)
                loss, own, gnorm = per_sample(domain, logits.float(), y.to(device).float()
                                              if domain == 'uav' else y.to(device), feat.float())
                # 'gnorm' is the last-layer weight-gradient norm; 'share' its fraction of the batch total
                share = gnorm / gnorm.sum()
                for i, f in enumerate(files):
                    rows.append(dict(domain=domain, encoder=enc, head=arm, image=f,
                                     is_target=bool(tgt[i]), is_alt=bool(alt[i]),
                                     loss=float(loss[i]), own_prob=float(own[i]),
                                     grad_norm=float(gnorm[i]), grad_share=float(share[i])))
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    write(rows)


if __name__ == '__main__':
    main()
