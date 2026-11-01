"""Query-swap control (E): check the swap queries before spending GPU time.

For each domain the swap head is rigged for a query that names a DIFFERENT class present in
the same held-out batch (medical: pleural effusion, 1 of 8 images; UAV: airport runway, 2 of 8),
under the same attacker VLM (BiomedCLIP / vanilla CLIP). This reports (a) how well the swap
query retrieves its class on the auxiliary half (AUROC, precision in the top 10%) and (b) how
the swap query and the base query rank the 8 held-out batch images, so that the expected
re-targeting is visible before the attack runs. The UAV auxiliary half has no airstrip images,
so the UAV positive control is exploratory (AUROC undefined there).

Outputs: results/query_swap/swap_query_retrieval.md
Usage:   python prototype/analyze_swap_query_retrieval.py --gpu 0
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from prototype.train_medical import _get_text_features as medical_text  # noqa: E402
from prototype.train_uav import _get_text_features as uav_text  # noqa: E402
from prototype.analyze_vlm_retrieval import retrieval  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'query_swap', 'swap_query_retrieval.md')
SPEC = {
    'medical': dict(base='Any chest X-ray showing pneumonia', swap='Any chest X-ray showing pleural effusion', tag='qeff',
                    base_cls=6, swap_cls=2, names={6: 'pneumonia', 2: 'effusion'}, batch=os.path.join(ROOT, 'assets', 'medical_heldout_pna')),
    'uav': dict(base='aerial drone image showing solar panels on rooftops', swap='aerial drone image showing an airport runway', tag='qair',
                base_cls=11, swap_cls=7, names={11: 'solar_board', 7: 'airstrip'}, batch=os.path.join(ROOT, 'assets', 'uav_heldout_solar')),
}


def biomedclip_image_features(pils, device):
    import open_clip
    from prototype.vlm_medical import BIOMEDCLIP_NAME
    model, _, preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP_NAME)
    model = model.to(device).eval()
    with torch.no_grad():
        x = torch.stack([preprocess(p) for p in pils]).to(device)
        f = model.encode_image(x)
    return (f / f.norm(dim=-1, keepdim=True)).float().cpu()


def clip_image_features(pils, device):
    from prototype.vlm_simclip import get_image_features
    return get_image_features(pils, vlm='clip', device=device).float().cpu()


def batch_images(domain):
    files = sorted(glob.glob(os.path.join(SPEC[domain]['batch'], '*.png')), key=lambda f: int(os.path.basename(f).split('-')[0]))
    with open(SPEC[domain]['batch'].rstrip('/') + '_manifest.json') as f:
        manifest = json.load(f)['samples']
    labels = {s['file']: s['labels'] for s in manifest}
    return files, [labels[os.path.basename(f)] for f in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    lines = ['# Query-swap control: swap-query retrieval on the auxiliary set and the held-out batch', '']
    for domain, sp in SPEC.items():
        text_fn = medical_text if domain == 'medical' else uav_text
        vlm = 'biomedclip' if domain == 'medical' else 'clip'
        # auxiliary-set retrieval for base and swap classes
        if domain == 'medical':
            from prototype.dataset_medical import GeminioChestMNIST
            ds = GeminioChestMNIST(root=os.path.join(ROOT, 'data'), split='test', train=True, transform=None, size=64,
                                   embed_path=os.path.join(ROOT, 'data', 'medical-biomedclip-test.pt'))
            aux = np.asarray(ds.samples)
            emb = torch.load(os.path.join(ROOT, 'data', 'medical-biomedclip-test.pt')).float()[aux]
            cls = np.asarray(ds.targets)[aux]
            pos = {c: cls == c for c in (sp['base_cls'], sp['swap_cls'])}
        else:
            cache = torch.load(os.path.join(ROOT, 'results', 'client_encoder', 'cache', 'uav_clip.pt'), weights_only=True)
            emb = cache['embeds'].float()
            pos = {c: cache['targets'][:, c].numpy() > 0 for c in (sp['base_cls'], sp['swap_cls'])}
        lines += [f'## {domain}: attacker VLM {vlm}', '', f"base query: \"{sp['base']}\"  ·  swap query: \"{sp['swap']}\" (tag {sp['tag']})", '',
                  '| Query | class | AUROC on aux | prec@10% | base rate | positives in aux |', '|---|---|---|---|---|---|']
        txt = {}
        for name, q in [('base', sp['base']), ('swap', sp['swap'])]:
            txt[name] = text_fn(q, vlm, device).float().cpu()
            sims = (emb @ txt[name].t()).squeeze(-1).numpy()
            for c in (sp['base_cls'], sp['swap_cls']):
                r = retrieval(sims, pos[c])
                lines.append(f"| {name} | {sp['names'][c]} | {r['auroc']:.3f} | {r['prec_top10']:.4f} | {r['base_rate']:.4f} | {int(pos[c].sum())} |")
        # held-out batch ranks
        files, labels = batch_images(domain)
        pils = [Image.open(f).convert('RGB') for f in files]
        feats = biomedclip_image_features(pils, device) if domain == 'medical' else clip_image_features(pils, device)
        lines += ['', '| batch image | labels | rank under base query | rank under swap query |', '|---|---|---|---|']
        ranks = {}
        for name in ('base', 'swap'):
            s = (feats @ txt[name].t()).squeeze(-1).numpy()
            order = np.argsort(-s)
            ranks[name] = {int(i): int(k) + 1 for k, i in enumerate(order)}
        for i, (f, lab) in enumerate(zip(files, labels)):
            mark = ' **(swap class)**' if sp['names'][sp['swap_cls']] in lab else (' **(base target)**' if sp['names'][sp['base_cls']] in lab else '')
            lines.append(f"| {os.path.basename(f)} | {', '.join(lab)}{mark} | {ranks['base'][i]}/8 | {ranks['swap'][i]}/8 |")
        lines.append('')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines)); print(f'\nWrote {OUT}')


if __name__ == '__main__':
    main()
