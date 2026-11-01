"""Control B follow-up: do robust guides pick auxiliary images that LOOK like the targets?

If retrieval of the target class does not explain why robust guides strengthen the medical
attack (analyze_vlm_retrieval.py), a second candidate is appearance: the images a guide ranks
highest might resemble the held-out target X-rays even when they are not labelled pneumonia.
For each guide this takes the 16 highest-similarity auxiliary images, reports their label mix,
and measures their LPIPS (AlexNet) to the two held-out pneumonia targets against 16 random
auxiliary images, then correlates the mean LPIPS with the guide's held-out attack strength.

Medical only (UAV auxiliary images need the UAVScenes root). Outputs:
results/complementarity/aux_appearance.{md,csv}
Usage: python prototype/analyze_aux_appearance.py --gpu 0
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from prototype.train_medical import _get_text_features, EMBED_PATHS  # noqa: E402
from prototype.analyze_vlm_retrieval import GUIDES, QUERY, PNEUMONIA, attack_strength  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'complementarity', 'aux_appearance')
TARGET_DIR = os.path.join(ROOT, 'assets', 'medical_heldout_pna')
TOPK = 16
LABELS = {0: 'atelectasis', 1: 'cardiomegaly', 2: 'effusion', 3: 'infiltration', 4: 'mass', 5: 'nodule', 6: 'pneumonia',
          7: 'pneumothorax', 8: 'consolidation', 9: 'edema', 10: 'emphysema', 11: 'fibrosis', 12: 'pleural', 13: 'hernia', 14: 'normal'}


def to_tensor(img):
    img = img.convert('RGB').resize((224, 224), Image.BICUBIC)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    import lpips
    lp = lpips.LPIPS(net='alex', verbose=False).to(device)

    from prototype.dataset_medical import GeminioChestMNIST
    ds = GeminioChestMNIST(root=os.path.join(ROOT, 'data'), split='test', train=True, transform=None, size=64,
                           embed_path=os.path.join(ROOT, 'data', 'medical-biomedclip-test.pt'))
    aux = np.asarray(ds.samples)
    targets_all = np.asarray(ds.targets)
    # held-out pneumonia targets (file names encode the label: {pos}-{label}.png)
    tfiles = sorted(f for f in glob.glob(os.path.join(TARGET_DIR, '*.png')) if os.path.basename(f).split('-')[1].split('.')[0] == str(PNEUMONIA))
    tgt = torch.stack([to_tensor(Image.open(f)) for f in tfiles]).to(device)
    print(f'{len(tfiles)} held-out pneumonia targets: {[os.path.basename(f) for f in tfiles]}')

    def mean_lpips(indices):
        imgs = torch.stack([to_tensor(Image.fromarray(ds.images[i])) for i in indices]).to(device)
        vals = []
        with torch.no_grad():
            for t in tgt:
                vals.append(lp(imgs, t[None].expand_as(imgs), normalize=True).view(-1).cpu())
        return float(torch.cat(vals).mean())

    rng = np.random.RandomState(0)
    rand_idx = aux[rng.choice(len(aux), TOPK, replace=False)]
    rand_lpips = mean_lpips(rand_idx)
    strength = attack_strength('medical')
    rows = []
    for g in ['biomedclip'] + GUIDES:
        emb = torch.load(os.path.join(ROOT, 'data', EMBED_PATHS[g])).float()[aux]
        txt = _get_text_features(QUERY['medical'], g, device).float().cpu()
        sims = (emb @ txt.t()).squeeze(-1).numpy()
        top = aux[np.argsort(-sims)[:TOPK]]
        labels = targets_all[top]
        mix = ', '.join(f'{LABELS[int(k)]} {int(v)}' for k, v in zip(*np.unique(labels, return_counts=True)))
        row = dict(guide=g, top_lpips_to_targets=mean_lpips(top), n_pneumonia_in_top=int((labels == PNEUMONIA).sum()),
                   label_mix=mix, random_lpips=rand_lpips, **strength.get(g, {}))
        rows.append(row)
        print(f"[{g}] top-{TOPK} LPIPS to targets {row['top_lpips_to_targets']:.3f} (random {rand_lpips:.3f}); {mix}")
    both = [r for r in rows if 'target_lpips_hung' in r]
    lines = ['# Control B follow-up: appearance of each guide\'s top-ranked auxiliary images', '',
             f'Top-{TOPK} auxiliary images by query similarity per guide; mean LPIPS (AlexNet, 224 px) to the two '
             f'held-out pneumonia targets vs {TOPK} random auxiliary images (seed 0). Attack strength = held-out '
             'target LPIPS of the ResNet-18 attack guided by that VLM.', '',
             '| Guide | top-16 LPIPS to targets | random LPIPS | pneumonia in top-16 | label mix | attack target LPIPS (Hung.) |',
             '|---|---|---|---|---|---|']
    for r in rows:
        th = r.get('target_lpips_hung')
        lines.append(f"| {r['guide']} | {r['top_lpips_to_targets']:.3f} | {r['random_lpips']:.3f} | {r['n_pneumonia_in_top']} | "
                     f"{r['label_mix']} | {'n/a' if th is None else f'{th:.3f}'} |")
    if len(both) >= 4:
        for key in ['target_lpips', 'target_lpips_hung']:
            rho, p = stats.spearmanr([r['top_lpips_to_targets'] for r in both], [r[key] for r in both])
            lines.append(f'\nSpearman top-16 LPIPS vs {key} over {len(both)} CLIP-family guides: rho = {rho:+.2f}, p = {p:.3f}')
    with open(OUT + '.csv', 'w', newline='') as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != 'guide', k))
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(OUT + '.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines)); print(f'\nWrote {OUT}.md/.csv')


if __name__ == '__main__':
    main()
