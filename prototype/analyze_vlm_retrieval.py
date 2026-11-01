"""Control B: is the attacker-side robust-VLM effect explained by retrieval precision?

In the ResNet-18 pipeline the guide VLM only scores the auxiliary images while the rigged
head is trained, so any effect of swapping the guide must run through WHICH auxiliary images
get the high loss. The simplest story is that robust CLIPs retrieve the target class (pneumonia /
solar panels) better than vanilla CLIP. This script measures, for every guide, how well the
query similarity ranks the target class on the auxiliary half (AUROC, precision in the top 10%
that Geminio's loss ratio uses, base rate), and correlates that with the held-out attack
strength of the corresponding guide (results/heldout_rerun/runs.csv, {domain}_vlm{vlm}_ho).

Medical embeddings: data/medical-{vlm}-test.pt (all 22,433 test images; the auxiliary half is
GeminioChestMNIST(train=True).samples). UAV embeddings per guide live under the UAVScenes root;
when that root is unreadable only the vanilla-CLIP embeddings cached in
results/client_encoder/cache/uav_clip.pt are used and the UAV correlation is skipped.

Outputs: results/complementarity/vlm_retrieval.{md,csv}
Usage:   python prototype/analyze_vlm_retrieval.py --gpu 0
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from prototype.train_medical import _get_text_features as medical_text, EMBED_PATHS  # noqa: E402
from prototype.train_uav import _get_text_features as uav_text, DATA_ROOT, SCENES  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'complementarity', 'vlm_retrieval')
GUIDES = ['clip', 'simclip2', 'simclip4', 'fare2', 'fare4', 'tecoa2', 'tecoa4']
QUERY = {'medical': 'Any chest X-ray showing pneumonia', 'uav': 'aerial drone image showing solar panels on rooftops'}
PNEUMONIA, SOLAR_IDX = 6, 11


def auroc(scores, pos):
    pos = np.asarray(pos, bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = stats.rankdata(scores)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def retrieval(sims, pos):
    pos = np.asarray(pos, bool)
    thr = np.percentile(sims, 90)
    top = sims >= thr
    return dict(auroc=auroc(sims, pos), prec_top10=float(pos[top].mean()), base_rate=float(pos.mean()),
                n_pos_top10=int(pos[top].sum()), n_top10=int(top.sum()), n_aux=int(len(sims)))


def attack_strength(domain):
    """Held-out attack strength per guide: mean over seeds of target LPIPS (best-match and Hungarian)."""
    out = {}
    with open(os.path.join(ROOT, 'results', 'heldout_rerun', 'runs.csv')) as f:
        rows = list(csv.DictReader(f))
    for g in GUIDES:
        rr = [r for r in rows if r['condition'] == f'{domain}_vlm{g}_ho']
        if rr:
            out[g] = dict(target_lpips=np.mean([float(r['target_lpips']) for r in rr]),
                          target_lpips_hung=np.mean([float(r['target_lpips_hung']) for r in rr]))
    return out


def medical_rows(device):
    from prototype.dataset_medical import GeminioChestMNIST
    ds = GeminioChestMNIST(root=os.path.join(ROOT, 'data'), split='test', train=True, transform=None, size=64,
                           embed_path=os.path.join(ROOT, 'data', 'medical-biomedclip-test.pt'))
    aux = np.asarray(ds.samples)
    # Two positive definitions: the single-label target used for the held-out batches (pneumonia is
    # the primary, first-active label) and the multi-label flag (pneumonia among the image's labels).
    pos = {'primary': np.asarray(ds.targets)[aux] == PNEUMONIA,
           'any': np.asarray(ds.labels_multi)[aux, PNEUMONIA] == 1}
    rows = []
    for g in ['biomedclip'] + GUIDES:
        path = os.path.join(ROOT, 'data', EMBED_PATHS[g])
        if not os.path.exists(path):
            print(f'[medical] no embeddings for {g}: {path}')
            continue
        emb = torch.load(path).float()[aux]
        txt = medical_text(QUERY['medical'], g, device).float().cpu()
        sims = (emb @ txt.t()).squeeze(-1).numpy()
        row = dict(domain='medical', guide=g, **retrieval(sims, pos['primary']))
        row.update({'any_' + k: v for k, v in retrieval(sims, pos['any']).items() if k in ('auroc', 'prec_top10', 'base_rate')})
        rows.append(row)
        print(f"[medical] {g:10s} AUROC {row['auroc']:.3f} (any-label {row['any_auroc']:.3f})  prec@10% {row['prec_top10']:.4f} "
              f"(any {row['any_prec_top10']:.4f})  base {row['base_rate']:.4f} / {row['any_base_rate']:.4f}")
    return rows


def uav_rows(device):
    rows = []
    scene_tag = '_'.join(SCENES)
    cache = torch.load(os.path.join(ROOT, 'results', 'client_encoder', 'cache', 'uav_clip.pt'), weights_only=True)
    pos_cache = cache['targets'][:, SOLAR_IDX].numpy() > 0
    readable = os.access(DATA_ROOT, os.R_OK)
    for g in GUIDES:
        path = os.path.join(DATA_ROOT, f'uav_{g}_embeddings_{scene_tag}.pt')
        if readable and os.path.exists(path):
            from prototype.dataset_uav import GeminioUAVScenes
            ds = GeminioUAVScenes(data_root=DATA_ROOT, scenes=SCENES, embed_path=path, transform=None, train=True)
            emb = ds.img_embeds[ds.indices].float()
            pos = ds.targets[ds.indices][:, SOLAR_IDX] > 0
        elif g == 'clip':
            emb, pos = cache['embeds'].float(), pos_cache   # attacker-VLM embeddings cached by client_encoder.py
        else:
            print(f'[uav] embeddings for {g} unavailable (UAVScenes root {DATA_ROOT} unreadable: {not readable}); skipped')
            continue
        txt = uav_text(QUERY['uav'], g, device).float().cpu()
        sims = (emb @ txt.t()).squeeze(-1).numpy()
        rows.append(dict(domain='uav', guide=g, **retrieval(sims, pos)))
        print(f"[uav] {g:10s} AUROC {rows[-1]['auroc']:.3f}  prec@10% {rows[-1]['prec_top10']:.4f}  base {rows[-1]['base_rate']:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    rows = medical_rows(device) + uav_rows(device)
    lines = ['# Control B: retrieval precision of the attacker VLM on the auxiliary set', '',
             'Query similarity of each guide VLM ranks the auxiliary half of the data; AUROC and precision in the '
             'top 10% (the slice Geminio\'s loss ratio weights) for the target class, against the base rate. '
             'Attack strength is the held-out target LPIPS of the ResNet-18 attack guided by that VLM (mean of 3 seeds, '
             'lower = stronger). Spearman rho over the CLIP-family guides.', '']
    for domain in ['medical', 'uav']:
        dr = [r for r in rows if r['domain'] == domain]
        if not dr:
            continue
        strength = attack_strength(domain)
        for r in dr:
            r.update(strength.get(r['guide'], {}))
        any_cols = 'any_auroc' in dr[0]
        lines += [f'## {domain}', '', '| Guide | AUROC | prec@10% | base rate | positives in top 10% |'
                  + (' AUROC (any-label) | prec@10% (any) | base rate (any) |' if any_cols else '')
                  + ' target LPIPS (best) | target LPIPS (Hung.) |',
                  '|---|---|---|---|---|' + ('---|---|---|' if any_cols else '') + '---|---|']
        for r in dr:
            tl = r.get('target_lpips'); th = r.get('target_lpips_hung')
            extra = (f" {r['any_auroc']:.3f} | {r['any_prec_top10']:.4f} | {r['any_base_rate']:.4f} |" if any_cols else '')
            lines.append(f"| {r['guide']} | {r['auroc']:.3f} | {r['prec_top10']:.4f} | {r['base_rate']:.4f} | "
                         f"{r['n_pos_top10']}/{r['n_top10']} |{extra} {'n/a' if tl is None else f'{tl:.3f}'} | {'n/a' if th is None else f'{th:.3f}'} |")
        both = [r for r in dr if r['guide'] in GUIDES and 'target_lpips' in r]
        if len(both) >= 4:
            for key in ['target_lpips', 'target_lpips_hung']:
                for pred in (['auroc', 'prec_top10', 'any_auroc', 'any_prec_top10'] if 'any_auroc' in both[0] else ['auroc', 'prec_top10']):
                    x = [r[pred] for r in both]; y = [r[key] for r in both]
                    rho, p = stats.spearmanr(x, y)
                    lines.append(f'\nSpearman {pred} vs {key} over {len(both)} CLIP-family guides: rho = {rho:+.2f}, p = {p:.3f}')
        else:
            lines.append(f'\n_Correlation skipped: only {len(both)} guides with both retrieval and attack numbers._')
        lines.append('')
    with open(OUT + '.csv', 'w', newline='') as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('domain', 'guide'), k))
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(OUT + '.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines)); print(f'\nWrote {OUT}.md/.csv')


if __name__ == '__main__':
    main()
