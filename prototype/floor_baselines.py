"""Reference metric values for uninformative reconstruction sets.

Scores several fixed image sets against each held-out batch with the same
matching code the real runs use (scripts/summarize_ci.match_metrics), to give
the scale on which a reported target LPIPS should be read.

Output: results/complementarity/floor_baselines.{md,csv}
Usage:  python prototype/floor_baselines.py --gpu 0
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from summarize_ci import match_metrics, target_mask, to_unit  # noqa: E402
from prototype.domains import MEDICAL  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'complementarity', 'floor_baselines')
RUN = {'medical': 'results/medical_pna_base_encclip_ce_s42',
       'pathmnist': 'results/pathmnist_tum_base_ho_s42',
       'uav': 'results/uav_sol_base_encclip_ce_s42'}
KEYS = ['target_lpips', 'selectivity', 'target_lpips_hung', 'selectivity_hung',
        'nontarget_lpips', 'n_unique_best']


def _to224(pil):
    return TF.to_tensor(pil.convert('RGB').resize((224, 224), Image.BICUBIC))


def sample_images(domain, rng, n):
    """n images drawn from the head-training half of the domain's dataset."""
    if domain in MEDICAL:
        from prototype.dataset_medical import GeminioChestMNIST, GeminioMedMNIST
        from prototype.domains import embed_path
        kw = dict(root=os.path.join(ROOT, 'data'), split='test', train=True, transform=None, size=64,
                  embed_path=embed_path(domain, MEDICAL[domain]['attacker_vlm'], os.path.join(ROOT, 'data')))
        ds = GeminioChestMNIST(**kw) if domain == 'medical' else GeminioMedMNIST(flag=MEDICAL[domain]['flag'], **kw)
        idx = rng.choice(len(ds.samples), n, replace=False)
        return torch.stack([_to224(Image.fromarray(ds.images[ds.samples[i]])) for i in idx])
    from prototype.dataset_uav import GeminioUAVScenes
    from prototype.train_uav import DATA_ROOT, SCENES
    ds = GeminioUAVScenes(data_root=DATA_ROOT, scenes=SCENES, embed_path=None,
                          transform=None, train=True)
    idx = rng.choice(len(ds.indices), n, replace=False)
    return torch.stack([_to224(Image.open(ds.image_paths[ds.indices[i]])) for i in idx])


def score(lp_fn, gt, cand, mask, device):
    """match_metrics for a candidate set scored against ground truth gt."""
    n = gt.shape[0]
    lp = torch.zeros(n, cand.shape[0])
    with torch.no_grad():
        for g in range(n):
            lp[g] = lp_fn(cand.to(device), gt[g:g + 1].expand_as(cand).to(device),
                          normalize=True).view(-1).cpu()
    mse = ((cand[None] - gt[:, None]) ** 2).flatten(2).mean(-1)
    psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
    return match_metrics(lp.numpy(), psnr.numpy(), mask)


def leave_one_out(lp_fn, gt, mask, device):
    """Each batch image scored against the other seven (self pairing excluded)."""
    n = gt.shape[0]
    lp = torch.zeros(n, n)
    with torch.no_grad():
        for g in range(n):
            lp[g] = lp_fn(gt.to(device), gt[g:g + 1].expand_as(gt).to(device),
                          normalize=True).view(-1).cpu()
    mse = ((gt[None] - gt[:, None]) ** 2).flatten(2).mean(-1)
    psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
    lp.fill_diagonal_(float('inf'))
    psnr.fill_diagonal_(-1.0)
    return match_metrics(lp.numpy(), psnr.numpy(), mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--draws', type=int, default=50)
    args = ap.parse_args()
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    import lpips
    lp_fn = lpips.LPIPS(net='alex', verbose=False).to(device)

    rows = []
    lines = ['# Reference values for uninformative reconstruction sets', '',
             'Same matching code as the real runs. `dataset sample` draws 8 images from the '
             'head-training half (mean over draws, with the 5th and 95th percentile). '
             'A reported target LPIPS is informative only below these values.', '']
    for domain, run in RUN.items():
        if not os.path.exists(os.path.join(ROOT, run, 'true_data.pt')):
            print(f'skip {domain}: {run} has no true_data.pt yet')
            continue
        true = torch.load(os.path.join(ROOT, run, 'true_data.pt'), map_location='cpu',
                          weights_only=True)
        gt = to_unit(true['data'])
        mask = target_mask(domain, true['labels'])
        rng = np.random.RandomState(0)
        draws = [score(lp_fn, gt, sample_images(domain, rng, gt.shape[0]), mask, device)
                 for _ in range(args.draws)]
        table = {'dataset sample': {k: (float(np.mean([d[k] for d in draws])),
                                        float(np.percentile([d[k] for d in draws], 5)),
                                        float(np.percentile([d[k] for d in draws], 95)))
                                    for k in KEYS}}
        loo = leave_one_out(lp_fn, gt, mask, device)
        table['other batch images'] = {k: (loo[k], float('nan'), float('nan')) for k in KEYS}
        for name, cand in [('uniform grey', torch.full_like(gt, 0.5)),
                           ('uniform noise', torch.rand_like(gt)),
                           ('batch mean', gt.mean(0, keepdim=True).expand_as(gt).clone())]:
            r = score(lp_fn, gt, cand, mask, device)
            table[name] = {k: (r[k], float('nan'), float('nan')) for k in KEYS}

        lines += [f'## {domain} ({int(mask.sum())} targets of {gt.shape[0]})', '',
                  '| set | ' + ' | '.join(KEYS) + ' |', '|---|' + '---|' * len(KEYS)]
        for name, vals in table.items():
            cells = []
            for k in KEYS:
                m, lo, hi = vals[k]
                cells.append(f'{m:.3f}' if np.isnan(lo) else f'{m:.3f} [{lo:.3f}, {hi:.3f}]')
                rows.append(dict(domain=domain, set=name, metric=k, mean=m, p5=lo, p95=hi))
            lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
        lines.append('')

    with open(OUT + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUT + '.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
