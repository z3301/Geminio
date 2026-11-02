"""
Summarize multi-seed reconstruction runs with 95% confidence intervals.

Experiments:
  client_encoder  FL client model = CLIP-family ViT-L/14 encoder + Geminio head
                  (results/{domain}_{query}_enc{encoder}_ce_s{seed}/, scripts/run_client_encoder.sh)
  vlm_attacker    ResNet-18 client, attacker VLM varied (Section 5 runs:
                  results/{medical_pneumonia_descriptive|uav_solar_panels}_vlm_{vlm}[_s{seed}]/)
  client_encoder_plain
                  Control D: honest (unweighted) head on the same encoders, runs
                  results/{domain}_{query}_enc{encoder}_ce_plain_s{seed}/, contrasted with the rigged
                  client_encoder runs (same encoder, same seed). Sign convention: plain minus rigged.
                  Duplicate-seed rigged runs (tag ce_dup_s42) give the same-seed floor.
  client_encoder_swap
                  Control E: head rigged for a different query (--tag medical=qeff,uav=qair), attacked on
                  the SAME solar/pneumonia batches; --alt-target names the swap query's class so the
                  swap target's own metrics are reported too (alt_* keys). Sign: swap minus rigged, and
                  swap minus plain when plain runs exist.

Per run it reads metrics.json and recomputes target-specific image metrics from the saved
tensors. Targets are pneumonia (ChestMNIST class 6) for medical and images containing
solar_board (UAVScenes class index 11) for UAV. Two matchings are reported:
  best-match  for each target image, the best-matching reconstruction in the batch
              (min LPIPS, max PSNR, chosen independently per image), then averaged
              over targets. Several ground-truth images may map to the same
              reconstruction; `n_unique_best` counts the distinct reconstructions used
              (8 = one-to-one, 1 = everything collapsed onto one image).
  Hungarian   one-to-one assignment of reconstructions to ground-truth images that
              minimizes total LPIPS (scipy linear_sum_assignment); PSNR is taken at the
              same pairing. Keys carry the suffix `_hung`. Use these when selectivity
              matters, since best-match selectivity is weakly interpretable whenever
              `n_unique_best` is small.
Both are architecture-independent, unlike Geminio's attack F1, which compares last-layer
gradients of the model under attack and can be near 1 for unrecognizable reconstructions
on CLIP features.

Intervals: mean +- t(0.975, n-1) * sd / sqrt(n) over seeds. Paired differences against
vanilla CLIP use the same seed, with a two-sided paired t-test. With 3 seeds (df = 2)
the t multiplier is 4.30, so intervals are wide.

Usage:
    python scripts/summarize_ci.py --experiment client_encoder --gpu 0
    python scripts/summarize_ci.py --experiment vlm_attacker
    python scripts/summarize_ci.py --experiment client_encoder_plain --gpu 0
    python scripts/summarize_ci.py --experiment client_encoder_swap --tag medical=qeff,uav=qair \\
        --alt-target medical=effusion,uav=airstrip --gpu 0
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch
from scipy import stats
from scipy.optimize import linear_sum_assignment

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

SEEDS = [42, 123, 256]
COMPARATORS = ['laion2b', 'dinov2', 'in21k']  # non-robust ViT-L client encoders (prototype/client_encoder.py)
ENCODERS = ['clip', 'simclip2', 'simclip4', 'fare2', 'fare4', 'tecoa2', 'tecoa4', 'slade2', 'slade4'] + COMPARATORS + ['resnet18']
PNEUMONIA = 6
SOLAR_BOARD_IDX = 11
CONTRAST = {'client_encoder_plain': 'plain', 'client_encoder_swap': 'swap'}
# Domains: ChestMNIST ('medical'), the second medical dataset ('pathmnist') and UAVScenes ('uav').
sys.path.insert(0, REPO)
from prototype.domains import MEDICAL  # noqa: E402
DOMAINS = ['medical', 'pathmnist', 'uav']
BASE_QUERY = {'medical': 'pna_base', 'uav': 'sol_base', 'pathmnist': MEDICAL['pathmnist']['tag']}
# ChestMNIST single-label indices (prototype/dataset_medical.py: first active label, 14 = normal)
MEDICAL_CLASSES = {'atelectasis': 0, 'cardiomegaly': 1, 'effusion': 2, 'infiltration': 3, 'mass': 4, 'nodule': 5,
                   'pneumonia': 6, 'pneumothorax': 7, 'consolidation': 8, 'edema': 9, 'emphysema': 10,
                   'fibrosis': 11, 'pleural': 12, 'hernia': 13, 'normal': 14}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

METRICS = [
    ('target_lpips', 'Target LPIPS (lower = more leakage)'),
    ('target_psnr', 'Target PSNR dB (higher = more leakage)'),
    ('nontarget_lpips', 'Non-target LPIPS'),
    ('selectivity', 'Selectivity: non-target minus target LPIPS (higher = targets favored)'),
    ('target_lpips_hung', 'Target LPIPS, Hungarian one-to-one matching (lower = more leakage)'),
    ('selectivity_hung', 'Selectivity, Hungarian matching (higher = targets favored)'),
    ('n_unique_best', 'Distinct reconstructions used by best-match (8 = one-to-one, 1 = collapsed)'),
    ('lpips', 'Batch LPIPS'),
    ('psnr', 'Batch PSNR'),
    ('attack_f1', 'Attack F1 (Geminio metric)'),
    ('avg_cos_sim', 'Avg last-layer grad cosine'),
]


def run_dirs(experiment, domains=DOMAINS):
    """Yield (domain, condition, seed, directory) for every expected run."""
    for domain in domains:
        for cond in ENCODERS:
            for seed in SEEDS:
                if experiment == 'client_encoder':
                    query = BASE_QUERY[domain]
                    if cond == 'resnet18':  # reference: ResNet-18 client, language-sweep head
                        d = f'results/{domain}_{query}_ce_resnet18_s{seed}'
                    else:
                        d = f'results/{domain}_{query}_enc{cond}_ce_s{seed}'
                else:
                    stem = 'medical_pneumonia_descriptive' if domain == 'medical' else 'uav_solar_panels'
                    d = f'results/{stem}_vlm_{cond}' + ('' if seed == 42 else f'_s{seed}')
                yield domain, cond, seed, os.path.join(REPO, d)


def class_index(domain, name):
    """Class index for a class name (or a bare integer string) in either domain."""
    if str(name).isdigit():
        return int(name)
    if domain == 'medical':
        return MEDICAL_CLASSES[name]
    if domain in MEDICAL:
        from medmnist import INFO
        return {v: int(k) for k, v in INFO[MEDICAL[domain]['flag']]['label'].items()}[name]
    from prototype.dataset_uav import IDX_TO_CLASS_NAME
    return {v: k for k, v in IDX_TO_CLASS_NAME.items()}[name]


def target_mask(domain, labels, cls=None):
    if domain in MEDICAL:
        if cls is None:
            cls = MEDICAL[domain]['target']
        return (labels.view(-1) == cls).numpy()
    return (labels[:, SOLAR_BOARD_IDX if cls is None else cls] > 0).numpy()


def to_unit(x):
    return torch.clamp(x.cpu().float() * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def image_metrics(domain, run_dir, lpips_fn, device, alt_cls=None):
    """Target metrics for the domain's primary target; with alt_cls, also for that class (alt_* keys)."""
    rec = torch.load(os.path.join(run_dir, 'reconstructed.pt'), map_location='cpu', weights_only=True)
    true = torch.load(os.path.join(run_dir, 'true_data.pt'), map_location='cpu', weights_only=True)
    rec_img, gt_img = to_unit(rec['data']), to_unit(true['data'])
    mask = target_mask(domain, true['labels'])
    n = gt_img.shape[0]
    with torch.no_grad():
        lp = torch.zeros(n, rec_img.shape[0])
        for g in range(n):
            lp[g] = lpips_fn(rec_img.to(device), gt_img[g:g + 1].expand_as(rec_img).to(device),
                             normalize=True).view(-1).cpu()
    mse = ((rec_img[None] - gt_img[:, None]) ** 2).flatten(2).mean(-1)  # [gt, rec]
    psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
    out = match_metrics(lp.numpy(), psnr.numpy(), mask)
    if alt_cls is not None:
        alt = match_metrics(lp.numpy(), psnr.numpy(), target_mask(domain, true['labels'], alt_cls))
        out.update({'alt_' + k: v for k, v in alt.items() if k != 'n_unique_best'})
    return out


def match_metrics(lp, psnr, mask):
    """Target/non-target image metrics from [n_gt, n_rec] LPIPS and PSNR matrices.

    Best-match keys (`target_lpips`, `target_psnr`, `nontarget_lpips`, `selectivity`)
    choose the best reconstruction independently for each ground-truth image, so several
    images may share one reconstruction; `n_unique_best` is the number of distinct
    reconstructions chosen. Hungarian keys (`*_hung`) use the one-to-one assignment that
    minimizes total LPIPS; PSNR is read at the same pairing. If there are fewer
    reconstructions than ground-truth images, unassigned images are NaN and ignored.
    """
    lp = np.asarray(lp, dtype=float)
    psnr = np.asarray(psnr, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    n_gt = lp.shape[0]

    best_idx = lp.argmin(axis=1)
    best_lp = lp[np.arange(n_gt), best_idx]
    best_psnr = psnr.max(axis=1)

    rows, cols = linear_sum_assignment(lp)
    hung_lp = np.full(n_gt, np.nan)
    hung_psnr = np.full(n_gt, np.nan)
    hung_lp[rows] = lp[rows, cols]
    hung_psnr[rows] = psnr[rows, cols]

    def mean_over(values, sel):
        v = values[sel]
        v = v[~np.isnan(v)]
        return float(v.mean()) if v.size else float('nan')

    out = dict(n_targets=int(mask.sum()), n_unique_best=int(np.unique(best_idx).size))
    out['target_lpips'] = mean_over(best_lp, mask)
    out['target_psnr'] = mean_over(best_psnr, mask)
    out['nontarget_lpips'] = mean_over(best_lp, ~mask)
    out['selectivity'] = out['nontarget_lpips'] - out['target_lpips']
    out['target_lpips_hung'] = mean_over(hung_lp, mask)
    out['target_psnr_hung'] = mean_over(hung_psnr, mask)
    out['nontarget_lpips_hung'] = mean_over(hung_lp, ~mask)
    out['selectivity_hung'] = out['nontarget_lpips_hung'] - out['target_lpips_hung']
    return out


def ci(values):
    v = np.asarray([x for x in values if not math.isnan(x)], dtype=float)
    if len(v) == 0:
        return dict(n=0, mean=float('nan'), sd=float('nan'), half=float('nan'))
    if len(v) == 1:
        return dict(n=1, mean=float(v[0]), sd=float('nan'), half=float('nan'))
    sd = float(v.std(ddof=1))
    half = float(stats.t.ppf(0.975, len(v) - 1) * sd / math.sqrt(len(v)))
    return dict(n=len(v), mean=float(v.mean()), sd=sd, half=half)


def paired(a, b):
    """a - b over matched seeds (dicts seed -> value)."""
    seeds = sorted(s for s in a if s in b and not math.isnan(a[s]) and not math.isnan(b[s]))
    if len(seeds) < 2:
        return dict(n=len(seeds), diff=float('nan'), half=float('nan'), p=float('nan'))
    d = np.array([a[s] - b[s] for s in seeds])
    c = ci(d)
    p = float(stats.ttest_rel([a[s] for s in seeds], [b[s] for s in seeds]).pvalue) if d.std() > 0 else float('nan')
    return dict(n=len(seeds), diff=c['mean'], half=c['half'], p=p)


def fmt(m, h, digits=3):
    if math.isnan(m):
        return 'n/a'
    return f'{m:.{digits}f}' if math.isnan(h) else f'{m:.{digits}f} ± {h:.{digits}f}'


# ---------------------------------------------------------------------------------------------
# Head-contrast experiments (controls D and E): the same encoder and seed under two heads.
# ---------------------------------------------------------------------------------------------
CONTRAST_ENCODERS = ['clip', 'simclip2', 'simclip4', 'fare2', 'fare4', 'tecoa2', 'tecoa4'] + COMPARATORS
CONTRAST_METRICS = [
    ('target_lpips', 'Target LPIPS, best-match (lower = more leakage)'),
    ('selectivity', 'Selectivity, best-match (non-target minus target LPIPS)'),
    ('target_lpips_hung', 'Target LPIPS, Hungarian one-to-one'),
    ('selectivity_hung', 'Selectivity, Hungarian one-to-one'),
    ('nontarget_lpips', 'Non-target LPIPS, best-match'),
    ('nontarget_lpips_hung', 'Non-target LPIPS, Hungarian'),
    ('n_unique_best', 'Distinct reconstructions used by best-match (8 = one-to-one)'),
    ('lpips', 'Batch LPIPS (metrics.json)'),
]
ALT_METRICS = [
    ('alt_target_lpips', 'Swap-query class: target LPIPS, best-match'),
    ('alt_selectivity', 'Swap-query class: selectivity, best-match'),
    ('alt_target_lpips_hung', 'Swap-query class: target LPIPS, Hungarian'),
    ('alt_selectivity_hung', 'Swap-query class: selectivity, Hungarian'),
]


def parse_map(text, default):
    """'medical=qeff,uav=qair' -> dict; None -> default."""
    if not text:
        return dict(default)
    out = dict(default)
    for part in text.split(','):
        k, v = part.split('=')
        out[k.strip()] = v.strip()
    return out


def ce_dir(domain, enc, seed, tag):
    query = BASE_QUERY[domain]
    return os.path.join(REPO, f'results/{domain}_{query}_enc{enc}_{tag}_s{seed}')


def head_stats(domain, enc, suffix):
    path = os.path.join(REPO, 'malicious_models_client_encoder', f'{domain}_{enc}{suffix}.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def wilcoxon_p(d):
    d = np.asarray([x for x in d if not math.isnan(x)], dtype=float)
    if len(d) < 5 or np.all(d == 0):
        return float('nan')
    try:
        return float(stats.wilcoxon(d, zero_method='wilcox').pvalue)
    except ValueError:
        return float('nan')


def contrast_block(lines, rows, domain, arm_a, arm_b, metrics):
    """Tables of arm_a minus arm_b per encoder (paired over seeds) with an aggregate line."""
    drows = [r for r in rows if r['domain'] == domain]
    encs = [e for e in CONTRAST_ENCODERS if any(r['encoder'] == e and r['arm'] == arm_a for r in drows)
            and any(r['encoder'] == e and r['arm'] == arm_b for r in drows)]
    if not encs:
        lines += [f'_No {arm_a} vs {arm_b} pairs for {domain}._', '']
        return
    for key, label in metrics:
        lines += [f'### {domain}: {label}', '',
                  f'| Encoder | {arm_b} | {arm_a} | Δ ({arm_a} − {arm_b}) ± 95% CI | paired p |', '|---|---|---|---|---|']
        enc_means, run_diffs = [], []
        for enc in encs:
            a = {r['seed']: r.get(key, float('nan')) for r in drows if r['encoder'] == enc and r['arm'] == arm_a}
            b = {r['seed']: r.get(key, float('nan')) for r in drows if r['encoder'] == enc and r['arm'] == arm_b}
            ca, cb, pd = ci(list(a.values())), ci(list(b.values())), paired(a, b)
            if not math.isnan(pd['diff']):
                enc_means.append(pd['diff'])
            run_diffs += [a[s] - b[s] for s in a if s in b and not math.isnan(a[s]) and not math.isnan(b[s])]
            p_txt = 'n/a' if math.isnan(pd['p']) else f"{pd['p']:.3f}"
            lines.append(f"| {enc} | {fmt(cb['mean'], cb['half'])} | {fmt(ca['mean'], ca['half'])} | "
                         f"{fmt(pd['diff'], pd['half'])} | {p_txt} |")
        if enc_means:
            pos_e = sum(d > 0 for d in enc_means)
            pos_r = sum(d > 0 for d in run_diffs)
            pw_e, pw_r = wilcoxon_p(enc_means), wilcoxon_p(run_diffs)
            lines.append(f"| **all** | | | **{np.mean(enc_means):+.3f}** ({pos_e}/{len(enc_means)} encoders > 0, "
                         f"Wilcoxon p = {'n/a' if math.isnan(pw_e) else f'{pw_e:.3f}'}) | "
                         f"{pos_r}/{len(run_diffs)} run pairs > 0, Wilcoxon p = "
                         f"{'n/a' if math.isnan(pw_r) else f'{pw_r:.3f}'} |")
        lines.append('')


def run_contrast(args, lpips_fn, device):
    kind = CONTRAST[args.experiment]
    if kind == 'plain':
        tags = {d: 'plain' for d in DOMAINS}
        alt = {}
    else:
        tags = parse_map(args.tag, {'medical': 'qeff', 'uav': 'qair'})
        alt_names = parse_map(args.alt_target, {'medical': 'effusion', 'uav': 'airstrip'})
        alt = {d: class_index(d, n) for d, n in alt_names.items()}
    out_dir = args.out_dir or os.path.join(REPO, 'results', 'client_encoder_plain' if kind == 'plain'
                                           else f"client_encoder_swap_{tags['uav']}_{tags['medical']}")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for domain in args.domains:
        arms = {'rigged': 'ce', kind: f'ce_{tags[domain]}', 'dup': 'ce_dup'}
        if kind == 'swap':
            arms['plain'] = 'ce_plain'
        for enc in CONTRAST_ENCODERS:
            for seed in SEEDS:
                for arm, tag in arms.items():
                    d = ce_dir(domain, enc, seed, tag)
                    mpath = os.path.join(d, 'metrics.json')
                    if not os.path.exists(mpath):
                        continue
                    with open(mpath) as f:
                        m = json.load(f)
                    row = dict(domain=domain, encoder=enc, seed=seed, arm=arm, run_dir=os.path.relpath(d, REPO),
                               lpips=float(m.get('lpips', float('nan'))), psnr=float(m.get('psnr', float('nan'))),
                               attack_f1=float(m.get('attack_f1', float('nan'))))
                    row.update(image_metrics(domain, d, lpips_fn, device, alt.get(domain)))
                    rows.append(row)
    if not rows:
        sys.exit('No finished runs found.')
    with open(os.path.join(out_dir, 'runs.csv'), 'w', newline='') as f:
        keys = sorted({k for r in rows for k in r},
                      key=lambda k: (k not in ('domain', 'encoder', 'seed', 'arm'), k))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    metrics = CONTRAST_METRICS + (ALT_METRICS if kind == 'swap' else [])
    lines = [f'# {args.experiment}: {kind} head vs rigged head, same encoder and seed', '',
             'All image metrics are re-scored from the saved tensors (best-match and Hungarian matching). '
             'Per-encoder Δ is paired over seeds (t-test, df = n−1); the aggregate line reports the mean of the '
             'per-encoder Δs with a Wilcoxon signed-rank test over encoders and over all run pairs. '
             f'Sign convention: {kind} minus rigged (positive target-LPIPS Δ = the {kind} head leaks less).', '']
    if kind == 'swap':
        lines += [f"Swap queries: medical tag {tags['medical']}, UAV tag {tags['uav']}. Primary target metrics are "
                  f"for the ORIGINAL targets (pneumonia / solar); alt_* metrics are for the swap query's class "
                  f"({alt_names['medical']} / {alt_names['uav']}).", '']
    for domain in args.domains:
        lines += [f'## {domain}: {kind} − rigged', '']
        contrast_block(lines, rows, domain, kind, 'rigged', metrics)
        if kind == 'swap' and any(r['arm'] == 'plain' and r['domain'] == domain for r in rows):
            lines += [f'## {domain}: swap − plain', '']
            contrast_block(lines, rows, domain, kind, 'plain', metrics)
        # same-seed floor from duplicate rigged runs
        dups = [r for r in rows if r['domain'] == domain and r['arm'] == 'dup']
        if dups:
            lines += [f'## {domain}: same-seed floor (duplicate rigged run minus original)', '',
                      '| Encoder | seed | ' + ' | '.join(k for k, _ in metrics) + ' |',
                      '|---|---|' + '---|' * len(metrics)]
            for dr in dups:
                orig = next((r for r in rows if r['domain'] == domain and r['arm'] == 'rigged'
                             and r['encoder'] == dr['encoder'] and r['seed'] == dr['seed']), None)
                if orig is None:
                    continue
                cells = [f"{dr.get(k, float('nan')) - orig.get(k, float('nan')):+.4f}" for k, _ in metrics]
                lines.append(f"| {dr['encoder']} | {dr['seed']} | " + ' | '.join(cells) + ' |')
            lines.append('')
        # head loss ratios
        suffix = '_plain' if kind == 'plain' else f"_{tags[domain]}"
        lines += [f'## {domain}: head loss ratios (top 10% by query similarity vs rest)', '',
                  f'| Encoder | rigged (base query) | {kind} head, own query | {kind} head, base query |', '|---|---|---|---|']
        for enc in CONTRAST_ENCODERS:
            r0, r1 = head_stats(domain, enc, ''), head_stats(domain, enc, suffix)
            f0 = lambda d, k: 'n/a' if k not in d else f"{d[k]:.2f}x"
            lines.append(f"| {enc} | {f0(r0, 'loss_ratio')} | {f0(r1, 'loss_ratio')} | {f0(r1, 'base_query_loss_ratio')} |")
        lines.append('')
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nWrote {out_dir}/runs.csv, summary.md ({len(rows)} runs)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True,
                        choices=['client_encoder', 'vlm_attacker'] + list(CONTRAST))
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--tag', default=None, help='client_encoder_swap: swap head tags, e.g. medical=qeff,uav=qair')
    parser.add_argument('--alt-target', default=None,
                        help='client_encoder_swap: class of the swap query, e.g. medical=effusion,uav=airstrip')
    parser.add_argument('--domains', nargs='+', default=['medical', 'uav'], choices=DOMAINS,
                        help='Domains to summarize (default: medical uav; add pathmnist for the second medical dataset)')
    parser.add_argument('--encoders', nargs='+', default=None,
                        help='Restrict the client-encoder conditions (default: every known encoder)')
    args = parser.parse_args()
    if args.encoders:
        global ENCODERS, CONTRAST_ENCODERS
        ENCODERS = [e for e in ENCODERS if e in args.encoders]
        CONTRAST_ENCODERS = [e for e in CONTRAST_ENCODERS if e in args.encoders]

    import lpips
    device = f'cuda:{args.gpu}' if args.gpu is not None and torch.cuda.is_available() else 'cpu'
    lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
    if args.experiment in CONTRAST:
        return run_contrast(args, lpips_fn, device)
    out_dir = args.out_dir or os.path.join(REPO, 'results', args.experiment)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for domain, cond, seed, run_dir in run_dirs(args.experiment, args.domains):
        mpath = os.path.join(run_dir, 'metrics.json')
        if not os.path.exists(mpath):
            continue
        with open(mpath) as f:
            m = json.load(f)
        row = dict(domain=domain, condition=cond, seed=seed, run_dir=os.path.relpath(run_dir, REPO))
        for key, _ in METRICS:
            row[key] = float(m.get(key, float('nan')))
        row.update(image_metrics(domain, run_dir, lpips_fn, device))
        if args.experiment == 'client_encoder':
            hpath = os.path.join(REPO, 'malicious_models_client_encoder', f'{domain}_{cond}.json')
            if os.path.exists(hpath):
                with open(hpath) as f:
                    row['loss_ratio'] = json.load(f)['loss_ratio']
        rows.append(row)

    if not rows:
        sys.exit('No finished runs found.')
    with open(os.path.join(out_dir, 'runs.csv'), 'w', newline='') as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('domain', 'condition', 'seed'), k))
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    lines = [f'# {args.experiment}: mean ± 95% CI over seeds', '',
             'Intervals use t(0.975, n-1); with n = 3 the multiplier is 4.30. '
             '"Δ vs CLIP" is the paired difference (same seed) with its 95% CI and paired t-test p.', '']
    summary = []
    for domain in args.domains:
        drows = [r for r in rows if r['domain'] == domain]
        if not drows:
            continue
        conds = [c for c in ENCODERS if any(r['condition'] == c for r in drows)]
        for key, label in METRICS:
            lines += [f'## {domain}: {label}', '',
                      '| Condition | n | Mean ± 95% CI | Δ vs CLIP ± 95% CI | p |', '|---|---|---|---|---|']
            ref = {r['seed']: r[key] for r in drows if r['condition'] == 'clip'}
            for cond in conds:
                vals = {r['seed']: r[key] for r in drows if r['condition'] == cond}
                c = ci(list(vals.values()))
                pd = paired(vals, ref) if cond != 'clip' else dict(diff=float('nan'), half=float('nan'), p=float('nan'))
                p_txt = 'n/a' if math.isnan(pd['p']) else f"{pd['p']:.3f}"
                d_txt = '—' if cond == 'clip' else fmt(pd['diff'], pd['half'])
                lines.append(f"| {cond} | {c['n']} | {fmt(c['mean'], c['half'])} | {d_txt} | {p_txt} |")
                summary.append(dict(domain=domain, metric=key, condition=cond, n=c['n'], mean=c['mean'],
                                    ci95_half=c['half'], sd=c['sd'], diff_vs_clip=pd['diff'],
                                    diff_ci95_half=pd['half'], paired_p=pd['p']))
            lines.append('')
        if args.experiment == 'client_encoder':
            lines += [f'## {domain}: rigged-head loss ratio (top 10% vs rest)', '', '| Condition | Loss ratio |', '|---|---|']
            for cond in conds:
                lr = next((r.get('loss_ratio') for r in drows if r['condition'] == cond), None)
                lines.append(f"| {cond} | {'n/a' if lr is None else f'{lr:.2f}x'} |")
            lines.append('')

    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nWrote {out_dir}/runs.csv, summary.csv, summary.md ({len(rows)} runs)')


if __name__ == '__main__':
    main()
