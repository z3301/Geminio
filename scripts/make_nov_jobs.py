"""
Job lists for the November additions (second medical dataset + non-robust ViT-L comparators),
in priority order for scripts/run_job_queue.sh. One jobs.tsv per group plus the combined list.

  A  pathmnist_resnet    PathMNIST, ResNet-18 client: 8 guide heads (BiomedCLIP base + 7 CLIP-family) x 3 seeds,
                         plus the defense pairs on the base head (NBFU eps=5; IG; IG + NBFU).
                         Result dirs: results/pathmnist_{tag}_ho_s{seed}, results/pathmnist_vlm{guide}_ho_s{seed},
                         results/pathmnist_{tag}[_ig][_dpsgd_eps5.0]_ho_s{seed}.
  B  uav_comparators     UAV, ViT-L client on laion2b / dinov2 / in21k: rigged and plain heads x 3 seeds.
                         Result dirs: results/uav_sol_base_enc{enc}_ce[_plain]_s{seed}.
  C  pathmnist_vit       PathMNIST, ViT-L client on the 7 CLIP-family encoders, rigged heads x 3 seeds.
  D  medical_comparators ChestMNIST + PathMNIST, rigged heads on the 3 comparators x 3 seeds.
  E  plain_tail          PathMNIST plain heads (10 encoders) + ChestMNIST comparator plain heads, seed 42 first.

Only jobs whose head file exists are written; missing heads are listed. Usage:
    python scripts/make_nov_jobs.py
    GPUS="0 1 2 3" bash scripts/run_job_queue.sh results/nov_runs/jobs.tsv
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from prototype.domains import MEDICAL  # noqa: E402

SEEDS = [42, 123, 256]
ITERS = '24000'
CLIP_FAMILY = ['clip', 'simclip2', 'simclip4', 'fare2', 'fare4', 'tecoa2', 'tecoa4']
COMPARATORS = ['laion2b', 'dinov2', 'in21k']
TAG = MEDICAL['pathmnist']['tag']
PATH_DATA = './assets/pathmnist_heldout/'
DOMAINS = {
    'medical': dict(script='prototype/reconstruct_medical.py', query='pna_base', data='./assets/medical_heldout_pna/', extra=[]),
    'pathmnist': dict(script='prototype/reconstruct_medical.py', query=TAG, data=PATH_DATA, extra=['--dataset', 'pathmnist']),
    'uav': dict(script='prototype/reconstruct_uav.py', query='sol_base', data='./assets/uav_heldout_solar/', extra=[]),
}
missing = []


def exists(rel):
    ok = os.path.exists(os.path.join(REPO, rel))
    if not ok:
        missing.append(rel)
    return ok


def client_job(domain, enc, seed, plain=False):
    spec = DOMAINS[domain]
    head = f"malicious_models_client_encoder/{domain}_{enc}{'_plain' if plain else ''}.pt"
    if not exists(head):
        return None
    tag = f"ce{'_plain' if plain else ''}_s{seed}"
    out = f"./results/{domain}_{spec['query']}_enc{enc}_{tag}"
    return [out, spec['script'], '--geminio-query', spec['query'], '--model-path', head, '--client-encoder', enc,
            '--data-dir', spec['data'], '--attack', 'hfgradinv', '--max-iterations', ITERS, '--seed', str(seed),
            '--num-samples', '8', '--batch-tag', tag] + spec['extra']


def resnet_job(seed, query, head, attack='hfgradinv', eps=None):
    """PathMNIST ResNet-18 run; naming mirrors scripts/make_heldout_jobs.py."""
    spec = DOMAINS['pathmnist']
    if not exists(head):
        return None
    attack_tag = '' if attack == 'hfgradinv' else f'_{attack}'
    defense_tag = '' if eps is None else f'_dpsgd_eps{float(eps)}'
    tag = f'ho_s{seed}'
    out = f'./results/pathmnist_{query}{attack_tag}{defense_tag}_{tag}'
    args = [out, spec['script'], '--geminio-query', query, '--model-path', head, '--attack', attack,
            '--max-iterations', ITERS, '--seed', str(seed), '--data-dir', spec['data'], '--num-samples', '8',
            '--batch-tag', tag] + spec['extra']
    if eps is not None:
        args += ['--dpsgd-epsilon', str(eps), '--dpsgd-max-grad-norm', '1.0']
    return args


def write(name, jobs):
    out_dir = os.path.join(REPO, 'results', 'nov_runs')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{name}.tsv')
    with open(path, 'w') as f:
        for j in jobs:
            f.write('\t'.join(j) + '\n')
    print(f'{len(jobs):3d} jobs -> results/nov_runs/{name}.tsv')
    return jobs


def main():
    assert TAG, 'set the pathmnist target/tag in prototype/domains.py first'
    base_head = f'malicious_models_pathmnist/{TAG}.pt'
    A = []
    for seed in SEEDS:
        A.append(resnet_job(seed, TAG, base_head))
        for g in CLIP_FAMILY:
            A.append(resnet_job(seed, f'vlm{g}', f'malicious_models_pathmnist/{TAG}_{g}.pt'))
        A.append(resnet_job(seed, TAG, base_head, eps=5.0))
        A.append(resnet_job(seed, TAG, base_head, attack='ig'))
        A.append(resnet_job(seed, TAG, base_head, attack='ig', eps=5.0))
    B = [client_job('uav', e, s, p) for s in SEEDS for e in COMPARATORS for p in (False, True)]
    C = [client_job('pathmnist', e, s) for s in SEEDS for e in CLIP_FAMILY]
    D = [client_job(d, e, s) for s in SEEDS for d in ('medical', 'pathmnist') for e in COMPARATORS]
    E = [client_job('pathmnist', e, s, True) for s in SEEDS for e in CLIP_FAMILY + COMPARATORS]
    E += [client_job('medical', e, s, True) for s in SEEDS for e in COMPARATORS]
    groups = {'pathmnist_resnet': A, 'uav_comparators': B, 'pathmnist_vit': C, 'medical_comparators': D, 'plain_tail': E}
    combined = []
    for name, jobs in groups.items():
        jobs = [j for j in jobs if j]
        write(name, jobs)
        combined += jobs
    outs = [j[0] for j in combined]
    dupes = {o for o in outs if outs.count(o) > 1}
    assert not dupes, f'duplicate result dirs: {sorted(dupes)[:5]}'
    write('jobs', combined)
    if missing:
        print(f'\n{len(set(missing))} head files missing; their jobs were NOT written:')
        for m in sorted(set(missing)):
            print('   ', m)


if __name__ == '__main__':
    main()
