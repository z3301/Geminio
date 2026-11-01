"""
Job lists for the complementarity controls (re-created 2026-10-25 after the Oct 23 storage
restore removed the Oct 17-19 originals). Three experiments, one jobs.tsv each, plus a
combined list in priority order for scripts/run_job_queue.sh:

  D  plain-head control    results/plain_head/jobs.tsv
     Honest (unweighted) head on each of the 7 frozen client encoders, attacked exactly like the
     rigged heads of results/client_encoder (HF-GradInv, 24k iterations, held-out batches,
     seeds 42/123/256). Result dirs: results/{domain}_{query}_enc{enc}_ce_plain_s{seed}.
     Two duplicate-seed rigged jobs (medical/clip and uav/clip, seed 42, tag ce_dup_s42) measure
     the same-seed floor of the ViT-L attack.
  C  temperature sweep     results/temp_sweep/jobs.tsv
     ResNet-18 heads at T in {10,30,300,1000} (rep 1) and {10,30,100,300,1000} (rep 2, second
     init), base query, held-out batch, 3 seeds. Result dirs: results/{domain}_{tag}_T{T}[r2]_s{seed}.
     T = 100 rep 1 is the existing results/{domain}_{tag}_ho_s{seed} run (same head).
  E  query-swap control    results/query_swap/jobs.tsv
     Heads rigged for a different query under the same attacker VLM (medical: pleural effusion,
     tag qeff; UAV: airport runway, tag qair), attacked on the SAME solar/pneumonia batches.
     Result dirs: results/{domain}_{query}_enc{enc}_ce_{tag}_s{seed}.

Only jobs whose head file exists are written; missing heads are listed so nothing silently
drops out. Usage:
    python scripts/make_rerun_jobs.py
    GPUS="4 5 7" bash scripts/run_job_queue.sh results/temp_sweep/jobs.tsv      # C, ResNet-18
    GPUS="2 3 6" bash scripts/run_job_queue.sh results/rerun_oct24/jobs.tsv     # D then E, ViT-L
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [42, 123, 256]
ITERS = '24000'
ENCODERS = ['clip', 'simclip2', 'simclip4', 'fare2', 'fare4', 'tecoa2', 'tecoa4']
SWAP_TAG = {'medical': 'qeff', 'uav': 'qair'}
DOMAINS = {
    'medical': dict(script='prototype/reconstruct_medical.py', query='pna_base', data='./assets/medical_heldout_pna/'),
    'uav': dict(script='prototype/reconstruct_uav.py', query='sol_base', data='./assets/uav_heldout_solar/'),
}
TEMPS_REP1 = [10, 30, 300, 1000]
TEMPS_REP2 = [10, 30, 100, 300, 1000]


def exists(rel):
    return os.path.exists(os.path.join(REPO, rel))


def client_job(domain, enc, seed, head, batch_tag):
    spec = DOMAINS[domain]
    out = f"./results/{domain}_{spec['query']}_enc{enc}_{batch_tag}"
    return [out, spec['script'], '--geminio-query', spec['query'], '--model-path', head, '--client-encoder', enc,
            '--data-dir', spec['data'], '--attack', 'hfgradinv', '--max-iterations', ITERS, '--seed', str(seed),
            '--num-samples', '8', '--batch-tag', batch_tag]


def resnet_job(domain, seed, head, batch_tag):
    spec = DOMAINS[domain]
    out = f"./results/{domain}_{spec['query']}_{batch_tag}"
    return [out, spec['script'], '--geminio-query', spec['query'], '--model-path', head, '--data-dir', spec['data'],
            '--attack', 'hfgradinv', '--max-iterations', ITERS, '--seed', str(seed), '--num-samples', '8',
            '--batch-tag', batch_tag]


def write(name, jobs):
    out_dir = os.path.join(REPO, 'results', name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'jobs.tsv')
    with open(path, 'w') as f:
        for j in jobs:
            f.write('\t'.join(j) + '\n')
    print(f'{len(jobs):3d} jobs -> results/{name}/jobs.tsv')
    return jobs


def main():
    missing = []
    # D: plain heads (seed-major so the seed-42 runs for every encoder land first) + 2 dup jobs
    plain = []
    for seed in SEEDS:
        for domain in DOMAINS:
            for enc in ENCODERS:
                head = f'malicious_models_client_encoder/{domain}_{enc}_plain.pt'
                if exists(head):
                    plain.append(client_job(domain, enc, seed, head, f'ce_plain_s{seed}'))
                else:
                    missing.append(head)
    for domain in DOMAINS:
        head = f'malicious_models_client_encoder/{domain}_clip.pt'
        plain.append(client_job(domain, 'clip', 42, head, 'ce_dup_s42'))
    # C: temperature sweep
    temp = []
    for seed in SEEDS:
        for domain, spec in DOMAINS.items():
            for rep, temps in [(1, TEMPS_REP1), (2, TEMPS_REP2)]:
                for T in temps:
                    r = '_r2' if rep == 2 else ''
                    head = f"malicious_models_tempsweep/{domain}/{spec['query']}_T{T}{r}.pt"
                    if exists(head):
                        temp.append(resnet_job(domain, seed, head, f"T{T}{'r2' if rep == 2 else ''}_s{seed}"))
                    else:
                        missing.append(head)
    # E: query swap
    swap = []
    for seed in SEEDS:
        for domain in DOMAINS:
            for enc in ENCODERS:
                tag = SWAP_TAG[domain]
                head = f'malicious_models_client_encoder/{domain}_{enc}_{tag}.pt'
                if exists(head):
                    swap.append(client_job(domain, enc, seed, head, f'ce_{tag}_s{seed}'))
                else:
                    missing.append(head)

    write('plain_head', plain)
    write('temp_sweep', temp)
    write('query_swap', swap)
    # The ViT-L controls (D then E) share one queue; the ResNet-18 temperature sweep runs from its own
    # jobs file so it can start as soon as its heads exist. Several run_job_queue.sh instances may
    # serve the same jobs file (they share the flock-protected counter), so workers can be added later.
    combined = plain + swap
    outs = [j[0] for j in combined + temp]
    dupes = {o for o in outs if outs.count(o) > 1}
    assert not dupes, f'duplicate result dirs: {sorted(dupes)[:5]}'
    write('rerun_oct24', combined)
    if missing:
        print(f'\n{len(missing)} head files missing; their jobs were NOT written:')
        for m in sorted(set(missing)):
            print('   ', m)


if __name__ == '__main__':
    main()
