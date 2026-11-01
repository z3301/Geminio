"""
Job list for rerunning the paper's ResNet-18 experiments on held-out private batches.

Why: the original medical batch (assets/medical_samples/) contains no pneumonia image,
and 7 of 8 images in the original UAV batch (both solar-panel targets) come from the
half of the data the rigged head is trained on. The new batches
(prototype/prepare_target_batch.py) are drawn from the held-out half and contain the
targets (2 of 8; 4 of 16; 8 of 32). Everything else matches the original scripts
(scripts/run_2c_multiseed.sh, run_cliponly_multiseed.sh, run_eps10_multiseed.sh,
run_2c_epssweep.sh, run_record_level_dpsgd.sh, run_bn_trainmode_multiseed.sh,
run_frozen_backbone.sh, run_c_sensitivity.sh, run_batch_2c.sh, run_vlm_*.sh,
prototype/run_language_reconstruct.py), now with 3 seeds everywhere. NBFU noise uses the
analytic Gaussian calibration currently in prototype/defenses.py.

Output: results/heldout_rerun/jobs.tsv, one job per line, tab-separated:
    <result dir> <script> <arg> <arg> ...
(--gpu is appended by scripts/run_job_queue.sh). Jobs are ordered by priority:
headline NBFU table, robust-VLM table, language sweep, then robustness checks.

Usage:
    python scripts/make_heldout_jobs.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from prototype.language_sweep_config import LANGUAGE_SWEEP_QUERIES

SEEDS = [42, 123, 256]
ITERS = '24000'
VLMS = ['clip', 'simclip2', 'simclip4', 'fare2', 'fare4', 'tecoa2', 'tecoa4']
DOMAINS = {
    'medical': dict(script='prototype/reconstruct_medical.py', query='pneumonia_descriptive',
                    data={8: './assets/medical_heldout_pna/', 16: './assets/medical_heldout_pna_B16/',
                          32: './assets/medical_heldout_pna_B32/'},
                    vlm_head=lambda v: f'malicious_models_medical_v2/Any_chest_X-ray_showing_pneumonia_{v}.pt'),
    'uav': dict(script='prototype/reconstruct_uav.py', query='solar_panels',
                data={8: './assets/uav_heldout_solar/', 16: './assets/uav_heldout_solar_B16/',
                      32: './assets/uav_heldout_solar_B32/'},
                vlm_head=lambda v: 'malicious_models_uav/aerial_drone_image_showing_solar_panels_on_rooftops'
                                   + ('' if v == 'clip' else f'_{v}') + '.pt'),
}


def defense_args(eps=None, C=1.0, record_level=False):
    if eps is None:
        return []
    args = ['--dpsgd-epsilon', str(eps), '--dpsgd-max-grad-norm', str(C)]
    return args + (['--dpsgd-record-level'] if record_level else [])


def result_dir(domain, query, attack, variant, seed, eps=None, record_level=False, frozen=False):
    """Mirror the save_dir naming in prototype/reconstruct_{medical,uav}.py."""
    attack_tag = '' if attack == 'hfgradinv' else f'_{attack}'
    defense_tag = ''
    if eps is not None and not record_level:  # record-level zeroes epsilon before naming
        defense_tag += '_dpsgd_cliponly' if eps < 0 else f'_dpsgd_eps{float(eps)}'
    if frozen:
        defense_tag += '_frozen'
    if record_level:
        defense_tag += '_reclvl'
    batch_tag = f'_ho{variant}_s{seed}'
    return f'./results/{domain}_{query}{attack_tag}{defense_tag}{batch_tag}', batch_tag[1:]


def job(domain, seed, attack='hfgradinv', eps=None, C=1.0, record_level=False, frozen=False,
        bn_train=False, batch=8, query=None, model_path=None, variant=''):
    spec = DOMAINS[domain]
    query = query or spec['query']
    out, batch_tag = result_dir(domain, query, attack, variant, seed, eps, record_level, frozen)
    args = ['--geminio-query', query, '--attack', attack, '--max-iterations', ITERS, '--seed', str(seed),
            '--data-dir', spec['data'][batch], '--num-samples', str(batch), '--batch-tag', batch_tag]
    if model_path:
        args += ['--model-path', model_path]
    if bn_train:
        args.append('--bn-train-mode')
    if frozen:
        args.append('--freeze-backbone')
    args += defense_args(eps, C, record_level)
    return [out, spec['script']] + args


def main():
    jobs = []
    # 1. Headline: NBFU eps=5 vs clean across three reconstruction algorithms
    for seed in SEEDS:
        for domain in DOMAINS:
            for attack in ['hfgradinv', 'ig', 'gradinversion']:
                jobs.append(job(domain, seed, attack))
                jobs.append(job(domain, seed, attack, eps=5.0))
    # 2. Robust VLMs as the attacker's guide (Section 5 / Appendix J)
    for seed in SEEDS:
        for domain, spec in DOMAINS.items():
            for vlm in VLMS:
                head = spec['vlm_head'](vlm)
                assert os.path.exists(os.path.join(REPO, head)), head
                jobs.append(job(domain, seed, query=f'vlm{vlm}', model_path=head))
    # 3. Language sweep: 24 queries, clean and NBFU eps=5
    for seed in SEEDS:
        for domain, queries in LANGUAGE_SWEEP_QUERIES.items():
            for tag, *_ in queries:
                head = f'malicious_models_lang_sweep/{domain}/{tag}.pt'
                assert os.path.exists(os.path.join(REPO, head)), head
                jobs.append(job(domain, seed, query=tag, model_path=head))
                jobs.append(job(domain, seed, query=tag, model_path=head, eps=5.0))
    # 4. Robustness checks
    for seed in SEEDS:
        for domain in DOMAINS:
            for attack in ['hfgradinv', 'ig', 'gradinversion']:
                jobs.append(job(domain, seed, attack, eps=-1))                       # clip only (sigma = 0)
            for eps in [0.1, 1.0, 10.0, 50.0]:                                        # budget sweep (5.0 is in 1.)
                jobs.append(job(domain, seed, eps=eps))
            jobs.append(job(domain, seed, eps=5.0, record_level=True))               # record-level DP-SGD
            jobs.append(job(domain, seed, bn_train=True, variant='bntrain'))         # BatchNorm train mode
            jobs.append(job(domain, seed, eps=5.0, bn_train=True, variant='bntrain'))
            for eps in [None, 5.0, 10.0]:                                             # frozen backbone
                jobs.append(job(domain, seed, eps=eps, frozen=True))
            for C in [0.1, 0.5, 2.0, 5.0]:                                            # clipping bound (1.0 is in 1.)
                jobs.append(job(domain, seed, eps=5.0, C=C, variant=f'C{C}'))
            for batch in [16, 32]:                                                    # batch size
                jobs.append(job(domain, seed, batch=batch, variant=f'B{batch}'))
                jobs.append(job(domain, seed, eps=5.0, batch=batch, variant=f'B{batch}'))

    outs = [j[0] for j in jobs]
    dupes = {o for o in outs if outs.count(o) > 1}
    assert not dupes, f'duplicate result dirs: {sorted(dupes)[:5]}'
    out_dir = os.path.join(REPO, 'results', 'heldout_rerun')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'jobs.tsv'), 'w') as f:
        for j in jobs:
            f.write('\t'.join(j) + '\n')
    print(f'{len(jobs)} jobs -> {out_dir}/jobs.tsv')


if __name__ == '__main__':
    main()
