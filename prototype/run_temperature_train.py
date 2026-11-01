"""Train Geminio heads at several softmax temperatures for the temperature sweep (control C).

The temperature T in Geminio's weight (1 - softmax(sim * T)) sets how sharply the head
concentrates loss on query-matched images, and therefore the head's loss ratio, at a FIXED
query and backbone (ResNet-18). The sweep tests corollary (ii) of the complementarity
framing: selectivity and target leakage should follow the loss ratio.

T = 100 is the paper default and already exists as the language-sweep base head
(malicious_models_lang_sweep/{domain}/{pna_base|sol_base}.pt, unseeded). Replicate 1 therefore
trains T in {10, 30, 300, 1000} with seed 0; replicate 2 trains all five T, including 100,
with seed 1, so that head-to-head variation at fixed T can be measured.

Outputs:
    malicious_models_tempsweep/{domain}/{base_tag}_T{T}[_r2].pt
    results/temp_sweep/train_logs[_rep2]/{base_tag}_T{T}.log
    malicious_models_tempsweep/manifest.csv   (tag, domain, T, rep, seed, loss_ratio, path)

Usage:
    python prototype/run_temperature_train.py --domain medical --rep 1 --gpu 2
    python prototype/run_temperature_train.py --domain medical --rep 2 --gpu 3
    python prototype/run_temperature_train.py --domain uav --rep 1 --gpu 4   # needs UAVScenes access
"""
import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = '/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python'
HEAD_DIR = os.path.join(ROOT, 'malicious_models_tempsweep')
MANIFEST = os.path.join(HEAD_DIR, 'manifest.csv')
LOG_ROOT = os.path.join(ROOT, 'results', 'temp_sweep')

BASE = {
    'medical': dict(tag='pna_base', text='Any chest X-ray showing pneumonia', vlm='biomedclip'),
    'uav': dict(tag='sol_base', text='aerial drone image showing solar panels on rooftops', vlm='clip'),
}
TEMPS = [10, 30, 100, 300, 1000]
REP_SEED = {1: 0, 2: 1}


def parse_loss_ratio(text):
    """First post-training loss ratio: medical prints 'Ratio: X.XXx', UAV 'Loss ratio: X.XXx'."""
    m = re.search(r'(?:Loss\s+)?Ratio:\s*([\d.]+)x', text, re.IGNORECASE)
    return float(m.group(1)) if m else float('nan')


def head_path(domain, T, rep):
    return os.path.join(HEAD_DIR, domain, f"{BASE[domain]['tag']}_T{T}" + ('_r2' if rep == 2 else '') + '.pt')


def train_one(domain, T, rep, gpu, epochs):
    spec = BASE[domain]
    final = head_path(domain, T, rep)
    if os.path.exists(final):
        print(f'[skip] {final} exists')
        return None
    tmp = os.path.join(HEAD_DIR, '_tmp', f'{domain}_T{T}_r{rep}')
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    log_dir = os.path.join(LOG_ROOT, 'train_logs' if rep == 1 else 'train_logs_rep2')
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, f"{spec['tag']}_T{T}.log")
    cmd = [PY, os.path.join(ROOT, 'prototype', f'train_{domain}.py'), '--query', spec['text'], '--gpu', str(gpu),
           '--epochs', str(epochs), '--vlm', spec['vlm'], '--temperature', str(T), '--seed', str(REP_SEED[rep]),
           '--output-dir', tmp]
    print(f"[{time.strftime('%H:%M:%S')}] train {domain} T={T} rep={rep} seed={REP_SEED[rep]} on GPU {gpu}")
    t0 = time.time()
    with open(log, 'w') as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    dt = time.time() - t0
    if rc != 0:
        print(f'[FAIL] {domain} T={T} rep={rep} rc={rc}; see {log}')
        return dict(ok=False)
    produced = glob.glob(os.path.join(tmp, '*.pt'))
    if len(produced) != 1:
        print(f'[FAIL] expected one .pt in {tmp}, found {produced}')
        return dict(ok=False)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    shutil.move(produced[0], final)
    shutil.rmtree(tmp, ignore_errors=True)
    lr = parse_loss_ratio(open(log).read())
    print(f"[{time.strftime('%H:%M:%S')}] done  {domain} T={T} rep={rep} loss_ratio={lr:.2f}x ({dt:.0f}s) -> {final}")
    return dict(tag=f"{spec['tag']}_T{T}" + ('_r2' if rep == 2 else ''), domain=domain, T=T, rep=rep,
                seed=REP_SEED[rep], vlm=spec['vlm'], loss_ratio=lr, elapsed_sec=int(dt), path=final, ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True, choices=list(BASE))
    ap.add_argument('--rep', type=int, default=1, choices=[1, 2])
    ap.add_argument('--temps', type=int, nargs='+', default=None,
                    help='Temperatures to train (default: all but 100 for rep 1, all five for rep 2)')
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=5)
    args = ap.parse_args()
    temps = args.temps or ([t for t in TEMPS if t != 100] if args.rep == 1 else TEMPS)
    rows = [r for T in temps if (r := train_one(args.domain, T, args.rep, args.gpu, args.epochs)) and r.get('ok')]
    if rows:
        fields = ['tag', 'domain', 'T', 'rep', 'seed', 'vlm', 'loss_ratio', 'elapsed_sec', 'path']
        new = not os.path.exists(MANIFEST)
        with open(MANIFEST, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in fields})
        print(f'manifest: {MANIFEST}')


if __name__ == '__main__':
    main()
