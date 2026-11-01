"""Threshold-free attack metrics (AUC, top-k retrieval) computed post-hoc.

Loads reconstructed.pt + true_data.pt + the malicious model from each result
directory, computes per-sample gradient cosine similarities, and reports:
  - F1 at the default cosine threshold (0.90) for sanity vs metrics.json
  - F1 at threshold 0.80 and 0.70 (sensitivity sweep)
  - AUC of "matched pair cosine > random pair cosine" across all pair pairings
  - Top-1 and top-2 retrieval accuracy: for each reconstruction, rank against
    all truths and check if the matched truth is in the top-k

Usage:
    python prototype/threshold_free_metrics.py --domain medical|uav \
        --dirs results/medical_pneumonia_descriptive_dpsgd_eps5.0*

Outputs JSON per result dir at <dir>/threshold_free_metrics.json plus
a summary CSV at results/threshold_free_summary.csv.
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.models import GeminioResNet18  # noqa: E402

MEDICAL_HEAD = ROOT / "malicious_models_medical_v2" / "pneumonia_descriptive.pt"
UAV_HEAD = ROOT / "malicious_models_uav" / "aerial_drone_image_showing_solar_panels_on_rooftops.pt"

DOMAIN_TO_HEAD = {"medical": MEDICAL_HEAD, "uav": UAV_HEAD}
DOMAIN_TO_NCLASS = {"medical": 15, "uav": 18}
DOMAIN_TO_LOSS = {"medical": torch.nn.CrossEntropyLoss(), "uav": torch.nn.BCEWithLogitsLoss()}


def per_sample_grad(model, x_i, y_i, loss_fn, trainable_params):
    model.zero_grad()
    out = model(x_i)
    loss = loss_fn(out, y_i)
    g = torch.autograd.grad(loss, trainable_params, create_graph=False)
    return g[-2].flatten()  # output-layer weight gradient, same as compute_attack_f1


def compute_metrics(result_dir: Path, domain: str, device):
    rec_path = result_dir / "reconstructed.pt"
    true_path = result_dir / "true_data.pt"
    if not rec_path.exists() or not true_path.exists():
        return None

    rec = torch.load(rec_path, map_location=device)
    truth = torch.load(true_path, map_location=device)

    model = GeminioResNet18(num_classes=DOMAIN_TO_NCLASS[domain]).to(device).eval()
    head_state = torch.load(DOMAIN_TO_HEAD[domain], map_location=device)
    model.clf.load_state_dict(head_state, strict=True)

    loss_fn = DOMAIN_TO_LOSS[domain]
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    rec_data, true_data, labels = rec["data"], truth["data"], truth["labels"]
    B = rec_data.shape[0]

    # Compute B truth gradients and B reconstruction gradients
    rec_grads = torch.stack([per_sample_grad(model, rec_data[i:i+1], labels[i:i+1], loss_fn, trainable_params) for i in range(B)])
    true_grads = torch.stack([per_sample_grad(model, true_data[i:i+1], labels[i:i+1], loss_fn, trainable_params) for i in range(B)])

    # cosine matrix [B, B]: row i = rec[i], col j = truth[j]
    cos_mat = F.cosine_similarity(rec_grads.unsqueeze(1), true_grads.unsqueeze(0), dim=-1)  # [B, B]
    diag = cos_mat.diag()  # matched-pair cos sims

    # F1 at multiple thresholds (matched only)
    f1_at = {f"f1_thr_{t:.2f}": float((diag >= t).float().mean()) for t in (0.70, 0.80, 0.90)}

    # Top-k retrieval accuracy: for each row, rank columns and check if matched is in top-k
    ranks = cos_mat.argsort(dim=1, descending=True)  # [B, B]
    matched_rank = (ranks == torch.arange(B, device=device).unsqueeze(1)).float().argmax(dim=1)  # [B]
    top1 = float((matched_rank < 1).float().mean())
    top2 = float((matched_rank < 2).float().mean())

    # AUC: how often is the matched-pair cos similarity strictly larger than a random non-matched pair cos similarity?
    # Equivalently, the mean rank of matched-pair cos among the row's B candidates.
    # AUC = Pr(diag[i] > cos_mat[i, j]) for random j != i.
    off_diag = []
    for i in range(B):
        for j in range(B):
            if i != j:
                off_diag.append(cos_mat[i, j].item())
    diag_vals = [diag[i].item() for i in range(B)]
    n_better = sum(1 for d in diag_vals for o in off_diag if d > o)
    n_total = len(diag_vals) * len(off_diag) if off_diag else 1
    auc = n_better / n_total if n_total else 0.0

    return {
        "B": B,
        "matched_cos_mean": float(diag.mean()),
        "matched_cos_min": float(diag.min()),
        "matched_cos_max": float(diag.max()),
        **f1_at,
        "top1_retrieval": top1,
        "top2_retrieval": top2,
        "auc_matched_vs_random": auc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["medical", "uav"])
    ap.add_argument("--dirs", required=True, nargs="+")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--summary-csv", default=None)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    expanded = []
    for pattern in args.dirs:
        expanded.extend(glob.glob(pattern))
    expanded = sorted(set(expanded))

    rows = []
    for d in expanded:
        p = Path(d)
        if not p.is_dir():
            continue
        try:
            r = compute_metrics(p, args.domain, device)
        except Exception as e:
            print(f"  [FAIL] {p.name}: {e}")
            continue
        if r is None:
            continue
        (p / "threshold_free_metrics.json").write_text(json.dumps(r, indent=2))
        rows.append({"dir": p.name, **r})
        print(f"  {p.name:80s} F1@90={r['f1_thr_0.90']:.3f}  top1={r['top1_retrieval']:.3f}  top2={r['top2_retrieval']:.3f}  AUC={r['auc_matched_vs_random']:.3f}")

    if args.summary_csv and rows:
        with open(args.summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote summary to {args.summary_csv}")


if __name__ == "__main__":
    main()
