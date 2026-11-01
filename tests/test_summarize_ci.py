"""Unit tests for the matching helper in scripts/summarize_ci.py.

Run:  python tests/test_summarize_ci.py      (no pytest needed)
"""
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from summarize_ci import match_metrics  # noqa: E402


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_diagonal_best_equals_hungarian():
    # Each GT image has its own clearly best reconstruction on the diagonal.
    lp = np.full((4, 4), 0.9)
    np.fill_diagonal(lp, [0.1, 0.2, 0.3, 0.4])
    psnr = 10.0 - 10.0 * lp
    mask = np.array([True, False, True, False])
    out = match_metrics(lp, psnr, mask)
    assert out['n_targets'] == 2
    assert out['n_unique_best'] == 4
    assert approx(out['target_lpips'], 0.2)            # mean of 0.1, 0.3
    assert approx(out['target_lpips_hung'], 0.2)
    assert approx(out['nontarget_lpips'], 0.3)         # mean of 0.2, 0.4
    assert approx(out['nontarget_lpips_hung'], 0.3)
    assert approx(out['selectivity'], 0.1)
    assert approx(out['selectivity_hung'], 0.1)
    assert approx(out['target_psnr_hung'], out['target_psnr'])


def test_collapsed_best_match_is_spread_by_hungarian():
    # Reconstruction 0 is the closest match for EVERY GT image (the degenerate case the
    # analyst flagged). Best-match collapses onto column 0; Hungarian must assign one
    # reconstruction per GT and therefore report a worse (higher) target LPIPS.
    lp = np.array([[0.30, 0.50, 0.60, 0.70],
                   [0.31, 0.55, 0.65, 0.75],
                   [0.32, 0.52, 0.62, 0.72],
                   [0.33, 0.58, 0.68, 0.78]])
    psnr = 20.0 - 10.0 * lp
    mask = np.array([True, True, False, False])
    out = match_metrics(lp, psnr, mask)
    assert out['n_unique_best'] == 1
    assert approx(out['target_lpips'], 0.305)
    # Hungarian: 4 distinct columns, total LPIPS minimized; target mean must exceed 0.305.
    assert out['target_lpips_hung'] > out['target_lpips']
    assert approx(out['selectivity'], (0.32 + 0.33) / 2 - 0.305)
    # Hungarian PSNR is taken at the LPIPS-assigned pairing (one assignment, not two).
    assert out['target_psnr_hung'] <= out['target_psnr'] + 1e-9


def test_fewer_reconstructions_than_targets():
    # 3 GT images, 2 reconstructions: one GT image gets no Hungarian partner (NaN, ignored).
    lp = np.array([[0.1, 0.9],
                   [0.9, 0.2],
                   [0.5, 0.5]])
    psnr = 10.0 - 10.0 * lp
    mask = np.array([True, True, True])
    out = match_metrics(lp, psnr, mask)
    assert out['n_targets'] == 3
    assert approx(out['target_lpips'], (0.1 + 0.2 + 0.5) / 3)
    assert approx(out['target_lpips_hung'], (0.1 + 0.2) / 2)
    assert math.isnan(out['nontarget_lpips'])
    assert math.isnan(out['selectivity'])


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'ok  {name}')
    print('all tests passed')
