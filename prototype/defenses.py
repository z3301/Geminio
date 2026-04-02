"""
Defense mechanisms against gradient inversion attacks.

Implements gradient-level defenses that are applied to shared gradients
before the attacker can run gradient inversion. These simulate what a
privacy-conscious FL client would do before sharing gradient updates.

Defenses:
  - Gradient pruning (Zhu et al., NeurIPS 2019)
  - Gradient noise injection (Laplacian/Gaussian)
  - DP-SGD (Abadi et al., CCS 2016) — per-sample clipping + calibrated Gaussian noise

Usage:
    from prototype.defenses import apply_dpsgd, apply_gradient_pruning, apply_gradient_noise

    # DP-SGD with epsilon=1.0
    defended = apply_dpsgd(gradients, epsilon=1.0, delta=1e-5, max_grad_norm=1.0, batch_size=8)

    # Gradient pruning at 90%
    defended = apply_gradient_pruning(gradients, prune_rate=0.90)
"""
import math
import torch


def apply_gradient_pruning(gradients, prune_rate):
    """
    Prune smallest gradients by magnitude (defense).

    Zeros out the smallest-magnitude gradient entries. For example,
    prune_rate=0.90 removes 90% of gradient entries.

    Reference: Zhu et al., "Deep Leakage from Gradients", NeurIPS 2019.

    Args:
        gradients: List of gradient tensors
        prune_rate: Fraction of entries to zero out (0.0 to 0.99)

    Returns:
        List of pruned gradient tensors
    """
    pruned = []
    for grad in gradients:
        g = grad.clone()
        n = int(prune_rate * g.numel())
        if n > 0:
            threshold = torch.sort(torch.abs(g).flatten())[0][n - 1]
            mask = torch.abs(g) > threshold
            g *= mask
        pruned.append(g)
    return pruned


def apply_gradient_noise(gradients, noise_scale, distribution='laplacian'):
    """
    Add random noise to gradients (defense).

    Args:
        gradients: List of gradient tensors
        noise_scale: Standard deviation / scale of noise distribution
        distribution: 'laplacian' or 'gaussian'

    Returns:
        List of noisy gradient tensors
    """
    noisy = []
    for grad in gradients:
        g = grad.clone()
        if distribution == 'laplacian':
            noise = torch.distributions.Laplace(0, noise_scale).sample(g.shape).to(g.device)
        else:
            noise = torch.distributions.Normal(0, noise_scale).sample(g.shape).to(g.device)
        noisy.append(g + noise)
    return noisy


def compute_noise_multiplier(epsilon, delta, sensitivity=1.0):
    """
    Compute the Gaussian noise multiplier for (epsilon, delta)-DP.

    Uses the analytic Gaussian mechanism (Balle & Wang, 2018):
        sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

    Args:
        epsilon: Privacy budget. Smaller = more privacy, more noise.
        delta: Failure probability (typically 1/n^2 or 1e-5).
        sensitivity: L2 sensitivity of the query (= max_grad_norm for DP-SGD).

    Returns:
        float: Noise standard deviation (sigma)
    """
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def apply_dpsgd(gradients, epsilon, delta=1e-5, max_grad_norm=1.0, batch_size=1):
    """
    Apply DP-SGD defense to shared gradients.

    Implements the two core steps of DP-SGD (Abadi et al., CCS 2016):
      1. Gradient clipping: clip the global L2 norm of the gradient to max_grad_norm
      2. Gaussian noise: add calibrated noise with sigma = max_grad_norm * sqrt(2*ln(1.25/delta)) / epsilon

    The noise is scaled by 1/batch_size to account for gradient averaging over the batch,
    matching the standard DP-SGD formulation where noise is added to the average gradient.

    Args:
        gradients: List of gradient tensors (the shared gradient update)
        epsilon: Privacy budget. Common values:
                   0.1  = very strong privacy (heavy noise, likely destroys utility)
                   1.0  = strong privacy (significant noise)
                   5.0  = moderate privacy
                   10.0 = weak privacy (light noise)
                   50.0 = very weak privacy (minimal noise, for reference)
        delta: Failure probability. Should be < 1/n where n = dataset size.
               Default 1e-5 is suitable for datasets with >100K samples.
        max_grad_norm: Maximum L2 norm for gradient clipping (C in the DP-SGD paper).
                       Controls sensitivity. Default 1.0 is standard.
        batch_size: Number of samples in the batch. Noise is scaled by 1/batch_size.

    Returns:
        List of defended gradient tensors

    Reference:
        Abadi et al., "Deep Learning with Differential Privacy", CCS 2016.
        Balle & Wang, "Improving the Gaussian Mechanism for Differential Privacy", ICML 2018.
    """
    # Step 1: Gradient clipping (bound the L2 norm)
    clipped = []
    flat_grads = torch.cat([g.flatten() for g in gradients])
    global_norm = torch.norm(flat_grads, p=2).item()
    clip_factor = min(1.0, max_grad_norm / (global_norm + 1e-8))

    for grad in gradients:
        clipped.append(grad.clone() * clip_factor)

    # Step 2: Add calibrated Gaussian noise
    sigma = compute_noise_multiplier(epsilon, delta, sensitivity=max_grad_norm)
    # Scale noise by 1/batch_size (noise is added to the average gradient)
    noise_std = sigma / batch_size

    defended = []
    for grad in clipped:
        noise = torch.randn_like(grad) * noise_std
        defended.append(grad + noise)

    return defended, {
        'epsilon': epsilon,
        'delta': delta,
        'max_grad_norm': max_grad_norm,
        'global_norm_before_clip': global_norm,
        'clip_factor': clip_factor,
        'sigma': sigma,
        'noise_std_per_element': noise_std,
        'batch_size': batch_size,
    }
