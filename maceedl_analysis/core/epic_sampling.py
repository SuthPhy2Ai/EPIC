"""Differentiable Monte Carlo sampling for EPIC atomic NIG outputs."""

from __future__ import annotations

import torch


def _check_nig_domain(
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> None:
    if not (gamma.shape == nu.shape == alpha.shape == beta.shape):
        raise ValueError("gamma, nu, alpha, and beta must have identical shapes")
    for name, value in (("gamma", gamma), ("nu", nu), ("alpha", alpha), ("beta", beta)):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    if torch.any(nu <= 0) or torch.any(alpha <= 1) or torch.any(beta <= 0):
        raise ValueError("NIG requires nu > 0, alpha > 1, beta > 0")


def sample_atomic_nig_means(
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    n_samples: int,
    seed: int,
) -> torch.Tensor:
    """Sample differentiable latent atomic means from a Normal-Inverse-Gamma.

    Reusing ``seed`` for every displaced configuration defines common random
    numbers: sample ``k`` follows the same random stream throughout one phonon
    calculation. The returned tensor remains connected to all NIG parameters.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    _check_nig_domain(gamma, nu, alpha, beta)

    cuda_devices: list[int] = []
    if gamma.is_cuda:
        cuda_devices = [gamma.device.index if gamma.device.index is not None else torch.cuda.current_device()]

    # Isolate the common-random-number stream from model inference and callers.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        if gamma.is_cuda:
            torch.cuda.manual_seed(int(seed))
        variance = torch.distributions.InverseGamma(alpha, beta).rsample((n_samples,))
        epsilon = torch.randn_like(variance)

    return gamma.unsqueeze(0) + torch.sqrt(variance / nu.unsqueeze(0)) * epsilon


def compose_sampled_total_energies(
    mean_total_energy: torch.Tensor,
    atomic_gamma: torch.Tensor,
    sampled_atomic_means: torch.Tensor,
) -> torch.Tensor:
    """Replace the mean atomic EPIC contribution while retaining E0/pair terms."""
    if mean_total_energy.numel() != 1:
        raise ValueError("mean_total_energy must be scalar")
    if atomic_gamma.ndim != 1:
        raise ValueError("atomic_gamma must have shape [n_atoms]")
    if sampled_atomic_means.ndim != 2 or sampled_atomic_means.shape[1] != len(atomic_gamma):
        raise ValueError("sampled_atomic_means must have shape [n_samples, n_atoms]")
    baseline = mean_total_energy.reshape(()) - atomic_gamma.sum()
    return baseline + sampled_atomic_means.sum(dim=1)
