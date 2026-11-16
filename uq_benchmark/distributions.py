from __future__ import annotations

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm, t

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def nig_predictive_parameters(
    gamma: np.ndarray, nu: np.ndarray, alpha: np.ndarray, beta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gamma, nu, alpha, beta = np.broadcast_arrays(gamma, nu, alpha, beta)
    if np.any(nu <= 0) or np.any(alpha <= 1) or np.any(beta <= 0):
        raise ValueError("NIG requires nu > 0, alpha > 1, beta > 0")
    scale = np.sqrt(beta * (nu + 1.0) / (alpha * nu))
    return gamma.astype(float), scale.astype(float), (2.0 * alpha).astype(float)


def nig_variances(
    nu: np.ndarray, alpha: np.ndarray, beta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nu, alpha, beta = np.broadcast_arrays(nu, alpha, beta)
    if np.any(nu <= 0) or np.any(alpha <= 1) or np.any(beta <= 0):
        raise ValueError("NIG requires nu > 0, alpha > 1, beta > 0")
    aleatoric = beta / (alpha - 1.0)
    epistemic = beta / (nu * (alpha - 1.0))
    return aleatoric, epistemic, aleatoric + epistemic


def student_t_nll(y: np.ndarray, location: np.ndarray, scale: np.ndarray, df: np.ndarray) -> np.ndarray:
    y, location, scale, df = np.broadcast_arrays(y, location, scale, df)
    if np.any(scale <= 0) or np.any(df <= 0):
        raise ValueError("Student-t scale and degrees of freedom must be positive")
    z = (y - location) / scale
    return (
        np.log(scale)
        + 0.5 * np.log(df * np.pi)
        + gammaln(df / 2.0)
        - gammaln((df + 1.0) / 2.0)
        + 0.5 * (df + 1.0) * np.log1p(z * z / df)
    )


def gaussian_nll(y: np.ndarray, location: np.ndarray, scale: np.ndarray) -> np.ndarray:
    y, location, scale = np.broadcast_arrays(y, location, scale)
    if np.any(scale <= 0):
        raise ValueError("Gaussian scale must be positive")
    return 0.5 * np.log(2.0 * np.pi) + np.log(scale) + 0.5 * ((y - location) / scale) ** 2


def central_coverage(
    y: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
    levels: tuple[float, ...],
    family: str,
    df: np.ndarray | None = None,
) -> dict[float, np.ndarray]:
    result = {}
    for level in levels:
        if not 0 < level < 1:
            raise ValueError("coverage levels must lie in (0, 1)")
        if family == "gaussian":
            quantile = norm.ppf((1.0 + level) / 2.0)
        elif family == "student_t":
            if df is None:
                raise ValueError("Student-t coverage requires df")
            quantile = t.ppf((1.0 + level) / 2.0, df)
        else:
            raise ValueError(f"unsupported distribution family: {family}")
        result[level] = np.abs(np.asarray(y) - np.asarray(location)) <= quantile * np.asarray(scale)
    return result


def full_nig_nll_torch(
    target: torch.Tensor,
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    if torch.any(nu <= 0) or torch.any(alpha <= 1) or torch.any(beta <= 0):
        raise ValueError("NIG requires nu > 0, alpha > 1, beta > 0")
    two_blambda = 2.0 * beta * (1.0 + nu)
    nll = (
        0.5 * torch.log(torch.pi / nu)
        - alpha * torch.log(two_blambda)
        + (alpha + 0.5) * torch.log(nu * (target - gamma).square() + two_blambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.mean()
    raise ValueError(f"unknown reduction: {reduction}")


def full_nig_loss_torch(
    target: torch.Tensor,
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    regularization: float,
) -> torch.Tensor:
    nll = full_nig_nll_torch(target, gamma, nu, alpha, beta, reduction="none")
    evidence = 2.0 * nu + alpha
    penalty = torch.abs(target - gamma) * evidence
    return (nll + float(regularization) * penalty).mean()


def independent_atomic_moment_loss_torch(
    target: torch.Tensor,
    location: torch.Tensor,
    aleatoric_variance: torch.Tensor,
    epistemic_variance: torch.Tensor,
    atomic_nu: torch.Tensor,
    atomic_alpha: torch.Tensor,
    batch: torch.Tensor,
    atoms_per_graph: torch.Tensor,
    regularization: float,
) -> torch.Tensor:
    """Moment-matched graph likelihood for atomwise evidential predictions.

    Atomic predictive variances are added under a conditional-independence
    assumption. This is deliberately a Gaussian graph likelihood rather than
    an invalid sum of NIG parameters.
    """
    if not (
        target.shape
        == location.shape
        == aleatoric_variance.shape
        == epistemic_variance.shape
        == atoms_per_graph.shape
    ):
        raise ValueError("graph targets, moments, and atom counts must have matching shapes")
    if not (atomic_nu.shape == atomic_alpha.shape == batch.shape):
        raise ValueError("atomic evidence tensors and batch must have matching shapes")
    if torch.any(aleatoric_variance <= 0) or torch.any(epistemic_variance <= 0):
        raise ValueError("aggregated atomic variances must be positive")
    if torch.any(atomic_nu <= 0) or torch.any(atomic_alpha <= 1):
        raise ValueError("atomic evidence is outside the NIG domain")

    total_variance = aleatoric_variance + epistemic_variance
    error = target - location
    nll = 0.5 * (
        torch.log(2.0 * torch.pi * total_variance)
        + error.square() / total_variance
    )

    atomic_evidence = 2.0 * atomic_nu + atomic_alpha
    graph_evidence = location.new_zeros(location.shape)
    graph_evidence.index_add_(0, batch, atomic_evidence)
    graph_evidence = graph_evidence / atoms_per_graph
    penalty = (error.abs() / atoms_per_graph) * graph_evidence
    return (nll + float(regularization) * penalty).mean()


def annealed_regularization(lambda_max: float, epoch: int, anneal_epochs: int) -> float:
    if lambda_max < 0 or epoch < 0 or anneal_epochs <= 0:
        raise ValueError("invalid regularization schedule arguments")
    return float(lambda_max) * min((epoch + 1) / anneal_epochs, 1.0)
