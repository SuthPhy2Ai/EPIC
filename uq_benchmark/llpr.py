from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import apply_variance_map, fit_gaussian_variance_map
from .distributions import gaussian_nll
from .types import CalibrationState


class StreamingGram:
    def __init__(self, dimension: int, dtype: np.dtype = np.float64):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = int(dimension)
        self.matrix = np.zeros((dimension, dimension), dtype=dtype)
        self.n_samples = 0

    def update(self, features: np.ndarray) -> None:
        features = np.asarray(features, dtype=self.matrix.dtype)
        if features.ndim != 2 or features.shape[1] != self.dimension:
            raise ValueError(f"features must have shape [n, {self.dimension}]")
        self.matrix += features.T @ features
        self.n_samples += len(features)

    @property
    def trace_scale(self) -> float:
        return max(float(np.trace(self.matrix)) / self.dimension, np.finfo(self.matrix.dtype).eps)


@dataclass(frozen=True)
class LLPRPosterior:
    covariance_raw: np.ndarray
    ridge: float
    calibration: CalibrationState
    n_training_samples: int
    force_calibration: CalibrationState | None = None

    def energy_variance(self, features: np.ndarray, calibrated: bool = True) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        raw = np.einsum("ni,ij,nj->n", features, self.covariance_raw, features)
        raw = np.maximum(raw, 0.0)
        return apply_variance_map(raw, self.calibration) if calibrated else raw


def fit_llpr_posterior(
    gram: StreamingGram,
    calibration_features: np.ndarray,
    calibration_target: np.ndarray,
    calibration_location: np.ndarray,
    ridge_multipliers: tuple[float, ...],
) -> LLPRPosterior:
    features = np.asarray(calibration_features, dtype=float)
    best: tuple[float, float, np.ndarray, CalibrationState] | None = None
    for multiplier in ridge_multipliers:
        ridge = float(multiplier) * gram.trace_scale
        precision = gram.matrix + ridge * np.eye(gram.dimension)
        covariance = np.linalg.inv(precision)
        raw_variance = np.einsum("ni,ij,nj->n", features, covariance, features)
        state = fit_gaussian_variance_map(
            calibration_target, calibration_location, np.maximum(raw_variance, 0.0), method="llpr"
        )
        scale = np.sqrt(apply_variance_map(raw_variance, state))
        score = float(np.mean(gaussian_nll(calibration_target, calibration_location, scale)))
        candidate = (score, ridge, covariance, state)
        if best is None or score < best[0]:
            best = candidate
    assert best is not None
    return LLPRPosterior(best[2], best[1], best[3], gram.n_samples)


def antithetic_weight_samples(
    mean: np.ndarray, covariance: np.ndarray, n_samples: int = 32, seed: int = 42
) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if n_samples <= 0 or n_samples % 2:
        raise ValueError("n_samples must be positive and even")
    if covariance.shape != (len(mean), len(mean)):
        raise ValueError("covariance shape does not match mean")
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
    if eigenvalues.min() < -1e-10:
        raise ValueError("covariance is not positive semidefinite")
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    rng = np.random.default_rng(seed)
    half = rng.standard_normal((n_samples // 2, len(mean))) @ root.T
    return np.concatenate((mean + half, mean - half), axis=0)
