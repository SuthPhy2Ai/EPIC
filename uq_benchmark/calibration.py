from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .distributions import gaussian_nll, student_t_nll
from .types import CalibrationState


def fit_student_t_scale(
    target: np.ndarray, location: np.ndarray, raw_scale: np.ndarray, df: np.ndarray, method: str = "epic"
) -> CalibrationState:
    target, location, raw_scale, df = map(np.asarray, (target, location, raw_scale, df))
    if np.any(raw_scale <= 0):
        raise ValueError("raw_scale must be positive")

    def objective(log_tau: float) -> float:
        return float(np.mean(student_t_nll(target, location, raw_scale * np.exp(log_tau), df)))

    result = minimize_scalar(objective, bounds=(-12.0, 12.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Student-t calibration failed: {result.message}")
    return CalibrationState(method=method, parameters={"tau": float(np.exp(result.x)), "sigma0": 0.0})


def fit_gaussian_variance_map(
    target: np.ndarray,
    location: np.ndarray,
    raw_variance: np.ndarray,
    method: str,
    minimum_scale: float = 1e-8,
) -> CalibrationState:
    target, location, raw_variance = map(np.asarray, (target, location, raw_variance))
    if np.any(raw_variance < 0):
        raise ValueError("raw_variance must be non-negative")
    residual_scale = max(float(np.std(target - location)), minimum_scale)

    def objective(params: np.ndarray) -> float:
        tau = np.exp(params[0])
        sigma0 = np.exp(params[1])
        scale = np.sqrt(tau * tau * raw_variance + sigma0 * sigma0)
        return float(np.mean(gaussian_nll(target, location, scale)))

    initial = np.log([1.0, residual_scale * 0.1])
    result = minimize(objective, initial, method="L-BFGS-B", bounds=[(-12, 12), (-18, 12)])
    if not result.success:
        raise RuntimeError(f"Gaussian calibration failed: {result.message}")
    return CalibrationState(
        method=method,
        parameters={"tau": float(np.exp(result.x[0])), "sigma0": float(np.exp(result.x[1]))},
    )


def apply_variance_map(raw_variance: np.ndarray, state: CalibrationState) -> np.ndarray:
    tau = float(state.parameters["tau"])
    sigma0 = float(state.parameters.get("sigma0", 0.0))
    return tau * tau * np.asarray(raw_variance) + sigma0 * sigma0

