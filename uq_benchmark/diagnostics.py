from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def _finite_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def diagnose_nig_predictions(
    prediction_path: str | Path,
    output_path: str | Path | None = None,
) -> dict:
    """Summarize NIG domain, tail, variance, and structure-size diagnostics."""
    with np.load(prediction_path) as data:
        required = {"nu", "alpha", "beta", "natoms"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"prediction file is missing NIG fields: {sorted(missing)}")
        nu = np.asarray(data["nu"], dtype=np.float64).reshape(-1)
        alpha = np.asarray(data["alpha"], dtype=np.float64).reshape(-1)
        beta = np.asarray(data["beta"], dtype=np.float64).reshape(-1)
        natoms = np.asarray(data["natoms"], dtype=np.float64).reshape(-1)

    if not (len(nu) == len(alpha) == len(beta) == len(natoms)) or len(alpha) == 0:
        raise ValueError("NIG arrays and natoms must be non-empty with equal length")
    if not all(np.isfinite(values).all() for values in (nu, alpha, beta, natoms)):
        raise ValueError("NIG diagnostics require finite arrays")
    if np.any(nu <= 0) or np.any(alpha <= 1) or np.any(beta <= 0):
        raise ValueError("NIG domain violation: require nu > 0, alpha > 1, beta > 0")
    if np.any(natoms <= 0):
        raise ValueError("natoms must be positive")

    aleatoric = beta / (alpha - 1.0)
    epistemic = beta / (nu * (alpha - 1.0))
    quantile_levels = np.asarray([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    alpha_quantiles = np.quantile(alpha, quantile_levels)
    size_correlation = (
        np.corrcoef(natoms, epistemic)[0, 1]
        if len(alpha) > 1 and np.std(natoms) > 0 and np.std(epistemic) > 0
        else np.nan
    )
    result = {
        "prediction_path": str(Path(prediction_path)),
        "n_structures": int(len(alpha)),
        "domain_valid": True,
        "alpha_quantiles": {
            f"q{int(level * 100):02d}": float(value)
            for level, value in zip(quantile_levels, alpha_quantiles, strict=True)
        },
        "fraction_alpha_le_1_5": float(np.mean(alpha <= 1.5)),
        "fraction_alpha_le_2": float(np.mean(alpha <= 2.0)),
        "fourth_moment_exists_fraction": float(np.mean(alpha > 2.0)),
        "aleatoric_variance_min": float(np.min(aleatoric)),
        "aleatoric_variance_max": float(np.max(aleatoric)),
        "epistemic_variance_min": float(np.min(epistemic)),
        "epistemic_variance_max": float(np.max(epistemic)),
        "natoms_min": int(np.min(natoms)),
        "natoms_max": int(np.max(natoms)),
        "pearson_natoms_epistemic_variance": _finite_float(size_correlation),
        "requires_mc_convergence_study": bool(np.any(alpha <= 2.0)),
        "force_posterior_identified": False,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output_path)
    return result
