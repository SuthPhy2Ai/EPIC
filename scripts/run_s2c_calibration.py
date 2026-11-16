#!/usr/bin/env python3
"""Post-hoc affine variance recalibration for EPIC energy and Scheme 2c force predictions.

Implements Section 2.1, Section 4.6 (Eq. 19), and Supplementary Note 3:
  - Fits two-parameter affine variance map sigma_cal = sqrt(tau^2 * sigma^2 + sigma_0^2)
    by minimizing Gaussian negative log-likelihood on the disjoint calibration split.
  - Evaluates calibrated NLL and empirical 95% interval coverage on the calibration
    and held-out test splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uq_benchmark.calibration import apply_variance_map, fit_gaussian_variance_map
from uq_benchmark.distributions import gaussian_nll
from uq_benchmark.s2c_covariance import BASELINE_ID


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def evaluate_gaussian_metrics(
    target: np.ndarray,
    location: np.ndarray,
    raw_variance: np.ndarray,
    parameters: dict[str, float],
) -> dict[str, float | int]:
    err = np.asarray(location, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    cal_var = apply_variance_map(
        np.asarray(raw_variance, dtype=np.float64),
        type("State", (), {"parameters": parameters})(),
    )
    cal_std = np.sqrt(np.maximum(cal_var, 1e-30))
    nll = gaussian_nll(target, location, cal_std)
    cov95 = float(np.mean(np.abs(err) <= norm.ppf(0.975) * cal_std))
    return {
        "n_elements": int(err.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "nll": float(np.mean(nll)),
        "coverage95": cov95,
    }


def extract_force_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        y = z["force_true"].astype(np.float64).reshape(-1)
        mu = (z["force_mean"] if "force_mean" in z else z["force_pred"]).astype(np.float64).reshape(-1)
        if "force_variance_raw" in z:
            var = z["force_variance_raw"].astype(np.float64).reshape(-1)
        else:
            var = np.square(z["force_scale_raw"].astype(np.float64).reshape(-1))
    return y, mu, var


def extract_energy_arrays(
    path: Path,
    component: str = "total",
    finite_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        natoms = z["natoms"].astype(np.float64)
        y_pa = z["energy_true_total"].astype(np.float64) / natoms
        mu_pa = z["energy_pred_total"].astype(np.float64) / natoms
        alpha = z["alpha"].astype(np.float64)
        epi_tot = z["energy_epistemic_variance"].astype(np.float64)
        alea_tot = z["energy_aleatoric_variance"].astype(np.float64)

    if component == "epistemic":
        var_pa = epi_tot / np.square(natoms)
    elif component == "aleatoric":
        var_pa = alea_tot / np.square(natoms)
    elif component == "total":
        var_pa = (epi_tot + alea_tot) / np.square(natoms)
    else:
        raise ValueError(f"Unsupported energy variance component: {component}")

    if finite_only:
        mask = (alpha - 1.0) > 1e-5
        return y_pa[mask], mu_pa[mask], var_pa[mask]
    return y_pa, mu_pa, var_pa


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-calibration",
        type=Path,
        required=True,
        help="Path to calibration Scheme 2c force aggregated.npz.",
    )
    parser.add_argument(
        "--force-test",
        type=Path,
        help="Optional path to test Scheme 2c force aggregated.npz.",
    )
    parser.add_argument(
        "--energy-calibration",
        type=Path,
        help="Optional path to calibration point energy aggregated.npz.",
    )
    parser.add_argument(
        "--energy-test",
        type=Path,
        help="Optional path to test point energy aggregated.npz.",
    )
    parser.add_argument(
        "--energy-component",
        choices=["total", "epistemic", "aleatoric"],
        default="total",
        help="Energy variance component to calibrate (default: total).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output JSON calibration report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cal_y, cal_mu, cal_var = extract_force_arrays(args.force_calibration)
    f_state = fit_gaussian_variance_map(cal_y, cal_mu, cal_var, method=f"{BASELINE_ID}_force")
    f_params = {k: float(v) for k, v in f_state.parameters.items()}

    report: dict = {
        "method": BASELINE_ID,
        "force_parameters": {
            "tau_F": f_params["tau"],
            "sigma0_F": f_params["sigma0"],
            "a": f_params["tau"] ** 2,
            "b": f_params["sigma0"] ** 2,
        },
        "force_calibration_metrics": evaluate_gaussian_metrics(cal_y, cal_mu, cal_var, f_params),
        "inputs_sha256": {
            str(Path(args.force_calibration).resolve()): sha256_file(args.force_calibration)
        },
    }

    if args.force_test is not None:
        test_y, test_mu, test_var = extract_force_arrays(args.force_test)
        report["force_test_metrics"] = evaluate_gaussian_metrics(
            test_y, test_mu, test_var, f_params
        )
        report["inputs_sha256"][str(Path(args.force_test).resolve())] = sha256_file(args.force_test)

    if args.energy_calibration is not None:
        e_cal_y, e_cal_mu, e_cal_var = extract_energy_arrays(
            args.energy_calibration, component=args.energy_component
        )
        e_state = fit_gaussian_variance_map(
            e_cal_y, e_cal_mu, e_cal_var, method=f"{BASELINE_ID}_energy_{args.energy_component}"
        )
        e_params = {k: float(v) for k, v in e_state.parameters.items()}
        report["energy_component"] = args.energy_component
        report["energy_parameters"] = {
            "tau_E": e_params["tau"],
            "sigma0_E": e_params["sigma0"],
        }
        report["energy_calibration_metrics"] = evaluate_gaussian_metrics(
            e_cal_y, e_cal_mu, e_cal_var, e_params
        )
        report["inputs_sha256"][str(Path(args.energy_calibration).resolve())] = sha256_file(
            args.energy_calibration
        )
        if args.energy_test is not None:
            e_test_y, e_test_mu, e_test_var = extract_energy_arrays(
                args.energy_test, component=args.energy_component, finite_only=False
            )
            e_test_fin_y, e_test_fin_mu, e_test_fin_var = extract_energy_arrays(
                args.energy_test, component=args.energy_component, finite_only=True
            )
            report["energy_test_full_metrics"] = evaluate_gaussian_metrics(
                e_test_y, e_test_mu, e_test_var, e_params
            )
            report["energy_test_finite_metrics"] = evaluate_gaussian_metrics(
                e_test_fin_y, e_test_fin_mu, e_test_fin_var, e_params
            )
            report["inputs_sha256"][str(Path(args.energy_test).resolve())] = sha256_file(
                args.energy_test
            )

    atomic_json(Path(args.output), report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
