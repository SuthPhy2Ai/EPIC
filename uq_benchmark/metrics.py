from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, t
from sklearn.metrics import average_precision_score, roc_auc_score

from .distributions import gaussian_nll, student_t_nll


@dataclass(frozen=True)
class MetricEstimate:
    method: str
    target: str
    metric: str
    estimate: float
    unit: str
    n: int
    distribution_family: str
    calibration: str


def point_metrics(
    method: str,
    energy_true_pa: np.ndarray,
    energy_pred_pa: np.ndarray,
    force_true: np.ndarray | None = None,
    force_pred: np.ndarray | None = None,
) -> list[MetricEstimate]:
    energy_error = np.asarray(energy_pred_pa) - np.asarray(energy_true_pa)
    rows = [
        MetricEstimate(method, "energy_per_atom", "MAE", float(np.mean(np.abs(energy_error))), "eV/atom", len(energy_error), "point", "none"),
        MetricEstimate(method, "energy_per_atom", "RMSE", float(np.sqrt(np.mean(energy_error**2))), "eV/atom", len(energy_error), "point", "none"),
    ]
    if force_true is not None and force_pred is not None:
        force_error = np.asarray(force_pred) - np.asarray(force_true)
        if force_error.ndim != 2 or force_error.shape[1] != 3:
            raise ValueError("force arrays must have shape [n_atoms, 3]")
        vector = np.linalg.norm(force_error, axis=1)
        rows.extend(
            [
                MetricEstimate(method, "force_component", "MAE", float(np.mean(np.abs(force_error))), "eV/Angstrom", force_error.size, "point", "none"),
                MetricEstimate(method, "force_component", "RMSE", float(np.sqrt(np.mean(force_error**2))), "eV/Angstrom", force_error.size, "point", "none"),
                MetricEstimate(method, "force_vector", "MAE", float(np.mean(vector)), "eV/Angstrom", len(vector), "point", "none"),
                MetricEstimate(method, "force_vector", "RMSE", float(np.sqrt(np.mean(vector**2))), "eV/Angstrom", len(vector), "point", "none"),
            ]
        )
    return rows


def calibration_metrics(
    method: str,
    target: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
    family: str,
    levels: tuple[float, ...],
    calibrated: bool,
    df: np.ndarray | None = None,
    unit: str = "eV/atom",
) -> list[MetricEstimate]:
    target, location, scale = map(np.asarray, (target, location, scale))
    tag = "calibrated" if calibrated else "raw"
    if family == "gaussian":
        nll = gaussian_nll(target, location, scale)
        pit = norm.cdf((target - location) / scale)
    elif family == "student_t":
        if df is None:
            raise ValueError("Student-t metrics require degrees of freedom")
        nll = student_t_nll(target, location, scale, df)
        pit = t.cdf((target - location) / scale, df)
    else:
        raise ValueError(f"unsupported family: {family}")
    rows = [
        MetricEstimate(method, "energy_per_atom", "NLL", float(np.mean(nll)), "nats", len(target), family, tag),
        MetricEstimate(method, "energy_per_atom", "PIT_ECE_20", float(_pit_ece(pit, 20)), "absolute_probability", len(target), family, tag),
    ]
    coverage_errors = []
    for level in levels:
        quantile = norm.ppf((1 + level) / 2) if family == "gaussian" else t.ppf((1 + level) / 2, df)
        covered = np.abs(target - location) <= quantile * scale
        empirical = float(np.mean(covered))
        width = float(np.mean(2.0 * quantile * scale))
        coverage_errors.append(abs(empirical - level))
        rows.append(MetricEstimate(method, "energy_per_atom", f"coverage_{int(level*100):02d}", empirical, "fraction", len(target), family, tag))
        rows.append(MetricEstimate(method, "energy_per_atom", f"interval_width_{int(level*100):02d}", width, unit, len(target), family, tag))
    rows.append(MetricEstimate(method, "energy_per_atom", "mean_absolute_coverage_error", float(np.mean(coverage_errors)), "fraction", len(target), family, tag))
    return rows


def ranking_metrics(
    method: str,
    absolute_error: np.ndarray,
    risk: np.ndarray,
    danger_quantile: float = 0.95,
) -> tuple[list[MetricEstimate], pd.DataFrame]:
    absolute_error, risk = map(np.asarray, (absolute_error, risk))
    if len(absolute_error) != len(risk) or len(risk) == 0 or np.any(~np.isfinite(risk)):
        raise ValueError("error and risk must be finite matching arrays")
    threshold = float(np.quantile(absolute_error, danger_quantile, method="linear"))
    danger = absolute_error >= threshold
    rho = (
        float("nan")
        if np.ptp(risk) == 0 or np.ptp(absolute_error) == 0
        else float(spearmanr(risk, absolute_error).statistic)
    )
    auroc = float("nan") if np.all(danger == danger[0]) else float(roc_auc_score(danger, risk))
    rows = [
        MetricEstimate(method, "energy_per_atom", "Spearman", rho, "correlation", len(risk), "risk_score", "none"),
        MetricEstimate(method, "energy_per_atom", "top5_AUROC", auroc, "area", len(risk), "risk_score", "none"),
        MetricEstimate(method, "energy_per_atom", "top5_AUPRC", float(average_precision_score(danger, risk)), "area", len(risk), "risk_score", "none"),
    ]
    order = np.argsort(risk, kind="stable")
    coverages = np.linspace(0.01, 1.0, 100)
    curve = []
    for coverage in coverages:
        retained = max(1, int(np.ceil(coverage * len(risk))))
        curve.append((coverage, retained, float(np.mean(absolute_error[order[:retained]]))))
    curve_frame = pd.DataFrame(curve, columns=["coverage", "n_retained", "mean_absolute_error"])
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    aurc = float(trapz_fn(curve_frame.mean_absolute_error, curve_frame.coverage))
    rows.append(MetricEstimate(method, "energy_per_atom", "AURC_0p01_1", aurc, "eV/atom", len(risk), "risk_score", "none"))
    return rows, curve_frame


def budget_recall(
    method: str,
    absolute_error: np.ndarray,
    risk: np.ndarray,
    budgets: tuple[float, ...],
    danger_quantile: float = 0.95,
) -> pd.DataFrame:
    absolute_error, risk = map(np.asarray, (absolute_error, risk))
    danger = absolute_error >= np.quantile(absolute_error, danger_quantile, method="linear")
    rows = []
    for budget in budgets:
        target_n = max(1, int(np.ceil(budget * len(risk))))
        boundary = np.partition(risk, len(risk) - target_n)[len(risk) - target_n]
        selected = risk >= boundary
        rows.append(
            {
                "method": method,
                "target_budget_fraction": budget,
                "effective_budget_fraction": float(selected.mean()),
                "n_selected": int(selected.sum()),
                "danger_recall": float(np.count_nonzero(selected & danger) / np.count_nonzero(danger)),
                "random_expected_recall": float(selected.mean()),
                "tie_rule": "include complete score group at threshold",
            }
        )
    return pd.DataFrame(rows)


def cluster_bootstrap(
    values: dict[str, np.ndarray],
    material_id: np.ndarray,
    replicates: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    material_id = np.asarray(material_id).astype(str)
    labels, cluster = np.unique(material_id, return_inverse=True)
    if any(len(np.asarray(value)) != len(material_id) for value in values.values()):
        raise ValueError("all bootstrap values must have one row per structure")
    rng = np.random.default_rng(seed)
    output = []
    for replicate in range(replicates):
        cluster_weight = np.bincount(rng.integers(0, len(labels), len(labels)), minlength=len(labels))
        weight = cluster_weight[cluster].astype(float)
        row = {"replicate": replicate, "n_clusters": len(labels), "effective_rows": int(weight.sum())}
        for name, value in values.items():
            value = np.asarray(value, dtype=float)
            row[name] = float(np.dot(weight, value) / weight.sum())
        output.append(row)
    return pd.DataFrame(output)


def confidence_interval(samples: np.ndarray) -> tuple[float, float]:
    return tuple(np.quantile(np.asarray(samples), [0.025, 0.975]).astype(float))


def _pit_ece(pit: np.ndarray, bins: int) -> float:
    counts, _ = np.histogram(np.asarray(pit), bins=np.linspace(0, 1, bins + 1))
    return float(np.mean(np.abs(counts / counts.sum() - 1.0 / bins)))


def metric_frame(rows: list[MetricEstimate]) -> pd.DataFrame:
    return pd.DataFrame([row.__dict__ for row in rows])
