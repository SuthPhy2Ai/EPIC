from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from .distributions import gaussian_nll
from .supplement import (
    COVERAGE_LEVELS,
    SIZE_BINS,
    SIZE_LABELS,
    _material_ids_for_predictions,
    _ranking_groups,
    _validate_offsets,
    _weighted_auroc_auprc,
    _weighted_spearman,
)


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def _structure_sums(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 2:
        values = values.sum(axis=1)
    return np.add.reduceat(values, offsets[:-1])


def _force_point_statistics(record: dict[str, np.ndarray], location_field: str) -> dict[str, np.ndarray]:
    truth = np.asarray(record["force_true"], dtype=np.float64)
    location = np.asarray(record[location_field], dtype=np.float64)
    offsets = _validate_offsets(record["force_offsets"], len(truth), len(record["db_index"]))
    if truth.shape != location.shape:
        raise ValueError("external point-force truth and location shapes differ")
    counts = (3 * np.diff(offsets)).astype(np.float64)
    error = location - truth
    return {
        "counts": counts,
        "absolute_sum": _structure_sums(np.abs(error), offsets),
        "squared_sum": _structure_sums(np.square(error), offsets),
    }


def _external_force_uq_statistics(
    force: dict[str, np.ndarray],
    parameters: dict[str, float],
    coverage_levels: tuple[float, ...],
) -> dict[str, object]:
    truth = np.asarray(force["force_true"], dtype=np.float64)
    location = np.asarray(force["force_mean"], dtype=np.float64)
    raw_scale = np.asarray(force["force_scale_raw"], dtype=np.float64)
    offsets = _validate_offsets(force["force_offsets"], len(truth), len(force["db_index"]))
    if truth.shape != location.shape or truth.shape != raw_scale.shape:
        raise ValueError("external force UQ arrays are not aligned")
    calibrated_scale = np.sqrt(
        float(parameters["tau"]) ** 2 * np.square(raw_scale)
        + float(parameters["sigma0"]) ** 2
    )
    absolute = np.abs(location - truth)
    counts = (3 * np.diff(offsets)).astype(np.float64)
    structure_error = _structure_sums(absolute, offsets) / counts
    structure_risk = np.sqrt(_structure_sums(np.square(raw_scale), offsets) / counts)
    risk_order, risk_starts = _ranking_groups(structure_risk)
    return {
        "counts": counts,
        "structure_error": structure_error,
        "structure_risk": structure_risk,
        "error_rank": rankdata(structure_error, method="average"),
        "risk_rank": rankdata(structure_risk, method="average"),
        "danger": structure_error
        >= np.quantile(structure_error, 0.95, method="linear"),
        "risk_order": risk_order,
        "risk_starts": risk_starts,
        "nll_sum": _structure_sums(
            gaussian_nll(truth, location, calibrated_scale), offsets
        ),
        "coverage_sum": {
            level: _structure_sums(
                absolute <= norm.ppf((1 + level) / 2) * calibrated_scale,
                offsets,
            )
            for level in coverage_levels
        },
    }


def _evaluate_external(
    deterministic: dict[str, np.ndarray],
    hierarchical: dict[str, np.ndarray],
    uncertainty: dict[str, object],
    structure_weights: np.ndarray,
    coverage_levels: tuple[float, ...],
) -> dict[str, float]:
    row = {}
    for name, values in (
        ("deterministic_mace", deterministic),
        ("hierarchical_epic", hierarchical),
    ):
        denominator = float(np.dot(structure_weights, values["counts"]))
        row[f"{name}_force_component_mae"] = float(
            np.dot(structure_weights, values["absolute_sum"]) / denominator
        )
        row[f"{name}_force_component_rmse"] = float(
            np.sqrt(np.dot(structure_weights, values["squared_sum"]) / denominator)
        )
    auroc, auprc = _weighted_auroc_auprc(
        uncertainty["danger"],
        structure_weights,
        uncertainty["risk_order"],
        uncertainty["risk_starts"],
    )
    row["hierarchical_force_risk_spearman"] = _weighted_spearman(
        uncertainty["error_rank"], uncertainty["risk_rank"], structure_weights
    )
    row["hierarchical_force_top5_auroc"] = auroc
    row["hierarchical_force_top5_auprc"] = auprc
    denominator = float(np.dot(structure_weights, uncertainty["counts"]))
    row["hierarchical_force_in_domain_calibrated_nll"] = float(
        np.dot(structure_weights, uncertainty["nll_sum"]) / denominator
    )
    for level in coverage_levels:
        name = f"hierarchical_force_in_domain_calibrated_coverage_{int(level * 100):02d}"
        coverage = float(
            np.dot(structure_weights, uncertainty["coverage_sum"][level]) / denominator
        )
        row[name] = coverage
        row[name + "_absolute_error"] = abs(coverage - level)
    return row


def _energy_risk(record: dict[str, np.ndarray]) -> np.ndarray:
    natoms = np.asarray(record["natoms"], dtype=np.float64)
    variance = np.asarray(
        record["energy_aleatoric_variance"] + record["energy_epistemic_variance"],
        dtype=np.float64,
    )
    return np.sqrt(np.maximum(variance, 0.0)) / natoms


def _force_risk(record: dict[str, np.ndarray]) -> np.ndarray:
    offsets = np.asarray(record["force_offsets"], dtype=np.int64)
    scale = np.asarray(record["force_scale_raw"], dtype=np.float64)
    return np.sqrt(_structure_sums(np.square(scale), offsets) / (3 * np.diff(offsets)))


def _detection_setup(
    in_score: np.ndarray,
    external_score: np.ndarray,
    in_material: np.ndarray,
    external_material: np.ndarray,
    in_natoms: np.ndarray,
    external_natoms: np.ndarray,
) -> dict[str, object]:
    score = np.concatenate((in_score, external_score))
    label = np.concatenate(
        (np.zeros(len(in_score), dtype=bool), np.ones(len(external_score), dtype=bool))
    )
    materials = np.concatenate(
        (
            np.char.add("in::", np.asarray(in_material, dtype=str)),
            np.char.add("external::", np.asarray(external_material, dtype=str)),
        )
    )
    cluster_labels, cluster = np.unique(materials, return_inverse=True)
    order, starts = _ranking_groups(score)
    natoms = np.concatenate((in_natoms, external_natoms))
    bins = np.digitize(natoms, SIZE_BINS[1:-1], right=True)
    bin_records = []
    for index, name in enumerate(SIZE_LABELS):
        selected = bins == index
        if np.any(label[selected]) and np.any(~label[selected]):
            local_order, local_starts = _ranking_groups(score[selected])
            bin_records.append((name, selected, local_order, local_starts))
    return {
        "label": label,
        "cluster": cluster,
        "n_clusters": len(cluster_labels),
        "order": order,
        "starts": starts,
        "bins": bin_records,
        "n_in": len(in_score),
        "n_external": len(external_score),
        "in_score": in_score,
        "external_score": external_score,
    }


def _evaluate_detection(setup: dict[str, object], cluster_weights: np.ndarray) -> dict[str, float]:
    weights = cluster_weights[setup["cluster"]]
    auroc, auprc = _weighted_auroc_auprc(
        setup["label"], weights, setup["order"], setup["starts"]
    )
    bin_aurocs, bin_weights = [], []
    for _, selected, order, starts in setup["bins"]:
        bin_auroc, _ = _weighted_auroc_auprc(
            setup["label"][selected], weights[selected], order, starts
        )
        if np.isfinite(bin_auroc):
            bin_aurocs.append(bin_auroc)
            bin_weights.append(float(np.sum(weights[selected])))
    n_in = int(setup["n_in"])
    in_median = _weighted_median(setup["in_score"], weights[:n_in])
    external_median = _weighted_median(setup["external_score"], weights[n_in:])
    return {
        "auroc": auroc,
        "auprc": auprc,
        "size_stratified_auroc": float(np.average(bin_aurocs, weights=bin_weights))
        if bin_aurocs
        else float("nan"),
        "external_to_in_domain_median_risk_ratio": float(
            external_median / max(in_median, 1e-30)
        ),
    }


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("weighted median requires aligned values and positive weights")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    position = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
    return float(values[order[position]])


def _write_figure(metrics: pd.DataFrame, risk_curve: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    estimates = metrics.set_index("metric")["estimate"]
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    names = ("deterministic_mace", "hierarchical_epic")
    axes[0].bar(
        names,
        [estimates[f"{name}_force_component_mae"] for name in names],
        color=("#355C7D", "#0B6E69"),
    )
    axes[0].set_ylabel("Force component MAE (eV/Angstrom)")
    axes[0].set_title("External force point accuracy")
    axes[0].tick_params(axis="x", rotation=18)
    axes[1].plot(
        risk_curve["coverage"],
        risk_curve["mean_structure_absolute_force_error"],
        color="#0B6E69",
        linewidth=2,
    )
    axes[1].set_xlabel("Retained coverage")
    axes[1].set_ylabel("Mean structure force error")
    axes[1].set_title("External risk-coverage")
    axes[1].grid(alpha=0.22)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"external_ood_preliminary.{suffix}", dpi=220)
    plt.close(figure)


def analyze_external_ood(
    deterministic_point_path: str | Path,
    hierarchical_point_path: str | Path,
    hierarchical_force_path: str | Path,
    in_domain_point_path: str | Path,
    in_domain_force_path: str | Path,
    in_domain_store_path: str | Path,
    in_domain_calibration_parameters_path: str | Path,
    selection_manifest_path: str | Path,
    leakage_audit_path: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_replicates: int = 1000,
    seed: int = 42,
    coverage_levels: tuple[float, ...] = COVERAGE_LEVELS,
) -> Path:
    deterministic = _load_npz(deterministic_point_path)
    hierarchical = _load_npz(hierarchical_point_path)
    external_force = _load_npz(hierarchical_force_path)
    in_domain_point = _load_npz(in_domain_point_path)
    in_domain_force = _load_npz(in_domain_force_path)
    selection = json.loads(Path(selection_manifest_path).read_text())
    leakage = json.loads(Path(leakage_audit_path).read_text())
    calibration = json.loads(Path(in_domain_calibration_parameters_path).read_text())
    if calibration.get("fit_split") != "calibration" or calibration.get(
        "test_used_for_selection"
    ) is not False:
        raise ValueError("external evaluation requires calibration-only in-domain parameters")
    if leakage.get("strict_geometry_overlap_count") != 0:
        raise ValueError("external subset has strict geometry overlap with MPtrj")
    if leakage.get("translation_cell_invariant_geometry_overlap_count") != 0:
        raise ValueError(
            "external subset has translation/cell-basis invariant geometry overlap with MPtrj"
        )
    if selection.get("selection_uses_model_predictions") is not False:
        raise ValueError("external subset selection must not use predictions")
    for field in ("db_index", "natoms", "energy_true_total", "force_true", "force_offsets"):
        if not np.array_equal(deterministic[field], hierarchical[field]):
            raise ValueError(f"external point models disagree on {field}")
    for field in ("db_index", "force_true", "force_offsets"):
        if not np.array_equal(external_force[field], hierarchical[field]):
            raise ValueError(f"external force UQ disagrees with point record on {field}")

    deterministic_stats = _force_point_statistics(deterministic, "force_pred")
    hierarchical_stats = _force_point_statistics(hierarchical, "force_pred")
    uncertainty = _external_force_uq_statistics(
        external_force,
        calibration["force"]["gaussian_affine_global"],
        coverage_levels,
    )
    external_material = np.asarray(selection["selected_materials"], dtype=str)
    if len(external_material) != len(hierarchical["db_index"]):
        raise ValueError("external material manifest and predictions differ in length")
    labels, external_cluster = np.unique(external_material, return_inverse=True)
    estimate = _evaluate_external(
        deterministic_stats,
        hierarchical_stats,
        uncertainty,
        np.ones(len(external_material), dtype=np.float64),
        coverage_levels,
    )

    in_energy_material = _material_ids_for_predictions(
        in_domain_store_path, in_domain_point["db_index"]
    )
    in_force_material = _material_ids_for_predictions(
        in_domain_store_path, in_domain_force["db_index"]
    )
    external_natoms = np.asarray(hierarchical["natoms"], dtype=np.int64)
    external_point_index = np.asarray(hierarchical["db_index"], dtype=np.int64)
    external_force_index = np.asarray(external_force["db_index"], dtype=np.int64)
    if np.any(np.diff(external_point_index) <= 0):
        raise ValueError("external point db_index must be strictly increasing")
    force_positions = np.searchsorted(external_point_index, external_force_index)
    if np.any(force_positions >= len(external_point_index)) or not np.array_equal(
        external_point_index[force_positions], external_force_index
    ):
        raise ValueError("external force db_index is not aligned with point predictions")
    external_force_natoms = external_natoms[force_positions]
    in_point_index = np.asarray(in_domain_point["db_index"], dtype=np.int64)
    in_force_index = np.asarray(in_domain_force["db_index"], dtype=np.int64)
    if np.any(np.diff(in_point_index) <= 0):
        raise ValueError("in-domain point db_index must be strictly increasing")
    in_force_positions = np.searchsorted(in_point_index, in_force_index)
    if np.any(in_force_positions >= len(in_point_index)) or not np.array_equal(
        in_point_index[in_force_positions], in_force_index
    ):
        raise ValueError("in-domain force db_index is not aligned with point predictions")
    detection = {
        "energy": _detection_setup(
            _energy_risk(in_domain_point),
            _energy_risk(hierarchical),
            in_energy_material,
            external_material,
            np.asarray(in_domain_point["natoms"], dtype=np.int64),
            external_natoms,
        ),
        "force": _detection_setup(
            _force_risk(in_domain_force),
            _force_risk(external_force),
            in_force_material,
            external_material[force_positions],
            np.asarray(in_domain_point["natoms"], dtype=np.int64)[in_force_positions],
            external_force_natoms,
        ),
    }
    for target, setup in detection.items():
        values = _evaluate_detection(
            setup, np.ones(setup["n_clusters"], dtype=np.float64)
        )
        for metric, value in values.items():
            estimate[f"ood_detection_{target}_{metric}"] = value

    rng = np.random.default_rng(seed)
    bootstrap_rows = []
    for replicate in range(bootstrap_replicates):
        external_cluster_weights = np.bincount(
            rng.integers(0, len(labels), len(labels)), minlength=len(labels)
        ).astype(np.float64)
        row = {
            "replicate": replicate,
            **_evaluate_external(
                deterministic_stats,
                hierarchical_stats,
                uncertainty,
                external_cluster_weights[external_cluster],
                coverage_levels,
            ),
        }
        for target, setup in detection.items():
            cluster_weights = np.bincount(
                rng.integers(0, setup["n_clusters"], setup["n_clusters"]),
                minlength=setup["n_clusters"],
            ).astype(np.float64)
            for metric, value in _evaluate_detection(setup, cluster_weights).items():
                row[f"ood_detection_{target}_{metric}"] = value
        bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows)
    metric_rows = []
    for metric, value in estimate.items():
        samples = bootstrap[metric].to_numpy(dtype=np.float64)
        samples = samples[np.isfinite(samples)]
        low, high = np.quantile(samples, [0.025, 0.975])
        metric_rows.append(
            {
                "metric": metric,
                "estimate": value,
                "ci_low_95": float(low),
                "ci_high_95": float(high),
                "valid_bootstrap_replicates": len(samples),
                "bootstrap_unit": "material_id",
                "bootstrap_replicates": bootstrap_replicates,
                "seed": seed,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    order = np.argsort(uncertainty["structure_risk"], kind="stable")
    risk_rows = []
    for coverage_value in np.linspace(0.01, 1.0, 100):
        retained = max(1, int(np.ceil(coverage_value * len(order))))
        risk_rows.append(
            {
                "coverage": coverage_value,
                "n_retained": retained,
                "mean_structure_absolute_force_error": float(
                    np.mean(uncertainty["structure_error"][order[:retained]])
                ),
            }
        )
    risk_curve = pd.DataFrame(risk_rows)

    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "external_ood_metrics.csv", index=False)
    bootstrap.to_csv(tables / "external_ood_bootstrap.csv", index=False)
    risk_curve.to_csv(tables / "external_ood_risk_coverage.csv", index=False)
    summary = {
        "schema_version": 1,
        "analysis": "elementa_open_external_domain_preliminary",
        "result_status": "preliminary_not_for_formal_ood_claim",
        "formal_external_ood_claim_permitted": False,
        "selection_uses_model_predictions": False,
        "strict_geometry_overlap_count": 0,
        "translation_cell_invariant_geometry_overlap_count": 0,
        "in_domain_calibration_reused_without_external_refit": True,
        "fit_split": "in_domain_calibration",
        "evaluation_split": "external_test",
        "external_test_used_for_selection": False,
        "n_external_structures": len(external_material),
        "bootstrap_unit": "material_id",
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "energy_error_reporting_limit": (
            "Absolute cross-dataset energy errors are withheld until DFT energy conventions "
            "and elemental reference offsets are verified."
        ),
        "remaining_gates": leakage.get("remaining_gates", []),
        "headline": estimate,
    }
    (output_dir / "external_ood_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_figure(metrics, risk_curve, output_dir / "figures")
    return output_dir
