from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .metrics import (
    budget_recall,
    calibration_metrics,
    cluster_bootstrap,
    confidence_interval,
    metric_frame,
    point_metrics,
    ranking_metrics,
)
from .figures import write_report_figures


FORCE_SIZE_BINS = np.asarray([0, 20, 40, 80, 160, np.inf])
FORCE_SIZE_LABELS = ("1-20", "21-40", "41-80", "81-160", ">160")


def structure_force_error(force_true: np.ndarray, force_pred: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    component_error = np.abs(np.asarray(force_pred) - np.asarray(force_true))
    return np.asarray(
        [component_error[offsets[i] : offsets[i + 1]].mean() for i in range(len(offsets) - 1)]
    )


def generate_report(
    store_path: str | Path,
    output_dir: str | Path,
    coverage_levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95),
    budget_fractions: tuple[float, ...] = (0.01, 0.05, 0.10),
    bootstrap_replicates: int = 1000,
    seed: int = 42,
) -> Path:
    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    rows = []
    risk_curves, budget_tables = [], []
    force_size_rows = []
    runtime_rows = []
    bootstrap_values = {}
    with h5py.File(store_path, "r") as handle:
        structure = handle["structures"]
        db_index = structure["db_index"][:]
        material_id = structure["material_id"].asstr()[:]
        energy_true = structure["energy_true_pa"][:]
        force_true = structure["force_true"][:]
        offsets = structure["force_offsets"][:]
        for method, group in handle["methods"].items():
            energy_pred = group["energy_location"][:]
            force_pred = group["force_location"][:] if "force_location" in group else None
            rows.extend(point_metrics(method, energy_true, energy_pred, force_true, force_pred))
            runtime_rows.append(
                {
                    "method": method,
                    "stage": "point_test",
                    "n_structures": len(db_index),
                    "summed_shard_device_seconds": float(
                        group.attrs.get("inference_seconds", 0.0)
                    ),
                    "peak_memory_bytes_per_device": int(
                        group.attrs.get("peak_memory_bytes", 0)
                    ),
                    "timing_scope": "sum of per-shard inference timers; not allocation wall time",
                }
            )
            energy_abs = np.abs(energy_pred - energy_true)
            bootstrap_values[f"{method}__energy_MAE"] = energy_abs
            bootstrap_values[f"{method}__energy_MSE"] = (energy_pred - energy_true) ** 2
            family = group.attrs["distribution_family"]
            if family in ("gaussian", "student_t"):
                df = group["energy_df"][:] if "energy_df" in group else None
                for scale_name, calibration in (("energy_scale_raw", "raw"), ("energy_scale_calibrated", "calibrated")):
                    if scale_name in group:
                        rows.extend(
                            calibration_metrics(
                                method,
                                energy_true,
                                energy_pred,
                                group[scale_name][:],
                                family,
                                coverage_levels,
                                calibrated=calibration == "calibrated",
                                df=df,
                            )
                        )
            if "energy_risk" in group:
                ranking_rows, curve = ranking_metrics(method, energy_abs, group["energy_risk"][:])
                rows.extend(ranking_rows)
                curve.insert(0, "method", method)
                curve.insert(1, "target", "energy_per_atom")
                risk_curves.append(curve)
                energy_budget = budget_recall(
                    method,
                    energy_abs,
                    group["energy_risk"][:],
                    budget_fractions,
                )
                energy_budget.insert(1, "target", "energy_per_atom")
                budget_tables.append(energy_budget)
                if force_pred is not None:
                    force_error = structure_force_error(force_true, force_pred, offsets)
                    force_rows, force_curve = ranking_metrics(method, force_error, group["energy_risk"][:])
                    force_rows = [row.__class__(row.method, "force_component_structure_MAE", row.metric, row.estimate, "eV/Angstrom" if row.metric.startswith("AURC") else row.unit, row.n, row.distribution_family, row.calibration) for row in force_rows]
                    rows.extend(force_rows)
                    force_curve.insert(0, "method", method)
                    force_curve.insert(1, "target", "force_component_structure_MAE")
                    risk_curves.append(force_curve)

            if "force_uq_subset" in group:
                subset = group["force_uq_subset"]
                force_metadata = json.loads(subset.attrs.get("metadata_json", "{}"))
                runtime_rows.extend(
                    [
                        {
                            "method": method,
                            "stage": "force_uq_calibration",
                            "n_structures": int(
                                force_metadata.get("calibration_n_structures", 0)
                            ),
                            "summed_shard_device_seconds": float(
                                force_metadata.get(
                                    "calibration_inference_seconds_summed_shards", 0.0
                                )
                            ),
                            "peak_memory_bytes_per_device": int(
                                force_metadata.get(
                                    "calibration_peak_memory_bytes_per_device", 0
                                )
                            ),
                            "timing_scope": "sum of per-shard inference timers; not allocation wall time",
                        },
                        {
                            "method": method,
                            "stage": "force_uq_test",
                            "n_structures": len(subset["db_index"]),
                            "summed_shard_device_seconds": float(
                                force_metadata.get(
                                    "test_inference_seconds_summed_shards", 0.0
                                )
                            ),
                            "peak_memory_bytes_per_device": int(
                                force_metadata.get(
                                    "test_peak_memory_bytes_per_device", 0
                                )
                            ),
                            "timing_scope": "sum of per-shard inference timers; not allocation wall time",
                        },
                    ]
                )
                force_family = subset.attrs.get("distribution_family", "gaussian")
                subset_force_true = subset["force_true"][:]
                subset_force_location = subset["force_location"][:]
                subset_offsets = subset["force_offsets"][:]
                subset_natoms = np.diff(subset_offsets)
                subset_force_scale_raw = subset["force_scale_raw"][:].astype(
                    np.float64
                )
                force_component_error = np.abs(
                    subset_force_location - subset_force_true
                ).reshape(-1)
                force_component_risk = np.square(subset_force_scale_raw).reshape(-1)
                force_ranking_rows, force_component_curve = ranking_metrics(
                    method,
                    force_component_error,
                    force_component_risk,
                )
                force_ranking_rows = [
                    row.__class__(
                        row.method,
                        "force_component_subset",
                        row.metric,
                        row.estimate,
                        "eV/Angstrom" if row.metric.startswith("AURC") else row.unit,
                        row.n,
                        row.distribution_family,
                        row.calibration,
                    )
                    for row in force_ranking_rows
                ]
                rows.extend(force_ranking_rows)
                force_component_curve.insert(0, "method", method)
                force_component_curve.insert(1, "target", "force_component_subset")
                risk_curves.append(force_component_curve)

                structure_error = structure_force_error(
                    subset_force_true,
                    subset_force_location,
                    subset_offsets,
                )
                structure_risk = np.asarray(
                    [
                        np.mean(
                            np.square(subset_force_scale_raw[start:stop])
                        )
                        for start, stop in zip(
                            subset_offsets[:-1], subset_offsets[1:], strict=True
                        )
                    ]
                )
                structure_ranking_rows, structure_curve = ranking_metrics(
                    method,
                    structure_error,
                    structure_risk,
                )
                structure_ranking_rows = [
                    row.__class__(
                        row.method,
                        "force_component_structure_MAE_subset",
                        row.metric,
                        row.estimate,
                        "eV/Angstrom" if row.metric.startswith("AURC") else row.unit,
                        row.n,
                        row.distribution_family,
                        row.calibration,
                    )
                    for row in structure_ranking_rows
                ]
                rows.extend(structure_ranking_rows)
                structure_curve.insert(0, "method", method)
                structure_curve.insert(
                    1, "target", "force_component_structure_MAE_subset"
                )
                risk_curves.append(structure_curve)
                structure_budget = budget_recall(
                    method,
                    structure_error,
                    structure_risk,
                    budget_fractions,
                )
                structure_budget.insert(
                    1, "target", "force_component_structure_MAE_subset"
                )
                budget_tables.append(structure_budget)

                atom_size_group = np.repeat(
                    np.digitize(subset_natoms, FORCE_SIZE_BINS[1:-1], right=True),
                    subset_natoms,
                )
                structure_size_group = np.digitize(
                    subset_natoms, FORCE_SIZE_BINS[1:-1], right=True
                )
                component_size_group = np.repeat(atom_size_group, 3)
                for scale_name, calibration in (("force_scale_raw", "raw"), ("force_scale_calibrated", "calibrated")):
                    target = subset_force_true.reshape(-1)
                    location = subset_force_location.reshape(-1)
                    scale = subset[scale_name][:].reshape(-1)
                    force_rows = calibration_metrics(
                        method,
                        target,
                        location,
                        np.maximum(scale, 1e-12),
                        force_family,
                        coverage_levels,
                        calibrated=calibration == "calibrated",
                        unit="eV/Angstrom",
                    )
                    force_rows = [row.__class__(row.method, "force_component", row.metric, row.estimate, row.unit, row.n, row.distribution_family, row.calibration) for row in force_rows]
                    rows.extend(force_rows)
                    for size_index, size_label in enumerate(FORCE_SIZE_LABELS):
                        selected_structures = structure_size_group == size_index
                        selected_components = component_size_group == size_index
                        if not np.any(selected_components):
                            continue
                        size_metrics = calibration_metrics(
                            method,
                            target[selected_components],
                            location[selected_components],
                            np.maximum(scale[selected_components], 1e-12),
                            force_family,
                            coverage_levels,
                            calibrated=calibration == "calibrated",
                            unit="eV/Angstrom",
                        )
                        size_row = {
                            "method": method,
                            "size_bin": size_label,
                            "calibration": calibration,
                            "n_structures": int(selected_structures.sum()),
                            "n_force_components": int(selected_components.sum()),
                            "natoms_min": int(subset_natoms[selected_structures].min()),
                            "natoms_max": int(subset_natoms[selected_structures].max()),
                            "force_component_mae_ev_per_angstrom": float(
                                force_component_error[selected_components].mean()
                            ),
                            "force_component_rmse_ev_per_angstrom": float(
                                np.sqrt(
                                    np.mean(
                                        np.square(
                                            location[selected_components]
                                            - target[selected_components]
                                        )
                                    )
                                )
                            ),
                        }
                        size_row.update(
                            {metric.metric: metric.estimate for metric in size_metrics}
                        )
                        force_size_rows.append(size_row)

        # Paired differences use identical material-cluster bootstrap draws.
        methods = sorted(handle["methods"].keys())
        for left_index, left in enumerate(methods):
            for right in methods[left_index + 1 :]:
                bootstrap_values[f"delta_MAE__{left}__minus__{right}"] = (
                    bootstrap_values[f"{left}__energy_MAE"] - bootstrap_values[f"{right}__energy_MAE"]
                )
        bootstrap = cluster_bootstrap(bootstrap_values, material_id, bootstrap_replicates, seed)

    metrics = metric_frame(rows)
    metrics.to_csv(tables / "metrics.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(tables / "runtime_cost.csv", index=False)
    risk_frame = pd.concat(risk_curves, ignore_index=True) if risk_curves else None
    if risk_frame is not None:
        risk_frame.to_csv(tables / "risk_coverage.csv", index=False)
    if budget_tables:
        pd.concat(budget_tables, ignore_index=True).to_csv(tables / "budget_recall.csv", index=False)
    force_size_frame = pd.DataFrame(force_size_rows) if force_size_rows else None
    if force_size_frame is not None:
        force_size_frame.to_csv(tables / "force_uq_size_stratified.csv", index=False)
    bootstrap.to_csv(tables / "paired_cluster_bootstrap.csv", index=False)
    ci_rows = []
    for column in bootstrap.columns:
        if column in ("replicate", "n_clusters", "effective_rows"):
            continue
        low, high = confidence_interval(bootstrap[column])
        ci_rows.append(
            {
                "metric": column,
                "estimate": float(np.mean(bootstrap_values[column])),
                "ci_low_95": low,
                "ci_high_95": high,
                "bootstrap_unit": "test material_id",
                "bootstrap_replicates": bootstrap_replicates,
                "seed": seed,
                "n": len(db_index),
            }
        )
    pd.DataFrame(ci_rows).to_csv(tables / "paired_cluster_confidence_intervals.csv", index=False)
    manifest = {
        "schema_version": 1,
        "store": str(Path(store_path).resolve()),
        "n_structures": len(db_index),
        "coverage_levels": coverage_levels,
        "budget_fractions": budget_fractions,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "labels": {
            "budget": "retrospective ranking only; not an active-learning gain",
            "downstream": "MLIP-predicted; no DFT validation",
        },
    }
    (output_dir / "report_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_report_figures(metrics, risk_frame, force_size_frame, output_dir / "figures")
    return output_dir
