from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .calibration import fit_gaussian_variance_map
from .method_comparison import (
    MethodFiles,
    _evaluate,
    _force_arrays,
    _point_arrays,
    _statistics,
    _verify_point_force_alignment,
)
from .supplement import COVERAGE_LEVELS, _material_ids_for_predictions


TRAINING_RUNS = (
    "seed42_reference",
    "seed43",
    "seed44",
    "allocation_weight_0_seed42",
)
SEED_RUNS = TRAINING_RUNS[:3]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def _align_records(records: dict[str, dict[str, dict[str, np.ndarray]]]) -> None:
    reference = records["seed42_reference"]
    for split in ("calibration", "test"):
        reference_energy = reference[f"energy_{split}"]
        reference_force = reference[f"force_{split}"]
        point_index = np.asarray(reference_energy["db_index"], dtype=np.int64)
        force_index = np.asarray(reference_force["db_index"], dtype=np.int64)
        if np.any(np.diff(point_index) <= 0):
            raise ValueError(f"reference {split} point db_index must be strictly increasing")
        positions = np.searchsorted(point_index, force_index)
        if np.any(positions >= len(point_index)) or not np.array_equal(
            point_index[positions], force_index
        ):
            raise ValueError(f"reference {split} force subset is not aligned")
        for run in TRAINING_RUNS:
            energy = records[run][f"energy_{split}"]
            force = records[run][f"force_{split}"]
            for field in ("db_index", "natoms", "energy_true_total"):
                if not np.array_equal(energy[field], reference_energy[field]):
                    raise ValueError(f"{run} {split} point record disagrees on {field}")
            for field in ("db_index", "force_offsets", "force_true"):
                if not np.array_equal(force[field], reference_force[field]):
                    raise ValueError(f"{run} {split} force record disagrees on {field}")
            if not np.array_equal(np.diff(force["force_offsets"]), energy["natoms"][positions]):
                raise ValueError(f"{run} {split} force offsets disagree with natoms")
            point_force = np.asarray(energy["force_pred"], dtype=np.float64)
            point_offsets = np.asarray(energy["force_offsets"], dtype=np.int64)
            selected = np.concatenate(
                [point_force[point_offsets[index] : point_offsets[index + 1]] for index in positions]
            )
            _verify_point_force_alignment(
                selected,
                force["force_mean"],
                f"{run} {split}",
            )


def _write_figure(seed_summary: pd.DataFrame, allocation: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    selected = seed_summary[
        (seed_summary.target == "force_component")
        & seed_summary.metric.isin(["point_mae", "risk_spearman"])
    ]
    for metric, group in selected.groupby("metric"):
        axes[0].errorbar(
            group.metric,
            group.seed_mean,
            yerr=group.seed_std,
            marker="o",
            capsize=4,
            label=metric,
        )
    axes[0].set_title("Seed 42/43/44 robustness")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)
    allocation_selected = allocation[
        (allocation.target == "force_component")
        & allocation.metric.isin(["point_mae", "risk_spearman", "gaussian_affine_nll"])
    ]
    axes[1].bar(
        allocation_selected.metric,
        allocation_selected.estimate,
        color="#C65D3B",
    )
    axes[1].axhline(0, color="#263238", linewidth=1)
    axes[1].set_title("Weight-0 minus learned seed-42")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.22)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"training_ablation_comparison.{suffix}", dpi=220)
    plt.close(figure)


def compare_training_ablations(
    files: dict[str, MethodFiles],
    atomic_summary_paths: dict[str, Path],
    store_path: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_replicates: int = 1000,
    seed: int = 42,
    coverage_levels: tuple[float, ...] = COVERAGE_LEVELS,
) -> Path:
    if tuple(files) != TRAINING_RUNS or tuple(atomic_summary_paths) != TRAINING_RUNS:
        raise ValueError(f"training ablation inputs must be ordered as {TRAINING_RUNS}")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    records = {
        run: {
            "energy_calibration": _load_npz(paths.energy_calibration),
            "energy_test": _load_npz(paths.energy_test),
            "force_calibration": _load_npz(paths.force_calibration),
            "force_test": _load_npz(paths.force_test),
        }
        for run, paths in files.items()
    }
    _align_records(records)
    statistics = {"energy_per_atom": {}, "force_component": {}}
    parameters = {}
    for run, paths in files.items():
        energy_calibration = _point_arrays(records[run]["energy_calibration"], None)
        energy_test = _point_arrays(records[run]["energy_test"], None)
        energy_state = fit_gaussian_variance_map(
            *energy_calibration, method=f"{run}_energy"
        )
        statistics["energy_per_atom"][run] = _statistics(
            *energy_test, dict(energy_state.parameters), coverage_levels
        )
        force_calibration = _force_arrays(records[run]["force_calibration"])
        force_test = _force_arrays(records[run]["force_test"])
        force_state = fit_gaussian_variance_map(
            force_calibration[0].reshape(-1),
            force_calibration[1].reshape(-1),
            np.square(force_calibration[2].reshape(-1)),
            method=f"{run}_force",
        )
        statistics["force_component"][run] = _statistics(
            force_test[0],
            force_test[1],
            np.square(force_test[2]),
            dict(force_state.parameters),
            coverage_levels,
            offsets=force_test[3],
        )
        parameters[run] = {
            "energy_per_atom": dict(energy_state.parameters),
            "force_component": dict(force_state.parameters),
        }

    energy_index = records["seed42_reference"]["energy_test"]["db_index"]
    force_index = records["seed42_reference"]["force_test"]["db_index"]
    target_groups = {}
    for target, index in (("energy_per_atom", energy_index), ("force_component", force_index)):
        material = _material_ids_for_predictions(store_path, index)
        labels, cluster = np.unique(material, return_inverse=True)
        target_groups[target] = (labels, cluster)

    estimate = {}
    for target in statistics:
        labels, cluster = target_groups[target]
        weights = np.ones(len(labels), dtype=np.float64)
        for run in TRAINING_RUNS:
            for metric, value in _evaluate(
                statistics[target][run], weights, cluster, coverage_levels
            ).items():
                estimate[f"{target}__{run}__{metric}"] = value

    rng = np.random.default_rng(seed)
    bootstrap_rows = []
    for replicate in range(bootstrap_replicates):
        row = {"replicate": replicate}
        for target in statistics:
            labels, cluster = target_groups[target]
            weights = np.bincount(
                rng.integers(0, len(labels), len(labels)), minlength=len(labels)
            ).astype(np.float64)
            evaluated = {
                run: _evaluate(statistics[target][run], weights, cluster, coverage_levels)
                for run in TRAINING_RUNS
            }
            for run, values in evaluated.items():
                for metric, value in values.items():
                    row[f"{target}__{run}__{metric}"] = value
            for run in TRAINING_RUNS[1:]:
                for metric, value in evaluated[run].items():
                    row[f"{target}__{run}_minus_seed42__{metric}"] = (
                        value - evaluated["seed42_reference"][metric]
                    )
        bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows)

    metric_rows = []
    for key, value in estimate.items():
        target, run, metric = key.split("__", 2)
        samples = bootstrap[key].to_numpy(dtype=np.float64)
        samples = samples[np.isfinite(samples)]
        low, high = np.quantile(samples, [0.025, 0.975])
        metric_rows.append(
            {
                "target": target,
                "run": run,
                "metric": metric,
                "estimate": value,
                "ci_low_95": float(low),
                "ci_high_95": float(high),
                "valid_bootstrap_replicates": len(samples),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    difference_rows = []
    for column in bootstrap.columns:
        if "_minus_seed42__" not in column:
            continue
        target, comparison, metric = column.split("__", 2)
        run = comparison.removesuffix("_minus_seed42")
        samples = bootstrap[column].to_numpy(dtype=np.float64)
        samples = samples[np.isfinite(samples)]
        point_difference = estimate[f"{target}__{run}__{metric}"] - estimate[
            f"{target}__seed42_reference__{metric}"
        ]
        low, high = np.quantile(samples, [0.025, 0.975])
        difference_rows.append(
            {
                "target": target,
                "comparison": comparison,
                "metric": metric,
                "estimate": point_difference,
                "ci_low_95": float(low),
                "ci_high_95": float(high),
                "valid_bootstrap_replicates": len(samples),
            }
        )
    differences = pd.DataFrame(difference_rows)
    seed_rows = []
    for (target, metric), group in metrics[metrics.run.isin(SEED_RUNS)].groupby(
        ["target", "metric"]
    ):
        values = group.set_index("run").loc[list(SEED_RUNS), "estimate"].to_numpy()
        seed_rows.append(
            {
                "target": target,
                "metric": metric,
                "seed_mean": float(np.mean(values)),
                "seed_std": float(np.std(values, ddof=1)),
                "seed_min": float(np.min(values)),
                "seed_max": float(np.max(values)),
                "n_seeds": 3,
            }
        )
    seed_summary = pd.DataFrame(seed_rows)
    allocation = differences[
        differences.comparison == "allocation_weight_0_seed42_minus_seed42"
    ].reset_index(drop=True)

    atomic_frames = []
    atomic_bootstrap = {}
    for run, path in atomic_summary_paths.items():
        path = Path(path)
        summary = json.loads(path.read_text())
        if (
            summary["bootstrap_unit"] != "material_id"
            or summary["bootstrap_replicates"] != bootstrap_replicates
            or summary["seed"] != seed
        ):
            raise ValueError(f"{run} atomic-attribution bootstrap protocol differs")
        table_dir = path.parent / "tables"
        confidence = pd.read_csv(
            table_dir / "atomic_attribution_confidence_intervals.csv"
        )
        confidence.insert(0, "run", run)
        atomic_frames.append(confidence)
        atomic_bootstrap[run] = pd.read_csv(
            table_dir / "atomic_attribution_bootstrap.csv"
        )
    atomic = pd.concat(atomic_frames, ignore_index=True)
    reference_atomic_bootstrap = atomic_bootstrap["seed42_reference"]
    identity_columns = ["replicate", "n_clusters"]
    metric_columns = [
        column
        for column in reference_atomic_bootstrap.columns
        if column not in identity_columns
    ]
    atomic_estimate = atomic.set_index(["run", "metric"])["estimate"]
    atomic_difference_rows = []
    atomic_bootstrap_differences = reference_atomic_bootstrap[identity_columns].copy()
    for run in TRAINING_RUNS[1:]:
        observed = atomic_bootstrap[run]
        if list(observed.columns) != list(reference_atomic_bootstrap.columns):
            raise ValueError(f"{run} atomic-attribution bootstrap columns differ")
        for field in identity_columns:
            if not np.array_equal(observed[field], reference_atomic_bootstrap[field]):
                raise ValueError(f"{run} atomic-attribution bootstrap {field} differs")
        for metric in metric_columns:
            comparison = f"{run}_minus_seed42"
            samples = observed[metric].to_numpy(dtype=np.float64) - (
                reference_atomic_bootstrap[metric].to_numpy(dtype=np.float64)
            )
            column = f"{comparison}__{metric}"
            atomic_bootstrap_differences[column] = samples
            finite = samples[np.isfinite(samples)]
            if len(finite) == 0:
                raise ValueError(f"{run} atomic-attribution difference is non-finite")
            low, high = np.quantile(finite, [0.025, 0.975])
            atomic_difference_rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "estimate": float(
                        atomic_estimate.loc[(run, metric)]
                        - atomic_estimate.loc[("seed42_reference", metric)]
                    ),
                    "ci_low_95": float(low),
                    "ci_high_95": float(high),
                    "valid_bootstrap_replicates": len(finite),
                }
            )
    atomic_differences = pd.DataFrame(atomic_difference_rows)
    atomic_seed_rows = []
    for metric in metric_columns:
        values = np.asarray(
            [atomic_estimate.loc[(run, metric)] for run in SEED_RUNS],
            dtype=np.float64,
        )
        atomic_seed_rows.append(
            {
                "metric": metric,
                "seed_mean": float(np.mean(values)),
                "seed_std": float(np.std(values, ddof=1)),
                "seed_min": float(np.min(values)),
                "seed_max": float(np.max(values)),
                "n_seeds": 3,
            }
        )
    atomic_seed_summary = pd.DataFrame(atomic_seed_rows)

    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "training_ablation_metrics.csv", index=False)
    differences.to_csv(tables / "training_ablation_paired_differences.csv", index=False)
    seed_summary.to_csv(tables / "seed_robustness_summary.csv", index=False)
    atomic.to_csv(tables / "training_ablation_atomic_attribution.csv", index=False)
    atomic_differences.to_csv(
        tables / "training_ablation_atomic_paired_differences.csv", index=False
    )
    atomic_seed_summary.to_csv(
        tables / "atomic_seed_robustness_summary.csv", index=False
    )
    atomic_bootstrap_differences.to_csv(
        tables / "training_ablation_atomic_bootstrap_differences.csv", index=False
    )
    bootstrap.to_csv(tables / "training_ablation_bootstrap.csv", index=False)
    summary = {
        "schema_version": 1,
        "comparison": "hierarchical_training_seed_and_allocation_weight_ablations",
        "runs": list(TRAINING_RUNS),
        "seed_runs": list(SEED_RUNS),
        "fit_split": "calibration",
        "evaluation_split": "test",
        "test_used_for_selection": False,
        "force_subset_alignment_verified": True,
        "point_force_alignment_verified": True,
        "atomic_attribution_paired_bootstrap_verified": True,
        "bootstrap_unit": "material_id",
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "calibration_parameters": parameters,
        "interpretation_limit": (
            "The allocation_weight=0 result is a separately trained model and complements, "
            "rather than replaces, the inference-time learned-versus-uniform isolation."
        ),
    }
    (output_dir / "training_ablation_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_figure(seed_summary, allocation, output_dir / "figures")
    return output_dir
