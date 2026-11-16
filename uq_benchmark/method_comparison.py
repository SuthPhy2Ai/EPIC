from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore[assignment]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from .calibration import fit_gaussian_variance_map
from .distributions import gaussian_nll
from .supplement import (
    COVERAGE_LEVELS,
    _material_ids_for_predictions,
    _ranking_groups,
    _validate_offsets,
    _weighted_auroc_auprc,
    _weighted_spearman,
)


METHODS = ("hierarchical_epic", "graph_nig", "llpr", "llpr_force_aware")
METHODS_WITH_ENSEMBLE = METHODS + ("ensemble5",)
HIGHER_IS_BETTER = {"risk_spearman", "top5_auroc", "top5_auprc"}
# Point predictions and Monte Carlo means are accumulated along different
# float32 reduction paths. Keep this gate tight relative to force magnitudes
# while allowing the measured reduction envelope across 256 samples.
POINT_FORCE_ALIGNMENT_RMSE_LIMIT = 2e-6
POINT_FORCE_ALIGNMENT_MAX_LIMIT = 1e-3


@dataclass(frozen=True)
class MethodFiles:
    energy_calibration: Path
    energy_test: Path
    force_calibration: Path
    force_test: Path
    posterior: Path | None = None


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def _verify_point_force_alignment(
    point_force: np.ndarray,
    sampled_force_mean: np.ndarray,
    context: str,
) -> dict[str, float]:
    point_force = np.asarray(point_force, dtype=np.float64)
    sampled_force_mean = np.asarray(sampled_force_mean, dtype=np.float64)
    if point_force.shape != sampled_force_mean.shape:
        raise ValueError(f"{context} point and sampled force shapes differ")
    difference = point_force - sampled_force_mean
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    maximum = float(np.max(np.abs(difference)))
    if rmse > POINT_FORCE_ALIGNMENT_RMSE_LIMIT or maximum > POINT_FORCE_ALIGNMENT_MAX_LIMIT:
        raise ValueError(
            f"{context} sampled mean force disagrees with point force: "
            f"rmse={rmse:.6g}, max_abs={maximum:.6g}"
        )
    return {"rmse": rmse, "max_absolute_difference": maximum}


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_posterior(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as posterior:
        required = {"covariance_raw", "ridge", "n_training_samples"}
        if missing := required - set(posterior.files):
            raise ValueError(f"LLPR posterior is missing {sorted(missing)}")
        covariance = np.asarray(posterior["covariance_raw"], dtype=np.float64)
        ridge = float(posterior["ridge"])
        n_training_samples = int(posterior["n_training_samples"])
        posterior_kind = (
            str(posterior["posterior_kind"])
            if "posterior_kind" in posterior.files
            else "energy_only_llpr"
        )
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("LLPR posterior covariance must be square")
    if np.any(~np.isfinite(covariance)) or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("LLPR posterior covariance and ridge must be finite")
    if n_training_samples <= 0:
        raise ValueError("LLPR posterior training sample count must be positive")
    symmetry_error = float(np.max(np.abs(covariance - covariance.T)))
    eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2)
    minimum_eigenvalue = float(eigenvalues.min())
    if symmetry_error > 1e-10 or minimum_eigenvalue < -1e-10:
        raise ValueError("LLPR posterior covariance is not symmetric positive semidefinite")
    return covariance, {
        "posterior_kind": posterior_kind,
        "dimension": covariance.shape[0],
        "n_training_samples": n_training_samples,
        "ridge": ridge,
        "symmetry_max_absolute_error": symmetry_error,
        "minimum_eigenvalue": minimum_eigenvalue,
        "maximum_eigenvalue": float(eigenvalues.max()),
        "finite_symmetric_positive_semidefinite_verified": True,
    }


def _point_arrays(
    record: dict[str, np.ndarray],
    posterior_path: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    natoms = np.asarray(record["natoms"], dtype=np.float64)
    truth = np.asarray(record["energy_true_total"], dtype=np.float64) / natoms
    location = np.asarray(record["energy_pred_total"], dtype=np.float64) / natoms
    if posterior_path is None:
        required = {"energy_aleatoric_variance", "energy_epistemic_variance"}
        missing = required - set(record)
        if not missing:
            total_variance = np.asarray(
                record["energy_aleatoric_variance"]
                + record["energy_epistemic_variance"],
                dtype=np.float64,
            )
        elif {"nu", "alpha", "beta"}.issubset(record):
            nu = np.asarray(record["nu"], dtype=np.float64)
            alpha = np.asarray(record["alpha"], dtype=np.float64)
            beta = np.asarray(record["beta"], dtype=np.float64)
            if (
                np.any(~np.isfinite(nu))
                or np.any(~np.isfinite(alpha))
                or np.any(~np.isfinite(beta))
                or np.any(nu <= 0)
                or np.any(alpha <= 1)
                or np.any(beta <= 0)
            ):
                raise ValueError(
                    "NIG domain violation: require finite nu > 0, alpha > 1, beta > 0"
                )
            aleatoric = beta / (alpha - 1.0)
            epistemic = beta / (nu * (alpha - 1.0))
            total_variance = aleatoric + epistemic
        else:
            raise ValueError(f"NIG energy record is missing {sorted(missing)}")
    else:
        if "llpr_features" not in record:
            raise ValueError("LLPR energy record is missing last-layer features")
        covariance, _ = _validated_posterior(posterior_path)
        features = np.asarray(record["llpr_features"], dtype=np.float64)
        if covariance.shape != (features.shape[1], features.shape[1]):
            raise ValueError("LLPR covariance and feature dimensions differ")
        total_variance = np.einsum(
            "ni,ij,nj->n", features, covariance, features, optimize=True
        )
    raw_variance = total_variance / np.square(natoms)
    if np.any(~np.isfinite(raw_variance)) or np.any(raw_variance < -1e-10):
        raise ValueError("energy UQ variance must be finite and non-negative")
    return truth, location, np.maximum(raw_variance, 0.0)


def _force_arrays(
    record: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    truth = np.asarray(record["force_true"], dtype=np.float64)
    location = np.asarray(record["force_mean"], dtype=np.float64)
    raw_scale = np.asarray(record["force_scale_raw"], dtype=np.float64)
    offsets = _validate_offsets(
        record["force_offsets"], len(truth), len(record["db_index"])
    )
    if truth.shape != location.shape or truth.shape != raw_scale.shape or truth.ndim != 2:
        raise ValueError("force truth, location, and scale arrays must have matching [atom,3] shapes")
    if truth.shape[1] != 3 or np.any(~np.isfinite(raw_scale)) or np.any(raw_scale < 0):
        raise ValueError("force scales must be finite and non-negative Cartesian arrays")
    return truth, location, raw_scale, offsets


def _align_method_inputs(
    records: dict[str, dict[str, dict[str, np.ndarray]]],
    methods: tuple[str, ...],
) -> None:
    for split in ("calibration", "test"):
        energy_reference = records["hierarchical_epic"][f"energy_{split}"]
        force_reference = records["hierarchical_epic"][f"force_{split}"]
        for method in methods[1:]:
            energy = records[method][f"energy_{split}"]
            for field in ("db_index", "natoms", "energy_true_total"):
                if not np.array_equal(energy[field], energy_reference[field]):
                    raise ValueError(f"{method} {split} energy record disagrees on {field}")
            force = records[method][f"force_{split}"]
            for field in ("db_index", "force_offsets", "force_true"):
                if not np.array_equal(force[field], force_reference[field]):
                    raise ValueError(f"{method} {split} force record disagrees on {field}")
        force_index = np.asarray(force_reference["db_index"], dtype=np.int64)
        point_index = np.asarray(energy_reference["db_index"], dtype=np.int64)
        positions = np.searchsorted(point_index, force_index)
        if np.any(positions >= len(point_index)) or not np.array_equal(
            point_index[positions], force_index
        ):
            raise ValueError(f"{split} force subset is not contained in energy predictions")
        for method in methods:
            energy = records[method][f"energy_{split}"]
            force = records[method][f"force_{split}"]
            if not np.array_equal(np.diff(force["force_offsets"]), energy["natoms"][positions]):
                raise ValueError(f"{method} {split} force offsets disagree with point natoms")
            point_force = np.asarray(energy["force_pred"], dtype=np.float64)
            point_offsets = np.asarray(energy["force_offsets"], dtype=np.int64)
            selected_point_force = np.concatenate(
                [point_force[point_offsets[index] : point_offsets[index + 1]] for index in positions]
            )
            _verify_point_force_alignment(
                selected_point_force,
                force["force_mean"],
                f"{method} {split}",
            )


def _structure_sums(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 2:
        values = values.sum(axis=1)
    return np.add.reduceat(values, offsets[:-1])


def _statistics(
    truth: np.ndarray,
    location: np.ndarray,
    raw_variance: np.ndarray,
    calibration_parameters: dict[str, float],
    coverage_levels: tuple[float, ...],
    offsets: np.ndarray | None = None,
) -> dict[str, object]:
    truth = np.asarray(truth, dtype=np.float64)
    location = np.asarray(location, dtype=np.float64)
    raw_variance = np.asarray(raw_variance, dtype=np.float64)
    tau = float(calibration_parameters["tau"])
    sigma0 = float(calibration_parameters["sigma0"])
    scale = np.sqrt(tau * tau * raw_variance + sigma0 * sigma0)
    absolute = np.abs(location - truth)
    squared = np.square(location - truth)
    nll = gaussian_nll(truth, location, scale)
    if offsets is None:
        counts = np.ones(len(truth), dtype=np.float64)
        absolute_sum = absolute
        squared_sum = squared
        nll_sum = nll
        structure_error = absolute
        structure_risk = np.sqrt(raw_variance)
        coverage_sum = {
            level: absolute <= norm.ppf((1 + level) / 2) * scale
            for level in coverage_levels
        }
    else:
        component_count = 3 * np.diff(offsets)
        counts = component_count.astype(np.float64)
        absolute_sum = _structure_sums(absolute, offsets)
        squared_sum = _structure_sums(squared, offsets)
        nll_sum = _structure_sums(nll, offsets)
        structure_error = absolute_sum / counts
        structure_risk = np.sqrt(
            _structure_sums(raw_variance, offsets) / counts
        )
        coverage_sum = {
            level: _structure_sums(
                absolute <= norm.ppf((1 + level) / 2) * scale,
                offsets,
            )
            for level in coverage_levels
        }
    danger_threshold = np.quantile(structure_error, 0.95, method="linear")
    risk_order, risk_starts = _ranking_groups(structure_risk)
    return {
        "counts": counts,
        "absolute_sum": absolute_sum,
        "squared_sum": squared_sum,
        "nll_sum": nll_sum,
        "coverage_sum": coverage_sum,
        "structure_error": structure_error,
        "structure_risk": structure_risk,
        "error_rank": rankdata(structure_error, method="average"),
        "risk_rank": rankdata(structure_risk, method="average"),
        "danger": structure_error >= danger_threshold,
        "risk_order": risk_order,
        "risk_starts": risk_starts,
    }


def _evaluate(
    values: dict[str, object],
    cluster_weights: np.ndarray,
    structure_cluster: np.ndarray,
    coverage_levels: tuple[float, ...],
) -> dict[str, float]:
    weights = np.asarray(cluster_weights, dtype=np.float64)[structure_cluster]
    counts = np.asarray(values["counts"], dtype=np.float64)
    denominator = float(np.dot(weights, counts))
    auroc, auprc = _weighted_auroc_auprc(
        values["danger"], weights, values["risk_order"], values["risk_starts"]
    )
    row = {
        "point_mae": float(np.dot(weights, values["absolute_sum"]) / denominator),
        "point_rmse": float(
            np.sqrt(np.dot(weights, values["squared_sum"]) / denominator)
        ),
        "risk_spearman": _weighted_spearman(
            values["error_rank"], values["risk_rank"], weights
        ),
        "top5_auroc": auroc,
        "top5_auprc": auprc,
        "gaussian_affine_nll": float(
            np.dot(weights, values["nll_sum"]) / denominator
        ),
    }
    for level in coverage_levels:
        name = f"coverage_{int(level * 100):02d}"
        coverage = float(np.dot(weights, values["coverage_sum"][level]) / denominator)
        row[name] = coverage
        row[name + "_absolute_error"] = abs(coverage - level)
    return row


def _risk_coverage_rows(
    target: str,
    method: str,
    error: np.ndarray,
    risk: np.ndarray,
) -> list[dict[str, object]]:
    order = np.argsort(risk, kind="stable")
    rows = []
    for coverage in np.linspace(0.01, 1.0, 100):
        retained = max(1, int(np.ceil(coverage * len(order))))
        rows.append(
            {
                "target": target,
                "method": method,
                "coverage": coverage,
                "n_retained": retained,
                "mean_structure_absolute_error": float(np.mean(error[order[:retained]])),
            }
        )
    return rows


def _compute_cost_rows(
    records: dict[str, dict[str, dict[str, np.ndarray]]],
    methods: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in methods:
        for split in ("calibration", "test"):
            for target, record_name in (
                ("point_inference", f"energy_{split}"),
                ("force_uq_inference", f"force_{split}"),
            ):
                record = records[method][record_name]
                required = {"db_index", "inference_seconds", "peak_memory_bytes"}
                if target == "force_uq_inference" and method != "ensemble5":
                    required.add("n_samples")
                if method == "ensemble5":
                    required.add("n_members")
                missing = required - set(record)
                if missing:
                    raise ValueError(
                        f"{method} {split} {target} record is missing cost fields "
                        f"{sorted(missing)}"
                    )
                n_structures = len(record["db_index"])
                timed_n_structures = int(record.get("timed_n_structures", n_structures))
                seconds = float(record["inference_seconds"])
                peak_memory = int(record["peak_memory_bytes"])
                n_samples = (
                    int(record["n_members"])
                    if method == "ensemble5"
                    else int(record["n_samples"])
                    if target == "force_uq_inference"
                    else 1
                )
                timing_semantics = str(
                    record.get(
                        "timing_semantics",
                        "sum_of_shard_local_device_wall_times",
                    )
                )
                if n_structures <= 0 or timed_n_structures <= 0:
                    raise ValueError(f"{method} {split} {target} record is empty")
                if not np.isfinite(seconds) or seconds <= 0:
                    raise ValueError(
                        f"{method} {split} {target} inference time must be positive"
                    )
                if peak_memory < 0 or n_samples <= 0:
                    raise ValueError(
                        f"{method} {split} {target} cost metadata is invalid"
                    )
                rows.append(
                    {
                        "method": method,
                        "target": target,
                        "split": split,
                        "n_structures": n_structures,
                        "timed_n_structures": timed_n_structures,
                        "mc_samples": n_samples,
                        "summed_shard_device_seconds": seconds,
                        "milliseconds_per_structure": 1000.0
                        * seconds
                        / timed_n_structures,
                        "peak_memory_bytes_per_device": peak_memory,
                        "timing_semantics": timing_semantics,
                    }
                )
    return rows


def _write_figure(
    metrics: pd.DataFrame,
    output_dir: Path,
    methods: tuple[str, ...],
    filename: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ("#0B6E69", "#355C7D", "#D69B2D", "#C65D3B", "#7A5195")
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.5))
    for row, target in enumerate(("energy_per_atom", "force_component")):
        selected = metrics[metrics.target == target].set_index(["method", "metric"])
        ranking = [selected.loc[(method, "risk_spearman"), "estimate"] for method in methods]
        axes[row, 0].bar(methods, ranking, color=colors[: len(methods)])
        axes[row, 0].set_ylim(-0.1, 1.0)
        axes[row, 0].set_ylabel("Spearman")
        axes[row, 0].set_title(f"{target}: error ranking")
        axes[row, 0].tick_params(axis="x", rotation=20)
        for method, color in zip(methods, colors[: len(methods)], strict=True):
            coverage = [
                selected.loc[(method, f"coverage_{int(level * 100):02d}"), "estimate"]
                for level in COVERAGE_LEVELS
            ]
            axes[row, 1].plot(COVERAGE_LEVELS, coverage, marker="o", color=color, label=method)
        axes[row, 1].plot(COVERAGE_LEVELS, COVERAGE_LEVELS, "--", color="#263238")
        axes[row, 1].set_title(f"{target}: Gaussian affine calibration")
        axes[row, 1].set_xlabel("Nominal coverage")
        axes[row, 1].set_ylabel("Empirical coverage")
        axes[row, 1].grid(alpha=0.22)
    axes[0, 1].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"{filename}.{suffix}", dpi=220)
    plt.close(figure)


def _compare_methods(
    files: dict[str, MethodFiles],
    store_path: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_replicates: int = 1000,
    seed: int = 42,
    coverage_levels: tuple[float, ...] = COVERAGE_LEVELS,
    methods: tuple[str, ...],
    include_ensemble: bool,
) -> Path:
    if tuple(files) != methods:
        raise ValueError(f"method inputs must be ordered as {methods}")
    records: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    posterior_audits: dict[str, dict[str, object]] = {}
    for method, paths in files.items():
        records[method] = {
            "energy_calibration": _load_npz(paths.energy_calibration),
            "energy_test": _load_npz(paths.energy_test),
            "force_calibration": _load_npz(paths.force_calibration),
            "force_test": _load_npz(paths.force_test),
        }
        if paths.posterior is not None:
            _, posterior_audits[method] = _validated_posterior(paths.posterior)
    _align_method_inputs(records, methods)

    statistics: dict[str, dict[str, dict[str, object]]] = {
        "energy_per_atom": {},
        "force_component": {},
    }
    calibration_parameters: dict[str, dict[str, dict[str, float]]] = {}
    risk_rows = []
    for method, paths in files.items():
        energy_calibration = _point_arrays(records[method]["energy_calibration"], paths.posterior)
        energy_test = _point_arrays(records[method]["energy_test"], paths.posterior)
        energy_state = fit_gaussian_variance_map(
            energy_calibration[0], energy_calibration[1], energy_calibration[2], method=method
        )
        energy_statistics = _statistics(
            *energy_test,
            dict(energy_state.parameters),
            coverage_levels,
        )
        statistics["energy_per_atom"][method] = energy_statistics

        force_calibration = _force_arrays(records[method]["force_calibration"])
        force_test = _force_arrays(records[method]["force_test"])
        force_state = fit_gaussian_variance_map(
            force_calibration[0].reshape(-1),
            force_calibration[1].reshape(-1),
            np.square(force_calibration[2].reshape(-1)),
            method=method,
        )
        force_statistics = _statistics(
            force_test[0],
            force_test[1],
            np.square(force_test[2]),
            dict(force_state.parameters),
            coverage_levels,
            offsets=force_test[3],
        )
        statistics["force_component"][method] = force_statistics
        calibration_parameters[method] = {
            "energy_per_atom": dict(energy_state.parameters),
            "force_component": dict(force_state.parameters),
        }
        risk_rows.extend(
            _risk_coverage_rows(
                "energy_per_atom",
                method,
                energy_statistics["structure_error"],
                energy_statistics["structure_risk"],
            )
        )
        risk_rows.extend(
            _risk_coverage_rows(
                "force_component",
                method,
                force_statistics["structure_error"],
                force_statistics["structure_risk"],
            )
        )

    energy_index = records["hierarchical_epic"]["energy_test"]["db_index"]
    force_index = records["hierarchical_epic"]["force_test"]["db_index"]
    energy_material = _material_ids_for_predictions(store_path, energy_index)
    force_material = _material_ids_for_predictions(store_path, force_index)
    target_groups = {}
    for target, material in (
        ("energy_per_atom", energy_material),
        ("force_component", force_material),
    ):
        labels, cluster = np.unique(material, return_inverse=True)
        target_groups[target] = (labels, cluster)

    estimate: dict[str, float] = {}
    for target in statistics:
        labels, cluster = target_groups[target]
        weights = np.ones(len(labels), dtype=np.float64)
        for method in methods:
            for metric, value in _evaluate(
                statistics[target][method], weights, cluster, coverage_levels
            ).items():
                estimate[f"{target}__{method}__{metric}"] = value

    rng = np.random.default_rng(seed)
    bootstrap_rows = []
    for replicate in range(bootstrap_replicates):
        row: dict[str, float | int] = {"replicate": replicate}
        for target in statistics:
            labels, cluster = target_groups[target]
            weights = np.bincount(
                rng.integers(0, len(labels), len(labels)), minlength=len(labels)
            ).astype(np.float64)
            evaluated = {
                method: _evaluate(
                    statistics[target][method], weights, cluster, coverage_levels
                )
                for method in methods
            }
            for method, method_values in evaluated.items():
                for metric, value in method_values.items():
                    row[f"{target}__{method}__{metric}"] = value
            for baseline in methods[1:]:
                for metric, hierarchical_value in evaluated["hierarchical_epic"].items():
                    if metric.startswith("coverage_") and not metric.endswith(
                        "_absolute_error"
                    ):
                        continue
                    baseline_value = evaluated[baseline][metric]
                    advantage = (
                        hierarchical_value - baseline_value
                        if metric in HIGHER_IS_BETTER
                        else baseline_value - hierarchical_value
                    )
                    row[f"{target}__hierarchical_advantage_over_{baseline}__{metric}"] = advantage
        bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows)

    metric_rows = []
    for key, value in estimate.items():
        target, method, metric = key.split("__", 2)
        samples = bootstrap[key].to_numpy(dtype=np.float64)
        samples = samples[np.isfinite(samples)]
        low, high = np.quantile(samples, [0.025, 0.975])
        metric_rows.append(
            {
                "target": target,
                "method": method,
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
    differences = []
    for column in bootstrap.columns:
        if "__hierarchical_advantage_over_" not in column:
            continue
        target, comparison, metric = column.split("__", 2)
        samples = bootstrap[column].to_numpy(dtype=np.float64)
        samples = samples[np.isfinite(samples)]
        low, high = np.quantile(samples, [0.025, 0.975])
        baseline = comparison.removeprefix("hierarchical_advantage_over_")
        hierarchical_estimate = estimate[
            f"{target}__hierarchical_epic__{metric}"
        ]
        baseline_estimate = estimate[f"{target}__{baseline}__{metric}"]
        point_advantage = (
            hierarchical_estimate - baseline_estimate
            if metric in HIGHER_IS_BETTER
            else baseline_estimate - hierarchical_estimate
        )
        differences.append(
            {
                "target": target,
                "comparison": comparison,
                "metric": metric,
                "estimate": point_advantage,
                "ci_low_95": float(low),
                "ci_high_95": float(high),
                "positive_means_hierarchical_better": True,
                "valid_bootstrap_replicates": len(samples),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    difference_frame = pd.DataFrame(differences)
    risk_frame = pd.DataFrame(risk_rows)
    cost_frame = pd.DataFrame(_compute_cost_rows(records, methods))

    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    prefix = "method" if include_ensemble else "nonensemble_method"
    risk_prefix = "method" if include_ensemble else "nonensemble"
    metrics.to_csv(tables / f"{prefix}_metrics.csv", index=False)
    difference_frame.to_csv(tables / "hierarchical_paired_advantages.csv", index=False)
    bootstrap.to_csv(tables / f"{prefix}_bootstrap.csv", index=False)
    risk_frame.to_csv(tables / f"{risk_prefix}_risk_coverage.csv", index=False)
    cost_frame.to_csv(tables / f"{risk_prefix}_compute_cost.csv", index=False)
    input_hashes = {}
    for method, paths in files.items():
        method_inputs = [
            ("energy_calibration", paths.energy_calibration),
            ("energy_test", paths.energy_test),
            ("force_calibration", paths.force_calibration),
            ("force_test", paths.force_test),
        ]
        if paths.posterior is not None:
            method_inputs.append(("posterior", paths.posterior))
        input_hashes[method] = {
            name: _sha256_file(path) for name, path in method_inputs
        }
    summary = {
        "schema_version": 1,
        "comparison": (
            "aligned_uq_methods_with_ensemble5"
            if include_ensemble
            else "nonensemble_uq_methods"
        ),
        "methods": list(methods),
        "fit_split": "calibration",
        "evaluation_split": "test",
        "test_used_for_selection": False,
        "force_subset_alignment_verified": True,
        "point_force_alignment_verified": True,
        "common_calibration_family": "gaussian_affine",
        "bootstrap_unit": "material_id",
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "calibration_parameters": calibration_parameters,
        "posterior_audits": posterior_audits,
        "compute_cost_table": f"tables/{risk_prefix}_compute_cost.csv",
        "compute_cost_timing_semantics": sorted(set(cost_frame.timing_semantics)),
        "input_sha256": input_hashes,
        "interpretation_limit": (
            "Each uncertainty method is evaluated against its own point-prediction errors. "
            + (
                "The ensemble uses five independently trained deterministic members."
                if include_ensemble
                else "This comparison excludes deterministic ensembles by design."
            )
        ),
    }
    summary_name = (
        "method_comparison_summary.json"
        if include_ensemble
        else "nonensemble_method_comparison_summary.json"
    )
    (output_dir / summary_name).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_figure(
        metrics,
        output_dir / "figures",
        methods,
        "method_comparison" if include_ensemble else "nonensemble_method_comparison",
    )
    return output_dir


def compare_nonensemble_methods(
    files: dict[str, MethodFiles],
    store_path: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_replicates: int = 1000,
    seed: int = 42,
    coverage_levels: tuple[float, ...] = COVERAGE_LEVELS,
) -> Path:
    return _compare_methods(
        files,
        store_path,
        output_dir,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
        coverage_levels=coverage_levels,
        methods=METHODS,
        include_ensemble=False,
    )


def compare_methods_with_ensemble(
    files: dict[str, MethodFiles],
    store_path: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_replicates: int = 1000,
    seed: int = 42,
    coverage_levels: tuple[float, ...] = COVERAGE_LEVELS,
) -> Path:
    return _compare_methods(
        files,
        store_path,
        output_dir,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
        coverage_levels=coverage_levels,
        methods=METHODS_WITH_ENSEMBLE,
        include_ensemble=True,
    )
