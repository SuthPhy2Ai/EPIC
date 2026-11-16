from __future__ import annotations

import json
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
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, rankdata, spearmanr, t

from .calibration import fit_gaussian_variance_map, fit_student_t_scale
from .distributions import gaussian_nll, student_t_nll


SIZE_BINS = np.asarray([0, 20, 40, 80, 160, np.inf])
SIZE_LABELS = ("1-20", "21-40", "41-80", "81-160", ">160")
ALPHA_LABELS = ("alpha_le_1p5", "alpha_1p5_to_2", "alpha_gt_2")
COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)
ATTRIBUTION_BASELINES = (
    "learned_allocation",
    "force_magnitude",
    "atomic_energy_magnitude",
)


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {name: source[name] for name in source.files}


def _validate_offsets(offsets: np.ndarray, n_atoms: int, n_structures: int) -> np.ndarray:
    offsets = np.asarray(offsets, dtype=np.int64)
    if offsets.shape != (n_structures + 1,):
        raise ValueError("force_offsets must contain one boundary per structure")
    if offsets[0] != 0 or offsets[-1] != n_atoms or np.any(np.diff(offsets) <= 0):
        raise ValueError("force_offsets must describe non-empty contiguous structures")
    return offsets


def _material_ids_for_predictions(
    store_path: str | Path, db_index: np.ndarray
) -> np.ndarray:
    with h5py.File(store_path, "r") as handle:
        store_index = np.asarray(handle["structures/db_index"][:], dtype=np.int64)
        material_id = handle["structures/material_id"].asstr()[:]
    if np.any(np.diff(store_index) <= 0):
        raise ValueError("benchmark store db_index must be strictly increasing")
    positions = np.searchsorted(store_index, np.asarray(db_index, dtype=np.int64))
    if np.any(positions >= len(store_index)) or not np.array_equal(
        store_index[positions], db_index
    ):
        raise ValueError("prediction db_index is not a subset of the benchmark store")
    return np.asarray(material_id[positions], dtype=str)


def _ndcg(relevance: np.ndarray, score: np.ndarray) -> float:
    relevance = np.asarray(relevance, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    discount = 1.0 / np.log2(np.arange(len(score), dtype=np.float64) + 2.0)
    observed = float(np.dot(relevance[np.argsort(-score, kind="stable")], discount))
    ideal = float(np.dot(np.sort(relevance)[::-1], discount))
    return observed / ideal if ideal > 0 else float("nan")


def _top_fraction_recall(
    relevance: np.ndarray, score: np.ndarray, fraction: float
) -> float:
    n_selected = max(1, int(np.ceil(fraction * len(score))))
    truth = np.argpartition(relevance, len(relevance) - n_selected)[-n_selected:]
    selected = np.argpartition(score, len(score) - n_selected)[-n_selected:]
    return float(len(np.intersect1d(truth, selected, assume_unique=False)) / n_selected)


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def _cluster_bootstrap_means(
    frame: pd.DataFrame,
    material_id: np.ndarray,
    columns: list[str],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    labels, cluster = np.unique(np.asarray(material_id, dtype=str), return_inverse=True)
    sums = np.zeros((len(labels), len(columns)), dtype=np.float64)
    counts = np.zeros_like(sums)
    for column_index, column in enumerate(columns):
        values = frame[column].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        sums[:, column_index] = np.bincount(
            cluster[finite], weights=values[finite], minlength=len(labels)
        )
        counts[:, column_index] = np.bincount(
            cluster[finite], minlength=len(labels)
        )
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        weights = np.bincount(
            rng.integers(0, len(labels), len(labels)), minlength=len(labels)
        ).astype(np.float64)
        numerator = weights @ sums
        denominator = weights @ counts
        row = {"replicate": replicate, "n_clusters": len(labels)}
        row.update(
            {
                column: float(numerator[index] / denominator[index])
                if denominator[index] > 0
                else float("nan")
                for index, column in enumerate(columns)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_confidence_table(
    frame: pd.DataFrame,
    bootstrap: pd.DataFrame,
    columns: list[str],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for column in columns:
        values = frame[column].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        samples = bootstrap[column].to_numpy(dtype=np.float64)
        samples = samples[np.isfinite(samples)]
        if len(samples):
            low, high = np.quantile(samples, [0.025, 0.975])
        else:
            low = high = float("nan")
        rows.append(
            {
                "metric": column,
                "estimate": float(np.mean(values)) if len(values) else float("nan"),
                "median": float(np.median(values)) if len(values) else float("nan"),
                "ci_low_95": float(low),
                "ci_high_95": float(high),
                "n_structures": len(values),
                "valid_bootstrap_replicates": len(samples),
                "bootstrap_unit": "material_id",
                "bootstrap_replicates": replicates,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def _write_atomic_figures(
    confidence: pd.DataFrame, by_size: pd.DataFrame, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ("#0B6E69", "#D69B2D", "#C65D3B")
    labels = ("Learned allocation", "|Predicted force|", "|Atomic energy|")
    metrics = ("ndcg", "top10_recall")
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for axis, metric, title in zip(
        axes, metrics, ("Atomic error ranking", "Top-10% error-atom recall"), strict=True
    ):
        names = [f"{baseline}_{metric}" for baseline in ATTRIBUTION_BASELINES]
        selected = confidence.set_index("metric").loc[names]
        estimate = selected["estimate"].to_numpy()
        error = np.vstack(
            [
                estimate - selected["ci_low_95"].to_numpy(),
                selected["ci_high_95"].to_numpy() - estimate,
            ]
        )
        axis.bar(np.arange(3), estimate, color=colors, yerr=error, capsize=3)
        axis.set_xticks(np.arange(3), labels, rotation=18, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"atomic_attribution_baselines.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    order = list(SIZE_LABELS)
    for axis, metric, title in zip(
        axes,
        ("learned_allocation_ndcg", "learned_allocation_top10_recall"),
        ("Learned ranking by structure size", "Top-10% recall by structure size"),
        strict=True,
    ):
        selected = by_size.set_index("size_bin").reindex(order)
        axis.plot(order, selected[metric], marker="o", color="#0B6E69", linewidth=2)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Atoms per structure")
        axis.set_title(title)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("NDCG")
    axes[1].set_ylabel("Recall")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"atomic_attribution_by_size.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def analyze_atomic_attribution(
    predictions_path: str | Path,
    store_path: str | Path,
    output_dir: str | Path,
    *,
    force_error_floor: float = 1e-6,
    bootstrap_replicates: int = 1000,
    seed: int = 42,
) -> Path:
    """Evaluate learned atom-wise allocation against held-out force-error profiles."""
    record = _load_npz(predictions_path)
    required = {
        "db_index",
        "natoms",
        "alpha",
        "force_true",
        "force_pred",
        "force_offsets",
        "atomic_numbers",
        "atomic_energy_gamma",
        "atomic_epistemic_weight",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"atomic attribution predictions are missing {sorted(missing)}")
    db_index = np.asarray(record["db_index"], dtype=np.int64)
    natoms = np.asarray(record["natoms"], dtype=np.int64)
    n_structures = len(db_index)
    force_true = np.asarray(record["force_true"], dtype=np.float64)
    force_pred = np.asarray(record["force_pred"], dtype=np.float64)
    if force_true.shape != force_pred.shape or force_true.ndim != 2 or force_true.shape[1] != 3:
        raise ValueError("force_true and force_pred must be matching [n_atoms, 3] arrays")
    offsets = _validate_offsets(record["force_offsets"], len(force_true), n_structures)
    if not np.array_equal(np.diff(offsets), natoms):
        raise ValueError("natoms does not match force_offsets")
    material_id = _material_ids_for_predictions(store_path, db_index)
    learned = np.asarray(record["atomic_epistemic_weight"], dtype=np.float64)
    if learned.shape != (len(force_true),) or np.any(~np.isfinite(learned)) or np.any(learned < 0):
        raise ValueError("atomic_epistemic_weight must be a finite non-negative atom array")
    error_score = np.square(force_pred - force_true).sum(axis=1) + float(force_error_floor)
    force_magnitude = np.linalg.norm(force_pred, axis=1)
    atomic_energy_magnitude = np.abs(
        np.asarray(record["atomic_energy_gamma"], dtype=np.float64)
    )
    learned_sums = np.add.reduceat(learned, offsets[:-1])
    if not np.allclose(learned_sums, 1.0, rtol=2e-5, atol=2e-6):
        raise ValueError("atomic epistemic weights do not sum to one per structure")

    rows = []
    top10_atom = np.zeros(len(force_true), dtype=bool)
    teacher_multiplier = np.empty(len(force_true), dtype=np.float64)
    learned_multiplier = np.empty(len(force_true), dtype=np.float64)
    tiny = np.finfo(np.float64).tiny
    for index in range(n_structures):
        start, stop = offsets[index : index + 2]
        teacher = error_score[start:stop]
        teacher = teacher / teacher.sum()
        prediction = learned[start:stop]
        prediction = prediction / prediction.sum()
        teacher_multiplier[start:stop] = teacher * natoms[index]
        learned_multiplier[start:stop] = prediction * natoms[index]
        k10 = max(1, int(np.ceil(0.10 * natoms[index])))
        top10_local = np.argpartition(teacher, len(teacher) - k10)[-k10:]
        top10_atom[start + top10_local] = True
        row = {
            "db_index": int(db_index[index]),
            "material_id": material_id[index],
            "natoms": int(natoms[index]),
            "alpha": float(record["alpha"][index]),
            "teacher_entropy": float(-np.sum(teacher * np.log(np.maximum(teacher, tiny)))),
            "learned_kl_from_teacher": float(
                np.sum(teacher * (np.log(np.maximum(teacher, tiny)) - np.log(np.maximum(prediction, tiny))))
            ),
            "uniform_kl_from_teacher": float(
                np.sum(teacher * (np.log(np.maximum(teacher, tiny)) + np.log(natoms[index])))
            ),
            "learned_total_variation": float(0.5 * np.abs(teacher - prediction).sum()),
            "uniform_expected_top10_recall": float(k10 / natoms[index]),
            "uniform_expected_top20_recall": float(
                max(1, int(np.ceil(0.20 * natoms[index]))) / natoms[index]
            ),
        }
        scores = {
            "learned_allocation": prediction,
            "force_magnitude": force_magnitude[start:stop],
            "atomic_energy_magnitude": atomic_energy_magnitude[start:stop],
        }
        for name, score in scores.items():
            row[f"{name}_spearman"] = _safe_spearman(teacher, score)
            row[f"{name}_ndcg"] = _ndcg(teacher, score)
            row[f"{name}_top10_recall"] = _top_fraction_recall(teacher, score, 0.10)
            row[f"{name}_top20_recall"] = _top_fraction_recall(teacher, score, 0.20)
        rows.append(row)
    structure = pd.DataFrame(rows)
    structure["learned_kl_improvement_vs_uniform"] = (
        structure["uniform_kl_from_teacher"] - structure["learned_kl_from_teacher"]
    )
    attribution_difference_columns = []
    for baseline in ("force_magnitude", "atomic_energy_magnitude"):
        for metric in ("spearman", "ndcg", "top10_recall", "top20_recall"):
            column = f"{baseline}_minus_learned_allocation_{metric}"
            structure[column] = (
                structure[f"{baseline}_{metric}"]
                - structure[f"learned_allocation_{metric}"]
            )
            attribution_difference_columns.append(column)
    metric_columns = [
        "learned_kl_from_teacher",
        "uniform_kl_from_teacher",
        "learned_kl_improvement_vs_uniform",
        "learned_total_variation",
        "uniform_expected_top10_recall",
        "uniform_expected_top20_recall",
    ]
    for baseline in ATTRIBUTION_BASELINES:
        metric_columns.extend(
            [
                f"{baseline}_spearman",
                f"{baseline}_ndcg",
                f"{baseline}_top10_recall",
                f"{baseline}_top20_recall",
            ]
        )
    metric_columns.extend(attribution_difference_columns)
    bootstrap = _cluster_bootstrap_means(
        structure, material_id, metric_columns, bootstrap_replicates, seed
    )
    bootstrap["aggregate_kl_skill_vs_uniform"] = 1.0 - (
        bootstrap["learned_kl_from_teacher"]
        / bootstrap["uniform_kl_from_teacher"]
    )
    confidence = _mean_confidence_table(
        structure, bootstrap, metric_columns, bootstrap_replicates, seed
    )
    aggregate_skill_samples = bootstrap["aggregate_kl_skill_vs_uniform"].to_numpy()
    aggregate_skill_low, aggregate_skill_high = np.quantile(
        aggregate_skill_samples, [0.025, 0.975]
    )
    aggregate_skill = 1.0 - (
        structure["learned_kl_from_teacher"].mean()
        / structure["uniform_kl_from_teacher"].mean()
    )
    confidence = pd.concat(
        [
            confidence,
            pd.DataFrame(
                [
                    {
                        "metric": "aggregate_kl_skill_vs_uniform",
                        "estimate": float(aggregate_skill),
                        "median": float(np.median(aggregate_skill_samples)),
                        "ci_low_95": float(aggregate_skill_low),
                        "ci_high_95": float(aggregate_skill_high),
                        "n_structures": n_structures,
                        "valid_bootstrap_replicates": bootstrap_replicates,
                        "bootstrap_unit": "material_id",
                        "bootstrap_replicates": bootstrap_replicates,
                        "seed": seed,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    size_group = np.digitize(natoms, SIZE_BINS[1:-1], right=True)
    structure["size_bin"] = np.asarray(SIZE_LABELS, dtype=object)[size_group]
    alpha = structure["alpha"].to_numpy()
    structure["alpha_group"] = np.select(
        [alpha <= 1.5, alpha <= 2.0], ALPHA_LABELS[:2], default=ALPHA_LABELS[2]
    )
    grouped_columns = [
        "learned_kl_from_teacher",
        "uniform_kl_from_teacher",
        "learned_kl_improvement_vs_uniform",
        "learned_allocation_spearman",
        "learned_allocation_ndcg",
        "learned_allocation_top10_recall",
        "force_magnitude_ndcg",
        "atomic_energy_magnitude_ndcg",
    ]
    by_size = (
        structure.groupby("size_bin", observed=False)[grouped_columns]
        .agg(["count", "mean", "median"])
    )
    by_size.columns = [f"{name}_{stat}" for name, stat in by_size.columns]
    by_size = by_size.reset_index()
    for name in grouped_columns:
        by_size[name] = by_size[f"{name}_mean"]
    by_size["aggregate_kl_skill_vs_uniform"] = 1.0 - (
        by_size["learned_kl_from_teacher_mean"]
        / by_size["uniform_kl_from_teacher_mean"]
    )
    by_alpha = (
        structure.groupby("alpha_group", observed=False)[grouped_columns]
        .agg(["count", "mean", "median"])
    )
    by_alpha.columns = [f"{name}_{stat}" for name, stat in by_alpha.columns]
    by_alpha = by_alpha.reset_index()
    by_alpha["aggregate_kl_skill_vs_uniform"] = 1.0 - (
        by_alpha["learned_kl_from_teacher_mean"]
        / by_alpha["uniform_kl_from_teacher_mean"]
    )

    atomic_numbers = np.asarray(record["atomic_numbers"], dtype=np.int64)
    element_rows = []
    for atomic_number in np.unique(atomic_numbers):
        selected = atomic_numbers == atomic_number
        element_rows.append(
            {
                "atomic_number": int(atomic_number),
                "n_atoms": int(selected.sum()),
                "mean_force_error_l2": float(
                    np.linalg.norm(force_pred[selected] - force_true[selected], axis=1).mean()
                ),
                "mean_teacher_multiplier_vs_uniform": float(teacher_multiplier[selected].mean()),
                "mean_learned_multiplier_vs_uniform": float(learned_multiplier[selected].mean()),
                "top10_error_atom_fraction": float(top10_atom[selected].mean()),
            }
        )
    by_element = pd.DataFrame(element_rows)

    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    structure.to_csv(tables / "atomic_attribution_structure_metrics.csv", index=False)
    bootstrap.to_csv(tables / "atomic_attribution_bootstrap.csv", index=False)
    confidence.to_csv(tables / "atomic_attribution_confidence_intervals.csv", index=False)
    by_size.to_csv(tables / "atomic_attribution_by_size.csv", index=False)
    by_alpha.to_csv(tables / "atomic_attribution_by_alpha.csv", index=False)
    by_element.to_csv(tables / "atomic_attribution_by_element.csv", index=False)
    selected_metrics = confidence.set_index("metric")["estimate"].to_dict()
    summary = {
        "schema_version": 1,
        "analysis_kind": "held_out_force_error_informed_atomic_attribution",
        "n_structures": n_structures,
        "n_atoms": len(force_true),
        "force_error_teacher": "within_structure_normalized_squared_vector_force_error",
        "force_error_floor": force_error_floor,
        "bootstrap_unit": "material_id",
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "headline": {
            name: float(selected_metrics[name])
            for name in (
                "aggregate_kl_skill_vs_uniform",
                "learned_allocation_spearman",
                "learned_allocation_ndcg",
                "learned_allocation_top10_recall",
                "force_magnitude_ndcg",
                "atomic_energy_magnitude_ndcg",
                "force_magnitude_minus_learned_allocation_ndcg",
                "force_magnitude_minus_learned_allocation_top10_recall",
                "atomic_energy_magnitude_minus_learned_allocation_ndcg",
                "atomic_energy_magnitude_minus_learned_allocation_top10_recall",
            )
        },
        "interpretation_limit": (
            "This evaluates alignment with the detached relative force-error proxy. "
            "It does not turn atomic allocation weights into calibrated atomic force variances."
        ),
    }
    (output_dir / "atomic_attribution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_atomic_figures(confidence, by_size, output_dir / "figures")
    return output_dir


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(np.sum(weights))
    return float(np.dot(np.asarray(values, dtype=np.float64), weights) / denominator)


def _weighted_spearman(
    error_rank: np.ndarray, risk_rank: np.ndarray, weights: np.ndarray
) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    left_mean = np.dot(weights, error_rank) / total
    right_mean = np.dot(weights, risk_rank) / total
    left = error_rank - left_mean
    right = risk_rank - right_mean
    denominator = np.sqrt(np.dot(weights, left * left) * np.dot(weights, right * right))
    return float(np.dot(weights, left * right) / denominator) if denominator > 0 else float("nan")


def _ranking_groups(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-np.asarray(score), kind="stable")
    ordered = np.asarray(score)[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    return order, starts


def _weighted_auroc_auprc(
    danger: np.ndarray,
    weights: np.ndarray,
    order: np.ndarray,
    starts: np.ndarray,
) -> tuple[float, float]:
    labels = np.asarray(danger, dtype=bool)[order]
    ordered_weights = np.asarray(weights, dtype=np.float64)[order]
    positives = np.add.reduceat(ordered_weights * labels, starts)
    negatives = np.add.reduceat(ordered_weights * ~labels, starts)
    total_positive = positives.sum()
    total_negative = negatives.sum()
    if total_positive <= 0 or total_negative <= 0:
        return float("nan"), float("nan")
    cumulative_negative = np.cumsum(negatives)
    concordant = np.sum(
        positives * (total_negative - cumulative_negative + 0.5 * negatives)
    )
    auroc = float(concordant / (total_positive * total_negative))
    cumulative_positive = np.cumsum(positives)
    cumulative_total = np.cumsum(positives + negatives)
    precision = cumulative_positive / np.maximum(cumulative_total, 1e-30)
    auprc = float(np.sum((positives / total_positive) * precision))
    return auroc, auprc


def _force_structure_sums(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 2:
        values = values.sum(axis=1)
    return np.add.reduceat(values, np.asarray(offsets[:-1], dtype=np.int64))


def bootstrap_uq_metrics(
    store_path: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_replicates: int = 1000,
    seed: int = 42,
    coverage_levels: tuple[float, ...] = COVERAGE_LEVELS,
) -> Path:
    """Material-cluster bootstrap calibrated energy and force UQ metrics."""
    with h5py.File(store_path, "r") as handle:
        structure = handle["structures"]
        full_index = np.asarray(structure["db_index"][:], dtype=np.int64)
        material_id = structure["material_id"].asstr()[:]
        energy_true = np.asarray(structure["energy_true_pa"][:], dtype=np.float64)
        energy = handle["methods/hierarchical_epic"]
        energy_location = np.asarray(energy["energy_location"][:], dtype=np.float64)
        energy_scale = np.asarray(energy["energy_scale_calibrated"][:], dtype=np.float64)
        energy_df = np.asarray(energy["energy_df"][:], dtype=np.float64)
        energy_risk = np.asarray(energy["energy_risk"][:], dtype=np.float64)
        force = energy["force_uq_subset"]
        force_index = np.asarray(force["db_index"][:], dtype=np.int64)
        force_true = np.asarray(force["force_true"][:], dtype=np.float64)
        force_location = np.asarray(force["force_location"][:], dtype=np.float64)
        force_scale = np.asarray(force["force_scale_calibrated"][:], dtype=np.float64)
        force_scale_raw = np.asarray(force["force_scale_raw"][:], dtype=np.float64)
        force_offsets = np.asarray(force["force_offsets"][:], dtype=np.int64)
    if not np.array_equal(full_index, np.sort(full_index)):
        raise ValueError("benchmark store db_index must be sorted")
    force_positions = np.searchsorted(full_index, force_index)
    if np.any(force_positions >= len(full_index)) or not np.array_equal(
        full_index[force_positions], force_index
    ):
        raise ValueError("force subset is not aligned to benchmark structures")
    labels, energy_cluster = np.unique(np.asarray(material_id, dtype=str), return_inverse=True)
    label_to_cluster = {label: index for index, label in enumerate(labels)}
    force_cluster = np.asarray(
        [label_to_cluster[str(material_id[position])] for position in force_positions],
        dtype=np.int64,
    )

    energy_error = np.abs(energy_location - energy_true)
    energy_nll = student_t_nll(
        energy_true, energy_location, energy_scale, energy_df
    )
    energy_covered = {
        level: energy_error <= t.ppf((1.0 + level) / 2.0, energy_df) * energy_scale
        for level in coverage_levels
    }
    energy_danger = energy_error >= np.quantile(energy_error, 0.95, method="linear")
    energy_error_rank = rankdata(energy_error, method="average")
    energy_risk_rank = rankdata(energy_risk, method="average")
    energy_order, energy_starts = _ranking_groups(energy_risk)

    force_component_error = np.abs(force_location - force_true)
    force_nll_component = (
        0.5 * np.log(2.0 * np.pi)
        + np.log(force_scale)
        + 0.5 * np.square((force_true - force_location) / force_scale)
    )
    force_component_count = 3 * np.diff(force_offsets)
    force_nll_sum = _force_structure_sums(force_nll_component, force_offsets)
    force_covered_sum = {
        level: _force_structure_sums(
            force_component_error <= norm.ppf((1.0 + level) / 2.0) * force_scale,
            force_offsets,
        )
        for level in coverage_levels
    }
    force_structure_error = _force_structure_sums(
        force_component_error, force_offsets
    ) / force_component_count
    force_structure_risk = np.sqrt(
        _force_structure_sums(np.square(force_scale_raw), force_offsets)
        / force_component_count
    )
    force_danger = force_structure_error >= np.quantile(
        force_structure_error, 0.95, method="linear"
    )
    force_error_rank = rankdata(force_structure_error, method="average")
    force_risk_rank = rankdata(force_structure_risk, method="average")
    force_order, force_starts = _ranking_groups(force_structure_risk)

    def evaluate(cluster_weights: np.ndarray) -> dict[str, float]:
        energy_weights = cluster_weights[energy_cluster]
        force_weights = cluster_weights[force_cluster]
        row = {
            "energy_nll_calibrated": _weighted_mean(energy_nll, energy_weights),
            "energy_spearman": _weighted_spearman(
                energy_error_rank, energy_risk_rank, energy_weights
            ),
            "force_nll_calibrated_component_weighted": float(
                np.dot(force_weights, force_nll_sum)
                / np.dot(force_weights, force_component_count)
            ),
            "force_structure_spearman": _weighted_spearman(
                force_error_rank, force_risk_rank, force_weights
            ),
        }
        for level in coverage_levels:
            row[f"energy_coverage_{int(level * 100):02d}"] = _weighted_mean(
                energy_covered[level], energy_weights
            )
            row[f"force_coverage_{int(level * 100):02d}_component_weighted"] = float(
                np.dot(force_weights, force_covered_sum[level])
                / np.dot(force_weights, force_component_count)
            )
        row["energy_top5_auroc"], row["energy_top5_auprc"] = _weighted_auroc_auprc(
            energy_danger, energy_weights, energy_order, energy_starts
        )
        row["force_structure_top5_auroc"], row["force_structure_top5_auprc"] = (
            _weighted_auroc_auprc(
                force_danger, force_weights, force_order, force_starts
            )
        )
        return row

    estimate = evaluate(np.ones(len(labels), dtype=np.float64))
    rng = np.random.default_rng(seed)
    replicate_rows = []
    for replicate in range(bootstrap_replicates):
        cluster_weights = np.bincount(
            rng.integers(0, len(labels), len(labels)), minlength=len(labels)
        ).astype(np.float64)
        row = {
            "replicate": replicate,
            "n_clusters": len(labels),
            **evaluate(cluster_weights),
        }
        replicate_rows.append(row)
    bootstrap = pd.DataFrame(replicate_rows)
    confidence_rows = []
    for metric, value in estimate.items():
        samples = bootstrap[metric].to_numpy(dtype=np.float64)
        samples = samples[np.isfinite(samples)]
        if len(samples):
            low, high = np.quantile(samples, [0.025, 0.975])
        else:
            low = high = float("nan")
        confidence_rows.append(
            {
                "metric": metric,
                "estimate": value,
                "ci_low_95": float(low),
                "ci_high_95": float(high),
                "bootstrap_unit": "material_id",
                "bootstrap_replicates": bootstrap_replicates,
                "valid_bootstrap_replicates": len(samples),
                "seed": seed,
            }
        )
    confidence = pd.DataFrame(confidence_rows)
    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    bootstrap.to_csv(tables / "uq_bootstrap_replicates.csv", index=False)
    confidence.to_csv(tables / "uq_confidence_intervals.csv", index=False)
    summary = {
        "schema_version": 1,
        "bootstrap_unit": "material_id",
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "n_energy_structures": len(full_index),
        "n_force_structures": len(force_index),
        "n_force_components": int(force_component_count.sum()),
        "headline": estimate,
        "force_coverage_weighting": "Cartesian-component weighted within material-cluster bootstrap",
        "spearman_definition": "weighted Pearson correlation of fixed average ranks",
    }
    (output_dir / "uq_bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return output_dir


def _pit_ece(pit: np.ndarray, bins: int = 20) -> float:
    counts, _ = np.histogram(np.asarray(pit), bins=np.linspace(0.0, 1.0, bins + 1))
    return float(np.mean(np.abs(counts / counts.sum() - 1.0 / bins)))


def _conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0 or np.any(~np.isfinite(scores)):
        raise ValueError("conformal scores must be a non-empty finite array")
    probability = min(1.0, np.ceil((len(scores) + 1) * coverage) / len(scores))
    return float(np.quantile(scores, probability, method="higher"))


def _distribution_metrics(
    method: str,
    target: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
    family: str,
    levels: tuple[float, ...],
    *,
    df: np.ndarray | float | None = None,
    target_name: str,
    unit: str,
) -> list[dict[str, object]]:
    target, location, scale = map(
        lambda value: np.asarray(value, dtype=np.float64), (target, location, scale)
    )
    if np.any(scale <= 0) or np.any(~np.isfinite(scale)):
        raise ValueError("calibrated scales must be finite and positive")
    standardized = (target - location) / scale
    if family == "gaussian":
        nll = gaussian_nll(target, location, scale)
        pit = norm.cdf(standardized)
    elif family == "student_t":
        if df is None:
            raise ValueError("Student-t metrics require df")
        nll = student_t_nll(target, location, scale, np.asarray(df))
        pit = t.cdf(standardized, df)
    else:
        raise ValueError(f"unsupported distribution family: {family}")
    rows = [
        {
            "method": method,
            "target": target_name,
            "metric": "NLL",
            "estimate": float(np.mean(nll)),
            "unit": "nats",
            "n": len(target),
            "distribution_family": family,
        },
        {
            "method": method,
            "target": target_name,
            "metric": "PIT_ECE_20",
            "estimate": _pit_ece(pit),
            "unit": "absolute_probability",
            "n": len(target),
            "distribution_family": family,
        },
    ]
    absolute_error = np.abs(target - location)
    for level in levels:
        quantile = (
            norm.ppf((1.0 + level) / 2.0)
            if family == "gaussian"
            else t.ppf((1.0 + level) / 2.0, df)
        )
        radius = np.asarray(quantile) * scale
        rows.extend(
            [
                {
                    "method": method,
                    "target": target_name,
                    "metric": f"coverage_{int(level * 100):02d}",
                    "estimate": float(np.mean(absolute_error <= radius)),
                    "unit": "fraction",
                    "n": len(target),
                    "distribution_family": family,
                },
                {
                    "method": method,
                    "target": target_name,
                    "metric": f"interval_width_{int(level * 100):02d}",
                    "estimate": float(np.mean(2.0 * radius)),
                    "unit": unit,
                    "n": len(target),
                    "distribution_family": family,
                },
            ]
        )
    return rows


def _conformal_metrics(
    method: str,
    target: np.ndarray,
    location: np.ndarray,
    radii: dict[float, np.ndarray],
    *,
    target_name: str,
    unit: str,
) -> list[dict[str, object]]:
    error = np.abs(np.asarray(target) - np.asarray(location))
    rows = []
    for level, radius in radii.items():
        radius = np.broadcast_to(np.asarray(radius, dtype=np.float64), error.shape)
        rows.extend(
            [
                {
                    "method": method,
                    "target": target_name,
                    "metric": f"coverage_{int(level * 100):02d}",
                    "estimate": float(np.mean(error <= radius)),
                    "unit": "fraction",
                    "n": len(error),
                    "distribution_family": "split_conformal",
                },
                {
                    "method": method,
                    "target": target_name,
                    "metric": f"interval_width_{int(level * 100):02d}",
                    "estimate": float(np.mean(2.0 * radius)),
                    "unit": unit,
                    "n": len(error),
                    "distribution_family": "split_conformal",
                },
            ]
        )
    return rows


def _fit_student_t_affine(
    target: np.ndarray, location: np.ndarray, raw_scale: np.ndarray
) -> dict[str, float]:
    residual_scale = max(float(np.std(target - location)), 1e-8)

    def objective(parameters: np.ndarray) -> float:
        tau = np.exp(parameters[0])
        sigma0 = np.exp(parameters[1])
        df = 2.0 + np.exp(parameters[2])
        scale = np.sqrt(np.square(tau * raw_scale) + sigma0 * sigma0)
        return float(np.mean(student_t_nll(target, location, scale, df)))

    result = minimize(
        objective,
        np.asarray([0.0, np.log(0.1 * residual_scale), np.log(6.0)]),
        method="L-BFGS-B",
        bounds=[(-12, 12), (-18, 12), (np.log(0.01), np.log(198.0))],
    )
    if not result.success:
        raise RuntimeError(f"Student-t affine calibration failed: {result.message}")
    return {
        "tau": float(np.exp(result.x[0])),
        "sigma0": float(np.exp(result.x[1])),
        "df": float(2.0 + np.exp(result.x[2])),
    }


def _fit_size_power_scale(
    target: np.ndarray,
    location: np.ndarray,
    raw_scale: np.ndarray,
    natoms_per_value: np.ndarray,
    *,
    family: str,
    df: np.ndarray | None = None,
    additive_floor: bool,
) -> dict[str, float]:
    reference_natoms = float(np.median(natoms_per_value))
    log_size = np.log(np.asarray(natoms_per_value, dtype=np.float64) / reference_natoms)
    residual_scale = max(float(np.std(target - location)), 1e-8)

    def objective(parameters: np.ndarray) -> float:
        tau = np.exp(parameters[0] + parameters[1] * log_size)
        sigma0 = np.exp(parameters[2]) if additive_floor else 0.0
        scale = np.sqrt(np.square(tau * raw_scale) + sigma0 * sigma0)
        if family == "student_t":
            return float(np.mean(student_t_nll(target, location, scale, df)))
        return float(np.mean(gaussian_nll(target, location, scale)))

    initial = [0.0, 0.0]
    bounds = [(-12, 12), (-3, 3)]
    if additive_floor:
        initial.append(np.log(0.1 * residual_scale))
        bounds.append((-18, 12))
    result = minimize(objective, np.asarray(initial), method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise RuntimeError(f"size-power calibration failed: {result.message}")
    return {
        "tau_reference": float(np.exp(result.x[0])),
        "size_exponent": float(result.x[1]),
        "sigma0": float(np.exp(result.x[2])) if additive_floor else 0.0,
        "reference_natoms": reference_natoms,
    }


def _apply_size_power(
    raw_scale: np.ndarray, natoms_per_value: np.ndarray, parameters: dict[str, float]
) -> np.ndarray:
    tau = parameters["tau_reference"] * np.power(
        np.asarray(natoms_per_value, dtype=np.float64) / parameters["reference_natoms"],
        parameters["size_exponent"],
    )
    return np.sqrt(
        np.square(tau * np.asarray(raw_scale, dtype=np.float64))
        + parameters["sigma0"] ** 2
    )


def _size_rows(
    method: str,
    target: np.ndarray,
    location: np.ndarray,
    natoms_per_value: np.ndarray,
    radii: dict[float, np.ndarray],
    target_name: str,
) -> list[dict[str, object]]:
    groups = np.digitize(natoms_per_value, SIZE_BINS[1:-1], right=True)
    error = np.abs(np.asarray(target) - np.asarray(location))
    rows = []
    for group_index, label in enumerate(SIZE_LABELS):
        selected = groups == group_index
        if not np.any(selected):
            continue
        row: dict[str, object] = {
            "method": method,
            "target": target_name,
            "size_bin": label,
            "n_values": int(selected.sum()),
        }
        for level, radius in radii.items():
            radius = np.broadcast_to(np.asarray(radius), error.shape)
            row[f"coverage_{int(level * 100):02d}"] = float(
                np.mean(error[selected] <= radius[selected])
            )
            row[f"mean_radius_{int(level * 100):02d}"] = float(
                np.mean(radius[selected])
            )
        rows.append(row)
    return rows


def _distribution_radii(
    scale: np.ndarray,
    family: str,
    levels: tuple[float, ...],
    df: np.ndarray | float | None = None,
) -> dict[float, np.ndarray]:
    return {
        level: np.asarray(scale)
        * (
            norm.ppf((1.0 + level) / 2.0)
            if family == "gaussian"
            else t.ppf((1.0 + level) / 2.0, df)
        )
        for level in levels
    }


def _write_calibration_sensitivity_figure(
    metrics: pd.DataFrame, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True, sharey=True)
    colors = ("#0B6E69", "#C65D3B", "#D69B2D", "#355C7D", "#7B5E57", "#4F772D")
    for axis, target, title in (
        (axes[0], "energy_per_atom", "Energy calibration sensitivity"),
        (axes[1], "force_component", "Force calibration sensitivity"),
    ):
        selected = metrics[
            (metrics.target == target) & metrics.metric.str.startswith("coverage_")
        ]
        for color, (method, group) in zip(colors, selected.groupby("method"), strict=False):
            levels = group.metric.str.rsplit("_", n=1).str[-1].astype(float) / 100.0
            axis.plot(levels, group.estimate, marker="o", label=method, color=color)
        axis.plot(COVERAGE_LEVELS, COVERAGE_LEVELS, "--", color="#263238", label="ideal")
        axis.set_title(title)
        axis.set_xlabel("Nominal coverage")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Empirical coverage")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"calibration_family_sensitivity.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def analyze_calibration_sensitivity(
    energy_calibration_path: str | Path,
    energy_test_path: str | Path,
    force_calibration_path: str | Path,
    force_test_path: str | Path,
    output_dir: str | Path,
    *,
    coverage_levels: tuple[float, ...] = COVERAGE_LEVELS,
) -> Path:
    """Fit calibration alternatives on calibration only and evaluate frozen test."""
    from .pipeline import epic_prediction

    energy_calibration = _load_npz(energy_calibration_path)
    energy_test = _load_npz(energy_test_path)
    force_calibration = _load_npz(force_calibration_path)
    force_test = _load_npz(force_test_path)
    if str(energy_calibration.get("split", "")) != "calibration":
        raise ValueError("energy calibration predictions must declare calibration split")
    if str(energy_test.get("split", "")) != "test":
        raise ValueError("energy test predictions must declare test split")
    if str(force_calibration.get("split", "")) != "calibration":
        raise ValueError("force calibration predictions must declare calibration split")
    if str(force_test.get("split", "")) != "test":
        raise ValueError("force test predictions must declare test split")

    energy_cal_raw = epic_prediction(energy_calibration_path)
    energy_test_raw = epic_prediction(energy_test_path)
    energy_cal_natoms = np.asarray(energy_calibration["natoms"], dtype=np.float64)
    energy_test_natoms = np.asarray(energy_test["natoms"], dtype=np.float64)
    energy_cal_true = (
        np.asarray(energy_calibration["energy_true_total"], dtype=np.float64)
        / energy_cal_natoms
    )
    energy_test_true = (
        np.asarray(energy_test["energy_true_total"], dtype=np.float64)
        / energy_test_natoms
    )
    energy_global_state = fit_student_t_scale(
        energy_cal_true,
        energy_cal_raw.energy_location,
        energy_cal_raw.energy_scale_raw,
        energy_cal_raw.energy_df,
    )
    energy_global_scale = (
        energy_test_raw.energy_scale_raw * energy_global_state.parameters["tau"]
    )
    energy_size_parameters = _fit_size_power_scale(
        energy_cal_true,
        energy_cal_raw.energy_location,
        energy_cal_raw.energy_scale_raw,
        energy_cal_natoms,
        family="student_t",
        df=energy_cal_raw.energy_df,
        additive_floor=False,
    )
    energy_size_scale = _apply_size_power(
        energy_test_raw.energy_scale_raw, energy_test_natoms, energy_size_parameters
    )
    energy_global_conformal_scores = np.abs(
        energy_cal_true - energy_cal_raw.energy_location
    ) / energy_cal_raw.energy_scale_raw
    energy_global_conformal_radii = {
        level: _conformal_quantile(energy_global_conformal_scores, level)
        * energy_test_raw.energy_scale_raw
        for level in coverage_levels
    }
    energy_cal_bins = np.digitize(energy_cal_natoms, SIZE_BINS[1:-1], right=True)
    energy_test_bins = np.digitize(energy_test_natoms, SIZE_BINS[1:-1], right=True)
    energy_size_conformal_radii = {}
    energy_size_conformal_parameters: dict[str, dict[str, float]] = {}
    for level in coverage_levels:
        quantiles = np.asarray(
            [
                _conformal_quantile(
                    energy_global_conformal_scores[energy_cal_bins == index], level
                )
                for index in range(len(SIZE_LABELS))
            ]
        )
        energy_size_conformal_radii[level] = (
            quantiles[energy_test_bins] * energy_test_raw.energy_scale_raw
        )
        energy_size_conformal_parameters[str(level)] = {
            label: float(quantiles[index]) for index, label in enumerate(SIZE_LABELS)
        }

    rows = []
    size_rows = []
    energy_models = {
        "student_t_global": (
            energy_global_scale,
            _distribution_radii(
                energy_global_scale, "student_t", coverage_levels, energy_test_raw.energy_df
            ),
        ),
        "student_t_size_power": (
            energy_size_scale,
            _distribution_radii(
                energy_size_scale, "student_t", coverage_levels, energy_test_raw.energy_df
            ),
        ),
    }
    for method, (scale, radii) in energy_models.items():
        rows.extend(
            _distribution_metrics(
                method,
                energy_test_true,
                energy_test_raw.energy_location,
                scale,
                "student_t",
                coverage_levels,
                df=energy_test_raw.energy_df,
                target_name="energy_per_atom",
                unit="eV/atom",
            )
        )
        size_rows.extend(
            _size_rows(
                method,
                energy_test_true,
                energy_test_raw.energy_location,
                energy_test_natoms,
                radii,
                "energy_per_atom",
            )
        )
    for method, radii in (
        ("conformal_global", energy_global_conformal_radii),
        ("conformal_size_bin", energy_size_conformal_radii),
    ):
        rows.extend(
            _conformal_metrics(
                method,
                energy_test_true,
                energy_test_raw.energy_location,
                radii,
                target_name="energy_per_atom",
                unit="eV/atom",
            )
        )
        size_rows.extend(
            _size_rows(
                method,
                energy_test_true,
                energy_test_raw.energy_location,
                energy_test_natoms,
                radii,
                "energy_per_atom",
            )
        )

    force_cal_index = np.asarray(force_calibration["db_index"], dtype=np.int64)
    force_test_index = np.asarray(force_test["db_index"], dtype=np.int64)
    for point, force_index, label in (
        (energy_calibration, force_cal_index, "calibration"),
        (energy_test, force_test_index, "test"),
    ):
        point_index = np.asarray(point["db_index"], dtype=np.int64)
        positions = np.searchsorted(point_index, force_index)
        if np.any(positions >= len(point_index)) or not np.array_equal(
            point_index[positions], force_index
        ):
            raise ValueError(f"force {label} subset is not aligned to point predictions")
        if label == "calibration":
            force_cal_natoms = np.asarray(point["natoms"], dtype=np.int64)[positions]
        else:
            force_test_natoms = np.asarray(point["natoms"], dtype=np.int64)[positions]
    if not np.array_equal(np.diff(force_calibration["force_offsets"]), force_cal_natoms):
        raise ValueError("force calibration offsets do not match natoms")
    if not np.array_equal(np.diff(force_test["force_offsets"]), force_test_natoms):
        raise ValueError("force test offsets do not match natoms")
    force_cal_true = np.asarray(force_calibration["force_true"], dtype=np.float64).reshape(-1)
    force_cal_location = np.asarray(force_calibration["force_mean"], dtype=np.float64).reshape(-1)
    force_cal_raw = np.asarray(force_calibration["force_scale_raw"], dtype=np.float64).reshape(-1)
    force_test_true = np.asarray(force_test["force_true"], dtype=np.float64).reshape(-1)
    force_test_location = np.asarray(force_test["force_mean"], dtype=np.float64).reshape(-1)
    force_test_raw = np.asarray(force_test["force_scale_raw"], dtype=np.float64).reshape(-1)
    force_cal_natoms_component = np.repeat(force_cal_natoms, force_cal_natoms * 3)
    force_test_natoms_component = np.repeat(force_test_natoms, force_test_natoms * 3)

    force_global_state = fit_gaussian_variance_map(
        force_cal_true,
        force_cal_location,
        np.square(force_cal_raw),
        method="gaussian_affine_global",
    )
    force_global_scale = np.sqrt(
        force_global_state.parameters["tau"] ** 2 * np.square(force_test_raw)
        + force_global_state.parameters["sigma0"] ** 2
    )
    force_multiplicative_minimum = 1e-8
    force_cal_raw_multiplicative = np.maximum(
        force_cal_raw, force_multiplicative_minimum
    )
    force_test_raw_multiplicative = np.maximum(
        force_test_raw, force_multiplicative_minimum
    )
    force_multiplicative_tau = float(
        np.sqrt(
            np.mean(
                np.square(
                    (force_cal_true - force_cal_location)
                    / force_cal_raw_multiplicative
                )
            )
        )
    )
    force_multiplicative_scale = (
        force_multiplicative_tau * force_test_raw_multiplicative
    )
    force_student_parameters = _fit_student_t_affine(
        force_cal_true, force_cal_location, force_cal_raw
    )
    force_student_scale = np.sqrt(
        np.square(force_student_parameters["tau"] * force_test_raw)
        + force_student_parameters["sigma0"] ** 2
    )
    force_size_parameters = _fit_size_power_scale(
        force_cal_true,
        force_cal_location,
        force_cal_raw,
        force_cal_natoms_component,
        family="gaussian",
        additive_floor=True,
    )
    force_size_scale = _apply_size_power(
        force_test_raw, force_test_natoms_component, force_size_parameters
    )
    force_distribution_models = {
        "gaussian_affine_global": (force_global_scale, "gaussian", None),
        "gaussian_multiplicative": (force_multiplicative_scale, "gaussian", None),
        "student_t_affine_global": (
            force_student_scale,
            "student_t",
            force_student_parameters["df"],
        ),
        "gaussian_size_power_affine": (force_size_scale, "gaussian", None),
    }
    for method, (scale, family, df) in force_distribution_models.items():
        radii = _distribution_radii(scale, family, coverage_levels, df)
        rows.extend(
            _distribution_metrics(
                method,
                force_test_true,
                force_test_location,
                scale,
                family,
                coverage_levels,
                df=df,
                target_name="force_component",
                unit="eV/Angstrom",
            )
        )
        size_rows.extend(
            _size_rows(
                method,
                force_test_true,
                force_test_location,
                force_test_natoms_component,
                radii,
                "force_component",
            )
        )

    force_cal_global_scale = np.sqrt(
        force_global_state.parameters["tau"] ** 2 * np.square(force_cal_raw)
        + force_global_state.parameters["sigma0"] ** 2
    )
    force_conformal_scores = np.abs(force_cal_true - force_cal_location) / force_cal_global_scale
    force_global_conformal_radii = {
        level: _conformal_quantile(force_conformal_scores, level) * force_global_scale
        for level in coverage_levels
    }
    force_cal_bins = np.digitize(
        force_cal_natoms_component, SIZE_BINS[1:-1], right=True
    )
    force_test_bins = np.digitize(
        force_test_natoms_component, SIZE_BINS[1:-1], right=True
    )
    force_size_conformal_radii = {}
    force_size_conformal_parameters: dict[str, dict[str, float]] = {}
    for level in coverage_levels:
        quantiles = np.asarray(
            [
                _conformal_quantile(force_conformal_scores[force_cal_bins == index], level)
                for index in range(len(SIZE_LABELS))
            ]
        )
        force_size_conformal_radii[level] = quantiles[force_test_bins] * force_global_scale
        force_size_conformal_parameters[str(level)] = {
            label: float(quantiles[index]) for index, label in enumerate(SIZE_LABELS)
        }
    for method, radii in (
        ("force_conformal_global", force_global_conformal_radii),
        ("force_conformal_size_bin", force_size_conformal_radii),
    ):
        rows.extend(
            _conformal_metrics(
                method,
                force_test_true,
                force_test_location,
                radii,
                target_name="force_component",
                unit="eV/Angstrom",
            )
        )
        size_rows.extend(
            _size_rows(
                method,
                force_test_true,
                force_test_location,
                force_test_natoms_component,
                radii,
                "force_component",
            )
        )

    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    size_frame = pd.DataFrame(size_rows)
    metrics.to_csv(tables / "calibration_sensitivity_metrics.csv", index=False)
    size_frame.to_csv(tables / "calibration_sensitivity_by_size.csv", index=False)
    parameters = {
        "schema_version": 1,
        "fit_split": "calibration",
        "evaluation_split": "test",
        "test_used_for_selection": False,
        "energy": {
            "student_t_global": dict(energy_global_state.parameters),
            "student_t_size_power": energy_size_parameters,
            "conformal_size_bin": energy_size_conformal_parameters,
        },
        "force": {
            "gaussian_affine_global": dict(force_global_state.parameters),
            "gaussian_multiplicative": {
                "tau": force_multiplicative_tau,
                "minimum_raw_scale": force_multiplicative_minimum,
            },
            "student_t_affine_global": force_student_parameters,
            "gaussian_size_power_affine": force_size_parameters,
            "conformal_size_bin": force_size_conformal_parameters,
        },
    }
    (output_dir / "calibration_sensitivity_parameters.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True) + "\n"
    )
    _write_calibration_sensitivity_figure(metrics, output_dir / "figures")
    return output_dir


def _validate_force_allocation_pair(
    learned: dict[str, np.ndarray],
    uniform: dict[str, np.ndarray],
    split: str,
) -> dict[str, float]:
    required = {
        "db_index",
        "force_offsets",
        "force_true",
        "force_mean",
        "force_scale_raw",
        "split",
    }
    for name, record in (("learned", learned), ("uniform", uniform)):
        missing = required - set(record)
        if missing:
            raise ValueError(f"{name} {split} force record is missing {sorted(missing)}")
        if str(record["split"]) != split:
            raise ValueError(f"{name} force record must declare {split} split")
        scale = np.asarray(record["force_scale_raw"], dtype=np.float64)
        if np.any(~np.isfinite(scale)) or np.any(scale < 0):
            raise ValueError(f"{name} {split} force scales must be finite and non-negative")

    for field in ("db_index", "force_offsets", "force_true"):
        if not np.array_equal(learned[field], uniform[field]):
            raise ValueError(f"learned and uniform {split} records disagree on {field}")
    point_difference = np.asarray(learned["force_mean"], dtype=np.float64) - np.asarray(
        uniform["force_mean"], dtype=np.float64
    )
    point_rmse = float(np.sqrt(np.mean(np.square(point_difference))))
    point_maximum = float(np.max(np.abs(point_difference)))
    if point_rmse > 1e-6 or point_maximum > 1e-4:
        raise ValueError(
            f"learned and uniform {split} point forces differ beyond CUDA rerun "
            f"tolerance: rmse={point_rmse:.6g}, max_abs={point_maximum:.6g}"
        )
    if learned["force_scale_raw"].shape != learned["force_true"].shape:
        raise ValueError(f"learned {split} force scale shape does not match force truth")
    if uniform["force_scale_raw"].shape != uniform["force_true"].shape:
        raise ValueError(f"uniform {split} force scale shape does not match force truth")

    uniform_mode = str(uniform.get("allocation_mode", ""))
    if uniform_mode != "uniform":
        raise ValueError("uniform force record must declare allocation_mode=uniform")
    learned_mode = str(learned.get("allocation_mode", "learned"))
    if learned_mode != "learned":
        raise ValueError("learned force record must use learned allocation")
    uniform_sampling = str(uniform.get("sampling_method", ""))
    if uniform_sampling != "hierarchical_uniform_shared_scale_latent_mean_rsample":
        raise ValueError("uniform force record has an unexpected sampling method")
    for field in ("n_samples", "sample_seed"):
        if field in learned and field in uniform and int(learned[field]) != int(uniform[field]):
            raise ValueError(f"learned and uniform {split} records disagree on {field}")
    return {"rmse": point_rmse, "max_absolute_difference": point_maximum}


def _force_allocation_statistics(
    record: dict[str, np.ndarray],
    gaussian_parameters: dict[str, float],
    coverage_levels: tuple[float, ...],
) -> dict[str, object]:
    offsets = np.asarray(record["force_offsets"], dtype=np.int64)
    truth = np.asarray(record["force_true"], dtype=np.float64)
    location = np.asarray(record["force_mean"], dtype=np.float64)
    raw_scale = np.asarray(record["force_scale_raw"], dtype=np.float64)
    _validate_offsets(offsets, len(truth), len(record["db_index"]))
    component_count = 3 * np.diff(offsets)
    component_error = np.abs(location - truth)
    structure_error = _force_structure_sums(component_error, offsets) / component_count
    structure_risk = np.sqrt(
        _force_structure_sums(np.square(raw_scale), offsets) / component_count
    )
    calibrated_scale = np.sqrt(
        gaussian_parameters["tau"] ** 2 * np.square(raw_scale)
        + gaussian_parameters["sigma0"] ** 2
    )
    component_nll = gaussian_nll(truth, location, calibrated_scale)
    nll_sum = _force_structure_sums(component_nll, offsets)
    coverage_sum = {
        level: _force_structure_sums(
            component_error
            <= norm.ppf((1.0 + level) / 2.0) * calibrated_scale,
            offsets,
        )
        for level in coverage_levels
    }
    return {
        "component_count": component_count,
        "structure_error": structure_error,
        "structure_error_rank": rankdata(structure_error, method="average"),
        "structure_risk": structure_risk,
        "structure_risk_rank": rankdata(structure_risk, method="average"),
        "risk_order": _ranking_groups(structure_risk),
        "nll_sum": nll_sum,
        "coverage_sum": coverage_sum,
        "raw_scale": raw_scale,
        "calibrated_scale": calibrated_scale,
    }


def _write_force_allocation_figure(
    confidence: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    estimates = confidence.set_index("metric")["estimate"]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    allocations = ("learned", "uniform")
    colors = ("#0B6E69", "#C65D3B")
    ranking = np.asarray(
        [estimates[f"{name}_force_structure_spearman"] for name in allocations]
    )
    axes[0].bar(allocations, ranking, color=colors)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Spearman correlation")
    axes[0].set_title("Raw force-risk ranking")
    axes[0].grid(axis="y", alpha=0.22)

    x = np.asarray(COVERAGE_LEVELS)
    for name, color in zip(allocations, colors, strict=True):
        values = [estimates[f"{name}_coverage_{int(level * 100):02d}"] for level in x]
        axes[1].plot(x, values, marker="o", color=color, label=name)
    axes[1].plot(x, x, "--", color="#263238", label="ideal")
    axes[1].set_xlabel("Nominal coverage")
    axes[1].set_ylabel("Empirical coverage")
    axes[1].set_title("Gaussian affine calibration")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"learned_vs_uniform_force_allocation.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)


def compare_force_allocations(
    learned_calibration_path: str | Path,
    learned_test_path: str | Path,
    uniform_calibration_path: str | Path,
    uniform_test_path: str | Path,
    store_path: str | Path,
    learned_sensitivity_dir: str | Path,
    uniform_sensitivity_dir: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_replicates: int = 1000,
    seed: int = 42,
    coverage_levels: tuple[float, ...] = COVERAGE_LEVELS,
) -> Path:
    """Compare learned and uniform force allocation with paired cluster bootstrap."""
    learned_calibration = _load_npz(learned_calibration_path)
    learned_test = _load_npz(learned_test_path)
    uniform_calibration = _load_npz(uniform_calibration_path)
    uniform_test = _load_npz(uniform_test_path)
    point_force_alignment = {
        "calibration": _validate_force_allocation_pair(
            learned_calibration, uniform_calibration, "calibration"
        ),
        "test": _validate_force_allocation_pair(learned_test, uniform_test, "test"),
    }

    sensitivity_records = {}
    sensitivity_frames = []
    for allocation, directory in (
        ("learned", Path(learned_sensitivity_dir)),
        ("uniform", Path(uniform_sensitivity_dir)),
    ):
        parameters = json.loads(
            (directory / "calibration_sensitivity_parameters.json").read_text()
        )
        if (
            parameters.get("fit_split") != "calibration"
            or parameters.get("evaluation_split") != "test"
            or parameters.get("test_used_for_selection") is not False
        ):
            raise ValueError(f"{allocation} sensitivity provenance is invalid")
        sensitivity_records[allocation] = parameters
        frame = pd.read_csv(directory / "tables/calibration_sensitivity_metrics.csv")
        frame.insert(0, "allocation", allocation)
        sensitivity_frames.append(frame)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)
    energy_learned = sensitivity[
        (sensitivity.allocation == "learned")
        & (sensitivity.target == "energy_per_atom")
    ].drop(columns="allocation").reset_index(drop=True)
    energy_uniform = sensitivity[
        (sensitivity.allocation == "uniform")
        & (sensitivity.target == "energy_per_atom")
    ].drop(columns="allocation").reset_index(drop=True)
    if not energy_learned.equals(energy_uniform):
        raise ValueError("energy sensitivity changed under force allocation ablation")

    test_index = np.asarray(learned_test["db_index"], dtype=np.int64)
    material_id = _material_ids_for_predictions(store_path, test_index)
    labels, structure_cluster = np.unique(material_id, return_inverse=True)
    danger_threshold = np.quantile(
        _force_structure_sums(
            np.abs(learned_test["force_mean"] - learned_test["force_true"]),
            learned_test["force_offsets"],
        )
        / (3 * np.diff(learned_test["force_offsets"])),
        0.95,
        method="linear",
    )
    statistics = {
        allocation: _force_allocation_statistics(
            record,
            sensitivity_records[allocation]["force"]["gaussian_affine_global"],
            coverage_levels,
        )
        for allocation, record in (("learned", learned_test), ("uniform", uniform_test))
    }

    def evaluate(cluster_weights: np.ndarray) -> dict[str, float]:
        structure_weights = cluster_weights[structure_cluster]
        row: dict[str, float] = {}
        for allocation, values in statistics.items():
            component_count = values["component_count"]
            risk_order, risk_starts = values["risk_order"]
            danger = values["structure_error"] >= danger_threshold
            auroc, auprc = _weighted_auroc_auprc(
                danger, structure_weights, risk_order, risk_starts
            )
            prefix = f"{allocation}_"
            row[prefix + "force_structure_spearman"] = _weighted_spearman(
                values["structure_error_rank"],
                values["structure_risk_rank"],
                structure_weights,
            )
            row[prefix + "force_structure_top5_auroc"] = auroc
            row[prefix + "force_structure_top5_auprc"] = auprc
            denominator = np.dot(structure_weights, component_count)
            row[prefix + "gaussian_affine_nll"] = float(
                np.dot(structure_weights, values["nll_sum"]) / denominator
            )
            for level in coverage_levels:
                suffix = f"coverage_{int(level * 100):02d}"
                coverage = float(
                    np.dot(structure_weights, values["coverage_sum"][level])
                    / denominator
                )
                row[prefix + suffix] = coverage
                row[prefix + suffix + "_absolute_error"] = abs(coverage - level)
        metric_suffixes = [
            "force_structure_spearman",
            "force_structure_top5_auroc",
            "force_structure_top5_auprc",
            "gaussian_affine_nll",
        ]
        for level in coverage_levels:
            suffix = f"coverage_{int(level * 100):02d}"
            metric_suffixes.extend((suffix, suffix + "_absolute_error"))
        for suffix in metric_suffixes:
            row[f"uniform_minus_learned_{suffix}"] = (
                row[f"uniform_{suffix}"] - row[f"learned_{suffix}"]
            )
        return row

    estimate = evaluate(np.ones(len(labels), dtype=np.float64))
    rng = np.random.default_rng(seed)
    replicate_rows = []
    for replicate in range(bootstrap_replicates):
        cluster_weights = np.bincount(
            rng.integers(0, len(labels), len(labels)), minlength=len(labels)
        ).astype(np.float64)
        replicate_rows.append(
            {
                "replicate": replicate,
                "n_clusters": len(labels),
                **evaluate(cluster_weights),
            }
        )
    bootstrap = pd.DataFrame(replicate_rows)
    confidence_rows = []
    for metric, value in estimate.items():
        samples = bootstrap[metric].to_numpy(dtype=np.float64)
        samples = samples[np.isfinite(samples)]
        low, high = (
            np.quantile(samples, [0.025, 0.975])
            if len(samples)
            else (float("nan"), float("nan"))
        )
        confidence_rows.append(
            {
                "metric": metric,
                "estimate": value,
                "ci_low_95": float(low),
                "ci_high_95": float(high),
                "bootstrap_unit": "material_id",
                "bootstrap_replicates": bootstrap_replicates,
                "valid_bootstrap_replicates": len(samples),
                "seed": seed,
            }
        )
    confidence = pd.DataFrame(confidence_rows)

    scale_rows = []
    for split, learned, uniform in (
        ("calibration", learned_calibration, uniform_calibration),
        ("test", learned_test, uniform_test),
    ):
        left = np.asarray(learned["force_scale_raw"], dtype=np.float64).reshape(-1)
        right = np.asarray(uniform["force_scale_raw"], dtype=np.float64).reshape(-1)
        relative = np.abs(right - left) / np.maximum(left, 1e-12)
        scale_rows.append(
            {
                "split": split,
                "n_components": len(left),
                "learned_zero_components": int(np.sum(left == 0)),
                "uniform_zero_components": int(np.sum(right == 0)),
                "learned_scale_median": float(np.median(left)),
                "uniform_scale_median": float(np.median(right)),
                "median_absolute_difference": float(np.median(np.abs(right - left))),
                "median_absolute_relative_difference": float(np.median(relative)),
                "q95_absolute_relative_difference": float(np.quantile(relative, 0.95)),
                "component_scale_spearman": _safe_spearman(left, right),
                "learned_inference_seconds": float(learned["inference_seconds"]),
                "uniform_inference_seconds": float(uniform["inference_seconds"]),
                "learned_peak_memory_bytes": int(learned["peak_memory_bytes"]),
                "uniform_peak_memory_bytes": int(uniform["peak_memory_bytes"]),
            }
        )
    scale_comparison = pd.DataFrame(scale_rows)

    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    confidence.to_csv(tables / "force_allocation_confidence_intervals.csv", index=False)
    bootstrap.to_csv(tables / "force_allocation_bootstrap.csv", index=False)
    scale_comparison.to_csv(tables / "force_allocation_scale_comparison.csv", index=False)
    sensitivity.to_csv(tables / "force_allocation_calibration_sensitivity.csv", index=False)
    summary = {
        "schema_version": 1,
        "comparison": "learned_vs_uniform_force_allocation",
        "fit_split": "calibration",
        "evaluation_split": "test",
        "test_used_for_selection": False,
        "n_test_structures": len(test_index),
        "n_test_force_components": int(3 * np.diff(learned_test["force_offsets"]).sum()),
        "bootstrap_unit": "material_id",
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "point_force_alignment_verified": True,
        "point_force_alignment_tolerance": {
            "rmse_limit": 1e-6,
            "max_absolute_difference_limit": 1e-4,
        },
        "point_force_alignment_observed": point_force_alignment,
        "energy_uq_invariance_verified": True,
        "headline": estimate,
        "interpretation_limit": (
            "This isolates inference-time force allocation for one trained checkpoint. "
            "It does not replace the separately trained allocation_weight=0 ablation."
        ),
    }
    (output_dir / "force_allocation_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_force_allocation_figure(confidence, output_dir / "figures")
    return output_dir
