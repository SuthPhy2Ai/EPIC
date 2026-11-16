from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, t

from .distributions import student_t_nll
from .pipeline import epic_prediction, load_calibration_state_file, validate_epic_force_record
from .figures import (
    write_force_alpha_convergence_figures,
    write_force_convergence_figures,
    write_size_analysis_figures,
)


SIZE_BINS = np.asarray([0, 20, 40, 80, 160, np.inf])
SIZE_LABELS = ("1-20", "21-40", "41-80", "81-160", ">160")
COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)


def _structure_force_mae(force_true: np.ndarray, force_pred: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    error = np.abs(np.asarray(force_pred) - np.asarray(force_true))
    return np.asarray([error[offsets[i] : offsets[i + 1]].mean() for i in range(len(offsets) - 1)])


def _atomic_sums(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.add.reduceat(np.asarray(values, dtype=np.float64), np.asarray(offsets[:-1], dtype=np.int64))


def analyze_two_model_predictions(
    mace_test_path: str | Path,
    hierarchical_test_path: str | Path,
    calibration_state_path: str | Path,
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = load_calibration_state_file(calibration_state_path, "hierarchical_epic")
    hierarchical_prediction = epic_prediction(hierarchical_test_path, state)
    with np.load(mace_test_path) as source:
        mace = {name: source[name] for name in source.files}
    with np.load(hierarchical_test_path) as source:
        hierarchical = {name: source[name] for name in source.files}
    if not np.array_equal(mace["db_index"], hierarchical["db_index"]):
        raise ValueError("two-model analysis inputs are not aligned")
    if not np.array_equal(mace["force_offsets"], hierarchical["force_offsets"]):
        raise ValueError("two-model force offsets are not aligned")

    natoms = np.asarray(hierarchical["natoms"], dtype=np.int64)
    truth = np.asarray(hierarchical["energy_true_total"], dtype=np.float64) / natoms
    size_group = np.digitize(natoms, SIZE_BINS[1:-1], right=True)
    point_records = {
        "mace": mace,
        "hierarchical_epic": hierarchical,
    }
    size_rows = []
    quantile_rows = []
    quantiles = (0.50, 0.90, 0.95, 0.99, 0.999, 1.0)
    for method, record in point_records.items():
        energy_pred = np.asarray(record["energy_pred_total"], dtype=np.float64) / natoms
        energy_error = np.abs(energy_pred - truth)
        force_error = _structure_force_mae(
            hierarchical["force_true"],
            record["force_pred"],
            hierarchical["force_offsets"],
        )
        for index, label in enumerate(SIZE_LABELS):
            selected = size_group == index
            if not np.any(selected):
                continue
            row = {
                "method": method,
                "size_bin": label,
                "n_structures": int(selected.sum()),
                "natoms_min": int(natoms[selected].min()),
                "natoms_max": int(natoms[selected].max()),
                "energy_mae_ev_per_atom": float(energy_error[selected].mean()),
                "energy_rmse_ev_per_atom": float(np.sqrt(np.mean((energy_pred[selected] - truth[selected]) ** 2))),
                "force_structure_component_mae_ev_per_angstrom": float(force_error[selected].mean()),
            }
            if method == "hierarchical_epic":
                for calibration, scale in (
                    ("raw", hierarchical_prediction.energy_scale_raw),
                    ("calibrated", hierarchical_prediction.energy_scale_calibrated),
                ):
                    row[f"energy_nll_{calibration}"] = float(
                        np.mean(
                            student_t_nll(
                                truth[selected],
                                energy_pred[selected],
                                scale[selected],
                                hierarchical_prediction.energy_df[selected],
                            )
                        )
                    )
                    for level in COVERAGE_LEVELS:
                        radius = t.ppf(
                            (1.0 + level) / 2.0,
                            hierarchical_prediction.energy_df[selected],
                        ) * scale[selected]
                        row[f"coverage_{int(level * 100):02d}_{calibration}"] = float(
                            np.mean(energy_error[selected] <= radius)
                        )
            size_rows.append(row)
        for target, values, unit in (
            ("energy_per_atom_absolute_error", energy_error, "eV/atom"),
            ("force_structure_component_mae", force_error, "eV/Angstrom"),
        ):
            for quantile in quantiles:
                quantile_rows.append(
                    {
                        "method": method,
                        "target": target,
                        "quantile": quantile,
                        "estimate": float(np.quantile(values, quantile)),
                        "unit": unit,
                        "n": len(values),
                    }
                )

    offsets = np.asarray(hierarchical["force_offsets"], dtype=np.int64)
    reconstruction = {}
    for component in ("aleatoric", "epistemic", "total"):
        atomic = _atomic_sums(hierarchical[f"atomic_{component}_variance"], offsets)
        if component == "total" and "energy_total_variance" not in hierarchical:
            graph = np.asarray(
                hierarchical["energy_aleatoric_variance"]
                + hierarchical["energy_epistemic_variance"],
                dtype=np.float64,
            )
        else:
            graph = np.asarray(hierarchical[f"energy_{component}_variance"], dtype=np.float64)
        residual = atomic - graph
        reconstruction[component] = {
            "max_absolute_residual": float(np.max(np.abs(residual))),
            "mean_absolute_residual": float(np.mean(np.abs(residual))),
            "max_relative_residual": float(
                np.max(np.abs(residual) / np.maximum(np.abs(graph), 1e-30))
            ),
        }

    alpha = np.asarray(hierarchical["alpha"], dtype=np.float64)
    summary = {
        "schema_version": 1,
        "n_test_structures": int(len(natoms)),
        "natoms_min": int(natoms.min()),
        "natoms_max": int(natoms.max()),
        "fraction_alpha_le_1_5": float(np.mean(alpha <= 1.5)),
        "fraction_alpha_le_2": float(np.mean(alpha <= 2.0)),
        "calibration_tau": float(state.parameters["tau"]),
        "atomic_variance_reconstruction": reconstruction,
        "external_ood_evaluation_available": False,
        "external_ood_note": (
            "The frozen material-grouped test split is in-distribution. Large-size strata are "
            "reported as stress tests and are not relabeled as external OOD data."
        ),
    }
    size_frame = pd.DataFrame(size_rows)
    size_frame.to_csv(output_dir / "size_stratified_metrics.csv", index=False)
    pd.DataFrame(quantile_rows).to_csv(output_dir / "error_quantiles.csv", index=False)
    (output_dir / "two_model_analysis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_size_analysis_figures(size_frame, output_dir / "figures")
    return output_dir


def _structure_rms_scale(scale: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    squared = np.square(np.asarray(scale, dtype=np.float64))
    return np.asarray(
        [np.sqrt(squared[offsets[i] : offsets[i + 1]].mean()) for i in range(len(offsets) - 1)]
    )


def analyze_force_convergence(
    input_paths: list[str | Path],
    output_dir: str | Path,
    hierarchical_predictions_path: str | Path | None = None,
) -> Path:
    if not input_paths:
        raise ValueError("force convergence analysis requires input files")
    records = []
    for path in input_paths:
        with np.load(path) as source:
            record = {name: source[name] for name in source.files}
        validate_epic_force_record(record, str(path))
        record["path"] = str(Path(path).resolve())
        records.append(record)
    reference_index = records[0]["db_index"]
    reference_truth = records[0]["force_true"]
    reference_offsets = records[0]["force_offsets"]
    for record in records[1:]:
        if not np.array_equal(record["db_index"], reference_index):
            raise ValueError("force convergence inputs use different structure subsets")
        if not np.array_equal(record["force_offsets"], reference_offsets):
            raise ValueError("force convergence inputs use different atom offsets")
        if not np.allclose(record["force_true"], reference_truth, rtol=0.0, atol=1e-7):
            raise ValueError("force convergence inputs use different force targets")

    by_key = {(int(record["n_samples"]), int(record["sample_seed"])): record for record in records}
    sample_counts = sorted({key[0] for key in by_key})
    seeds = sorted({key[1] for key in by_key})
    if len(sample_counts) < 2 or len(seeds) < 2:
        raise ValueError("force convergence analysis requires multiple sample counts and seeds")
    if set(by_key) != {(samples, seed) for samples in sample_counts for seed in seeds}:
        raise ValueError("force convergence grid is incomplete")

    run_rows = []
    raw_coverage_by_key = {}
    comparison_rows = []
    for samples in sample_counts:
        for seed in seeds:
            record = by_key[(samples, seed)]
            scale = np.asarray(record["force_scale_raw"], dtype=np.float64)
            standardized = np.abs(record["force_true"] - record["force_mean"]) / np.maximum(scale, 1e-12)
            raw_coverage = float(np.mean(standardized <= norm.ppf(0.95)))
            raw_coverage_by_key[(samples, seed)] = raw_coverage
            run_rows.append(
                {
                    "n_samples": samples,
                    "sample_seed": seed,
                    "n_structures": len(reference_index),
                    "n_force_components": scale.size,
                    "scale_median": float(np.median(scale)),
                    "scale_q95": float(np.quantile(scale, 0.95)),
                    "scale_q99": float(np.quantile(scale, 0.99)),
                    "scale_max": float(np.max(scale)),
                    "coverage_90_raw_gaussian": raw_coverage,
                    "inference_seconds": float(record.get("inference_seconds", 0.0)),
                    "path": record["path"],
                }
            )
            if samples != sample_counts[-1]:
                reference_scale = np.asarray(
                    by_key[(sample_counts[-1], seed)]["force_scale_raw"],
                    dtype=np.float64,
                )
                relative = np.abs(scale - reference_scale) / np.maximum(reference_scale, 1e-12)
                correlation = spearmanr(scale.reshape(-1), reference_scale.reshape(-1)).statistic
                comparison_rows.append(
                    {
                        "n_samples": samples,
                        "sample_seed": seed,
                        "reference_n_samples": sample_counts[-1],
                        "median_absolute_relative_scale_difference": float(np.median(relative)),
                        "q95_absolute_relative_scale_difference": float(np.quantile(relative, 0.95)),
                        "spearman_scale_vs_reference": float(correlation),
                    }
                )

    alpha_subset = None
    if hierarchical_predictions_path is not None:
        with np.load(hierarchical_predictions_path) as source:
            prediction_index = np.asarray(source["db_index"], dtype=np.int64)
            prediction_alpha = np.asarray(source["alpha"], dtype=np.float64)
        positions = np.searchsorted(prediction_index, reference_index)
        if (
            np.any(positions >= len(prediction_index))
            or not np.array_equal(prediction_index[positions], reference_index)
        ):
            raise ValueError("force convergence subset is not aligned with NIG predictions")
        alpha_subset = prediction_alpha[positions]

    seed_rows = []
    alpha_rows = []
    for samples in sample_counts:
        structure_scales = np.stack(
            [
                _structure_rms_scale(by_key[(samples, seed)]["force_scale_raw"], reference_offsets)
                for seed in seeds
            ]
        )
        mean = structure_scales.mean(axis=0)
        cv = structure_scales.std(axis=0, ddof=1) / np.maximum(mean, 1e-12)
        pairwise = []
        for left, right in combinations(range(len(seeds)), 2):
            pairwise.append(spearmanr(structure_scales[left], structure_scales[right]).statistic)
        seed_rows.append(
            {
                "n_samples": samples,
                "n_seeds": len(seeds),
                "median_structure_seed_cv": float(np.median(cv)),
                "q95_structure_seed_cv": float(np.quantile(cv, 0.95)),
                "mean_pairwise_structure_scale_spearman": float(np.mean(pairwise)),
                "coverage_90_raw_seed_range": float(
                    max(raw_coverage_by_key[(samples, seed)] for seed in seeds)
                    - min(raw_coverage_by_key[(samples, seed)] for seed in seeds)
                ),
            }
        )
        if alpha_subset is not None:
            alpha_groups = (
                ("alpha_le_1p5", alpha_subset <= 1.5),
                ("alpha_1p5_to_2", (alpha_subset > 1.5) & (alpha_subset <= 2.0)),
                ("alpha_gt_2", alpha_subset > 2.0),
            )
            for label, selected in alpha_groups:
                if not np.any(selected):
                    continue
                alpha_rows.append(
                    {
                        "n_samples": samples,
                        "alpha_group": label,
                        "n_structures": int(selected.sum()),
                        "median_structure_seed_cv": float(np.median(cv[selected])),
                        "q95_structure_seed_cv": float(np.quantile(cv[selected], 0.95)),
                    }
                )

    seed_frame = pd.DataFrame(seed_rows)
    comparison_frame = pd.DataFrame(comparison_rows)
    highest = seed_frame.loc[seed_frame.n_samples == sample_counts[-1]].iloc[0]
    converged = bool(
        highest.median_structure_seed_cv <= 0.10
        and highest.q95_structure_seed_cv <= 0.25
    )
    recommended = None
    for samples in sample_counts[:-1]:
        seed_row = seed_frame.loc[seed_frame.n_samples == samples].iloc[0]
        comparisons = comparison_frame.loc[comparison_frame.n_samples == samples]
        reference_ok = (
            not comparisons.empty
            and comparisons.median_absolute_relative_scale_difference.max() <= 0.10
            and comparisons.q95_absolute_relative_scale_difference.max() <= 0.50
        )
        if (
            seed_row.median_structure_seed_cv <= 0.10
            and seed_row.q95_structure_seed_cv <= 0.25
            and reference_ok
        ):
            recommended = int(samples)
            break

    aggregate_recommended = None
    for samples in sample_counts[:-1]:
        seed_row = seed_frame.loc[seed_frame.n_samples == samples].iloc[0]
        comparisons = comparison_frame.loc[comparison_frame.n_samples == samples]
        reference_ok = (
            not comparisons.empty
            and comparisons.median_absolute_relative_scale_difference.max() <= 0.10
            and comparisons.q95_absolute_relative_scale_difference.max() <= 0.50
        )
        if (
            seed_row.median_structure_seed_cv <= 0.10
            and seed_row.mean_pairwise_structure_scale_spearman >= 0.98
            and seed_row.coverage_90_raw_seed_range <= 0.03
            and reference_ok
        ):
            aggregate_recommended = int(samples)
            break

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows).to_csv(output_dir / "force_mc_runs.csv", index=False)
    comparison_frame.to_csv(output_dir / "force_mc_reference_convergence.csv", index=False)
    seed_frame.to_csv(output_dir / "force_mc_seed_stability.csv", index=False)
    if alpha_rows:
        alpha_frame = pd.DataFrame(alpha_rows)
        alpha_frame.to_csv(
            output_dir / "force_mc_seed_stability_by_alpha.csv", index=False
        )
    summary = {
        "schema_version": 1,
        "split": "calibration",
        "n_structures": int(len(reference_index)),
        "sample_counts": sample_counts,
        "sample_seeds": seeds,
        "convergence_thresholds": {
            "median_structure_seed_cv_max": 0.10,
            "q95_structure_seed_cv_max": 0.25,
            "median_relative_difference_vs_max_max": 0.10,
            "q95_relative_difference_vs_max_max": 0.50,
            "aggregate_mean_pairwise_structure_scale_spearman_min": 0.98,
            "aggregate_coverage_90_raw_seed_range_max": 0.03,
        },
        "highest_sample_count_seed_stable": converged,
        "highest_sample_count_is_reference_only": True,
        "recommendation_requires_higher_sample_reference": True,
        "recommended_sample_count": recommended,
        "aggregate_metric_recommended_sample_count": aggregate_recommended,
        "aggregate_recommendation_scope": (
            "fixed-seed aggregate calibration, coverage, and ranking metrics only; "
            "not convergence of every per-structure force scale"
        ),
        "test_split_used_for_selection": False,
        "heavy_tail_stratification_available": alpha_subset is not None,
    }
    if alpha_subset is not None:
        summary["convergence_subset_fraction_alpha_le_1p5"] = float(
            np.mean(alpha_subset <= 1.5)
        )
        summary["convergence_subset_fraction_alpha_le_2"] = float(
            np.mean(alpha_subset <= 2.0)
        )
    (output_dir / "force_mc_convergence.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_force_convergence_figures(
        seed_frame,
        comparison_frame,
        output_dir / "figures",
    )
    if alpha_rows:
        write_force_alpha_convergence_figures(
            alpha_frame,
            output_dir / "figures",
        )
    return output_dir
