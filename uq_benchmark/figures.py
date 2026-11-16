from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "mace": "#C65D3B",
    "hierarchical_epic": "#0B6E69",
    "raw": "#C65D3B",
    "calibrated": "#0B6E69",
    "reference": "#263238",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "axes.facecolor": "#FBF7EE",
            "figure.facecolor": "#F7F1E3",
            "axes.edgecolor": "#263238",
            "axes.labelcolor": "#263238",
            "xtick.color": "#263238",
            "ytick.color": "#263238",
            "font.size": 10,
            "axes.titleweight": "bold",
        }
    )


def _save(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def write_report_figures(
    metrics: pd.DataFrame,
    risk_curves: pd.DataFrame | None,
    force_size: pd.DataFrame | None,
    output_dir: str | Path,
) -> Path:
    _style()
    output_dir = Path(output_dir)
    methods = ("mace", "hierarchical_epic")
    labels = ("MACE", "Hierarchical EPIC")

    if set(methods).issubset(set(metrics.method)):
        figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
        for axis, target, title, ylabel in (
            (axes[0], "energy_per_atom", "Energy point error", "eV/atom"),
            (axes[1], "force_component", "Force-component point error", "eV/Angstrom"),
        ):
            positions = np.arange(2)
            width = 0.34
            for method_index, (method, label) in enumerate(zip(methods, labels, strict=True)):
                values = []
                for metric in ("MAE", "RMSE"):
                    row = metrics[
                        (metrics.method == method)
                        & (metrics.target == target)
                        & (metrics.metric == metric)
                        & (metrics.distribution_family == "point")
                    ]
                    values.append(float(row.iloc[0].estimate))
                axis.bar(
                    positions + (method_index - 0.5) * width,
                    values,
                    width,
                    label=label,
                    color=COLORS[method],
                )
            axis.set_xticks(positions, ("MAE", "RMSE"))
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.22)
        axes[0].legend(frameon=False)
        figure.tight_layout()
        _save(figure, output_dir, "point_prediction_comparison")

    levels = np.asarray([0.50, 0.80, 0.90, 0.95])
    hierarchical_metrics = metrics[metrics.method == "hierarchical_epic"]
    has_energy_uq = np.any(
        (hierarchical_metrics.target == "energy_per_atom")
        & (hierarchical_metrics.distribution_family == "student_t")
        & (hierarchical_metrics.calibration == "calibrated")
    )
    has_force_uq = np.any(
        (hierarchical_metrics.target == "force_component")
        & (hierarchical_metrics.distribution_family == "gaussian")
        & (hierarchical_metrics.calibration == "calibrated")
    )
    if has_energy_uq and has_force_uq:
        figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True, sharey=True)
        for axis, target, family, title in (
            (axes[0], "energy_per_atom", "student_t", "Energy intervals"),
            (axes[1], "force_component", "gaussian", "Force-component intervals"),
        ):
            for calibration in ("raw", "calibrated"):
                observed = []
                for level in levels:
                    metric = f"coverage_{int(level * 100):02d}"
                    row = metrics[
                        (metrics.method == "hierarchical_epic")
                        & (metrics.target == target)
                        & (metrics.metric == metric)
                        & (metrics.calibration == calibration)
                        & (metrics.distribution_family == family)
                    ]
                    observed.append(float(row.iloc[0].estimate))
                axis.plot(
                    levels,
                    observed,
                    marker="o",
                    color=COLORS[calibration],
                    label=calibration.capitalize(),
                )
            axis.plot(levels, levels, "--", color=COLORS["reference"], label="Ideal")
            axis.set_title(title)
            axis.set_xlabel("Nominal coverage")
            axis.grid(alpha=0.22)
        axes[0].set_ylabel("Empirical coverage")
        axes[0].legend(frameon=False)
        figure.tight_layout()
        _save(figure, output_dir, "calibration_coverage")

    if risk_curves is not None and not risk_curves.empty:
        targets = (
            ("energy_per_atom", "Energy risk-coverage", "eV/atom"),
            (
                "force_component_structure_MAE_subset",
                "Force risk-coverage",
                "eV/Angstrom",
            ),
        )
        figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
        for axis, (target, title, ylabel) in zip(axes, targets, strict=True):
            selected = risk_curves[risk_curves.target == target]
            axis.plot(
                selected.coverage,
                selected.mean_absolute_error,
                color=COLORS["hierarchical_epic"],
                linewidth=2.0,
            )
            axis.set_title(title)
            axis.set_xlabel("Retained fraction")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.22)
        figure.tight_layout()
        _save(figure, output_dir, "risk_coverage")

    if (
        force_size is not None
        and not force_size.empty
        and "hierarchical_epic" in set(force_size.method)
    ):
        force_size = force_size[force_size.method == "hierarchical_epic"]
        figure, axis = plt.subplots(figsize=(6.2, 3.8))
        order = list(dict.fromkeys(force_size.size_bin.tolist()))
        for calibration in ("raw", "calibrated"):
            selected = force_size[force_size.calibration == calibration].set_index(
                "size_bin"
            )
            axis.plot(
                order,
                selected.loc[order, "coverage_95"],
                marker="o",
                color=COLORS[calibration],
                label=calibration.capitalize(),
            )
        axis.axhline(0.95, linestyle="--", color=COLORS["reference"], label="Nominal")
        axis.set_xlabel("Atoms per structure")
        axis.set_ylabel("95% force-component coverage")
        axis.set_title("Force calibration size stress test")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
        figure.tight_layout()
        _save(figure, output_dir, "force_coverage_by_size")
    return output_dir


def write_size_analysis_figures(
    size_metrics: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    _style()
    output_dir = Path(output_dir)
    order = list(dict.fromkeys(size_metrics.size_bin.tolist()))
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for method, label in (("mace", "MACE"), ("hierarchical_epic", "Hierarchical EPIC")):
        selected = size_metrics[size_metrics.method == method].set_index("size_bin")
        axes[0].plot(
            order,
            selected.loc[order, "energy_mae_ev_per_atom"],
            marker="o",
            color=COLORS[method],
            label=label,
        )
    hierarchical = size_metrics[size_metrics.method == "hierarchical_epic"].set_index(
        "size_bin"
    )
    axes[1].plot(
        order,
        hierarchical.loc[order, "coverage_95_raw"],
        marker="o",
        color=COLORS["raw"],
        label="Raw",
    )
    axes[1].plot(
        order,
        hierarchical.loc[order, "coverage_95_calibrated"],
        marker="o",
        color=COLORS["calibrated"],
        label="Calibrated",
    )
    axes[1].axhline(0.95, linestyle="--", color=COLORS["reference"], label="Nominal")
    axes[0].set_title("Energy MAE by size")
    axes[0].set_ylabel("eV/atom")
    axes[1].set_title("Energy 95% coverage by size")
    axes[1].set_ylabel("Empirical coverage")
    for axis in axes:
        axis.set_xlabel("Atoms per structure")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    figure.tight_layout()
    _save(figure, output_dir, "energy_size_stress_test")
    return output_dir


def write_force_convergence_figures(
    seed_stability: pd.DataFrame,
    reference_convergence: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    _style()
    output_dir = Path(output_dir)
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].plot(
        seed_stability.n_samples,
        seed_stability.median_structure_seed_cv,
        marker="o",
        label="Median",
        color=COLORS["calibrated"],
    )
    axes[0].plot(
        seed_stability.n_samples,
        seed_stability.q95_structure_seed_cv,
        marker="o",
        label="95th percentile",
        color=COLORS["raw"],
    )
    axes[0].axhline(0.10, linestyle="--", color=COLORS["calibrated"], alpha=0.7)
    axes[0].axhline(0.25, linestyle="--", color=COLORS["raw"], alpha=0.7)
    grouped = reference_convergence.groupby("n_samples", as_index=False).agg(
        median_max=("median_absolute_relative_scale_difference", "max"),
        q95_max=("q95_absolute_relative_scale_difference", "max"),
    )
    axes[1].plot(
        grouped.n_samples,
        grouped.median_max,
        marker="o",
        label="Median max across seeds",
        color=COLORS["calibrated"],
    )
    axes[1].plot(
        grouped.n_samples,
        grouped.q95_max,
        marker="o",
        label="95th percentile max",
        color=COLORS["raw"],
    )
    axes[1].axhline(0.10, linestyle="--", color=COLORS["calibrated"], alpha=0.7)
    axes[1].axhline(0.50, linestyle="--", color=COLORS["raw"], alpha=0.7)
    axes[0].set_title("Across-seed force-scale stability")
    axes[1].set_title("Difference from maximum-sample reference")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(seed_stability.n_samples)
    axes[0].set_xticklabels([str(int(value)) for value in seed_stability.n_samples])
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(grouped.n_samples)
    axes[1].set_xticklabels([str(int(value)) for value in grouped.n_samples])
    for axis in axes:
        axis.set_xlabel("Monte Carlo samples")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Coefficient of variation")
    axes[1].set_ylabel("Absolute relative difference")
    figure.tight_layout()
    _save(figure, output_dir, "force_mc_convergence")
    return output_dir


def write_force_alpha_convergence_figures(
    alpha_stability: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    _style()
    output_dir = Path(output_dir)
    labels = {
        "alpha_le_1p5": "alpha <= 1.5",
        "alpha_1p5_to_2": "1.5 < alpha <= 2",
        "alpha_gt_2": "alpha > 2",
    }
    colors = {
        "alpha_le_1p5": "#C65D3B",
        "alpha_1p5_to_2": "#D69B2D",
        "alpha_gt_2": "#0B6E69",
    }
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    for group in labels:
        selected = alpha_stability[alpha_stability.alpha_group == group]
        axes[0].plot(
            selected.n_samples,
            selected.median_structure_seed_cv,
            marker="o",
            color=colors[group],
            label=labels[group],
        )
        axes[1].plot(
            selected.n_samples,
            selected.q95_structure_seed_cv,
            marker="o",
            color=colors[group],
            label=labels[group],
        )
    axes[0].axhline(0.10, linestyle="--", color=COLORS["reference"])
    axes[1].axhline(0.25, linestyle="--", color=COLORS["reference"])
    axes[0].set_title("Median seed CV by NIG tail")
    axes[1].set_title("95th-percentile seed CV by NIG tail")
    ticks = sorted(alpha_stability.n_samples.unique())
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(ticks)
        axis.set_xticklabels([str(int(value)) for value in ticks])
        axis.set_xlabel("Monte Carlo samples")
        axis.set_ylabel("Coefficient of variation")
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    _save(figure, output_dir, "force_mc_convergence_by_alpha")
    return output_dir
