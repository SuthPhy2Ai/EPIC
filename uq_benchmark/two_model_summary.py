from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _metric(
    frame: pd.DataFrame,
    method: str,
    target: str,
    metric: str,
    *,
    calibration: str,
    family: str,
) -> float:
    selected = frame[
        (frame["method"] == method)
        & (frame["target"] == target)
        & (frame["metric"] == metric)
        & (frame["calibration"] == calibration)
        & (frame["distribution_family"] == family)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one metric row for {method}/{target}/{metric}/{calibration}/{family}, "
            f"got {len(selected)}"
        )
    return float(selected.iloc[0]["estimate"])


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return output


def write_two_model_summary(
    report_dir: str | Path,
    analysis_dir: str | Path,
    force_convergence_path: str | Path,
    checkpoint_audit_path: str | Path,
    force_calibration_path: str | Path,
    output_dir: str | Path,
    cost_dir: str | Path | None = None,
) -> Path:
    report_dir = Path(report_dir)
    analysis_dir = Path(analysis_dir)
    metrics = pd.read_csv(report_dir / "tables/metrics.csv")
    paired_ci = pd.read_csv(
        report_dir / "tables/paired_cluster_confidence_intervals.csv"
    )
    size_metrics = pd.read_csv(analysis_dir / "size_stratified_metrics.csv")
    force_size_metrics = pd.read_csv(
        report_dir / "tables/force_uq_size_stratified.csv"
    )
    analysis = json.loads((analysis_dir / "two_model_analysis.json").read_text())
    convergence = json.loads(Path(force_convergence_path).read_text())
    audit = json.loads(Path(checkpoint_audit_path).read_text())
    force_calibration = json.loads(Path(force_calibration_path).read_text())
    training_cost = None
    inference_cost = None
    runtime_cost = pd.read_csv(report_dir / "tables/runtime_cost.csv")
    if cost_dir is not None:
        cost_dir = Path(cost_dir)
        training_cost = pd.read_csv(cost_dir / "training_cost.csv")
        inference_cost = pd.read_csv(cost_dir / "inference_cost.csv")

    point = {}
    for method in ("mace", "hierarchical_epic"):
        point[method] = {
            "energy_mae_ev_per_atom": _metric(
                metrics, method, "energy_per_atom", "MAE", calibration="none", family="point"
            ),
            "energy_rmse_ev_per_atom": _metric(
                metrics, method, "energy_per_atom", "RMSE", calibration="none", family="point"
            ),
            "force_component_mae_ev_per_angstrom": _metric(
                metrics, method, "force_component", "MAE", calibration="none", family="point"
            ),
            "force_component_rmse_ev_per_angstrom": _metric(
                metrics, method, "force_component", "RMSE", calibration="none", family="point"
            ),
            "force_vector_mae_ev_per_angstrom": _metric(
                metrics, method, "force_vector", "MAE", calibration="none", family="point"
            ),
            "force_vector_rmse_ev_per_angstrom": _metric(
                metrics, method, "force_vector", "RMSE", calibration="none", family="point"
            ),
        }

    energy_uq = {
        "nll": _metric(
            metrics,
            "hierarchical_epic",
            "energy_per_atom",
            "NLL",
            calibration="calibrated",
            family="student_t",
        ),
        "pit_ece_20": _metric(
            metrics,
            "hierarchical_epic",
            "energy_per_atom",
            "PIT_ECE_20",
            calibration="calibrated",
            family="student_t",
        ),
        "mean_absolute_coverage_error": _metric(
            metrics,
            "hierarchical_epic",
            "energy_per_atom",
            "mean_absolute_coverage_error",
            calibration="calibrated",
            family="student_t",
        ),
    }
    force_uq = {
        "nll": _metric(
            metrics,
            "hierarchical_epic",
            "force_component",
            "NLL",
            calibration="calibrated",
            family="gaussian",
        ),
        "pit_ece_20": _metric(
            metrics,
            "hierarchical_epic",
            "force_component",
            "PIT_ECE_20",
            calibration="calibrated",
            family="gaussian",
        ),
        "mean_absolute_coverage_error": _metric(
            metrics,
            "hierarchical_epic",
            "force_component",
            "mean_absolute_coverage_error",
            calibration="calibrated",
            family="gaussian",
        ),
    }
    for level in (50, 80, 90, 95):
        energy_uq[f"coverage_{level:02d}"] = _metric(
            metrics,
            "hierarchical_epic",
            "energy_per_atom",
            f"coverage_{level:02d}",
            calibration="calibrated",
            family="student_t",
        )
        force_uq[f"coverage_{level:02d}"] = _metric(
            metrics,
            "hierarchical_epic",
            "force_component",
            f"coverage_{level:02d}",
            calibration="calibrated",
            family="gaussian",
        )

    risk_ranking = {}
    for name, target in (
        ("energy_per_atom", "energy_per_atom"),
        ("force_component", "force_component_subset"),
        ("force_structure", "force_component_structure_MAE_subset"),
    ):
        risk_ranking[name] = {
            metric: _metric(
                metrics,
                "hierarchical_epic",
                target,
                metric,
                calibration="none",
                family="risk_score",
            )
            for metric in ("Spearman", "top5_AUROC", "top5_AUPRC", "AURC_0p01_1")
        }
    paired_row = paired_ci[
        paired_ci.metric == "delta_MAE__hierarchical_epic__minus__mace"
    ]
    if len(paired_row) != 1:
        raise ValueError("paired Hierarchical-minus-MACE energy MAE interval is missing")
    paired_energy_mae = {
        key: float(paired_row.iloc[0][key])
        for key in ("estimate", "ci_low_95", "ci_high_95")
    }

    summary = {
        "schema_version": 1,
        "benchmark_scope": ["mace", "hierarchical_epic"],
        "deferred_methods": ["ensemble5", "llpr", "graph_nig_epic"],
        "point_metrics": point,
        "energy_uq_calibrated": energy_uq,
        "force_uq_calibrated": force_uq,
        "risk_ranking": risk_ranking,
        "paired_energy_mae_hierarchical_minus_mace": paired_energy_mae,
        "force_calibration": force_calibration,
        "force_mc_convergence": convergence,
        "checkpoint_audit": {
            key: audit[key]
            for key in (
                "finite_difference_step_angstrom",
                "max_point_autograd_error",
                "max_point_finite_difference_error",
                "max_sampled_finite_difference_error",
                "max_sampled_closed_loop_absolute_work",
                "point_prediction_permutation_equivariance_verified",
                "atomic_weight_permutation_equivariance_verified",
                "same_sample_id_permutation_equivariance_verified",
                "global_random_pes_verified",
            )
        },
        "heavy_tail": {
            "fraction_alpha_le_1_5": analysis["fraction_alpha_le_1_5"],
            "fraction_alpha_le_2": analysis["fraction_alpha_le_2"],
        },
        "external_ood_evaluation_available": analysis[
            "external_ood_evaluation_available"
        ],
        "external_ood_note": analysis["external_ood_note"],
        "runtime_cost": {
            "training": None if training_cost is None else training_cost.to_dict("records"),
            "point_inference": None if inference_cost is None else inference_cost.to_dict("records"),
            "benchmark_stages": runtime_cost.to_dict("records"),
        },
    }

    point_rows = []
    point_labels = (
        ("Energy MAE (eV/atom)", "energy_mae_ev_per_atom"),
        ("Energy RMSE (eV/atom)", "energy_rmse_ev_per_atom"),
        ("Force component MAE (eV/Angstrom)", "force_component_mae_ev_per_angstrom"),
        ("Force component RMSE (eV/Angstrom)", "force_component_rmse_ev_per_angstrom"),
        ("Force vector MAE (eV/Angstrom)", "force_vector_mae_ev_per_angstrom"),
        ("Force vector RMSE (eV/Angstrom)", "force_vector_rmse_ev_per_angstrom"),
    )
    for label, key in point_labels:
        point_rows.append(
            [
                label,
                f"{point['mace'][key]:.6f}",
                f"{point['hierarchical_epic'][key]:.6f}",
            ]
        )
    coverage_rows = [
        [
            f"{level}%",
            f"{energy_uq[f'coverage_{level:02d}']:.6f}",
            f"{force_uq[f'coverage_{level:02d}']:.6f}",
        ]
        for level in (50, 80, 90, 95)
    ]
    energy_size_rows = []
    for _, row in size_metrics[
        size_metrics["method"] == "hierarchical_epic"
    ].iterrows():
        energy_size_rows.append(
            [
                str(row["size_bin"]),
                str(int(row["n_structures"])),
                f"{row['energy_mae_ev_per_atom']:.6f}",
                f"{row['coverage_95_calibrated']:.6f}",
            ]
        )
    force_size_rows = []
    for _, row in force_size_metrics[
        force_size_metrics["calibration"] == "calibrated"
    ].iterrows():
        force_size_rows.append(
            [
                str(row["size_bin"]),
                str(int(row["n_structures"])),
                f"{row['force_component_mae_ev_per_angstrom']:.6f}",
                f"{row['coverage_95']:.6f}",
            ]
        )

    cost_markdown = []
    if training_cost is not None and inference_cost is not None:
        cost_rows = []
        for method, label in (("mace", "MACE"), ("hierarchical_epic", "Hierarchical EPIC")):
            training_row = training_cost[training_cost.method == method].iloc[0]
            inference_row = inference_cost[inference_cost.method == method].iloc[0]
            cost_rows.append(
                [
                    label,
                    str(int(training_row.epochs_completed)),
                    f"{training_row.training_gpu_hours:.2f}",
                    f"{inference_row.milliseconds_per_structure:.3f}",
                ]
            )
        cost_markdown = [
            "## Recorded compute cost",
            "",
            *_markdown_table(
                ["Method", "Training epochs", "Training GPU-hours", "Point ms/structure"],
                cost_rows,
            ),
            "",
            "Training time sums recorded epoch timers. Point inference time sums shard-local "
            "device timers and is not allocation wall time.",
            "",
        ]

    markdown = [
        "# Frozen two-model benchmark results",
        "",
        "This report compares only deterministic MACE and Hierarchical atom-wise EPIC. "
        "The ensemble-of-five, LLPR, and graph-NIG EPIC benchmark is deferred.",
        "",
        "## Point prediction",
        "",
        *_markdown_table(
            ["Metric", "MACE", "Hierarchical EPIC"], point_rows
        ),
        "",
        "## Calibrated uncertainty",
        "",
        *_markdown_table(
            ["Nominal coverage", "Energy Student-t", "Force Gaussian MC"],
            coverage_rows,
        ),
        "",
        f"Energy NLL: `{energy_uq['nll']:.6f}`; energy PIT ECE-20: "
        f"`{energy_uq['pit_ece_20']:.6f}`. Force NLL: `{force_uq['nll']:.6f}`; "
        f"force PIT ECE-20: `{force_uq['pit_ece_20']:.6f}`.",
        "",
        f"Paired material-cluster bootstrap energy-MAE difference "
        f"(Hierarchical - MACE): `{paired_energy_mae['estimate']:.6f}` eV/atom "
        f"(95% CI `{paired_energy_mae['ci_low_95']:.6f}` to "
        f"`{paired_energy_mae['ci_high_95']:.6f}`). Energy-risk Spearman: "
        f"`{risk_ranking['energy_per_atom']['Spearman']:.6f}`; force-structure-risk "
        f"Spearman: `{risk_ranking['force_structure']['Spearman']:.6f}`.",
        "",
        "## Size stress tests",
        "",
        "Energy calibration by structure size:",
        "",
        *_markdown_table(
            ["Atoms", "Structures", "Energy MAE", "95% coverage"],
            energy_size_rows,
        ),
        "",
        "Force calibration by structure size:",
        "",
        *_markdown_table(
            ["Atoms", "Structures", "Force component MAE", "95% coverage"],
            force_size_rows,
        ),
        "",
        *cost_markdown,
        "## Interpretation limits",
        "",
        f"- The formal force benchmark uses "
        f"{convergence['aggregate_metric_recommended_sample_count']} samples selected only "
        "from the calibration split for aggregate calibration, coverage, and ranking metrics.",
        "- The strict per-structure force-scale convergence criterion is reported separately; "
        "the aggregate recommendation is not evidence that every heavy-tail scale converged.",
        "- Headline force calibration and coverage are Cartesian-component weighted; "
        "atom-count strata are reported separately.",
        f"- `alpha <= 2` for {analysis['fraction_alpha_le_2']:.3%} of test structures; "
        "finite-MC force scales remain heavy-tail-sensitive.",
        "- Point prediction and atomic attribution permutation checks pass within the "
        "recorded numerical tolerances.",
        "- Same-sample-ID permutation equivariance and a single global random PES are "
        "not verified.",
        "- The frozen test split is in-distribution and material-group held out; no "
        "external OOD dataset is claimed.",
        "- Point accuracy is mixed: report each energy/force MAE and RMSE rather than "
        "claiming universal superiority.",
        "",
    ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "two_model_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "two_model_benchmark_summary.md").write_text(
        "\n".join(markdown)
    )
    return output_dir
