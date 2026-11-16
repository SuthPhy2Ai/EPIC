from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


NONENSEMBLE_METHODS = (
    "hierarchical_epic",
    "graph_nig",
    "llpr",
    "llpr_force_aware",
)
COMPARISON_METHODS = NONENSEMBLE_METHODS + ("ensemble5",)


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _normalized_table(path: Path, analysis: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"metric", "estimate"}
    if not required <= set(frame):
        raise ValueError(f"{path} lacks {sorted(required - set(frame))}")
    for column in ("estimate", "ci_low_95", "ci_high_95"):
        if column in frame and np.any(~np.isfinite(frame[column].to_numpy(dtype=float))):
            raise ValueError(f"{path} contains non-finite {column}")
    frame.insert(0, "analysis", analysis)
    return frame


def _format_interval(row: pd.Series) -> str:
    estimate = float(row["estimate"])
    if "ci_low_95" in row and pd.notna(row["ci_low_95"]):
        return (
            f"{estimate:.6g} [{float(row['ci_low_95']):.6g}, "
            f"{float(row['ci_high_95']):.6g}]"
        )
    return f"{estimate:.6g}"


def _markdown_rows(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    columns = [column for column in columns if column in frame]
    lines = [
        "| " + " | ".join(columns + ["estimate [95% CI]"]) + " |",
        "|" + "---|" * (len(columns) + 1),
    ]
    for _, row in frame.iterrows():
        labels = [str(row[column]) for column in columns]
        lines.append("| " + " | ".join(labels + [_format_interval(row)]) + " |")
    return lines


def _validated_compute_cost_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "method",
        "target",
        "split",
        "n_structures",
        "mc_samples",
        "summed_shard_device_seconds",
        "milliseconds_per_structure",
        "peak_memory_bytes_per_device",
        "timing_semantics",
    }
    if missing := required - set(frame):
        raise ValueError(f"{path} lacks {sorted(missing)}")
    expected = {
        (method, target, split)
        for method in COMPARISON_METHODS
        for target in ("point_inference", "force_uq_inference")
        for split in ("calibration", "test")
    }
    observed = set(zip(frame.method, frame.target, frame.split, strict=True))
    if observed != expected or len(frame) != len(expected):
        raise ValueError("non-ensemble compute-cost rows are incomplete or duplicated")
    numeric = (
        "n_structures",
        "mc_samples",
        "summed_shard_device_seconds",
        "milliseconds_per_structure",
        "peak_memory_bytes_per_device",
    )
    for column in numeric:
        values = frame[column].to_numpy(dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"non-ensemble compute cost contains invalid {column}")
    if np.any(frame.n_structures <= 0) or np.any(frame.milliseconds_per_structure <= 0):
        raise ValueError("non-ensemble compute cost must use positive workloads and timings")
    ensemble = frame[frame.method == "ensemble5"]
    nonensemble = frame[frame.method != "ensemble5"]
    point = nonensemble[nonensemble.target == "point_inference"]
    force = nonensemble[nonensemble.target == "force_uq_inference"]
    if (
        not np.all(point.mc_samples == 1)
        or not np.all(force.mc_samples == 256)
        or not np.all(ensemble.mc_samples == 5)
    ):
        raise ValueError("method cost table does not use the frozen sample/member protocol")
    allowed_timing = {
        "sum_of_shard_local_device_wall_times",
        "sum_of_member_shard_local_device_wall_times",
    }
    if not set(frame.timing_semantics) <= allowed_timing:
        raise ValueError("method cost timing semantics differ from the frozen protocol")
    return frame


def _markdown_cost_rows(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| method | workload | MC samples | ms/structure | peak MiB/device |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['method']} | {row['target']} | {int(row['mc_samples'])} | "
            f"{float(row['milliseconds_per_structure']):.6g} | "
            f"{float(row['peak_memory_bytes_per_device']) / 2**20:.6g} |"
        )
    return lines


def finalize_supplement_release(
    supplement_root: str | Path,
    source_root: str | Path,
    output_dir: str | Path,
) -> Path:
    supplement_root = Path(supplement_root)
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite supplement release: {output_dir}")

    external_root = supplement_root / "external_ood/elementa_open_preliminary_v1"
    method_root = supplement_root / "method_comparison/formal_comparison_with_ensemble5"
    summary_paths = {
        "force_allocation": supplement_root
        / "force_allocation_comparison/force_allocation_comparison_summary.json",
        "external_ood": external_root / "analysis/external_ood_summary.json",
        "methods": method_root / "method_comparison_summary.json",
        "ensemble5_training": supplement_root
        / "ensemble5/audit/ensemble5_training_manifest.json",
        "atomic_attribution": supplement_root
        / "atomic_attribution_v2/atomic_attribution_summary.json",
        "external_selection": external_root / "selection_manifest.json",
        "external_leakage": external_root / "leakage_audit_invariant.json",
        "external_element_support": external_root / "element_support_audit.json",
        "reference_training_continuity": supplement_root
        / "training_reference_audit/reference_training_continuity_audit.json",
    }
    summaries = {name: _load_json(path) for name, path in summary_paths.items()}
    force = summaries["force_allocation"]
    external = summaries["external_ood"]
    methods = summaries["methods"]
    ensemble_training = summaries["ensemble5_training"]
    atomic = summaries["atomic_attribution"]
    selection = summaries["external_selection"]
    leakage = summaries["external_leakage"]
    element_support = summaries["external_element_support"]
    training_continuity = summaries["reference_training_continuity"]

    if force.get("bootstrap_replicates") != 1000 or force.get(
        "test_used_for_selection"
    ) is not False:
        raise ValueError("learned-vs-uniform allocation protocol is incomplete")
    if tuple(methods.get("methods", ())) != COMPARISON_METHODS:
        raise ValueError("method set differs from the frozen five-method protocol")
    if methods.get("bootstrap_replicates") != 1000:
        raise ValueError("five-method bootstrap protocol is incomplete")
    if (
        ensemble_training.get("seeds") != [42, 43, 44, 45, 46]
        or ensemble_training.get("completed_members") != 5
        or ensemble_training.get("seed42_reused") is not True
        or ensemble_training.get("matched_seed42_protocol_verified") is not True
        or ensemble_training.get("all_checkpoints_loadable_and_finite") is not True
    ):
        raise ValueError("ensemble5 training audit is incomplete")
    posterior_audits = methods.get("posterior_audits", {})
    if set(posterior_audits) != {"llpr", "llpr_force_aware"} or not all(
        audit.get("finite_symmetric_positive_semidefinite_verified") is True
        for audit in posterior_audits.values()
    ):
        raise ValueError("LLPR posterior audits are missing or invalid")
    if external.get("result_status") != "preliminary_not_for_formal_ood_claim":
        raise ValueError("external result status is not conservatively gated")
    if external.get("formal_external_ood_claim_permitted") is not False:
        raise ValueError("formal external OOD claim gate must remain closed")
    if (
        external.get("selection_uses_model_predictions") is not False
        or external.get("in_domain_calibration_reused_without_external_refit") is not True
        or external.get("external_test_used_for_selection") is not False
        or external.get("bootstrap_replicates") != 1000
    ):
        raise ValueError("external evaluation protocol is incomplete")
    selected_materials = selection.get("selected_materials", [])
    if (
        selection.get("dataset") != "ELEMENTA_open"
        or selection.get("subset_size") != 10_000
        or selection.get("selection_uses_model_predictions") is not False
        or selection.get("formal_external_ood_claim_permitted") is not False
        or len(selected_materials) != 10_000
        or len(set(selected_materials)) != 10_000
    ):
        raise ValueError("external deterministic selection protocol is incomplete")
    provenance = selection.get("provenance", {})
    if provenance.get("verified") is not False:
        raise ValueError("external provenance gate must remain closed")
    if leakage.get("strict_geometry_overlap_count") != 0 or leakage.get(
        "translation_cell_invariant_geometry_overlap_count"
    ) != 0:
        raise ValueError("external subset has geometry leakage")
    required_external_gates = {
        "provenance_license_and_dft_protocol_verification",
        "cross_dataset_material_identity_mapping",
        "primitive_supercell_and_cross_dataset_material_identity_mapping",
    }
    if (
        leakage.get("selection_uses_model_predictions") is not False
        or leakage.get("formal_external_ood_claim_permitted") is not False
        or leakage.get("cross_dataset_material_identity_mapping_available") is not False
        or leakage.get("invariant_geometry_fingerprint", {}).get(
            "primitive_supercell_invariant"
        )
        is not False
        or not required_external_gates <= set(leakage.get("remaining_gates", []))
    ):
        raise ValueError("external leakage and identity gates are incomplete")
    if element_support.get("all_external_elements_seen_in_reference_train") is not True:
        raise ValueError("external subset contains elements absent from the reference train split")
    if element_support.get("external_structures_with_unsupported_elements") != 0:
        raise ValueError("external subset contains unsupported-element structures")
    if element_support.get("formal_external_ood_claim_permitted") is not False:
        raise ValueError("element-support audit must not open the formal OOD claim gate")
    if int(element_support.get("external_unique_compositions", 0)) <= 0:
        raise ValueError("external composition-diversity audit is missing")
    if atomic.get("bootstrap_replicates") != 1000:
        raise ValueError("atomic attribution bootstrap protocol is incomplete")
    if training_continuity.get(
        "reference_training_epoch_and_state_continuity_verified"
    ) is not True:
        raise ValueError("seed42 reference-training continuity audit is incomplete")
    reproducibility = training_continuity.get("reproducibility_claims", {})
    if (
        reproducibility.get("optimizer_scheduler_state_restoration_supported") is not True
        or reproducibility.get("rng_state_continuity_verified") is not False
        or reproducibility.get("sampler_data_order_continuity_verified") is not False
        or reproducibility.get("bitwise_lossless_resume_verified") is not False
    ):
        raise ValueError("seed42 reference-training resume claims are not conservatively gated")
    table_paths = {
        "force_allocation": supplement_root
        / "force_allocation_comparison/tables/force_allocation_confidence_intervals.csv",
        "external_ood": external_root / "analysis/tables/external_ood_metrics.csv",
        "methods": method_root / "tables/method_metrics.csv",
        "method_differences": method_root
        / "tables/hierarchical_paired_advantages.csv",
        "atomic_attribution": supplement_root
        / "atomic_attribution_v2/tables/atomic_attribution_confidence_intervals.csv",
    }
    compute_cost_path = method_root / "tables/method_compute_cost.csv"
    frames = {
        name: _normalized_table(path, name) for name, path in table_paths.items()
    }
    compute_cost = _validated_compute_cost_table(compute_cost_path)
    combined = pd.concat(frames.values(), ignore_index=True, sort=False)
    identity = [
        column
        for column in ("analysis", "target", "method", "run", "comparison", "metric")
        if column in combined
    ]
    remaining = [column for column in combined if column not in identity]
    combined = combined[identity + remaining]

    source_files = [
        source_root / "pyproject.toml",
        source_root / "uq_benchmark/cli.py",
        source_root / "uq_benchmark/supplement.py",
        source_root / "uq_benchmark/training_continuity.py",
        source_root / "uq_benchmark/external_ood.py",
        source_root / "uq_benchmark/external_ood_analysis.py",
        source_root / "uq_benchmark/ensemble_pipeline.py",
        source_root / "uq_benchmark/llpr_pipeline.py",
        source_root / "uq_benchmark/method_comparison.py",
        source_root / "uq_benchmark/supplement_release.py",
    ]
    input_files = (
        list(summary_paths.values())
        + list(table_paths.values())
        + [compute_cost_path]
        + source_files
    )
    for path in input_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True)
    combined_path = output_dir / "supplement_metrics_long.csv"
    combined.to_csv(combined_path, index=False)
    compute_cost_output = output_dir / "method_compute_cost.csv"
    compute_cost.to_csv(compute_cost_output, index=False)
    manifest = {
        "schema_version": 1,
        "release_scope": "revised_scope_default_seed42_epic_with_ensemble5",
        "completed_methods": list(COMPARISON_METHODS),
        "deterministic_ensemble5_status": "completed",
        "ensemble5_seeds": [42, 43, 44, 45, 46],
        "cancelled_training_ablations": [
            "hierarchical_seed43",
            "hierarchical_seed44",
            "allocation_weight_0_seed42",
        ],
        "external_ood_result_status": external["result_status"],
        "formal_external_ood_claim_permitted": False,
        "external_subset_size": 10_000,
        "external_selection_uses_model_predictions": False,
        "external_in_domain_calibration_reused_without_external_refit": True,
        "external_remaining_gates": leakage.get("remaining_gates", []),
        "external_element_support_verified": True,
        "external_structures_with_unsupported_elements": 0,
        "minimum_reference_train_structure_count_for_external_elements": element_support[
            "minimum_reference_train_structure_count_for_external_elements"
        ],
        "external_unique_compositions": element_support["external_unique_compositions"],
        "external_maximum_composition_multiplicity": element_support[
            "external_maximum_composition_multiplicity"
        ],
        "bootstrap_replicates": 1000,
        "method_compute_cost_table_included": True,
        "llpr_posterior_psd_verified": True,
        "reference_training_epoch_and_state_continuity_verified": True,
        "reference_training_rng_state_continuity_verified": False,
        "reference_training_bitwise_lossless_resume_verified": False,
        "ensemble5_matched_seed42_protocol_verified": True,
        "compute_cost_timing_semantics": sorted(set(compute_cost.timing_semantics)),
        "input_sha256": {
            **{
                f"supplement/{path.relative_to(supplement_root)}": _sha256_file(path)
                for path in list(summary_paths.values()) + list(table_paths.values())
            },
            f"supplement/{compute_cost_path.relative_to(supplement_root)}": _sha256_file(
                compute_cost_path
            ),
            **{
                f"source/{path.relative_to(source_root)}": _sha256_file(path)
                for path in source_files
            },
        },
    }
    manifest_path = output_dir / "supplement_release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    allocation_frame = frames["force_allocation"]
    allocation_selected = allocation_frame[
        allocation_frame.metric.isin(
            [
                "learned_force_structure_spearman",
                "uniform_force_structure_spearman",
                "uniform_minus_learned_force_structure_spearman",
                "learned_gaussian_affine_nll",
                "uniform_gaussian_affine_nll",
            ]
        )
    ]
    atomic_frame = frames["atomic_attribution"]
    atomic_selected = atomic_frame[
        atomic_frame.metric.isin(
            [
                "aggregate_kl_skill_vs_uniform",
                "learned_allocation_ndcg",
                "force_magnitude_ndcg",
                "atomic_energy_magnitude_ndcg",
                "force_magnitude_minus_learned_allocation_ndcg",
            ]
        )
    ]
    external_frame = frames["external_ood"]
    external_selected = external_frame[
        external_frame.metric.isin(
            [
                "deterministic_mace_force_component_mae",
                "hierarchical_epic_force_component_mae",
                "hierarchical_force_risk_spearman",
                "ood_detection_energy_auroc",
                "ood_detection_force_auroc",
            ]
        )
    ]
    method_frame = frames["methods"]
    method_selected = method_frame[
        (method_frame.target == "force_component")
        & method_frame.metric.isin(["point_mae", "risk_spearman", "gaussian_affine_nll"])
    ]
    report = [
        "# Supplementary benchmark release",
        "",
        "This release follows the revised scope: the formal seed42 Hierarchical EPIC model ",
        "is retained, the seed/allocation training ablations are cancelled, and a matched ",
        "five-member deterministic ensemble is included.",
        "",
        "All confidence intervals below are 1,000-replicate material-cluster bootstrap ",
        "intervals. Calibration parameters are fit on calibration data only.",
        "",
        "## Learned versus uniform force allocation",
        "",
        *_markdown_rows(allocation_selected, ["metric"]),
        "",
        "## Formal Hierarchical EPIC training scope",
        "",
        "The seed42 reference checkpoint has a contiguous logged epoch and regularization "
        "schedule, and its epoch-4 handoff preserves model, optimizer, and scheduler state. "
        "RNG state and sampler order were not archived, so bitwise-lossless continuation is "
        "not claimed.",
        "No additional Hierarchical EPIC seeds or allocation-loss-zero model enter this release.",
        "",
        "## Atomic attribution",
        "",
        *_markdown_rows(atomic_selected, ["metric"]),
        "",
        "Atomic allocation is a relative attribution only. It is not a calibrated atomic ",
        "force variance and does not define independent atomic posteriors.",
        "",
        "## Aligned methods including ensemble5",
        "",
        *_markdown_rows(method_selected, ["method", "target", "metric"]),
        "",
        "## Aligned method inference cost",
        "",
        *_markdown_cost_rows(compute_cost[compute_cost.split == "test"]),
        "",
        "Times are sums of shard-local device wall times, not allocation wall time. "
        "LLPR posterior-fitting time is omitted because the archived posterior does not "
        "contain auditable fit-time metadata.",
        "",
        "## External-domain evaluation",
        "",
        *_markdown_rows(external_selected, ["metric"]),
        "",
        "The external-domain result remains preliminary. Provenance, license, DFT protocol, ",
        "primitive/supercell identity, and cross-dataset material mapping gates are not all ",
        "closed, so no formal external-OOD or cross-dataset absolute-energy claim is made.",
        "All external elements occur in the MPTrj training split; no selected external "
        "structure contains an unseen element.",
        f"The selected subset contains {element_support['external_unique_compositions']:,} "
        f"unique stoichiometric compositions; the maximum composition multiplicity is "
        f"{element_support['external_maximum_composition_multiplicity']}.",
        "",
        "## Scientific limits",
        "",
        "- The graph NIG is the only distribution directly supervised by total-energy labels.",
        "- Force samples are energy-gradient-derived surrogates, not a global random PES.",
        "- Force intervals over-cover before conformal correction; report the full coverage vector.",
        "- The Student-t force fit reaches its lower degrees-of-freedom bound.",
        "- No universal superiority over deterministic MACE is claimed.",
        "- Ensemble5 uses independently trained deterministic members with seeds 42-46.",
        "",
    ]
    report_path = output_dir / "SUPPLEMENT_BENCHMARK.md"
    report_path.write_text("\n".join(report))
    checksum_path = output_dir / "SHA256SUMS"
    checksum_path.write_text(
        "".join(
            f"{_sha256_file(path)}  {path.name}\n"
            for path in (
                report_path,
                manifest_path,
                combined_path,
                compute_cost_output,
            )
        )
    )
    return output_dir
