from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from uq_benchmark.supplement_release import (
    COMPARISON_METHODS,
    finalize_supplement_release,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_table(path: Path, **columns) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = {
        "metric": ["point_mae"],
        "estimate": [0.1],
        "ci_low_95": [0.08],
        "ci_high_95": [0.12],
    }
    base.update(columns)
    pd.DataFrame(base).to_csv(path, index=False)


def test_finalize_supplement_release_validates_scope_and_writes_hashes(tmp_path):
    root = tmp_path / "supplement"
    external = root / "external_ood/elementa_open_preliminary_v1"
    methods = root / "method_comparison/formal_comparison_with_ensemble5"
    _write_json(
        root / "force_allocation_comparison/force_allocation_comparison_summary.json",
        {"bootstrap_replicates": 1000, "test_used_for_selection": False},
    )
    _write_json(
        external / "analysis/external_ood_summary.json",
        {
            "result_status": "preliminary_not_for_formal_ood_claim",
            "formal_external_ood_claim_permitted": False,
            "selection_uses_model_predictions": False,
            "in_domain_calibration_reused_without_external_refit": True,
            "external_test_used_for_selection": False,
            "bootstrap_replicates": 1000,
        },
    )
    _write_json(
        methods / "method_comparison_summary.json",
        {
            "methods": list(COMPARISON_METHODS),
            "bootstrap_replicates": 1000,
            "posterior_audits": {
                "llpr": {"finite_symmetric_positive_semidefinite_verified": True},
                "llpr_force_aware": {
                    "finite_symmetric_positive_semidefinite_verified": True
                },
            },
        },
    )
    _write_json(
        root / "ensemble5/audit/ensemble5_training_manifest.json",
        {
            "seeds": [42, 43, 44, 45, 46],
            "completed_members": 5,
            "seed42_reused": True,
            "matched_seed42_protocol_verified": True,
            "all_checkpoints_loadable_and_finite": True,
        },
    )
    _write_json(
        root / "atomic_attribution_v2/atomic_attribution_summary.json",
        {"bootstrap_replicates": 1000},
    )
    _write_json(
        external / "selection_manifest.json",
        {
            "dataset": "ELEMENTA_open",
            "subset_size": 10_000,
            "selection_uses_model_predictions": False,
            "formal_external_ood_claim_permitted": False,
            "selected_materials": [f"ext-{i}" for i in range(10_000)],
            "provenance": {"verified": False},
        },
    )
    _write_json(
        external / "leakage_audit_invariant.json",
        {
            "strict_geometry_overlap_count": 0,
            "translation_cell_invariant_geometry_overlap_count": 0,
            "selection_uses_model_predictions": False,
            "formal_external_ood_claim_permitted": False,
            "cross_dataset_material_identity_mapping_available": False,
            "invariant_geometry_fingerprint": {"primitive_supercell_invariant": False},
            "remaining_gates": [
                "provenance_license_and_dft_protocol_verification",
                "cross_dataset_material_identity_mapping",
                "primitive_supercell_and_cross_dataset_material_identity_mapping",
            ],
        },
    )
    _write_json(
        external / "element_support_audit.json",
        {
            "all_external_elements_seen_in_reference_train": True,
            "external_structures_with_unsupported_elements": 0,
            "minimum_reference_train_structure_count_for_external_elements": 10,
            "external_unique_compositions": 9,
            "external_maximum_composition_multiplicity": 2,
            "formal_external_ood_claim_permitted": False,
        },
    )
    _write_json(
        root / "training_reference_audit/reference_training_continuity_audit.json",
        {
            "reference_training_epoch_and_state_continuity_verified": True,
            "reproducibility_claims": {
                "optimizer_scheduler_state_restoration_supported": True,
                "rng_state_continuity_verified": False,
                "sampler_data_order_continuity_verified": False,
                "bitwise_lossless_resume_verified": False,
            },
        },
    )

    _write_table(
        root
        / "force_allocation_comparison/tables/force_allocation_confidence_intervals.csv"
    )
    _write_table(external / "analysis/tables/external_ood_metrics.csv")
    _write_table(
        methods / "tables/method_metrics.csv",
        target=["force_component"],
        method=["hierarchical_epic"],
    )
    _write_table(
        methods / "tables/hierarchical_paired_advantages.csv",
        target=["force_component"],
        comparison=["hierarchical_advantage_over_graph_nig"],
    )
    cost_rows = []
    for method in COMPARISON_METHODS:
        for target in ("point_inference", "force_uq_inference"):
            for split in ("calibration", "test"):
                mc_samples = (
                    5
                    if method == "ensemble5"
                    else (256 if target == "force_uq_inference" else 1)
                )
                timing_sem = (
                    "sum_of_member_shard_local_device_wall_times"
                    if method == "ensemble5"
                    else "sum_of_shard_local_device_wall_times"
                )
                cost_rows.append(
                    {
                        "method": method,
                        "target": target,
                        "split": split,
                        "n_structures": 10,
                        "mc_samples": mc_samples,
                        "summed_shard_device_seconds": 2.0,
                        "milliseconds_per_structure": 200.0,
                        "peak_memory_bytes_per_device": 1024,
                        "timing_semantics": timing_sem,
                    }
                )
    pd.DataFrame(cost_rows).to_csv(
        methods / "tables/method_compute_cost.csv", index=False
    )
    _write_table(
        root / "atomic_attribution_v2/tables/atomic_attribution_confidence_intervals.csv"
    )

    source = tmp_path / "source"
    for relative in (
        "pyproject.toml",
        "uq_benchmark/cli.py",
        "uq_benchmark/supplement.py",
        "uq_benchmark/training_continuity.py",
        "uq_benchmark/external_ood.py",
        "uq_benchmark/external_ood_analysis.py",
        "uq_benchmark/ensemble_pipeline.py",
        "uq_benchmark/llpr_pipeline.py",
        "uq_benchmark/method_comparison.py",
        "uq_benchmark/supplement_release.py",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n")

    output = finalize_supplement_release(root, source, tmp_path / "release")
    manifest = json.loads((output / "supplement_release_manifest.json").read_text())
    assert manifest["completed_methods"] == list(COMPARISON_METHODS)
    assert manifest["deterministic_ensemble5_status"] == "completed"
    assert manifest["formal_external_ood_claim_permitted"] is False
    assert manifest["external_element_support_verified"] is True
    assert manifest["external_unique_compositions"] == 9
    assert str(tmp_path) not in json.dumps(manifest)
    assert all(
        path.startswith(("supplement/", "source/"))
        for path in manifest["input_sha256"]
    )
    assert (output / "supplement_metrics_long.csv").is_file()
    assert (output / "method_compute_cost.csv").is_file()
    assert manifest["method_compute_cost_table_included"] is True
    assert manifest["llpr_posterior_psd_verified"] is True
    assert manifest["reference_training_epoch_and_state_continuity_verified"] is True
    assert manifest["reference_training_rng_state_continuity_verified"] is False
    assert manifest["reference_training_bitwise_lossless_resume_verified"] is False
    assert manifest["ensemble5_matched_seed42_protocol_verified"] is True
    assert "deterministic ensemble" in (output / "SUPPLEMENT_BENCHMARK.md").read_text()
    assert len((output / "SHA256SUMS").read_text().splitlines()) == 4
