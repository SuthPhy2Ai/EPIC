from __future__ import annotations

import json

import h5py
import numpy as np

from uq_benchmark.external_ood_analysis import analyze_external_ood


def _point_record(path, db_index, error, energy_scale):
    n = len(db_index)
    natoms = np.full(n, 2, dtype=np.int32)
    energy_true = np.linspace(-2.0, 2.0, n) * natoms
    force_true = np.linspace(-0.5, 0.5, n * 6).reshape(n * 2, 3)
    force_pred = force_true + error * np.linspace(0.5, 1.5, n * 6).reshape(n * 2, 3)
    record = {
        "db_index": np.asarray(db_index),
        "natoms": natoms,
        "energy_true_total": energy_true,
        "energy_pred_total": energy_true + error,
        "energy_aleatoric_variance": np.full(n, energy_scale**2),
        "energy_epistemic_variance": np.full(n, (0.5 * energy_scale) ** 2),
        "force_true": force_true,
        "force_pred": force_pred,
        "force_offsets": np.arange(0, 2 * n + 1, 2),
    }
    np.savez(path, **record)
    return record


def _force_record(path, point, scale):
    n_atoms = len(point["force_true"])
    np.savez(
        path,
        db_index=point["db_index"],
        force_true=point["force_true"],
        force_mean=point["force_pred"],
        force_scale_raw=np.linspace(scale, 2 * scale, n_atoms * 3).reshape(n_atoms, 3),
        force_offsets=point["force_offsets"],
    )


def test_external_analysis_reuses_in_domain_calibration_and_stays_preliminary(tmp_path):
    external_index = np.arange(20)
    in_index = np.arange(100, 140)
    deterministic_path = tmp_path / "deterministic.npz"
    hierarchical_path = tmp_path / "hierarchical.npz"
    deterministic = _point_record(deterministic_path, external_index, 0.08, 0.2)
    hierarchical = _point_record(hierarchical_path, external_index, 0.06, 0.2)
    # Point-model truths must align while predictions are allowed to differ.
    deterministic["force_true"] = hierarchical["force_true"]
    deterministic["energy_true_total"] = hierarchical["energy_true_total"]
    np.savez(deterministic_path, **deterministic)
    external_force_path = tmp_path / "external_force.npz"
    _force_record(external_force_path, hierarchical, 0.08)

    in_point_path = tmp_path / "in_point.npz"
    in_point = _point_record(in_point_path, in_index, 0.03, 0.03)
    in_force_path = tmp_path / "in_force.npz"
    _force_record(in_force_path, in_point, 0.02)
    store = tmp_path / "store.h5"
    with h5py.File(store, "w") as handle:
        structures = handle.create_group("structures")
        structures.create_dataset("db_index", data=in_index)
        structures.create_dataset(
            "material_id",
            data=np.asarray([f"in-{index}" for index in range(len(in_index))], dtype="S32"),
        )

    parameters = tmp_path / "parameters.json"
    parameters.write_text(
        json.dumps(
            {
                "fit_split": "calibration",
                "evaluation_split": "test",
                "test_used_for_selection": False,
                "force": {"gaussian_affine_global": {"tau": 1.2, "sigma0": 0.01}},
            }
        )
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selected_materials": [f"external-{index}" for index in range(len(external_index))],
                "selection_uses_model_predictions": False,
            }
        )
    )
    leakage = tmp_path / "leakage.json"
    leakage.write_text(
        json.dumps(
            {
                "strict_geometry_overlap_count": 0,
                "translation_cell_invariant_geometry_overlap_count": 0,
                "remaining_gates": ["provenance_license_and_dft_protocol_verification"],
            }
        )
    )
    output = analyze_external_ood(
        deterministic_path,
        hierarchical_path,
        external_force_path,
        in_point_path,
        in_force_path,
        store,
        parameters,
        selection,
        leakage,
        tmp_path / "analysis",
        bootstrap_replicates=25,
        seed=42,
    )
    summary = json.loads((output / "external_ood_summary.json").read_text())
    assert summary["result_status"] == "preliminary_not_for_formal_ood_claim"
    assert summary["formal_external_ood_claim_permitted"] is False
    assert summary["in_domain_calibration_reused_without_external_refit"] is True
    assert summary["external_test_used_for_selection"] is False
    assert summary["n_external_structures"] == 20
    assert summary["headline"]["ood_detection_force_auroc"] > 0.9
