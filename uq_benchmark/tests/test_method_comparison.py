from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd

from uq_benchmark.method_comparison import (
    METHODS,
    MethodFiles,
    _validated_posterior,
    compare_nonensemble_methods,
)


def _write_point(path, db_index, method_offset, *, llpr=False):
    n = len(db_index)
    natoms = np.full(n, 2, dtype=np.int32)
    truth_pa = np.linspace(-1.0, 1.0, n)
    prediction_pa = truth_pa + method_offset + np.linspace(-0.08, 0.08, n)
    force_true = np.linspace(-0.5, 0.5, n * 6).reshape(n * 2, 3)
    force_pred = force_true + method_offset + np.linspace(-0.04, 0.04, n * 6).reshape(n * 2, 3)
    record = {
        "db_index": np.asarray(db_index),
        "natoms": natoms,
        "energy_true_total": truth_pa * natoms,
        "energy_pred_total": prediction_pa * natoms,
        "force_true": force_true,
        "force_pred": force_pred,
        "force_offsets": np.arange(0, 2 * n + 1, 2),
        "inference_seconds": np.asarray(float(n) / 10),
        "peak_memory_bytes": np.asarray(1000),
    }
    if llpr:
        record["llpr_features"] = np.column_stack(
            (np.linspace(0.5, 1.5, n), np.linspace(1.5, 0.5, n))
        )
    else:
        record["energy_aleatoric_variance"] = np.linspace(0.02, 0.08, n)
        record["energy_epistemic_variance"] = np.linspace(0.01, 0.04, n)
    np.savez(path, **record)
    return record


def _write_force(path, point, scale_factor):
    n = len(point["db_index"])
    scale = scale_factor * np.linspace(0.01, 0.09, n * 6).reshape(n * 2, 3)
    force_mean = point["force_pred"].copy()
    force_mean.flat[0] += 5e-6
    np.savez(
        path,
        db_index=point["db_index"],
        force_true=point["force_true"],
        force_mean=force_mean,
        force_scale_raw=scale,
        force_offsets=point["force_offsets"],
        n_samples=np.asarray(256),
        inference_seconds=np.asarray(float(n) * 2),
        peak_memory_bytes=np.asarray(2000),
    )


def test_nonensemble_method_comparison_is_aligned_calibrated_and_bootstrapped(tmp_path):
    calibration_index = np.arange(20)
    test_index = np.arange(100, 140)
    store = tmp_path / "store.h5"
    with h5py.File(store, "w") as handle:
        structures = handle.create_group("structures")
        structures.create_dataset("db_index", data=test_index)
        structures.create_dataset(
            "material_id",
            data=np.asarray([f"material-{index}" for index in range(len(test_index))], dtype="S32"),
        )

    files = {}
    method_offsets = {
        "hierarchical_epic": 0.00,
        "graph_nig": 0.01,
        "llpr": 0.02,
        "llpr_force_aware": 0.02,
    }
    for method_index, method in enumerate(METHODS):
        llpr = method.startswith("llpr")
        energy_calibration = tmp_path / f"{method}_energy_calibration.npz"
        energy_test = tmp_path / f"{method}_energy_test.npz"
        calibration = _write_point(
            energy_calibration, calibration_index, method_offsets[method], llpr=llpr
        )
        test = _write_point(energy_test, test_index, method_offsets[method], llpr=llpr)
        force_calibration = tmp_path / f"{method}_force_calibration.npz"
        force_test = tmp_path / f"{method}_force_test.npz"
        _write_force(force_calibration, calibration, 1.0 + 0.1 * method_index)
        _write_force(force_test, test, 1.0 + 0.1 * method_index)
        posterior = None
        if llpr:
            posterior = tmp_path / f"{method}_posterior.npz"
            np.savez(
                posterior,
                covariance_raw=np.diag([0.02, 0.03]),
                ridge=np.asarray(0.01),
                n_training_samples=np.asarray(100),
                posterior_kind=np.asarray(
                    "force_aware_llpr" if method == "llpr_force_aware" else "energy_only_llpr"
                ),
            )
        files[method] = MethodFiles(
            energy_calibration,
            energy_test,
            force_calibration,
            force_test,
            posterior,
        )

    output = compare_nonensemble_methods(
        files,
        store,
        tmp_path / "comparison",
        bootstrap_replicates=25,
        seed=42,
    )
    summary = json.loads((output / "nonensemble_method_comparison_summary.json").read_text())
    assert summary["methods"] == list(METHODS)
    assert summary["test_used_for_selection"] is False
    assert summary["force_subset_alignment_verified"] is True
    metrics = np.genfromtxt(
        output / "tables/nonensemble_method_metrics.csv", delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    assert len(metrics) > 50
    differences = (output / "tables/hierarchical_paired_advantages.csv").read_text()
    assert "hierarchical_advantage_over_graph_nig" in differences
    assert "ensemble" not in summary["methods"]
    costs = pd.read_csv(output / "tables/nonensemble_compute_cost.csv")
    assert len(costs) == len(METHODS) * 2 * 2
    assert set(costs["target"]) == {"point_inference", "force_uq_inference"}
    assert set(costs["mc_samples"]) == {1, 256}
    assert np.all(costs["milliseconds_per_structure"] > 0)
    assert summary["compute_cost_table"] == "tables/nonensemble_compute_cost.csv"
    assert set(summary["posterior_audits"]) == {"llpr", "llpr_force_aware"}
    assert all(
        audit["finite_symmetric_positive_semidefinite_verified"]
        for audit in summary["posterior_audits"].values()
    )


def test_nonensemble_posterior_audit_rejects_indefinite_covariance(tmp_path):
    posterior = tmp_path / "indefinite.npz"
    np.savez(
        posterior,
        covariance_raw=np.diag([0.1, -0.01]),
        ridge=np.asarray(0.001),
        n_training_samples=np.asarray(10),
    )
    with np.testing.assert_raises_regex(ValueError, "positive semidefinite"):
        _validated_posterior(posterior)
