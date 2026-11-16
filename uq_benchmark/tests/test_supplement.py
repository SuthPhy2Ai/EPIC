from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd

from uq_benchmark.supplement import (
    _ranking_groups,
    _weighted_auroc_auprc,
    analyze_atomic_attribution,
    analyze_calibration_sensitivity,
    bootstrap_uq_metrics,
    compare_force_allocations,
)


def _write_structure_store(path, n: int) -> None:
    with h5py.File(path, "w") as handle:
        structures = handle.create_group("structures")
        structures.create_dataset("db_index", data=np.arange(n, dtype=np.int64))
        structures.create_dataset(
            "material_id",
            data=np.asarray([f"material-{index // 2}" for index in range(n)], dtype="S32"),
        )


def test_atomic_attribution_analysis_uses_held_out_force_error(tmp_path):
    n_structures = 8
    natoms = np.full(n_structures, 3, dtype=np.int32)
    offsets = np.arange(0, 3 * n_structures + 1, 3, dtype=np.int64)
    force_pred = np.zeros((3 * n_structures, 3), dtype=np.float32)
    force_true = np.zeros_like(force_pred)
    force_true[0::3, 0] = 3.0
    force_true[1::3, 0] = 1.0
    learned = np.tile(np.asarray([0.82, 0.16, 0.02], dtype=np.float32), n_structures)
    predictions = tmp_path / "predictions.npz"
    np.savez(
        predictions,
        db_index=np.arange(n_structures, dtype=np.int64),
        natoms=natoms,
        alpha=np.linspace(1.2, 2.4, n_structures),
        force_true=force_true,
        force_pred=force_pred,
        force_offsets=offsets,
        atomic_numbers=np.tile(np.asarray([1, 6, 8], dtype=np.int16), n_structures),
        atomic_energy_gamma=np.tile(
            np.asarray([0.1, 0.2, 0.3], dtype=np.float32), n_structures
        ),
        atomic_epistemic_weight=learned,
    )
    store = tmp_path / "store.h5"
    _write_structure_store(store, n_structures)
    output = analyze_atomic_attribution(
        predictions, store, tmp_path / "atomic", bootstrap_replicates=20
    )
    confidence = pd.read_csv(
        output / "tables/atomic_attribution_confidence_intervals.csv"
    ).set_index("metric")
    assert confidence.loc["learned_allocation_ndcg", "estimate"] > 0.99
    assert (
        confidence.loc["learned_allocation_ndcg", "estimate"]
        > confidence.loc["atomic_energy_magnitude_ndcg", "estimate"]
    )
    assert (output / "figures/atomic_attribution_baselines.png").is_file()
    summary = json.loads((output / "atomic_attribution_summary.json").read_text())
    assert summary["n_structures"] == n_structures
    assert summary["n_atoms"] == 3 * n_structures


def test_material_cluster_uq_bootstrap_outputs_nonlinear_metrics(tmp_path):
    rng = np.random.default_rng(7)
    n = 30
    store = tmp_path / "store.h5"
    truth = np.linspace(-1.0, 1.0, n)
    location = truth + rng.normal(scale=0.08, size=n)
    force_true = rng.normal(size=(n, 3))
    force_location = force_true + rng.normal(scale=0.1, size=(n, 3))
    with h5py.File(store, "w") as handle:
        structures = handle.create_group("structures")
        structures.create_dataset("db_index", data=np.arange(n, dtype=np.int64))
        structures.create_dataset(
            "material_id",
            data=np.asarray([f"material-{index // 3}" for index in range(n)], dtype="S32"),
        )
        structures.create_dataset("energy_true_pa", data=truth)
        methods = handle.create_group("methods")
        epic = methods.create_group("hierarchical_epic")
        epic.create_dataset("energy_location", data=location)
        epic.create_dataset("energy_scale_calibrated", data=np.full(n, 0.1))
        epic.create_dataset("energy_df", data=np.full(n, 8.0))
        epic.create_dataset("energy_risk", data=np.abs(location - truth) + 0.01)
        force = epic.create_group("force_uq_subset")
        force.create_dataset("db_index", data=np.arange(n, dtype=np.int64))
        force.create_dataset("force_true", data=force_true)
        force.create_dataset("force_location", data=force_location)
        force.create_dataset("force_scale_calibrated", data=np.full((n, 3), 0.12))
        force.create_dataset("force_scale_raw", data=np.full((n, 3), 0.1))
        force.create_dataset("force_offsets", data=np.arange(n + 1, dtype=np.int64))
    output = bootstrap_uq_metrics(store, tmp_path / "uq", bootstrap_replicates=20)
    confidence = pd.read_csv(output / "tables/uq_confidence_intervals.csv")
    assert {
        "energy_nll_calibrated",
        "energy_spearman",
        "energy_top5_auroc",
        "force_nll_calibrated_component_weighted",
        "force_structure_spearman",
        "force_structure_top5_auprc",
    } <= set(confidence.metric)
    finite = confidence.dropna(subset=["ci_low_95", "ci_high_95"])
    assert len(finite) >= 10
    assert np.all(finite.ci_low_95 <= finite.ci_high_95)
    summary = json.loads((output / "uq_bootstrap_summary.json").read_text())
    assert summary["n_force_components"] == 3 * n


def test_weighted_ranking_metrics_handle_ties_and_perfect_order():
    danger = np.asarray([True, True, False, False])
    score = np.asarray([3.0, 2.0, 1.0, 1.0])
    order, starts = _ranking_groups(score)
    auroc, auprc = _weighted_auroc_auprc(
        danger, np.ones(4), order, starts
    )
    assert auroc == 1.0
    assert auprc == 1.0


def _write_calibration_sensitivity_inputs(root, split: str, seed: int):
    rng = np.random.default_rng(seed)
    natoms = np.tile(np.asarray([5, 30, 60, 100, 180], dtype=np.int32), 4)
    n = len(natoms)
    truth_pa = rng.normal(scale=0.4, size=n)
    energy_error = rng.standard_t(df=6, size=n) * 0.05 * np.power(natoms / 30.0, 0.2)
    energy_path = root / f"energy_{split}.npz"
    np.savez(
        energy_path,
        db_index=np.arange(n, dtype=np.int64),
        natoms=natoms,
        energy_true_total=truth_pa * natoms,
        energy_pred_total=(truth_pa + energy_error) * natoms,
        nu=np.full(n, 2.0),
        alpha=np.full(n, 3.0),
        beta=np.full(n, 0.02),
        force_pred=np.zeros((int(natoms.sum()), 3), dtype=np.float32),
        inference_seconds=np.asarray(1.0),
        split=np.asarray(split),
        checkpoint_kind=np.asarray("epic_hierarchical"),
        epic_variant=np.asarray("hierarchical_atomwise_shared_scale"),
    )
    offsets = np.concatenate(([0], np.cumsum(natoms))).astype(np.int64)
    force_true = rng.normal(scale=0.15, size=(int(natoms.sum()), 3))
    force_mean = force_true + rng.standard_t(df=5, size=force_true.shape) * 0.08
    force_scale = 0.04 + 0.01 * rng.random(force_true.shape)
    force_path = root / f"force_{split}.npz"
    np.savez(
        force_path,
        db_index=np.arange(n, dtype=np.int64),
        force_true=force_true,
        force_mean=force_mean,
        force_scale_raw=force_scale,
        force_offsets=offsets,
        split=np.asarray(split),
    )
    return energy_path, force_path


def test_calibration_sensitivity_fits_only_calibration_and_reports_size(tmp_path):
    energy_cal, force_cal = _write_calibration_sensitivity_inputs(
        tmp_path, "calibration", 10
    )
    energy_test, force_test = _write_calibration_sensitivity_inputs(
        tmp_path, "test", 11
    )
    output = analyze_calibration_sensitivity(
        energy_cal,
        energy_test,
        force_cal,
        force_test,
        tmp_path / "sensitivity",
    )
    metrics = pd.read_csv(output / "tables/calibration_sensitivity_metrics.csv")
    assert {
        "student_t_global",
        "student_t_size_power",
        "conformal_global",
        "conformal_size_bin",
        "gaussian_affine_global",
        "gaussian_multiplicative",
        "student_t_affine_global",
        "gaussian_size_power_affine",
        "force_conformal_global",
        "force_conformal_size_bin",
    } <= set(metrics.method)
    by_size = pd.read_csv(output / "tables/calibration_sensitivity_by_size.csv")
    assert set(by_size.size_bin) == {"1-20", "21-40", "41-80", "81-160", ">160"}
    parameters = json.loads(
        (output / "calibration_sensitivity_parameters.json").read_text()
    )
    assert parameters["fit_split"] == "calibration"
    assert parameters["evaluation_split"] == "test"
    assert parameters["test_used_for_selection"] is False
    assert (output / "figures/calibration_family_sensitivity.png").is_file()


def test_force_allocation_comparison_is_paired_and_calibration_only(tmp_path):
    energy_cal, learned_cal = _write_calibration_sensitivity_inputs(
        tmp_path, "calibration", 20
    )
    energy_test, learned_test = _write_calibration_sensitivity_inputs(
        tmp_path, "test", 21
    )

    def make_uniform(source, destination):
        with np.load(source) as record:
            values = {name: record[name] for name in record.files}
        values["force_scale_raw"] = 1.25 * values["force_scale_raw"]
        values["allocation_mode"] = np.asarray("uniform")
        values["sampling_method"] = np.asarray(
            "hierarchical_uniform_shared_scale_latent_mean_rsample"
        )
        values["n_samples"] = np.asarray(256)
        values["sample_seed"] = np.asarray(42)
        values["inference_seconds"] = np.asarray(2.0)
        values["peak_memory_bytes"] = np.asarray(200)
        np.savez(destination, **values)

    def add_learned_metadata(path):
        with np.load(path) as record:
            values = {name: record[name] for name in record.files}
        values["allocation_mode"] = np.asarray("learned")
        values["sampling_method"] = np.asarray(
            "hierarchical_atomwise_shared_scale_latent_mean_rsample"
        )
        values["n_samples"] = np.asarray(256)
        values["sample_seed"] = np.asarray(42)
        values["inference_seconds"] = np.asarray(1.0)
        values["peak_memory_bytes"] = np.asarray(100)
        np.savez(path, **values)

    add_learned_metadata(learned_cal)
    add_learned_metadata(learned_test)
    uniform_cal = tmp_path / "uniform_calibration.npz"
    uniform_test = tmp_path / "uniform_test.npz"
    make_uniform(learned_cal, uniform_cal)
    make_uniform(learned_test, uniform_test)

    learned_sensitivity = analyze_calibration_sensitivity(
        energy_cal,
        energy_test,
        learned_cal,
        learned_test,
        tmp_path / "learned_sensitivity",
    )
    uniform_sensitivity = analyze_calibration_sensitivity(
        energy_cal,
        energy_test,
        uniform_cal,
        uniform_test,
        tmp_path / "uniform_sensitivity",
    )
    store = tmp_path / "store.h5"
    _write_structure_store(store, 20)
    output = compare_force_allocations(
        learned_cal,
        learned_test,
        uniform_cal,
        uniform_test,
        store,
        learned_sensitivity,
        uniform_sensitivity,
        tmp_path / "comparison",
        bootstrap_replicates=20,
    )
    confidence = pd.read_csv(
        output / "tables/force_allocation_confidence_intervals.csv"
    )
    assert {
        "learned_force_structure_spearman",
        "uniform_force_structure_spearman",
        "uniform_minus_learned_gaussian_affine_nll",
        "uniform_minus_learned_coverage_90_absolute_error",
    } <= set(confidence.metric)
    assert np.all(np.isfinite(confidence.estimate))
    scale = pd.read_csv(output / "tables/force_allocation_scale_comparison.csv")
    assert set(scale.split) == {"calibration", "test"}
    assert np.allclose(scale.uniform_scale_median / scale.learned_scale_median, 1.25)
    summary = json.loads(
        (output / "force_allocation_comparison_summary.json").read_text()
    )
    assert summary["point_force_alignment_verified"] is True
    assert summary["energy_uq_invariance_verified"] is True
    assert (output / "figures/learned_vs_uniform_force_allocation.png").is_file()
