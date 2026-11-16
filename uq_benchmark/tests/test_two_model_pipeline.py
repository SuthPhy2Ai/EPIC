from __future__ import annotations

import json
import sqlite3

import h5py
import numpy as np
import pandas as pd

from uq_benchmark.pipeline import (
    build_two_model_store,
    fit_epic_calibration_file,
    fit_epic_force_calibration_file,
    load_calibration_state_file,
)
from uq_benchmark.report import generate_report
from uq_benchmark.hierarchical_epic import SAMPLING_METHOD
from uq_benchmark.two_model_analysis import (
    analyze_force_convergence,
    analyze_two_model_predictions,
)
from uq_benchmark.two_model_summary import write_two_model_summary


def _write_database(path, n_structures: int) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE text_key_values (id INTEGER, key TEXT, value TEXT)")
        connection.executemany(
            "INSERT INTO text_key_values VALUES (?, 'material_id', ?)",
            [(index + 1, f"material-{index // 2}") for index in range(n_structures)],
        )
        connection.commit()
    finally:
        connection.close()


def _write_predictions(path, *, split: str, hierarchical: bool, n: int = 12) -> None:
    db_index = np.arange(n, dtype=np.int64)
    truth = np.linspace(-2.0, 2.0, n)
    prediction = truth + np.linspace(-0.15, 0.12, n)
    force_true = np.repeat(truth, 3).reshape(n, 3)
    record = {
        "db_index": db_index,
        "natoms": np.ones(n, dtype=np.int32),
        "energy_true_total": truth,
        "energy_pred_total": prediction,
        "force_true": force_true,
        "force_pred": force_true + 0.05,
        "force_offsets": np.arange(n + 1, dtype=np.int64),
        "inference_seconds": np.asarray(2.0),
        "peak_memory_bytes": np.asarray(100, dtype=np.int64),
        "split": np.asarray(split),
        "checkpoint_kind": np.asarray("epic_hierarchical" if hierarchical else "deterministic"),
    }
    if hierarchical:
        record.update(
            nu=np.full(n, 2.0),
            alpha=np.full(n, 4.0),
            beta=np.linspace(0.1, 0.3, n),
            energy_aleatoric_variance=np.linspace(0.04, 0.08, n),
            energy_epistemic_variance=np.linspace(0.02, 0.04, n),
            atomic_aleatoric_variance=np.linspace(0.04, 0.08, n),
            atomic_epistemic_variance=np.linspace(0.02, 0.04, n),
            atomic_total_variance=np.linspace(0.06, 0.12, n),
            epic_variant=np.asarray("hierarchical_atomwise_shared_scale"),
        )
    np.savez(path, **record)


def _write_force_record(
    path,
    predictions_path,
    *,
    seed: int = 42,
    samples: int = 32,
    scale: float = 0.08,
) -> None:
    with np.load(predictions_path) as predictions:
        record = {
            "db_index": predictions["db_index"],
            "force_true": predictions["force_true"],
            "force_mean": predictions["force_pred"],
            "force_scale_raw": scale
            * (1.0 + 0.01 * np.abs(predictions["force_true"])),
            "force_offsets": predictions["force_offsets"],
            "sample_seed": np.asarray(seed),
            "n_samples": np.asarray(samples),
            "sampling_method": np.asarray(SAMPLING_METHOD),
            "uncertainty_component": np.asarray("epistemic"),
            "rng_seed_replayed_per_structure": np.asarray(True),
            "cross_structure_base_random_variable_alignment_verified": np.asarray(False),
            "force_samples_energy_gradient_derived": np.asarray(True),
            "global_random_pes_verified": np.asarray(False),
            "force_posterior_identified": np.asarray(False),
            "samplewise_permutation_equivariance_verified": np.asarray(False),
            "aleatoric_force_noise": np.asarray(False),
        }
    np.savez(path, **record)


def test_two_model_store_uses_calibration_only_and_excludes_deferred_methods(tmp_path):
    database = tmp_path / "structures.db"
    _write_database(database, 12)
    mace_test = tmp_path / "mace_test.npz"
    hierarchical_calibration = tmp_path / "hierarchical_calibration.npz"
    hierarchical_test = tmp_path / "hierarchical_test.npz"
    _write_predictions(mace_test, split="test", hierarchical=False)
    _write_predictions(hierarchical_calibration, split="calibration", hierarchical=True)
    _write_predictions(hierarchical_test, split="test", hierarchical=True)

    calibration_state = fit_epic_calibration_file(
        hierarchical_calibration,
        tmp_path / "calibration.json",
    )
    state = load_calibration_state_file(calibration_state, "hierarchical_epic")
    assert state.parameters["tau"] > 0
    payload = json.loads(calibration_state.read_text())
    assert payload["split"] == "calibration"
    assert payload["n_structures"] == 12

    store_path = build_two_model_store(
        tmp_path / "benchmark.h5",
        database,
        mace_test,
        hierarchical_calibration,
        hierarchical_test,
        calibration_state,
        {"test": True},
        expected_force_calibration_size=None,
        expected_force_test_size=None,
    )
    with h5py.File(store_path, "r") as store:
        assert set(store["methods"]) == {"mace", "hierarchical_epic"}
        assert set(store["calibration"]) == {"hierarchical_epic"}
        manifest = json.loads(store.attrs["manifest_json"])
        assert manifest["benchmark_scope"] == ["mace", "hierarchical_epic"]
        assert manifest["deferred_methods"] == ["ensemble5", "llpr", "graph_nig_epic"]
        assert store["methods/hierarchical_epic"].attrs["distribution_family"] == "student_t"
        assert store["methods/mace"].attrs["distribution_family"] == "none"

    report = generate_report(store_path, tmp_path / "report", bootstrap_replicates=10)
    metrics = pd.read_csv(report / "tables/metrics.csv")
    assert set(metrics["method"]) == {"mace", "hierarchical_epic"}
    assert "ensemble5" not in set(metrics["method"])
    analysis = analyze_two_model_predictions(
        mace_test,
        hierarchical_test,
        calibration_state,
        tmp_path / "analysis",
    )
    assert (analysis / "size_stratified_metrics.csv").is_file()
    summary = json.loads((analysis / "two_model_analysis.json").read_text())
    assert summary["external_ood_evaluation_available"] is False
    assert (
        summary["atomic_variance_reconstruction"]["total"]["max_absolute_residual"]
        <= np.finfo(np.float64).eps
    )


def test_epic_calibration_rejects_test_predictions(tmp_path):
    predictions = tmp_path / "hierarchical_test.npz"
    _write_predictions(predictions, split="test", hierarchical=True)
    with np.testing.assert_raises_regex(ValueError, "calibration split"):
        fit_epic_calibration_file(predictions, tmp_path / "calibration.json")


def test_two_model_store_calibrates_energy_gradient_force_subset(tmp_path):
    database = tmp_path / "structures.db"
    _write_database(database, 12)
    mace_test = tmp_path / "mace_test.npz"
    hierarchical_calibration = tmp_path / "hierarchical_calibration.npz"
    hierarchical_test = tmp_path / "hierarchical_test.npz"
    force_calibration = tmp_path / "force_calibration.npz"
    force_test = tmp_path / "force_test.npz"
    _write_predictions(mace_test, split="test", hierarchical=False)
    _write_predictions(hierarchical_calibration, split="calibration", hierarchical=True)
    _write_predictions(hierarchical_test, split="test", hierarchical=True)
    _write_force_record(force_calibration, hierarchical_calibration)
    _write_force_record(force_test, hierarchical_test)
    calibration_state = fit_epic_calibration_file(
        hierarchical_calibration,
        tmp_path / "calibration.json",
    )
    force_calibration_state = fit_epic_force_calibration_file(
        force_calibration,
        tmp_path / "force_calibration.json",
    )
    force_payload = json.loads(force_calibration_state.read_text())
    assert force_payload["split"] == "calibration"
    assert force_payload["n_structures"] == 12
    assert force_payload["n_samples"] == 32

    store_path = build_two_model_store(
        tmp_path / "benchmark.h5",
        database,
        mace_test,
        hierarchical_calibration,
        hierarchical_test,
        calibration_state,
        {"test": True},
        force_calibration,
        force_test,
        expected_force_calibration_size=None,
        expected_force_test_size=None,
    )
    with h5py.File(store_path, "r") as store:
        subset = store["methods/hierarchical_epic/force_uq_subset"]
        assert subset.attrs["distribution_family"] == "gaussian"
        metadata = json.loads(subset.attrs["metadata_json"])
        assert metadata["force_samples_energy_gradient_derived"] is True
        assert metadata["n_samples"] == 32
        assert set(store["calibration"]) == {
            "hierarchical_epic",
            "hierarchical_epic_force",
        }
    report = generate_report(store_path, tmp_path / "force_report", bootstrap_replicates=10)
    assert (report / "tables/force_uq_size_stratified.csv").is_file()
    assert (report / "tables/runtime_cost.csv").is_file()
    assert (report / "figures/calibration_coverage.png").is_file()
    assert (report / "figures/risk_coverage.pdf").is_file()
    force_metrics = pd.read_csv(report / "tables/metrics.csv")
    assert "force_component_subset" in set(force_metrics["target"])
    force_risk = pd.read_csv(report / "tables/risk_coverage.csv")
    assert "force_component_structure_MAE_subset" in set(force_risk["target"])
    analysis = analyze_two_model_predictions(
        mace_test,
        hierarchical_test,
        calibration_state,
        tmp_path / "force_analysis",
    )
    convergence = tmp_path / "convergence.json"
    convergence.write_text(
        json.dumps(
            {
                "recommended_sample_count": 32,
                "aggregate_metric_recommended_sample_count": 32,
                "test_split_used_for_selection": False,
            }
        )
    )
    audit = tmp_path / "checkpoint_audit.json"
    audit.write_text(
        json.dumps(
            {
                "finite_difference_step_angstrom": 0.01,
                "max_point_autograd_error": 1e-7,
                "max_point_finite_difference_error": 1e-4,
                "max_sampled_finite_difference_error": 2e-4,
                "max_sampled_closed_loop_absolute_work": 1e-8,
                "point_prediction_permutation_equivariance_verified": True,
                "atomic_weight_permutation_equivariance_verified": True,
                "same_sample_id_permutation_equivariance_verified": False,
                "global_random_pes_verified": False,
            }
        )
    )
    cost_dir = tmp_path / "cost"
    cost_dir.mkdir()
    pd.DataFrame(
        [
            {"method": "mace", "epochs_completed": 10, "training_gpu_hours": 20.0},
            {
                "method": "hierarchical_epic",
                "epochs_completed": 8,
                "training_gpu_hours": 16.0,
            },
        ]
    ).to_csv(cost_dir / "training_cost.csv", index=False)
    pd.DataFrame(
        [
            {"method": "mace", "milliseconds_per_structure": 3.0},
            {"method": "hierarchical_epic", "milliseconds_per_structure": 4.0},
        ]
    ).to_csv(cost_dir / "inference_cost.csv", index=False)
    summary_dir = write_two_model_summary(
        report,
        analysis,
        convergence,
        audit,
        force_calibration_state,
        tmp_path / "summary",
        cost_dir,
    )
    assert (summary_dir / "two_model_benchmark_summary.json").is_file()
    assert "Point accuracy is mixed" in (
        summary_dir / "two_model_benchmark_summary.md"
    ).read_text()
    assert "Recorded compute cost" in (
        summary_dir / "two_model_benchmark_summary.md"
    ).read_text()


def test_force_convergence_analysis_requires_and_summarizes_complete_grid(tmp_path):
    predictions = tmp_path / "hierarchical_calibration.npz"
    _write_predictions(predictions, split="calibration", hierarchical=True)
    paths = []
    for samples in (4, 8):
        for seed in (42, 43):
            path = tmp_path / f"samples_{samples}_seed_{seed}.npz"
            _write_force_record(
                path,
                predictions,
                seed=seed,
                samples=samples,
                scale=0.08 * (1.0 + 1.0 / samples + 0.01 * (seed - 42)),
            )
            paths.append(path)
    output = analyze_force_convergence(paths, tmp_path / "convergence")
    summary = json.loads((output / "force_mc_convergence.json").read_text())
    assert summary["sample_counts"] == [4, 8]
    assert summary["sample_seeds"] == [42, 43]
    assert summary["test_split_used_for_selection"] is False
    assert pd.read_csv(output / "force_mc_runs.csv").shape[0] == 4
    assert (output / "figures/force_mc_convergence.png").is_file()
