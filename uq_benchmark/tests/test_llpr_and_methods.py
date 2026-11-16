from __future__ import annotations

import numpy as np

from uq_benchmark.llpr import StreamingGram, antithetic_weight_samples, fit_llpr_posterior
from uq_benchmark.llpr_pipeline import aggregate_force_shards, structure_mean
from uq_benchmark.methods import combine_ensemble
from uq_benchmark.types import PredictionBatch


def test_streaming_gram_and_llpr_closed_form():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(120, 4))
    gram = StreamingGram(4)
    gram.update(features[:37])
    gram.update(features[37:])
    np.testing.assert_allclose(gram.matrix, features.T @ features, rtol=1e-13, atol=1e-13)

    calibration_features = rng.normal(size=(40, 4))
    weights = np.array([0.2, -0.4, 0.8, 0.1])
    target = calibration_features @ weights + rng.normal(scale=0.05, size=40)
    location = calibration_features @ weights
    posterior = fit_llpr_posterior(
        gram, calibration_features, target, location, (1e-8, 1e-4, 1e-2)
    )
    eigenvalues = np.linalg.eigvalsh(posterior.covariance_raw)
    assert eigenvalues.min() >= -1e-12
    expected = np.einsum(
        "ni,ij,nj->n", calibration_features, posterior.covariance_raw, calibration_features
    )
    np.testing.assert_allclose(posterior.energy_variance(calibration_features, calibrated=False), expected)


def test_antithetic_samples_preserve_mean():
    mean = np.array([1.0, -2.0, 0.5])
    covariance = np.diag([0.1, 0.2, 0.3])
    samples = antithetic_weight_samples(mean, covariance, 32, seed=42)
    np.testing.assert_allclose(samples.mean(axis=0), mean, atol=1e-14)


def test_ensemble_uses_unbiased_variance():
    members = []
    for values in ([1.0, 2.0], [2.0, 4.0], [3.0, 6.0]):
        members.append(PredictionBatch(np.array([10, 20]), np.array(values)))
    combined = combine_ensemble(members)
    np.testing.assert_allclose(combined.energy_location, [2.0, 4.0])
    np.testing.assert_allclose(combined.energy_scale_raw**2, np.var(np.array([[1, 2], [2, 4], [3, 6]]), axis=0, ddof=1))
    assert combined.metadata["n_members"] == 3


def test_llpr_force_shards_aggregate_in_db_order(tmp_path):
    paths = []
    for shard, db_index in enumerate((np.array([2, 8]), np.array([1, 5]))):
        record = {
            "db_index": db_index,
            "force_true": np.repeat(db_index, 3).reshape(2, 3),
            "force_mean": np.repeat(db_index + 0.1, 3).reshape(2, 3),
            "force_scale_raw": np.ones((2, 3)),
            "force_offsets": np.array([0, 1, 2]),
            "sample_seed": np.array(42),
            "n_samples": np.array(32),
            "sampling_method": np.array("llpr_last_layer_joint_weight_samples"),
            "uncertainty_component": np.array("epistemic"),
            "shared_sample_ids": np.array(True),
            "split": np.array("test"),
            "subset_namespace": np.array("epic-force-test-v1"),
            "selected_indices_sha256": np.array("fixture-subset-hash"),
            "inference_seconds": np.array(1.0),
            "peak_memory_bytes": np.array(10),
            "shard": np.array(shard),
            "total_shards": np.array(2),
            "selected_subset_size": np.array(4),
        }
        path = tmp_path / f"shard_{shard}.npz"
        np.savez(path, **record)
        paths.append(path)
    output = aggregate_force_shards(paths, tmp_path / "aggregated.npz")
    with np.load(output) as data:
        np.testing.assert_array_equal(data["db_index"], [1, 2, 5, 8])
        np.testing.assert_array_equal(data["force_true"][:, 0], [1, 2, 5, 8])
        np.testing.assert_array_equal(data["force_offsets"], [0, 1, 2, 3, 4])
        assert data["subset_namespace"] == "epic-force-test-v1"


def test_llpr_force_aggregate_rejects_mixed_subset_namespaces(tmp_path):
    paths = []
    for shard, namespace in enumerate(("epic-force-test-v1", "llpr-force-test-v1")):
        path = tmp_path / f"shard_{shard}.npz"
        np.savez(
            path,
            db_index=np.array([shard]),
            force_true=np.zeros((1, 3)),
            force_mean=np.zeros((1, 3)),
            force_scale_raw=np.ones((1, 3)),
            force_offsets=np.array([0, 1]),
            sample_seed=np.array(42),
            n_samples=np.array(32),
            sampling_method=np.array("llpr_last_layer_joint_weight_samples"),
            uncertainty_component=np.array("epistemic"),
            shared_sample_ids=np.array(True),
            split=np.array("test"),
            subset_namespace=np.array(namespace),
            selected_indices_sha256=np.array("fixture-subset-hash"),
            inference_seconds=np.array(1.0),
            peak_memory_bytes=np.array(10),
            shard=np.array(shard),
            total_shards=np.array(2),
            selected_subset_size=np.array(2),
        )
        paths.append(path)

    with np.testing.assert_raises_regex(ValueError, "subset_namespace"):
        aggregate_force_shards(paths, tmp_path / "aggregated.npz")


def test_force_structure_mean_uses_atom_offsets_for_three_components():
    values = np.arange(15, dtype=float).reshape(5, 3)
    result = structure_mean(values, np.array([0, 2, 5]))
    np.testing.assert_allclose(result, [2.5, 10.0])
