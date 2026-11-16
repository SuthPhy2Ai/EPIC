from __future__ import annotations

import numpy as np

from uq_benchmark.inference import aggregate_shards


def _inference_shard(
    db_index: np.ndarray,
    atom_value: np.ndarray,
    *,
    shard: int,
    total_shards: int = 2,
    selected_split_size: int = 4,
) -> dict:
    n_structures = len(db_index)
    natoms = np.ones(n_structures, dtype=np.int32)
    graph = db_index.astype(float)
    atom_value = np.asarray(atom_value, dtype=float)
    return {
        "db_index": db_index,
        "natoms": natoms,
        "energy_true_total": graph,
        "energy_pred_total": graph + 0.1,
        "nu": np.full(n_structures, 2.0),
        "alpha": np.full(n_structures, 3.0),
        "beta": np.full(n_structures, 0.5),
        "energy_aleatoric_variance": atom_value,
        "energy_epistemic_variance": 2.0 * atom_value,
        "force_true": np.repeat(graph, 3).reshape(n_structures, 3),
        "force_pred": np.repeat(graph + 0.1, 3).reshape(n_structures, 3),
        "atomic_numbers": np.ones(n_structures, dtype=np.int16),
        "atomic_energy_gamma": graph,
        "atomic_energy_nu": np.full(n_structures, 2.0),
        "atomic_energy_alpha": np.full(n_structures, 3.0),
        "atomic_energy_beta": np.full(n_structures, 0.5),
        "atomic_aleatoric_weight": np.ones(n_structures),
        "atomic_epistemic_weight": np.ones(n_structures),
        "atomic_aleatoric_variance": atom_value,
        "atomic_epistemic_variance": 2.0 * atom_value,
        "atomic_total_variance": 3.0 * atom_value,
        "force_offsets": np.arange(n_structures + 1, dtype=np.int64),
        "inference_seconds": np.asarray(1.0),
        "peak_memory_bytes": np.asarray(10, dtype=np.int64),
        "checkpoint_kind": np.asarray("epic_hierarchical"),
        "atomwise_distribution_assumption": np.asarray(
            "hierarchical_shared_scale_latent_nig_marginals"
        ),
        "epic_variant": np.asarray("hierarchical_atomwise_shared_scale"),
        "split": np.asarray("validation"),
        "shard": np.asarray(shard, dtype=np.int64),
        "total_shards": np.asarray(total_shards, dtype=np.int64),
        "selected_split_size": np.asarray(selected_split_size, dtype=np.int64),
    }


def test_hierarchical_atomic_variances_remain_aligned_when_aggregating_shards(
    tmp_path,
):
    paths = []
    for shard, (db_index, variance) in enumerate(
        (
            (np.asarray([2, 8]), np.asarray([20.0, 80.0])),
            (np.asarray([1, 5]), np.asarray([10.0, 50.0])),
        )
    ):
        path = tmp_path / f"shard-{shard}.npz"
        np.savez(path, **_inference_shard(db_index, variance, shard=shard))
        paths.append(path)

    output = aggregate_shards(paths, tmp_path / "aggregate.npz")
    with np.load(output) as data:
        np.testing.assert_array_equal(data["db_index"], [1, 2, 5, 8])
        np.testing.assert_array_equal(data["force_offsets"], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(
            data["atomic_aleatoric_variance"], [10.0, 20.0, 50.0, 80.0]
        )
        np.testing.assert_array_equal(
            data["atomic_epistemic_variance"], [20.0, 40.0, 100.0, 160.0]
        )
        np.testing.assert_array_equal(
            data["atomic_total_variance"], [30.0, 60.0, 150.0, 240.0]
        )
        assert str(data["checkpoint_kind"]) == "epic_hierarchical"
        assert str(data["epic_variant"]) == "hierarchical_atomwise_shared_scale"
        np.testing.assert_array_equal(data["aggregated_shards"], [0, 1])
        assert str(data["split"]) == "validation"
        assert int(data["selected_split_size"]) == 4


def test_aggregate_rejects_duplicate_shard_ids(tmp_path):
    paths = []
    for index, db_index in enumerate((np.asarray([0, 2]), np.asarray([1, 3]))):
        path = tmp_path / f"duplicate-{index}.npz"
        np.savez(
            path,
            **_inference_shard(
                db_index,
                db_index.astype(float),
                shard=0,
            ),
        )
        paths.append(path)

    with np.testing.assert_raises_regex(ValueError, "duplicate inference shard id"):
        aggregate_shards(paths, tmp_path / "aggregate.npz")


def test_aggregate_rejects_missing_shard(tmp_path):
    path = tmp_path / "shard-0.npz"
    np.savez(
        path,
        **_inference_shard(
            np.asarray([0, 2]),
            np.asarray([1.0, 2.0]),
            shard=0,
        ),
    )

    with np.testing.assert_raises_regex(ValueError, "incomplete inference shard set"):
        aggregate_shards([path], tmp_path / "aggregate.npz")


def test_aggregate_rejects_duplicate_db_indices(tmp_path):
    paths = []
    for shard, db_index in enumerate((np.asarray([0, 2]), np.asarray([1, 2]))):
        path = tmp_path / f"shard-{shard}.npz"
        np.savez(
            path,
            **_inference_shard(
                db_index,
                db_index.astype(float),
                shard=shard,
            ),
        )
        paths.append(path)

    with np.testing.assert_raises_regex(ValueError, "duplicate db_index"):
        aggregate_shards(paths, tmp_path / "aggregate.npz")


def test_aggregate_rejects_misaligned_force_offsets(tmp_path):
    paths = []
    for shard, db_index in enumerate((np.asarray([0, 2]), np.asarray([1, 3]))):
        record = _inference_shard(
            db_index,
            db_index.astype(float),
            shard=shard,
        )
        if shard == 1:
            record["force_offsets"] = np.asarray([0, 2, 2], dtype=np.int64)
        path = tmp_path / f"shard-{shard}.npz"
        np.savez(path, **record)
        paths.append(path)

    with np.testing.assert_raises_regex(ValueError, "invalid force_offsets"):
        aggregate_shards(paths, tmp_path / "aggregate.npz")
