from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from uq_benchmark.epic_pipeline import (
    SAMPLING_METHOD,
    aggregate_epic_force_shards,
    sampled_force_statistics,
)
from uq_benchmark.pipeline import validate_epic_force_record
from uq_benchmark.cli import build_parser


class FakeEPIC(torch.nn.Module):
    def forward(self, batch, training=False, compute_force=False):
        del training, compute_force
        x = batch.positions[:, 0]
        gamma = x.square() + 0.2 * x
        baseline = 0.5 * batch.positions.square().sum()
        pooled = x.mean().square()
        return {
            "energy": (baseline + gamma.sum()).reshape(1),
            "energy_nu": (2.0 + 0.1 * pooled).reshape(1),
            "energy_alpha": (4.0 + 0.1 * pooled).reshape(1),
            "energy_beta": (0.7 + 0.1 * pooled).reshape(1),
            "node_feats": batch.positions[:, :1],
        }


class FakeAtomwiseEPIC(torch.nn.Module):
    def forward(self, batch, training=False, compute_force=False):
        del training, compute_force
        x = batch.positions[:, 0]
        gamma = x.square() + 0.2 * x
        baseline = 0.5 * batch.positions.square().sum()
        return {
            "energy": (baseline + gamma.sum()).reshape(1),
            "atomic_energy_gamma": gamma,
            "atomic_energy_nu": 2.0 + 0.1 * x.square(),
            "atomic_energy_alpha": 4.0 + 0.1 * x.square(),
            "atomic_energy_beta": 0.7 + 0.1 * x.square(),
            "node_feats": batch.positions[:, :1],
        }


class FakeHierarchicalEPIC(torch.nn.Module):
    def forward(self, batch, training=False, compute_force=False):
        del training, compute_force
        x = batch.positions[:, 0]
        energy = 0.5 * batch.positions.square().sum() + (x.square() + 0.2 * x).sum()
        return {
            "energy": energy.reshape(1),
            "energy_nu": (2.0 + 0.1 * x.mean().square()).reshape(1),
            "energy_alpha": (4.0 + 0.1 * x.mean().square()).reshape(1),
            "energy_beta": (0.7 + 0.1 * x.mean().square()).reshape(1),
            "atomic_epistemic_weight": torch.softmax(x, dim=0),
            "node_feats": batch.positions[:, :1],
        }


def _batch():
    return SimpleNamespace(
        positions=torch.tensor(
            [[0.3, -0.2, 0.1], [-0.1, 0.25, -0.35]],
            dtype=torch.float64,
        ),
        batch=torch.zeros(2, dtype=torch.int64),
        head=torch.zeros(1, dtype=torch.int64),
    )


def test_epic_force_statistics_are_reproducible_and_energy_derived():
    model = FakeEPIC()
    point_force, scale, mc_mean, energies = sampled_force_statistics(
        model, _batch(), n_samples=16, seed=42
    )
    point_force_2, scale_2, mc_mean_2, energies_2 = sampled_force_statistics(
        model, _batch(), n_samples=16, seed=42
    )
    torch.testing.assert_close(point_force, point_force_2)
    torch.testing.assert_close(scale, scale_2)
    torch.testing.assert_close(mc_mean, mc_mean_2)
    torch.testing.assert_close(energies, energies_2)
    assert torch.all(scale >= 0)
    assert torch.any(scale > 0)

    positions = _batch().positions.requires_grad_(True)
    output = model(SimpleNamespace(positions=positions, batch=torch.zeros(2, dtype=torch.int64), head=torch.zeros(1, dtype=torch.int64)))
    expected = -torch.autograd.grad(output["energy"].sum(), positions)[0]
    torch.testing.assert_close(point_force, expected)


def test_atomwise_epic_force_samples_are_energy_gradient_derived():
    model = FakeAtomwiseEPIC()
    point_force, scale, _, energies = sampled_force_statistics(
        model, _batch(), n_samples=16, seed=42
    )
    assert energies.shape == (16,)
    assert scale.shape == (2, 3)
    assert torch.any(scale > 0)

    positions = _batch().positions.requires_grad_(True)
    batch = SimpleNamespace(
        positions=positions,
        batch=torch.zeros(2, dtype=torch.int64),
        head=torch.zeros(1, dtype=torch.int64),
    )
    expected = -torch.autograd.grad(model(batch)["energy"].sum(), positions)[0]
    torch.testing.assert_close(point_force, expected)


def test_hierarchical_epic_shared_scale_force_samples_are_energy_gradient_derived():
    model = FakeHierarchicalEPIC()
    point_force, scale, _, energies = sampled_force_statistics(
        model, _batch(), n_samples=16, seed=42
    )
    assert energies.shape == (16,)
    assert scale.shape == (2, 3)
    assert torch.any(scale > 0)

    positions = _batch().positions.requires_grad_(True)
    batch = SimpleNamespace(
        positions=positions,
        batch=torch.zeros(2, dtype=torch.int64),
        head=torch.zeros(1, dtype=torch.int64),
    )
    expected = -torch.autograd.grad(model(batch)["energy"].sum(), positions)[0]
    torch.testing.assert_close(point_force, expected)


def _valid_force_record():
    return {
        "db_index": np.array([3]),
        "force_true": np.zeros((2, 3)),
        "force_mean": np.zeros((2, 3)),
        "force_scale_raw": np.ones((2, 3)),
        "force_offsets": np.array([0, 2]),
        "sample_seed": np.array(42),
        "n_samples": np.array(32),
        "split": np.array("test"),
        "sampling_method": np.array(SAMPLING_METHOD),
        "allocation_mode": np.array("learned"),
        "uncertainty_component": np.array("epistemic"),
        "rng_seed_replayed_per_structure": np.array(True),
        "cross_structure_base_random_variable_alignment_verified": np.array(False),
        "force_samples_energy_gradient_derived": np.array(True),
        "global_random_pes_verified": np.array(False),
        "force_posterior_identified": np.array(False),
        "samplewise_permutation_equivariance_verified": np.array(False),
        "aleatoric_force_noise": np.array(False),
    }


def test_epic_force_cli_accepts_convergence_overrides():
    args = build_parser().parse_args(
        [
            "epic-force",
            "--split",
            "calibration",
            "--checkpoint",
            "checkpoint.pt",
            "--output",
            "force.npz",
            "--subset-size",
            "256",
            "--n-samples",
            "128",
            "--sample-seed",
            "43",
        ]
    )
    assert args.subset_size == 256
    assert args.n_samples == 128
    assert args.sample_seed == 43


def test_epic_force_cli_accepts_uniform_allocation_ablation():
    args = build_parser().parse_args(
        [
            "epic-force",
            "--split",
            "test",
            "--checkpoint",
            "checkpoint.pt",
            "--output",
            "force.npz",
            "--allocation-mode",
            "uniform",
        ]
    )
    assert args.allocation_mode == "uniform"


def test_epic_force_record_rejects_legacy_force_noise():
    record = _valid_force_record()
    validate_epic_force_record(record, "test")
    record["sampling_method"] = np.array("independent_gaussian_force_noise")
    with pytest.raises(ValueError, match="energy-first NIG"):
        validate_epic_force_record(record, "test")


def test_epic_force_record_allows_path_label_for_convergence_analysis():
    record = _valid_force_record()
    record["split"] = np.array("calibration")
    validate_epic_force_record(record, "/tmp/calibration_force.npz")


def test_epic_force_record_rejects_unverified_cross_structure_alignment_claim():
    record = _valid_force_record()
    record["cross_structure_base_random_variable_alignment_verified"] = np.array(True)
    with pytest.raises(ValueError, match="unsupported cross-structure"):
        validate_epic_force_record(record, "test")


def test_epic_force_shards_aggregate_in_db_order(tmp_path):
    paths = []
    for shard, db_index in enumerate((np.array([2, 8]), np.array([1, 5]))):
        record = _valid_force_record()
        record.update(
            db_index=db_index,
            force_true=np.repeat(db_index, 3).reshape(2, 3),
            force_mean=np.repeat(db_index + 0.1, 3).reshape(2, 3),
            force_mc_mean=np.repeat(db_index + 0.2, 3).reshape(2, 3),
            force_scale_raw=np.ones((2, 3)),
            force_offsets=np.array([0, 1, 2]),
            energy_mc_mean=db_index.astype(float),
            energy_mc_scale=np.ones(2),
            distribution_approximation=np.array("gaussian_moment_matched_mc"),
            inference_seconds=np.array(1.0),
            peak_memory_bytes=np.array(10),
            shard=np.array(shard),
            total_shards=np.array(2),
            selected_subset_size=np.array(4),
        )
        path = tmp_path / f"shard_{shard}.npz"
        np.savez(path, **record)
        paths.append(path)
    output = aggregate_epic_force_shards(paths, tmp_path / "aggregated.npz")
    with np.load(output) as data:
        np.testing.assert_array_equal(data["db_index"], [1, 2, 5, 8])
        np.testing.assert_array_equal(data["force_true"][:, 0], [1, 2, 5, 8])
        np.testing.assert_array_equal(data["force_offsets"], [0, 1, 2, 3, 4])
