from __future__ import annotations

import pytest
import torch

from uq_benchmark.hierarchical_epic import (
    HierarchicalEPICReadout,
    force_error_allocation_loss,
    sample_hierarchical_latent_energies,
)
from uq_benchmark.training import _validate_hierarchical_output_dir


class FakeReadoutBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 1
        self.linear_1 = torch.nn.Identity()
        self.non_linearity = torch.nn.Identity()
        self.gamma_head = torch.nn.Linear(1, 1, bias=False)
        self.nu_head = torch.nn.Linear(1, 1, bias=False)
        self.alpha_head = torch.nn.Linear(1, 1, bias=False)
        self.beta_head = torch.nn.Linear(1, 1, bias=False)
        for layer in (
            self.gamma_head,
            self.nu_head,
            self.alpha_head,
            self.beta_head,
        ):
            layer.weight.data.fill_(0.5)


def _segment_sum(values, batch, num_graphs):
    result = values.new_zeros(num_graphs)
    result.index_add_(0, batch, values)
    return result


def test_hierarchical_atomic_variances_exactly_reconstruct_graph_variances():
    readout = HierarchicalEPICReadout(FakeReadoutBase())
    readout.epistemic_allocation_head.weight.data.fill_(0.7)
    features = torch.tensor([[0.2], [1.1], [-0.4], [0.3], [1.5]])
    batch = torch.tensor([0, 0, 1, 1, 1])
    result = readout.hierarchical_parameters(features, batch, num_graphs=2)

    torch.testing.assert_close(
        _segment_sum(result["atomic_aleatoric_weight"], batch, 2),
        torch.ones(2),
    )
    torch.testing.assert_close(
        _segment_sum(result["atomic_epistemic_weight"], batch, 2),
        torch.ones(2),
    )
    torch.testing.assert_close(
        _segment_sum(result["atomic_aleatoric_variance"], batch, 2),
        result["energy_aleatoric_variance"],
    )
    torch.testing.assert_close(
        _segment_sum(result["atomic_epistemic_variance"], batch, 2),
        result["energy_epistemic_variance"],
    )
    assert torch.all(result["atomic_energy_nu"] > 0)
    assert torch.all(result["atomic_energy_alpha"] > 1)
    assert torch.all(result["atomic_energy_beta"] > 0)


def test_force_error_teacher_trains_only_allocation_weights():
    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    weights = torch.softmax(logits, dim=0)
    force_prediction = torch.tensor(
        [[3.0, 0.0, 0.0], [0.1, 0.0, 0.0]], requires_grad=True
    )
    loss = force_error_allocation_loss(
        weights,
        force_prediction,
        torch.zeros_like(force_prediction),
        torch.tensor([0, 0]),
        num_graphs=1,
        floor=1e-6,
    )
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0] < 0
    assert force_prediction.grad is None


def test_shared_scale_samples_match_graph_epistemic_variance():
    output = {
        "energy": torch.tensor([1.5], dtype=torch.float64),
        "energy_nu": torch.tensor([2.0], dtype=torch.float64),
        "energy_alpha": torch.tensor([5.0], dtype=torch.float64),
        "energy_beta": torch.tensor([3.0], dtype=torch.float64),
        "atomic_epistemic_weight": torch.tensor(
            [0.2, 0.3, 0.5], dtype=torch.float64
        ),
    }
    samples = sample_hierarchical_latent_energies(
        output, n_samples=50_000, seed=42
    )
    expected = 3.0 / (2.0 * (5.0 - 1.0))
    torch.testing.assert_close(
        samples.mean(), torch.tensor(1.5, dtype=torch.float64), atol=0.015, rtol=0
    )
    torch.testing.assert_close(
        samples.var(unbiased=True),
        torch.tensor(expected, dtype=torch.float64),
        atol=0.02,
        rtol=0.06,
    )


def test_uniform_allocation_ablation_preserves_graph_variance_but_changes_samples():
    output = {
        "energy": torch.tensor([1.5], dtype=torch.float64),
        "energy_nu": torch.tensor([2.0], dtype=torch.float64),
        "energy_alpha": torch.tensor([5.0], dtype=torch.float64),
        "energy_beta": torch.tensor([3.0], dtype=torch.float64),
        "atomic_epistemic_weight": torch.tensor([0.9, 0.1], dtype=torch.float64),
    }
    learned = sample_hierarchical_latent_energies(
        output, n_samples=20_000, seed=42, allocation_mode="learned"
    )
    uniform = sample_hierarchical_latent_energies(
        output, n_samples=20_000, seed=42, allocation_mode="uniform"
    )
    expected = torch.tensor(3.0 / (2.0 * (5.0 - 1.0)), dtype=torch.float64)
    assert not torch.equal(learned, uniform)
    torch.testing.assert_close(uniform.var(unbiased=True), expected, atol=0.02, rtol=0.06)


def test_hierarchical_sampling_rejects_unknown_allocation_mode():
    output = {
        "energy": torch.tensor([1.5]),
        "energy_nu": torch.tensor([2.0]),
        "energy_alpha": torch.tensor([5.0]),
        "energy_beta": torch.tensor([3.0]),
        "atomic_epistemic_weight": torch.tensor([0.5, 0.5]),
    }
    with pytest.raises(ValueError, match="allocation_mode"):
        sample_hierarchical_latent_energies(
            output, n_samples=4, seed=42, allocation_mode="invalid"
        )


def test_shared_scale_sampling_rejects_nonfinite_variance(monkeypatch):
    output = {
        "energy": torch.tensor([1.5]),
        "energy_nu": torch.tensor([2.0]),
        "energy_alpha": torch.tensor([1.01]),
        "energy_beta": torch.tensor([1.0]),
        "atomic_epistemic_weight": torch.tensor([0.4, 0.6]),
    }

    def nonfinite_rsample(self, sample_shape=torch.Size()):
        del self
        return torch.full(sample_shape, float("inf"))

    monkeypatch.setattr(torch.distributions.InverseGamma, "rsample", nonfinite_rsample)
    with pytest.raises(FloatingPointError, match="non-finite variances"):
        sample_hierarchical_latent_energies(output, n_samples=4, seed=42)


def test_hierarchical_training_directory_is_isolated(tmp_path):
    isolated = tmp_path / "uq_benchmark_runs" / "epic_hierarchical" / "probe"
    _validate_hierarchical_output_dir(isolated, resume=None)

    with pytest.raises(ValueError, match="epic_hierarchical"):
        _validate_hierarchical_output_dir(tmp_path / "legacy" / "probe", resume=None)

    isolated.mkdir(parents=True)
    (isolated / "checkpoint_best.pt").write_text("occupied")
    with pytest.raises(FileExistsError, match="non-empty"):
        _validate_hierarchical_output_dir(isolated, resume=None)
    _validate_hierarchical_output_dir(
        isolated,
        resume=None,
        check_occupancy=False,
    )
    _validate_hierarchical_output_dir(
        isolated, resume=isolated / "checkpoint_best.pt"
    )
