from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import t

from uq_benchmark.distributions import (
    annealed_regularization,
    full_nig_nll_torch,
    independent_atomic_moment_loss_torch,
    nig_predictive_parameters,
    nig_variances,
    student_t_nll,
)
from uq_benchmark.training import _loss, _prediction_error_sums, _prediction_metrics
from uq_benchmark.modeling import (
    AtomwiseNIGReadout,
    GraphNIGReadout,
    aggregate_independent_atomic_nig_moments,
)


def test_full_nig_matches_student_t_marginal():
    y = np.array([-1.0, 0.2, 2.1])
    gamma = np.array([-0.8, 0.0, 2.0])
    nu = np.array([0.5, 2.0, 5.0])
    alpha = np.array([1.5, 3.0, 8.0])
    beta = np.array([0.2, 1.0, 3.0])
    location, scale, df = nig_predictive_parameters(gamma, nu, alpha, beta)
    expected = -t.logpdf(y, df=df, loc=location, scale=scale)
    numpy_nll = student_t_nll(y, location, scale, df)
    torch_nll = full_nig_nll_torch(
        *[torch.tensor(value, dtype=torch.float64) for value in (y, gamma, nu, alpha, beta)],
        reduction="none",
    ).numpy()
    np.testing.assert_allclose(numpy_nll, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(torch_nll, expected, rtol=1e-12, atol=1e-12)


def test_nig_variance_decomposition_and_schedule():
    aleatoric, epistemic, total = nig_variances(np.array([2.0]), np.array([3.0]), np.array([4.0]))
    np.testing.assert_allclose(aleatoric, [2.0])
    np.testing.assert_allclose(epistemic, [1.0])
    np.testing.assert_allclose(total, [3.0])
    assert annealed_regularization(0.01, 0, 20) == 0.0005
    assert annealed_regularization(0.01, 19, 20) == 0.01
    assert annealed_regularization(0.01, 200, 20) == 0.01


def test_deterministic_force_loss_is_attached():
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    output = {"energy": parameter[None] * 0.0, "forces": parameter.expand(2, 3)}
    batch = SimpleNamespace(energy=torch.zeros(1), forces=torch.zeros(2, 3))
    loss, _, force_loss = _loss("deterministic", output, batch, force_weight=100.0, regularization=0.0)
    loss.backward()
    assert force_loss.requires_grad
    assert parameter.grad is not None and parameter.grad.item() != 0.0


def test_conditional_huber_force_loss_is_attached():
    output = {
        "energy": torch.tensor([0.0], requires_grad=True),
        "forces": torch.tensor([[0.2, -0.1, 0.0]], requires_grad=True),
    }
    batch = SimpleNamespace(
        energy=torch.tensor([0.0]),
        forces=torch.zeros((1, 3)),
    )
    loss, energy_loss, force_loss = _loss(
        "deterministic",
        output,
        batch,
        force_weight=10.0,
        regularization=0.0,
        force_loss_kind="conditional_huber",
        force_huber_delta=0.01,
    )
    loss.backward()
    assert energy_loss.item() == 0.0
    assert force_loss.item() > 0.0
    assert output["forces"].grad is not None


def test_deterministic_huber_energy_loss_uses_per_atom_error():
    output = {
        "energy": torch.tensor([3.0, 6.0], requires_grad=True),
        "forces": torch.zeros((3, 3), requires_grad=True),
    }
    batch = SimpleNamespace(
        energy=torch.tensor([1.0, 2.0]),
        forces=torch.zeros((3, 3)),
        ptr=torch.tensor([0, 1, 3]),
    )
    loss, energy_loss, _ = _loss(
        "deterministic",
        output,
        batch,
        force_weight=10.0,
        regularization=0.0,
        force_loss_kind="conditional_huber",
        force_huber_delta=0.01,
        energy_loss_kind="huber_per_atom",
    )
    expected = torch.nn.functional.huber_loss(
        torch.tensor([3.0, 3.0]),
        torch.tensor([1.0, 1.0]),
        delta=0.01,
    )
    torch.testing.assert_close(energy_loss, expected)
    loss.backward()
    assert output["energy"].grad is not None


def test_prediction_metrics_use_energy_per_atom_and_force_components():
    output = {
        "energy": torch.tensor([4.0, 2.0]),
        "forces": torch.tensor([[1.0, -1.0, 0.0], [2.0, 0.0, -2.0]]),
    }
    batch = SimpleNamespace(
        energy=torch.tensor([2.0, 5.0]),
        forces=torch.zeros(2, 3),
        ptr=torch.tensor([0, 2, 3]),
    )
    metrics = _prediction_metrics(_prediction_error_sums(output, batch))
    np.testing.assert_allclose(metrics, (2.0, np.sqrt(5.0), 1.0, np.sqrt(5.0 / 3.0)))


def test_graph_nig_constraints_are_applied_after_size_invariant_pooling():
    class FakeBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 1
            self.linear_1 = torch.nn.Identity()
            self.non_linearity = torch.nn.Identity()
            self.gamma_head = torch.nn.Linear(1, 1, bias=False)
            self.nu_head = torch.nn.Linear(1, 1, bias=False)
            self.alpha_head = torch.nn.Linear(1, 1, bias=False)
            self.beta_head = torch.nn.Linear(1, 1, bias=False)

    readout = GraphNIGReadout(FakeBase())
    node_features = torch.ones(5, 1)
    batch = torch.tensor([0, 1, 1, 1, 1])
    nu, alpha, beta = readout.graph_parameters(node_features, batch, num_graphs=2)
    torch.testing.assert_close(nu[0], nu[1])
    torch.testing.assert_close(alpha[0], alpha[1])
    torch.testing.assert_close(beta[0], beta[1])
    assert torch.all(nu > 0.0)
    assert torch.all(alpha > 1.0)
    assert torch.all(beta > 0.0)
    assert alpha[1] < 4.0


def test_atomwise_nig_readout_returns_one_valid_tuple_per_atom():
    class FakeBase(torch.nn.Module):
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
                layer.weight.data.fill_(1.0)

    readout = AtomwiseNIGReadout(FakeBase())
    node_features = torch.tensor([[0.0], [1.0], [2.0]])
    gamma, nu, alpha, beta = readout(node_features)
    assert gamma.shape == nu.shape == alpha.shape == beta.shape == (3,)
    assert not torch.equal(nu[0], nu[1])
    assert torch.all(nu > 0)
    assert torch.all(alpha > 1)
    assert torch.all(beta > 0)


def test_atomwise_moments_sum_variances_without_summing_nig_parameters():
    gamma = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    nu = torch.tensor([2.0, 4.0, 1.0], dtype=torch.float64)
    alpha = torch.tensor([3.0, 3.0, 5.0], dtype=torch.float64)
    beta = torch.tensor([4.0, 2.0, 8.0], dtype=torch.float64)
    batch = torch.tensor([0, 0, 1])
    result = aggregate_independent_atomic_nig_moments(
        gamma, nu, alpha, beta, batch, num_graphs=2
    )
    torch.testing.assert_close(
        result["graph_atomic_gamma_sum"],
        torch.tensor([3.0, 4.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result["energy_aleatoric_variance"],
        torch.tensor([3.0, 2.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result["energy_epistemic_variance"],
        torch.tensor([1.25, 2.0], dtype=torch.float64),
    )
    assert "energy_nu" not in result
    assert "energy_alpha" not in result
    assert "energy_beta" not in result


def test_atomwise_moment_loss_backpropagates_to_atomic_evidence():
    target = torch.tensor([1.0])
    location = torch.tensor([0.5], requires_grad=True)
    aleatoric = torch.tensor([0.4], requires_grad=True)
    epistemic = torch.tensor([0.2], requires_grad=True)
    atomic_nu = torch.tensor([1.0, 2.0], requires_grad=True)
    atomic_alpha = torch.tensor([2.0, 3.0], requires_grad=True)
    loss = independent_atomic_moment_loss_torch(
        target,
        location,
        aleatoric,
        epistemic,
        atomic_nu,
        atomic_alpha,
        torch.tensor([0, 0]),
        torch.tensor([2.0]),
        regularization=1e-3,
    )
    loss.backward()
    for value in (location, aleatoric, epistemic, atomic_nu, atomic_alpha):
        assert value.grad is not None
