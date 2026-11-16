"""Isolated hierarchical atom-wise EPIC implementation.

The graph-level NIG is the only distribution trained against total energy.
Atom-wise NIG marginals are derived from normalized uncertainty-allocation
weights and a shared graph scale, so their variances reconstruct the graph
aleatoric and epistemic variances exactly.
"""

from __future__ import annotations

import copy

import torch

from .distributions import full_nig_loss_torch


VARIANT = "hierarchical_atomwise_shared_scale"
SAMPLING_METHOD = "hierarchical_atomwise_shared_scale_latent_mean_rsample"
UNIFORM_SAMPLING_METHOD = "hierarchical_uniform_shared_scale_latent_mean_rsample"


def segment_normalize(
    values: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """Normalize positive atomic values independently inside each graph."""
    if values.ndim != 1 or batch.shape != values.shape:
        raise ValueError("values and batch must be matching 1D tensors")
    if torch.any(values <= 0) or not torch.isfinite(values).all():
        raise ValueError("segment-normalized values must be finite and positive")
    denominators = values.new_zeros(num_graphs)
    denominators.index_add_(0, batch, values)
    return values / denominators[batch]


def segment_softmax(
    logits: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    minimum: float = 1e-8,
) -> torch.Tensor:
    """Stable graph-wise softmax with a positive floor and renormalization."""
    if logits.ndim != 1 or batch.shape != logits.shape:
        raise ValueError("logits and batch must be matching 1D tensors")
    if not torch.isfinite(logits).all():
        raise ValueError("allocation logits must be finite")
    weights = logits.new_zeros(logits.shape)
    for graph_index in range(num_graphs):
        indices = torch.nonzero(batch == graph_index, as_tuple=False).flatten()
        if indices.numel() == 0:
            raise ValueError(f"graph {graph_index} has no atoms")
        local = torch.softmax(logits.index_select(0, indices), dim=0)
        weights = weights.index_copy(0, indices, local)
    floor = torch.as_tensor(minimum, dtype=weights.dtype, device=weights.device)
    return segment_normalize(weights.clamp_min(floor), batch, num_graphs)


class HierarchicalEPICReadout(torch.nn.Module):
    """Atomic energy head plus graph NIG and atomic allocation heads."""

    def __init__(self, base: torch.nn.Module, epsilon: float = 1e-6):
        super().__init__()
        if int(base.num_heads) != 1:
            raise ValueError("hierarchical EPIC currently requires one MACE head")
        self.base = base
        self.epsilon = float(epsilon)
        self.epistemic_allocation_head = copy.deepcopy(base.nu_head)
        for parameter in self.epistemic_allocation_head.parameters():
            torch.nn.init.zeros_(parameter)

    def _hidden(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.base.non_linearity(self.base.linear_1(node_features))

    def forward(self, node_features: torch.Tensor, heads: torch.Tensor | None = None):
        """Provide atomic mean contributions to the vendored MACEEDL path."""
        del heads
        hidden = self._hidden(node_features)
        gamma = self.base.gamma_head(hidden).squeeze(-1)
        placeholder = torch.zeros_like(gamma)
        return gamma, placeholder, placeholder, placeholder

    def hierarchical_parameters(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> dict[str, torch.Tensor]:
        hidden = self._hidden(node_features)
        pooled = hidden.new_zeros((num_graphs, hidden.shape[-1]))
        pooled.index_add_(0, batch, hidden)
        counts = torch.bincount(batch, minlength=num_graphs).to(hidden.dtype)
        if torch.any(counts <= 0):
            raise ValueError("every graph must contain at least one atom")
        pooled = pooled / counts[:, None]

        epsilon = torch.as_tensor(self.epsilon, dtype=hidden.dtype, device=hidden.device)
        graph_nu = torch.nn.functional.softplus(
            self.base.nu_head(pooled).squeeze(-1)
        ) + epsilon
        graph_alpha = (
            torch.nn.functional.softplus(self.base.alpha_head(pooled).squeeze(-1))
            + 1.0
            + epsilon
        )
        graph_beta = torch.nn.functional.softplus(
            self.base.beta_head(pooled).squeeze(-1)
        ) + epsilon

        epistemic_logits = self.epistemic_allocation_head(hidden).squeeze(-1)
        epistemic_weight = segment_softmax(
            epistemic_logits, batch, num_graphs
        )
        aleatoric_weight = 1.0 / counts[batch]

        atomic_alpha = graph_alpha[batch]
        atomic_beta = aleatoric_weight * graph_beta[batch]
        atomic_nu = (
            graph_nu[batch] * aleatoric_weight / epistemic_weight
        )

        graph_aleatoric = graph_beta / (graph_alpha - 1.0)
        graph_epistemic = graph_beta / (
            graph_nu * (graph_alpha - 1.0)
        )
        atomic_aleatoric = aleatoric_weight * graph_aleatoric[batch]
        atomic_epistemic = epistemic_weight * graph_epistemic[batch]
        return {
            "energy_nu": graph_nu,
            "energy_alpha": graph_alpha,
            "energy_beta": graph_beta,
            "energy_aleatoric_variance": graph_aleatoric,
            "energy_epistemic_variance": graph_epistemic,
            "energy_total_variance": graph_aleatoric + graph_epistemic,
            "atomic_energy_nu": atomic_nu,
            "atomic_energy_alpha": atomic_alpha,
            "atomic_energy_beta": atomic_beta,
            "atomic_aleatoric_weight": aleatoric_weight,
            "atomic_epistemic_weight": epistemic_weight,
            "atomic_aleatoric_variance": atomic_aleatoric,
            "atomic_epistemic_variance": atomic_epistemic,
            "atomic_total_variance": atomic_aleatoric + atomic_epistemic,
        }


class HierarchicalAtomwiseEPIC(torch.nn.Module):
    """MACE point potential with a shared-scale hierarchical EPIC head."""

    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        base_model.edl_readout = HierarchicalEPICReadout(base_model.edl_readout)
        self.base_model = base_model

    def forward(self, data, *args, **kwargs):
        output = self.base_model(data, *args, **kwargs)
        gamma, _, _, _ = self.base_model.edl_readout(output["node_feats"], None)
        num_graphs = int(output["energy"].shape[0])
        parameters = self.base_model.edl_readout.hierarchical_parameters(
            output["node_feats"], data["batch"], num_graphs
        )
        output.update(
            atomic_energy_gamma=gamma,
            epic_variant=VARIANT,
            **parameters,
        )
        return output


def force_error_allocation_loss(
    epistemic_weight: torch.Tensor,
    force_prediction: torch.Tensor,
    force_target: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    floor: float,
) -> torch.Tensor:
    """Train atomic attribution weights against a detached force-error proxy."""
    if force_prediction.shape != force_target.shape:
        raise ValueError("force prediction and target must have matching shapes")
    if force_prediction.ndim != 2 or force_prediction.shape[1] != 3:
        raise ValueError("force arrays must have shape [n_atoms, 3]")
    error_score = (
        (force_prediction.detach() - force_target.detach()).square().sum(dim=1)
        + float(floor)
    )
    target_weight = segment_normalize(error_score, batch, num_graphs)
    tiny = torch.finfo(epistemic_weight.dtype).tiny
    per_atom = -target_weight * torch.log(epistemic_weight.clamp_min(tiny))
    per_graph = per_atom.new_zeros(num_graphs)
    per_graph.index_add_(0, batch, per_atom)
    return per_graph.mean()


def hierarchical_epic_energy_loss(
    target_energy: torch.Tensor,
    output: dict,
    force_target: torch.Tensor,
    batch: torch.Tensor,
    regularization: float,
    allocation_weight: float,
    force_error_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total energy-side loss, graph NIG loss, and allocation loss."""
    graph_nig = full_nig_loss_torch(
        target_energy,
        output["energy"],
        output["energy_nu"],
        output["energy_alpha"],
        output["energy_beta"],
        regularization,
    )
    allocation = force_error_allocation_loss(
        output["atomic_epistemic_weight"],
        output["forces"],
        force_target,
        batch,
        int(target_energy.numel()),
        force_error_floor,
    )
    return (
        graph_nig + float(allocation_weight) * allocation,
        graph_nig,
        allocation,
    )


def sample_hierarchical_latent_energies(
    output: dict,
    *,
    n_samples: int,
    seed: int,
    allocation_mode: str = "learned",
) -> torch.Tensor:
    """Sample conservative latent-mean energies from one shared graph scale."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    if output["energy"].numel() != 1:
        raise ValueError("hierarchical sampling currently requires batch_size=1")
    nu = output["energy_nu"].reshape(())
    alpha = output["energy_alpha"].reshape(())
    beta = output["energy_beta"].reshape(())
    learned_weights = output["atomic_epistemic_weight"]
    if nu <= 0 or alpha <= 1 or beta <= 0:
        raise ValueError("graph NIG parameters are outside their domain")
    if not torch.isclose(
        learned_weights.sum(), learned_weights.new_tensor(1.0), rtol=1e-5, atol=1e-6
    ):
        raise ValueError("atomic epistemic weights must sum to one")
    if allocation_mode == "learned":
        weights = learned_weights
    elif allocation_mode == "uniform":
        weights = torch.full_like(learned_weights, 1.0 / learned_weights.numel())
    else:
        raise ValueError("allocation_mode must be learned or uniform")

    cuda_devices: list[int] = []
    if nu.is_cuda:
        cuda_devices = [
            nu.device.index
            if nu.device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        if nu.is_cuda:
            torch.cuda.manual_seed(int(seed))
        shared_variance = torch.distributions.InverseGamma(alpha, beta).rsample(
            (n_samples,)
        )
        epsilon = torch.randn(
            (n_samples, weights.numel()),
            dtype=weights.dtype,
            device=weights.device,
        )
    if not torch.isfinite(shared_variance).all():
        raise FloatingPointError(
            "hierarchical shared-scale sampling produced non-finite variances; "
            "inspect graph NIG alpha, beta, and nu"
        )
    atomic_delta = torch.sqrt(
        shared_variance[:, None] * weights[None, :] / nu
    ) * epsilon
    sampled_energies = output["energy"].reshape(()) + atomic_delta.sum(dim=1)
    if not torch.isfinite(sampled_energies).all():
        raise FloatingPointError(
            "hierarchical shared-scale sampling produced non-finite energies; "
            "inspect graph NIG tails and allocation weights"
        )
    return sampled_energies
