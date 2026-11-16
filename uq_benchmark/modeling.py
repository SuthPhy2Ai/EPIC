from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .config import ModelConfig
from .hierarchical_epic import HierarchicalAtomwiseEPIC


class GraphNIGReadout(torch.nn.Module):
    """Pool invariant node features before constructing one NIG per structure."""

    def __init__(self, base: torch.nn.Module, epsilon: float = 1e-6):
        super().__init__()
        if int(base.num_heads) != 1:
            raise ValueError("the benchmark graph NIG readout currently requires one MACE head")
        self.base = base
        self.epsilon = float(epsilon)

    def _hidden(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.base.non_linearity(self.base.linear_1(node_features))

    def forward(self, node_features: torch.Tensor, heads: torch.Tensor | None = None):
        """Provide atomic mean contributions to the vendored MACEEDL energy path."""
        del heads
        hidden = self._hidden(node_features)
        gamma = self.base.gamma_head(hidden).squeeze(-1)
        # MACEEDL aggregates these placeholders, but GraphNIGMACE replaces them.
        placeholder = torch.zeros_like(gamma)
        return gamma, placeholder, placeholder, placeholder

    def graph_parameters(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self._hidden(node_features)
        pooled = torch.zeros(
            (num_graphs, hidden.shape[-1]),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        pooled.index_add_(0, batch, hidden)
        counts = torch.bincount(batch, minlength=num_graphs).to(hidden.dtype).clamp_min(1.0)
        pooled = pooled / counts[:, None]

        raw_nu = self.base.nu_head(pooled).squeeze(-1)
        raw_alpha = self.base.alpha_head(pooled).squeeze(-1)
        raw_beta = self.base.beta_head(pooled).squeeze(-1)
        epsilon = torch.as_tensor(self.epsilon, dtype=hidden.dtype, device=hidden.device)
        nu = torch.nn.functional.softplus(raw_nu) + epsilon
        alpha = torch.nn.functional.softplus(raw_alpha) + 1.0 + epsilon
        beta = torch.nn.functional.softplus(raw_beta) + epsilon
        return nu, alpha, beta


class AtomwiseNIGReadout(torch.nn.Module):
    """Construct one valid NIG parameter tuple for every atom."""

    def __init__(self, base: torch.nn.Module, epsilon: float = 1e-6):
        super().__init__()
        if int(base.num_heads) != 1:
            raise ValueError("the benchmark atomwise NIG readout currently requires one MACE head")
        self.base = base
        self.epsilon = float(epsilon)

    def forward(self, node_features: torch.Tensor, heads: torch.Tensor | None = None):
        del heads
        hidden = self.base.non_linearity(self.base.linear_1(node_features))
        epsilon = torch.as_tensor(self.epsilon, dtype=hidden.dtype, device=hidden.device)
        gamma = self.base.gamma_head(hidden).squeeze(-1)
        nu = torch.nn.functional.softplus(self.base.nu_head(hidden).squeeze(-1)) + epsilon
        alpha = (
            torch.nn.functional.softplus(self.base.alpha_head(hidden).squeeze(-1))
            + 1.0
            + epsilon
        )
        beta = torch.nn.functional.softplus(self.base.beta_head(hidden).squeeze(-1)) + epsilon
        return gamma, nu, alpha, beta


def aggregate_independent_atomic_nig_moments(
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> dict[str, torch.Tensor]:
    """Aggregate atomic predictive moments without pretending NIG is additive.

    The variance sums assume conditionally independent atomic contributions.
    The resulting graph distribution is represented by moments, not by a
    fabricated graph-level NIG parameter tuple.
    """
    if not (gamma.shape == nu.shape == alpha.shape == beta.shape == batch.shape):
        raise ValueError("atomic NIG parameters and batch must be matching 1D tensors")
    if not all(torch.isfinite(value).all() for value in (gamma, nu, alpha, beta)):
        raise ValueError("atomic NIG parameters must be finite")
    if torch.any(nu <= 0) or torch.any(alpha <= 1) or torch.any(beta <= 0):
        raise ValueError("atomic NIG requires nu > 0, alpha > 1, beta > 0")

    atomic_aleatoric = beta / (alpha - 1.0)
    atomic_epistemic = beta / (nu * (alpha - 1.0))
    graph_gamma = gamma.new_zeros(num_graphs)
    graph_aleatoric = gamma.new_zeros(num_graphs)
    graph_epistemic = gamma.new_zeros(num_graphs)
    graph_gamma.index_add_(0, batch, gamma)
    graph_aleatoric.index_add_(0, batch, atomic_aleatoric)
    graph_epistemic.index_add_(0, batch, atomic_epistemic)
    return {
        "atomic_aleatoric_variance": atomic_aleatoric,
        "atomic_epistemic_variance": atomic_epistemic,
        "atomic_total_variance": atomic_aleatoric + atomic_epistemic,
        "graph_atomic_gamma_sum": graph_gamma,
        "energy_aleatoric_variance": graph_aleatoric,
        "energy_epistemic_variance": graph_epistemic,
        "energy_total_variance": graph_aleatoric + graph_epistemic,
    }


class GraphNIGMACE(torch.nn.Module):
    """Benchmark adapter that replaces invalid atomwise NIG summation."""

    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        base_model.edl_readout = GraphNIGReadout(base_model.edl_readout)
        self.base_model = base_model

    def forward(self, data, *args, **kwargs):
        output = self.base_model(data, *args, **kwargs)
        num_graphs = int(output["energy"].shape[0])
        nu, alpha, beta = self.base_model.edl_readout.graph_parameters(
            output["node_feats"], data["batch"], num_graphs
        )
        output["energy_nu"] = nu
        output["energy_alpha"] = alpha
        output["energy_beta"] = beta
        return output


class AtomwiseNIGMACE(torch.nn.Module):
    """MACE potential with real per-atom NIG outputs and moment aggregation."""

    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        base_model.edl_readout = AtomwiseNIGReadout(base_model.edl_readout)
        self.base_model = base_model

    def forward(self, data, *args, **kwargs):
        output = self.base_model(data, *args, **kwargs)
        gamma, nu, alpha, beta = self.base_model.edl_readout(output["node_feats"], None)
        num_graphs = int(output["energy"].shape[0])
        moments = aggregate_independent_atomic_nig_moments(
            gamma, nu, alpha, beta, data["batch"], num_graphs
        )

        # MACEEDL's legacy graph NIG values are direct atomwise sums. Remove
        # them so no downstream code can mistake those sums for a valid NIG.
        output.pop("energy_nu", None)
        output.pop("energy_alpha", None)
        output.pop("energy_beta", None)
        output.update(
            atomic_energy_gamma=gamma,
            atomic_energy_nu=nu,
            atomic_energy_alpha=alpha,
            atomic_energy_beta=beta,
            **moments,
        )
        return output


def configure_mace_source() -> Path:
    candidates = []
    if os.environ.get("MACE_SOURCE_ROOT"):
        candidates.append(Path(os.environ["MACE_SOURCE_ROOT"]))
    repo_root = Path(__file__).resolve().parents[1]
    candidates.append(repo_root)
    for candidate in candidates:
        if (candidate / "mace" / "modules" / "models.py").exists():
            sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError(
        "MACE source with MACEEDL support was not found; use the bundled source "
        "checkout or set MACE_SOURCE_ROOT"
    )


ModelKind = Literal[
    "deterministic",
    "epic",
    "epic_atomwise",
    "epic_hierarchical",
]


def create_model(kind: ModelKind, config: ModelConfig, avg_num_neighbors: float):
    configure_mace_source()
    if hasattr(torch.serialization, "safe_globals"):
        with torch.serialization.safe_globals([slice]):
            from e3nn import o3
    else:  # PyTorch versions before safe_globals default to full loading.
        from e3nn import o3
    from mace.modules.blocks import RealAgnosticResidualInteractionBlock
    from mace.modules.models import MACE, MACEEDL

    cls = MACE if kind == "deterministic" else MACEEDL
    model = cls(
        r_max=config.r_max,
        num_bessel=config.num_bessel,
        num_polynomial_cutoff=config.num_polynomial_cutoff,
        max_ell=config.max_ell,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        interaction_cls_first=RealAgnosticResidualInteractionBlock,
        num_interactions=config.num_interactions,
        num_elements=config.num_elements,
        hidden_irreps=o3.Irreps(f"{config.num_channels}x0e + {config.num_channels}x1o"),
        MLP_irreps=o3.Irreps(f"{config.num_channels}x0e"),
        atomic_energies=np.zeros(config.num_elements),
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=list(range(1, config.num_elements + 1)),
        correlation=config.correlation,
        gate=torch.nn.functional.silu,
    )
    if kind == "epic":
        return GraphNIGMACE(model)
    if kind == "epic_atomwise":
        return AtomwiseNIGMACE(model)
    if kind == "epic_hierarchical":
        return HierarchicalAtomwiseEPIC(model)
    return model


def backbone_state_keys(model) -> set[str]:
    keys = set()
    for key in model.state_dict():
        normalized = key.removeprefix("base_model.")
        if normalized.startswith("edl_readout.") or normalized.startswith("readouts."):
            continue
        keys.add(normalized)
    return keys


def _normalized_state(model) -> dict[str, torch.Tensor]:
    return {key.removeprefix("base_model."): value for key, value in model.state_dict().items()}


def assert_matched_backbones(deterministic, epic) -> None:
    deterministic_state = _normalized_state(deterministic)
    epic_state = _normalized_state(epic)
    left = backbone_state_keys(deterministic)
    right = backbone_state_keys(epic)
    if left != right:
        raise AssertionError(f"backbone keys differ: only deterministic={left-right}, only EPIC={right-left}")
    mismatched = [key for key in left if deterministic_state[key].shape != epic_state[key].shape]
    if mismatched:
        raise AssertionError(f"backbone tensor shapes differ: {mismatched}")
