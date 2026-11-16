#!/usr/bin/env python3
"""Run a CPU/CUDA integration smoke test against the bundled MACE source."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uq_benchmark.config import ModelConfig
from uq_benchmark.distributions import (
    full_nig_loss_torch,
    independent_atomic_moment_loss_torch,
)
from uq_benchmark.epic_pipeline import sampled_force_statistics
from uq_benchmark.hierarchical_epic import hierarchical_epic_energy_loss
from uq_benchmark.modeling import assert_matched_backbones, configure_mace_source, create_model


def main() -> int:
    configure_mace_source()
    from mace import data as mace_data
    from mace.tools import AtomicNumberTable, torch_geometric

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ModelConfig()
    deterministic = create_model("deterministic", config, avg_num_neighbors=4.0).to(device)
    epic = create_model("epic", config, avg_num_neighbors=4.0).to(device)
    atomwise_epic = create_model("epic_atomwise", config, avg_num_neighbors=4.0).to(device)
    hierarchical_epic = create_model(
        "epic_hierarchical", config, avg_num_neighbors=4.0
    ).to(device)
    assert_matched_backbones(deterministic, epic)
    assert_matched_backbones(deterministic, atomwise_epic)
    assert_matched_backbones(deterministic, hierarchical_epic)

    atoms = Atoms("HHe", positions=[[0, 0, 0], [0, 0, 0.75]], cell=[8, 8, 8], pbc=True)
    atoms.calc = SinglePointCalculator(atoms, energy=-1.0, forces=np.zeros((2, 3)))
    data = mace_data.AtomicData.from_config(
        mace_data.config_from_atoms(atoms, head_name="Default"),
        z_table=AtomicNumberTable(list(range(1, 119))),
        cutoff=config.r_max,
        heads=["Default"],
    )
    batch = next(
        iter(torch_geometric.dataloader.DataLoader([data], batch_size=1, shuffle=False))
    ).to(device)

    deterministic.zero_grad(set_to_none=True)
    output = deterministic(batch, training=True)
    loss = torch.nn.functional.mse_loss(output["energy"], batch.energy)
    force_loss = torch.nn.functional.mse_loss(output["forces"], batch.forces)
    (loss + 100.0 * force_loss).backward()
    force_gradients = [
        parameter.grad
        for name, parameter in deterministic.named_parameters()
        if name.startswith("interactions.") and parameter.grad is not None
    ]
    if not force_gradients or not any(torch.any(gradient != 0) for gradient in force_gradients):
        raise AssertionError("force loss did not produce a nonzero backbone gradient")

    epic.zero_grad(set_to_none=True)
    output = epic(batch, training=True)
    epic_loss = full_nig_loss_torch(
        batch.energy,
        output["energy"],
        output["energy_nu"],
        output["energy_alpha"],
        output["energy_beta"],
        1e-3,
    ) + 100.0 * torch.nn.functional.mse_loss(output["forces"], batch.forces)
    epic_loss.backward()
    for key in ("energy_nu", "energy_alpha", "energy_beta"):
        if not torch.all(torch.isfinite(output[key])):
            raise AssertionError(f"non-finite EPIC output: {key}")
    if not torch.all(output["energy_nu"] > 0):
        raise AssertionError("EPIC nu is outside the NIG domain")
    if not torch.all(output["energy_alpha"] > 1):
        raise AssertionError("EPIC alpha is outside the NIG domain")
    if not torch.all(output["energy_beta"] > 0):
        raise AssertionError("EPIC beta is outside the NIG domain")

    with torch.enable_grad():
        point_output = epic(batch, training=False, compute_force=True)
        point_force, force_scale, mc_force_mean, sampled_energies = sampled_force_statistics(
            epic, batch, n_samples=4, seed=42
        )
    torch.testing.assert_close(point_force, point_output["forces"], rtol=2e-5, atol=2e-6)
    for name, value in (
        ("EPIC point force", point_force),
        ("EPIC MC force scale", force_scale),
        ("EPIC MC force mean", mc_force_mean),
        ("EPIC sampled energies", sampled_energies),
    ):
        if not torch.all(torch.isfinite(value)):
            raise AssertionError(f"non-finite {name}")
    if sampled_energies.shape != (4,):
        raise AssertionError(f"unexpected EPIC sampled-energy shape: {sampled_energies.shape}")

    atomwise_epic.zero_grad(set_to_none=True)
    atomwise_output = atomwise_epic(batch, training=True)
    n_atoms = int(batch.positions.shape[0])
    for key in (
        "atomic_energy_gamma",
        "atomic_energy_nu",
        "atomic_energy_alpha",
        "atomic_energy_beta",
    ):
        if atomwise_output[key].shape != (n_atoms,):
            raise AssertionError(f"unexpected atomwise output shape for {key}")
    atomwise_loss = independent_atomic_moment_loss_torch(
        batch.energy,
        atomwise_output["energy"],
        atomwise_output["energy_aleatoric_variance"],
        atomwise_output["energy_epistemic_variance"],
        atomwise_output["atomic_energy_nu"],
        atomwise_output["atomic_energy_alpha"],
        batch.batch,
        (batch.ptr[1:] - batch.ptr[:-1]).to(atomwise_output["energy"].dtype),
        1e-3,
    ) + 100.0 * torch.nn.functional.mse_loss(
        atomwise_output["forces"], batch.forces
    )
    atomwise_loss.backward()
    with torch.enable_grad():
        atomwise_point = atomwise_epic(batch, training=False, compute_force=True)
        atomwise_force, atomwise_scale, _, atomwise_energies = sampled_force_statistics(
            atomwise_epic, batch, n_samples=4, seed=42
        )
    torch.testing.assert_close(
        atomwise_force, atomwise_point["forces"], rtol=2e-5, atol=2e-6
    )
    if not torch.all(torch.isfinite(atomwise_scale)):
        raise AssertionError("non-finite atomwise EPIC force scale")
    if atomwise_energies.shape != (4,):
        raise AssertionError("unexpected atomwise EPIC sampled-energy shape")

    hierarchical_epic.zero_grad(set_to_none=True)
    hierarchical_output = hierarchical_epic(batch, training=True)
    allocation_force_target = batch.forces.detach().clone()
    allocation_force_target[0, 0] += 1.0
    hierarchical_energy_loss, _, allocation_loss = hierarchical_epic_energy_loss(
        batch.energy,
        hierarchical_output,
        allocation_force_target,
        batch.batch,
        regularization=1e-3,
        allocation_weight=0.1,
        force_error_floor=1e-6,
    )
    hierarchical_loss = hierarchical_energy_loss + 100.0 * torch.nn.functional.mse_loss(
        hierarchical_output["forces"], batch.forces
    )
    hierarchical_loss.backward()
    allocation_gradients = [
        parameter.grad
        for name, parameter in hierarchical_epic.named_parameters()
        if "epistemic_allocation_head" in name
    ]
    if not allocation_gradients or not any(
        gradient is not None and torch.any(gradient != 0)
        for gradient in allocation_gradients
    ):
        raise AssertionError("hierarchical allocation head did not receive a gradient")
    if not torch.isfinite(allocation_loss):
        raise AssertionError("non-finite hierarchical allocation loss")

    graph_epistemic = hierarchical_output["energy_epistemic_variance"]
    reconstructed_epistemic = hierarchical_output[
        "atomic_epistemic_variance"
    ].sum().reshape_as(graph_epistemic)
    torch.testing.assert_close(graph_epistemic, reconstructed_epistemic)
    torch.testing.assert_close(
        hierarchical_output["energy"],
        hierarchical_output["atomic_energy_gamma"].sum().reshape_as(
            hierarchical_output["energy"]
        ),
    )
    for key in (
        "atomic_aleatoric_variance",
        "atomic_epistemic_variance",
        "atomic_total_variance",
    ):
        if hierarchical_output[key].shape != (n_atoms,):
            raise AssertionError(f"unexpected hierarchical output shape for {key}")
        if not torch.all(torch.isfinite(hierarchical_output[key])):
            raise AssertionError(f"non-finite hierarchical output: {key}")
    with torch.enable_grad():
        hierarchical_point = hierarchical_epic(
            batch, training=False, compute_force=True
        )
        hierarchical_force, hierarchical_scale, _, hierarchical_energies = (
            sampled_force_statistics(
                hierarchical_epic, batch, n_samples=4, seed=42
            )
        )
    torch.testing.assert_close(
        hierarchical_force,
        hierarchical_point["forces"],
        rtol=2e-5,
        atol=2e-6,
    )
    if not torch.all(torch.isfinite(hierarchical_scale)):
        raise AssertionError("non-finite hierarchical EPIC force scale")
    if hierarchical_energies.shape != (4,):
        raise AssertionError("unexpected hierarchical sampled-energy shape")
    print(f"bundled MACE integration smoke test passed on {device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
