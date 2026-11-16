from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from .config import BenchmarkConfig
from .data import load_dataset, make_data_loader
from .epic_pipeline import sampled_force_ensemble
from .inference import load_checkpoint_model
from .split import hashed_subset, load_split


POINT_ENERGY_PER_ATOM_PERMUTATION_ATOL = 1e-5
POINT_FORCE_PERMUTATION_ATOL = 1e-4
ATOMIC_WEIGHT_PERMUTATION_ATOL = 1e-5


def _fresh_batch(batch):
    clone = batch.clone()
    clone.positions = batch.positions.detach().clone()
    return clone


def _displaced_batch(batch, atom: int, axis: int, displacement: float):
    clone = _fresh_batch(batch)
    clone.positions[atom, axis] += displacement
    return clone


def _permuted_batch(batch, permutation: torch.Tensor):
    clone = _fresh_batch(batch)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation), device=permutation.device)
    for field in ("positions", "node_attrs", "forces", "charges", "batch"):
        value = getattr(clone, field, None)
        if value is not None and value.shape[0] == len(permutation):
            setattr(clone, field, value.index_select(0, permutation))
    clone.edge_index = inverse[clone.edge_index]
    return clone, inverse


def audit_hierarchical_checkpoint(
    config: BenchmarkConfig,
    split_path: str | Path,
    checkpoint: str | Path,
    output_path: str | Path,
    *,
    split: str = "calibration",
    subset_size: int = 8,
    n_samples: int = 8,
    sample_seed: int = 42,
    finite_difference_step: float = 1e-2,
    device: str = "cuda",
) -> Path:
    if split not in ("validation", "calibration"):
        raise ValueError("checkpoint audit is restricted to validation or calibration")
    indices = hashed_subset(
        load_split(split_path, split, allow_test=False),
        subset_size,
        f"hierarchical-checkpoint-audit-{split}-v1",
    )
    model, state = load_checkpoint_model(checkpoint, device)
    if state["kind"] != "epic_hierarchical":
        raise ValueError("checkpoint audit requires an epic_hierarchical checkpoint")
    model.eval()
    loader = make_data_loader(
        Subset(load_dataset(config), indices.tolist()),
        config,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    rows = []
    generator = torch.Generator(device="cpu").manual_seed(20260808)
    for db_index, host_batch in zip(indices, loader, strict=True):
        batch = host_batch.to(device)
        n_atoms = int(batch.positions.shape[0])
        atom = int(db_index) % n_atoms
        axis_x = int(db_index) % 3
        axis_y = (axis_x + 1) % 3

        original = _fresh_batch(batch)
        original_output = model(original, training=False, compute_force=True)
        point_force, force_stack, sampled_energy = sampled_force_ensemble(
            model,
            _fresh_batch(batch),
            n_samples=n_samples,
            seed=sample_seed,
        )
        plus_force, plus_stack, plus_energy = sampled_force_ensemble(
            model,
            _displaced_batch(batch, atom, axis_x, finite_difference_step),
            n_samples=n_samples,
            seed=sample_seed,
        )
        minus_force, minus_stack, minus_energy = sampled_force_ensemble(
            model,
            _displaced_batch(batch, atom, axis_x, -finite_difference_step),
            n_samples=n_samples,
            seed=sample_seed,
        )
        del plus_force, minus_force, minus_stack
        sampled_fd = -(plus_energy - minus_energy) / (2.0 * finite_difference_step)
        sampled_gradient_error = torch.abs(sampled_fd - force_stack[:, atom, axis_x])
        sampled_gradient_relative_error = sampled_gradient_error / torch.clamp(
            torch.maximum(torch.abs(sampled_fd), torch.abs(force_stack[:, atom, axis_x])),
            min=1e-6,
        )

        plus_y_force, plus_y_stack, _ = sampled_force_ensemble(
            model,
            _displaced_batch(batch, atom, axis_y, finite_difference_step),
            n_samples=n_samples,
            seed=sample_seed,
        )
        diagonal = _displaced_batch(batch, atom, axis_x, finite_difference_step)
        diagonal.positions[atom, axis_y] += finite_difference_step
        diagonal_force, diagonal_stack, _ = sampled_force_ensemble(
            model,
            diagonal,
            n_samples=n_samples,
            seed=sample_seed,
        )
        del plus_y_force, diagonal_force
        h = finite_difference_step
        loop_work = (
            0.5 * (force_stack[:, atom, axis_x] + plus_stack[:, atom, axis_x]) * h
            + 0.5 * (plus_stack[:, atom, axis_y] + diagonal_stack[:, atom, axis_y]) * h
            - 0.5 * (diagonal_stack[:, atom, axis_x] + plus_y_stack[:, atom, axis_x]) * h
            - 0.5 * (plus_y_stack[:, atom, axis_y] + force_stack[:, atom, axis_y]) * h
        )

        permutation = torch.randperm(n_atoms, generator=generator).to(device)
        permuted, inverse = _permuted_batch(batch, permutation)
        permuted_output = model(permuted, training=False, compute_force=True)
        _, permuted_force_stack, permuted_sampled_energy = sampled_force_ensemble(
            model,
            _permuted_batch(batch, permutation)[0],
            n_samples=n_samples,
            seed=sample_seed,
        )
        restored_point_force = permuted_output["forces"].index_select(0, inverse)
        restored_weights = permuted_output["atomic_epistemic_weight"].index_select(0, inverse)
        restored_sampled_force = permuted_force_stack.index_select(1, inverse)

        point_energy = original_output["energy"].sum()
        plus_point_energy = model(
            _displaced_batch(batch, atom, axis_x, finite_difference_step),
            training=False,
            compute_force=False,
        )["energy"].sum()
        minus_point_energy = model(
            _displaced_batch(batch, atom, axis_x, -finite_difference_step),
            training=False,
            compute_force=False,
        )["energy"].sum()
        point_fd = -(plus_point_energy - minus_point_energy) / (2.0 * finite_difference_step)
        point_gradient_error = torch.abs(point_fd - point_force[atom, axis_x])
        point_gradient_relative_error = point_gradient_error / torch.clamp(
            torch.maximum(torch.abs(point_fd), torch.abs(point_force[atom, axis_x])),
            min=1e-6,
        )
        point_energy_permutation_error = torch.abs(
            point_energy - permuted_output["energy"].sum()
        )
        rows.append(
            {
                "db_index": int(db_index),
                "natoms": n_atoms,
                "atom": atom,
                "axis": axis_x,
                "point_force_autograd_absolute_error": float(
                    torch.abs(point_force[atom, axis_x] - original_output["forces"][atom, axis_x]).cpu()
                ),
                "point_force_finite_difference_absolute_error": float(
                    point_gradient_error.cpu()
                ),
                "point_force_finite_difference_relative_error": float(
                    point_gradient_relative_error.cpu()
                ),
                "sampled_force_finite_difference_median_absolute_error": float(
                    sampled_gradient_error.median().cpu()
                ),
                "sampled_force_finite_difference_max_absolute_error": float(
                    sampled_gradient_error.max().cpu()
                ),
                "sampled_force_finite_difference_median_relative_error": float(
                    sampled_gradient_relative_error.median().cpu()
                ),
                "sampled_force_finite_difference_max_relative_error": float(
                    sampled_gradient_relative_error.max().cpu()
                ),
                "sampled_closed_loop_median_absolute_work": float(loop_work.abs().median().cpu()),
                "sampled_closed_loop_max_absolute_work": float(loop_work.abs().max().cpu()),
                "point_energy_permutation_absolute_error": float(
                    point_energy_permutation_error.cpu()
                ),
                "point_energy_permutation_per_atom_absolute_error": float(
                    (point_energy_permutation_error / n_atoms).cpu()
                ),
                "point_force_permutation_max_absolute_error": float(
                    torch.max(torch.abs(point_force - restored_point_force)).cpu()
                ),
                "atomic_weight_permutation_max_absolute_error": float(
                    torch.max(
                        torch.abs(
                            original_output["atomic_epistemic_weight"] - restored_weights
                        )
                    ).cpu()
                ),
                "same_sample_id_energy_permutation_max_absolute_difference": float(
                    torch.max(torch.abs(sampled_energy - permuted_sampled_energy)).cpu()
                ),
                "same_sample_id_force_permutation_max_absolute_difference": float(
                    torch.max(torch.abs(force_stack - restored_sampled_force)).cpu()
                ),
            }
        )

    summary = {
        "schema_version": 2,
        "checkpoint_kind": state["kind"],
        "split": split,
        "n_structures": len(rows),
        "n_samples": n_samples,
        "sample_seed": sample_seed,
        "finite_difference_step_angstrom": finite_difference_step,
        "rows": rows,
        "max_point_autograd_error": max(row["point_force_autograd_absolute_error"] for row in rows),
        "max_point_finite_difference_error": max(
            row["point_force_finite_difference_absolute_error"] for row in rows
        ),
        "max_point_finite_difference_relative_error": max(
            row["point_force_finite_difference_relative_error"] for row in rows
        ),
        "max_sampled_finite_difference_error": max(
            row["sampled_force_finite_difference_max_absolute_error"] for row in rows
        ),
        "max_sampled_finite_difference_relative_error": max(
            row["sampled_force_finite_difference_max_relative_error"] for row in rows
        ),
        "max_sampled_closed_loop_absolute_work": max(
            row["sampled_closed_loop_max_absolute_work"] for row in rows
        ),
        "point_prediction_permutation_equivariance_verified": bool(
            max(
                row["point_energy_permutation_per_atom_absolute_error"] for row in rows
            )
            <= POINT_ENERGY_PER_ATOM_PERMUTATION_ATOL
            and max(row["point_force_permutation_max_absolute_error"] for row in rows)
            <= POINT_FORCE_PERMUTATION_ATOL
        ),
        "atomic_weight_permutation_equivariance_verified": bool(
            max(row["atomic_weight_permutation_max_absolute_error"] for row in rows)
            <= ATOMIC_WEIGHT_PERMUTATION_ATOL
        ),
        "permutation_tolerances": {
            "point_energy_per_atom_absolute_error": POINT_ENERGY_PER_ATOM_PERMUTATION_ATOL,
            "point_force_absolute_error": POINT_FORCE_PERMUTATION_ATOL,
            "atomic_weight_absolute_error": ATOMIC_WEIGHT_PERMUTATION_ATOL,
        },
        "same_sample_id_permutation_equivariance_verified": bool(
            max(
                row["same_sample_id_energy_permutation_max_absolute_difference"]
                for row in rows
            )
            <= 1e-4
            and max(
                row["same_sample_id_force_permutation_max_absolute_difference"]
                for row in rows
            )
            <= 1e-4
        ),
        "global_random_pes_verified": False,
        "interpretation": (
            "Finite differences and loop work test local fixed-topology executions. "
            "They do not establish one global random PES across independent configurations."
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return output
