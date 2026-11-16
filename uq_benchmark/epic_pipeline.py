from __future__ import annotations

import time
import json
import os
from pathlib import Path

import numpy as np
import torch

from maceedl_analysis.core.epic_sampling import (
    compose_sampled_total_energies,
    sample_atomic_nig_means,
)

from .config import BenchmarkConfig
from .data import load_dataset, make_data_loader
from .hierarchical_epic import (
    SAMPLING_METHOD as HIERARCHICAL_SAMPLING_METHOD,
    UNIFORM_SAMPLING_METHOD as HIERARCHICAL_UNIFORM_SAMPLING_METHOD,
    sample_hierarchical_latent_energies,
)
from .inference import load_checkpoint_model
from .split import hashed_subset, load_split


SAMPLING_METHOD = "graph_nig_latent_mean_rsample"
ATOMWISE_SAMPLING_METHOD = "atomwise_nig_latent_mean_rsample"
UNCERTAINTY_COMPONENT = "epistemic"


def sampled_force_statistics(
    model,
    batch,
    *,
    n_samples: int,
    seed: int,
    allocation_mode: str = "learned",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return point force, MC force std, MC force mean, and sampled energies."""
    point_force, force_stack, sampled_energies = sampled_force_ensemble(
        model,
        batch,
        n_samples=n_samples,
        seed=seed,
        allocation_mode=allocation_mode,
    )
    return (
        point_force,
        force_stack.std(dim=0, unbiased=True),
        force_stack.mean(dim=0),
        sampled_energies,
    )


def sampled_force_ensemble(
    model,
    batch,
    *,
    n_samples: int,
    seed: int,
    allocation_mode: str = "learned",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return point force, sampled energy-gradient forces, and sampled energies."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    batch.positions.requires_grad_(True)
    output = model(batch, training=False, compute_force=False)
    if output["energy"].numel() != 1:
        raise ValueError("EPIC force sampling requires batch_size=1")
    if "atomic_epistemic_weight" in output:
        sampled_energies = sample_hierarchical_latent_energies(
            output,
            n_samples=n_samples,
            seed=seed,
            allocation_mode=allocation_mode,
        )
    elif "atomic_energy_nu" in output:
        sampled_atomic_means = sample_atomic_nig_means(
            output["atomic_energy_gamma"],
            output["atomic_energy_nu"],
            output["atomic_energy_alpha"],
            output["atomic_energy_beta"],
            n_samples=n_samples,
            seed=seed,
        )
        sampled_energies = compose_sampled_total_energies(
            output["energy"].sum(),
            output["atomic_energy_gamma"],
            sampled_atomic_means,
        )
    else:
        sampled_graph_means = sample_atomic_nig_means(
            output["energy"],
            output["energy_nu"],
            output["energy_alpha"],
            output["energy_beta"],
            n_samples=n_samples,
            seed=seed,
        )
        sampled_energies = sampled_graph_means[:, 0]
    if not torch.isfinite(sampled_energies).all():
        raise FloatingPointError(
            "EPIC sampling produced non-finite energies; inspect NIG tails and "
            "increase numerical safeguards before using force uncertainty"
        )
    mean_energy = output["energy"].sum()

    mean_gradient = torch.autograd.grad(mean_energy, batch.positions, retain_graph=True)[0]
    sampled_forces = []
    for sample_index, sampled_energy in enumerate(sampled_energies):
        gradient = torch.autograd.grad(
            sampled_energy,
            batch.positions,
            retain_graph=sample_index + 1 < len(sampled_energies),
            create_graph=False,
        )[0]
        sampled_forces.append(-gradient.detach())
    force_stack = torch.stack(sampled_forces)
    if not torch.isfinite(mean_gradient).all() or not torch.isfinite(force_stack).all():
        raise FloatingPointError(
            "EPIC energy gradients produced non-finite point or sampled forces"
        )
    return -mean_gradient.detach(), force_stack, sampled_energies.detach()


def sample_epic_force_subset(
    config: BenchmarkConfig,
    split_path: str | Path,
    split: str,
    checkpoint: str | Path,
    output_path: str | Path,
    subset_size: int,
    device: str = "cuda",
    shard: int = 0,
    total_shards: int = 1,
    n_samples: int | None = None,
    sample_seed: int | None = None,
    allocation_mode: str = "learned",
) -> Path:
    if split not in ("calibration", "test"):
        raise ValueError("EPIC force uncertainty is only evaluated on calibration or test")
    if not 0 <= shard < total_shards:
        raise ValueError("invalid EPIC force shard")
    n_samples = config.epic.force_samples if n_samples is None else int(n_samples)
    sample_seed = config.epic.sample_seed if sample_seed is None else int(sample_seed)
    if n_samples < 2:
        raise ValueError("EPIC force sampling requires at least two samples")
    if allocation_mode not in ("learned", "uniform"):
        raise ValueError("allocation_mode must be learned or uniform")
    all_indices = load_split(split_path, split, allow_test=split == "test")
    selected_indices = hashed_subset(all_indices, subset_size, f"epic-force-{split}-v1")
    indices = selected_indices[shard::total_shards]
    if len(indices) == 0:
        raise ValueError("EPIC force shard is empty; reduce total_shards or increase subset size")
    model, state = load_checkpoint_model(checkpoint, device)
    if state["kind"] not in ("epic", "epic_atomwise", "epic_hierarchical"):
        raise ValueError("EPIC force sampling requires an EPIC checkpoint")
    if state["kind"] == "epic_hierarchical":
        sampling_method = (
            HIERARCHICAL_SAMPLING_METHOD
            if allocation_mode == "learned"
            else HIERARCHICAL_UNIFORM_SAMPLING_METHOD
        )
    elif state["kind"] == "epic_atomwise":
        if allocation_mode != "learned":
            raise ValueError("uniform allocation is only defined for hierarchical EPIC")
        sampling_method = ATOMWISE_SAMPLING_METHOD
    else:
        if allocation_mode != "learned":
            raise ValueError("uniform allocation is only defined for hierarchical EPIC")
        sampling_method = SAMPLING_METHOD

    from torch.utils.data import Subset

    loader = make_data_loader(
        Subset(load_dataset(config), indices.tolist()),
        config,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    point_means, mc_means, scales, truths, offsets = [], [], [], [], [0]
    energy_mc_means, energy_mc_scales = [], []
    started = time.perf_counter()
    progress_path = Path(output_path).with_suffix(".progress.json")
    progress_interval = max(1, len(indices) // 100)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    for structure_index, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        point_force, force_scale, mc_force_mean, sampled_energies = sampled_force_statistics(
            model,
            batch,
            n_samples=n_samples,
            seed=sample_seed,
            allocation_mode=allocation_mode,
        )
        point_means.append(point_force.cpu().numpy())
        mc_means.append(mc_force_mean.cpu().numpy())
        scales.append(force_scale.cpu().numpy())
        truths.append(batch.forces.detach().cpu().numpy())
        offsets.append(offsets[-1] + int(batch.positions.shape[0]))
        energy_mc_means.append(float(sampled_energies.mean().cpu()))
        energy_mc_scales.append(float(sampled_energies.std(unbiased=True).cpu()))
        if structure_index == 1 or structure_index == len(indices) or structure_index % progress_interval == 0:
            elapsed = time.perf_counter() - started
            payload = {
                "stage": "epic_force",
                "split": split,
                "shard": shard,
                "total_shards": total_shards,
                "completed_structures": structure_index,
                "total_structures": len(indices),
                "percent": round(100.0 * structure_index / len(indices), 2),
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round(elapsed * (len(indices) - structure_index) / structure_index, 1),
            }
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = progress_path.with_suffix(progress_path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
            os.replace(temporary, progress_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        db_index=indices,
        force_mean=np.concatenate(point_means),
        force_mc_mean=np.concatenate(mc_means),
        force_scale_raw=np.concatenate(scales),
        force_true=np.concatenate(truths),
        force_offsets=np.asarray(offsets, dtype=np.int64),
        energy_mc_mean=np.asarray(energy_mc_means),
        energy_mc_scale=np.asarray(energy_mc_scales),
        sample_seed=np.asarray(sample_seed),
        n_samples=np.asarray(n_samples),
        sampling_method=np.asarray(sampling_method),
        allocation_mode=np.asarray(allocation_mode),
        uncertainty_component=np.asarray(UNCERTAINTY_COMPONENT),
        rng_seed_replayed_per_structure=np.asarray(True),
        cross_structure_base_random_variable_alignment_verified=np.asarray(False),
        force_samples_energy_gradient_derived=np.asarray(True),
        global_random_pes_verified=np.asarray(False),
        force_posterior_identified=np.asarray(False),
        samplewise_permutation_equivariance_verified=np.asarray(False),
        aleatoric_force_noise=np.asarray(False),
        distribution_approximation=np.asarray("gaussian_moment_matched_mc"),
        split=np.asarray(split),
        inference_seconds=np.asarray(time.perf_counter() - started),
        peak_memory_bytes=np.asarray(
            torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else 0,
            dtype=np.int64,
        ),
        shard=np.asarray(shard),
        total_shards=np.asarray(total_shards),
        selected_subset_size=np.asarray(len(selected_indices)),
    )
    return output_path


def aggregate_epic_force_shards(paths: list[str | Path], output_path: str | Path) -> Path:
    if not paths:
        raise ValueError("at least one EPIC force shard is required")
    records = [dict(np.load(path)) for path in paths]
    metadata_fields = (
        "sample_seed",
        "n_samples",
        "sampling_method",
        "allocation_mode",
        "uncertainty_component",
        "rng_seed_replayed_per_structure",
        "cross_structure_base_random_variable_alignment_verified",
        "force_samples_energy_gradient_derived",
        "global_random_pes_verified",
        "force_posterior_identified",
        "samplewise_permutation_equivariance_verified",
        "aleatoric_force_noise",
        "distribution_approximation",
        "split",
        "total_shards",
        "selected_subset_size",
    )
    for field in metadata_fields:
        reference = records[0][field]
        if any(not np.array_equal(record[field], reference) for record in records[1:]):
            raise ValueError(f"EPIC force shard metadata differs for {field}")
    expected_shards = int(records[0]["total_shards"])
    observed_shards = sorted(int(record["shard"]) for record in records)
    if observed_shards != list(range(expected_shards)):
        raise ValueError(f"expected EPIC force shards 0..{expected_shards - 1}, got {observed_shards}")

    db_index = np.concatenate([record["db_index"] for record in records])
    if len(np.unique(db_index)) != len(db_index):
        raise ValueError("EPIC force shards contain duplicate db_index values")
    if len(db_index) != int(records[0]["selected_subset_size"]):
        raise ValueError("EPIC force shards do not cover the selected subset")
    order = np.argsort(db_index, kind="stable")
    atom_starts, atom_lengths = [], []
    atom_cursor = 0
    for record in records:
        lengths = np.diff(record["force_offsets"])
        atom_starts.extend((record["force_offsets"][:-1] + atom_cursor).tolist())
        atom_lengths.extend(lengths.tolist())
        atom_cursor += len(record["force_true"])
    atom_starts = np.asarray(atom_starts, dtype=np.int64)
    atom_lengths = np.asarray(atom_lengths, dtype=np.int64)
    atom_order = np.concatenate(
        [np.arange(atom_starts[index], atom_starts[index] + atom_lengths[index]) for index in order]
    )
    combined = {
        "db_index": db_index[order],
        "force_offsets": np.concatenate(([0], np.cumsum(atom_lengths[order]))).astype(np.int64),
        "energy_mc_mean": np.concatenate([record["energy_mc_mean"] for record in records])[order],
        "energy_mc_scale": np.concatenate([record["energy_mc_scale"] for record in records])[order],
        "inference_seconds": np.asarray(sum(float(record["inference_seconds"]) for record in records)),
        "peak_memory_bytes": np.asarray(max(int(record["peak_memory_bytes"]) for record in records)),
    }
    for field in ("force_true", "force_mean", "force_mc_mean", "force_scale_raw"):
        combined[field] = np.concatenate([record[field] for record in records], axis=0)[atom_order]
    for field in metadata_fields:
        combined[field] = records[0][field]
    combined["shard"] = np.asarray(-1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **combined)
    return output_path
