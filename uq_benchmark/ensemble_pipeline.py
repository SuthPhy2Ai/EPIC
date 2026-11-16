from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _aligned_members(member_paths: list[Path]) -> list[dict[str, np.ndarray]]:
    if len(member_paths) != 5:
        raise ValueError("ensemble5 requires exactly five deterministic members")
    members = [_load_npz(path) for path in member_paths]
    reference = members[0]
    required = {
        "db_index",
        "natoms",
        "energy_true_total",
        "energy_pred_total",
        "force_true",
        "force_pred",
        "force_offsets",
        "inference_seconds",
        "peak_memory_bytes",
    }
    for index, member in enumerate(members):
        if missing := required - set(member):
            raise ValueError(f"ensemble member {index} lacks {sorted(missing)}")
        for field in (
            "db_index",
            "natoms",
            "energy_true_total",
            "force_true",
            "force_offsets",
        ):
            if not np.array_equal(member[field], reference[field]):
                raise ValueError(f"ensemble member {index} disagrees on {field}")
    return members


def build_ensemble_records(
    member_paths: list[str | Path],
    force_reference_path: str | Path,
    point_output: str | Path,
    force_output: str | Path,
) -> tuple[Path, Path]:
    member_paths = [Path(path) for path in member_paths]
    members = _aligned_members(member_paths)
    reference = members[0]
    force_reference = _load_npz(force_reference_path)

    energy_stack = np.stack(
        [np.asarray(member["energy_pred_total"], dtype=np.float64) for member in members]
    )
    force_stack = np.stack(
        [np.asarray(member["force_pred"], dtype=np.float64) for member in members]
    )
    member_seconds = np.asarray(
        [float(member["inference_seconds"]) for member in members], dtype=np.float64
    )
    member_peak_memory = np.asarray(
        [int(member["peak_memory_bytes"]) for member in members], dtype=np.int64
    )
    if np.any(~np.isfinite(member_seconds)) or np.any(member_seconds <= 0):
        raise ValueError("ensemble member inference times must be finite and positive")
    if np.any(member_peak_memory < 0):
        raise ValueError("ensemble member peak memory must be non-negative")

    point_output = Path(point_output)
    point_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        point_output,
        db_index=reference["db_index"],
        natoms=reference["natoms"],
        energy_true_total=reference["energy_true_total"],
        energy_pred_total=energy_stack.mean(axis=0),
        energy_aleatoric_variance=np.zeros(energy_stack.shape[1], dtype=np.float64),
        energy_epistemic_variance=energy_stack.var(axis=0, ddof=1),
        force_true=reference["force_true"],
        force_pred=force_stack.mean(axis=0),
        force_offsets=reference["force_offsets"],
        inference_seconds=member_seconds.sum(),
        peak_memory_bytes=member_peak_memory.max(),
        n_members=np.asarray(5),
        timed_n_structures=np.asarray(len(reference["db_index"])),
        timing_semantics=np.asarray("sum_of_member_shard_local_device_wall_times"),
        member_input_sha256=np.asarray([_sha256_file(path) for path in member_paths]),
    )

    point_index = np.asarray(reference["db_index"], dtype=np.int64)
    subset_index = np.asarray(force_reference["db_index"], dtype=np.int64)
    positions = np.searchsorted(point_index, subset_index)
    if np.any(positions >= len(point_index)) or not np.array_equal(
        point_index[positions], subset_index
    ):
        raise ValueError("force reference is not a subset of ensemble point predictions")
    point_offsets = np.asarray(reference["force_offsets"], dtype=np.int64)
    selected_member_forces = []
    selected_truth = []
    force_offsets = [0]
    for position in positions:
        start, stop = point_offsets[position : position + 2]
        selected_member_forces.append(force_stack[:, start:stop])
        selected_truth.append(np.asarray(reference["force_true"])[start:stop])
        force_offsets.append(force_offsets[-1] + int(stop - start))
    selected_stack = np.concatenate(selected_member_forces, axis=1)
    selected_truth_array = np.concatenate(selected_truth, axis=0)
    if not np.array_equal(selected_truth_array, force_reference["force_true"]):
        raise ValueError("ensemble member force truth disagrees with the force reference")
    if not np.array_equal(np.asarray(force_offsets), force_reference["force_offsets"]):
        raise ValueError("ensemble force offsets disagree with the force reference")

    force_output = Path(force_output)
    force_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        force_output,
        db_index=subset_index,
        force_true=selected_truth_array,
        force_mean=selected_stack.mean(axis=0),
        force_scale_raw=np.sqrt(np.maximum(selected_stack.var(axis=0, ddof=1), 0.0)),
        force_offsets=np.asarray(force_offsets, dtype=np.int64),
        n_samples=np.asarray(5),
        n_members=np.asarray(5),
        inference_seconds=member_seconds.sum(),
        peak_memory_bytes=member_peak_memory.max(),
        timed_n_structures=np.asarray(len(point_index)),
        timing_semantics=np.asarray("sum_of_member_shard_local_device_wall_times"),
        member_input_sha256=np.asarray([_sha256_file(path) for path in member_paths]),
    )
    return point_output, force_output
