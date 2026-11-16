from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .config import BenchmarkConfig
from .data import load_dataset, make_data_loader
from .inference import extract_last_layer_features, load_checkpoint_model
from .llpr import LLPRPosterior, StreamingGram, antithetic_weight_samples, fit_llpr_posterior
from .split import hashed_subset, load_split
from .types import CalibrationState


def structure_mean(values: np.ndarray, atom_offsets: np.ndarray) -> np.ndarray:
    """Average atom-wise values, including trailing components, by structure."""
    values = np.asarray(values)
    atom_offsets = np.asarray(atom_offsets)
    if atom_offsets.ndim != 1 or len(atom_offsets) < 2:
        raise ValueError("atom_offsets must be a one-dimensional boundary array")
    if atom_offsets[0] != 0 or atom_offsets[-1] != len(values):
        raise ValueError("atom_offsets must start at zero and end at len(values)")
    if np.any(np.diff(atom_offsets) <= 0):
        raise ValueError("each structure must contain at least one atom")
    return np.asarray(
        [values[start:stop].mean() for start, stop in zip(atom_offsets[:-1], atom_offsets[1:])]
    )


def accumulate_gram_shard(
    config: BenchmarkConfig,
    split_path: str | Path,
    checkpoint: str | Path,
    output_path: str | Path,
    shard: int,
    total_shards: int,
    device: str = "cuda",
) -> Path:
    indices = load_split(split_path, "train")[shard::total_shards]
    model, state = load_checkpoint_model(checkpoint, device)
    if state["kind"] != "deterministic":
        raise ValueError("LLPR requires a deterministic MACE checkpoint")
    from torch.utils.data import Subset

    loader = make_data_loader(
        Subset(load_dataset(config), indices.tolist()),
        config,
        batch_size=config.training.per_gpu_batch_size,
        shuffle=False,
        drop_last=False,
    )
    gram = StreamingGram(config.model.num_channels)
    completed = 0
    started = time.perf_counter()
    progress_path = Path(output_path).with_suffix(".progress.json")
    progress_interval = max(1, len(indices) // 100)
    next_progress = 1
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            output = model(batch, training=False, compute_force=False)
            features = extract_last_layer_features(model, output, batch)
        gram.update(features.cpu().numpy())
        completed += len(features)
        if completed >= next_progress or completed == len(indices):
            elapsed = time.perf_counter() - started
            payload = {
                "stage": "llpr_gram",
                "shard": shard,
                "total_shards": total_shards,
                "completed_structures": completed,
                "total_structures": len(indices),
                "percent": round(100.0 * completed / len(indices), 2),
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round(elapsed * (len(indices) - completed) / completed, 1),
            }
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = progress_path.with_suffix(progress_path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
            os.replace(temporary, progress_path)
            next_progress = completed + progress_interval
    if completed != len(indices):
        raise RuntimeError(f"Gram shard produced {completed} rows for {len(indices)} indices")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, gram=gram.matrix, n_samples=gram.n_samples)
    return output_path


def aggregate_gram_shards(paths: list[str | Path], output_path: str | Path) -> Path:
    matrices, counts = [], []
    for path in paths:
        with np.load(path) as data:
            matrices.append(data["gram"])
            counts.append(int(data["n_samples"]))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, gram=np.sum(matrices, axis=0), n_samples=sum(counts))
    return output_path


def fit_posterior_from_files(
    gram_path: str | Path,
    calibration_predictions: str | Path,
    ridge_multipliers: tuple[float, ...],
    output_path: str | Path,
) -> Path:
    with np.load(gram_path) as data:
        matrix = data["gram"]
        n_samples = int(data["n_samples"])
    gram = StreamingGram(matrix.shape[0])
    gram.matrix[:] = matrix
    gram.n_samples = n_samples
    with np.load(calibration_predictions) as data:
        features = data["llpr_features"]
        target = data["energy_true_total"] / data["natoms"]
        location = data["energy_pred_total"] / data["natoms"]
    posterior = fit_llpr_posterior(gram, features, target, location, ridge_multipliers)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        covariance_raw=posterior.covariance_raw,
        ridge=posterior.ridge,
        n_training_samples=posterior.n_training_samples,
        calibration_parameters=json.dumps(dict(posterior.calibration.parameters), sort_keys=True),
    )
    return output_path


def load_posterior(path: str | Path) -> LLPRPosterior:
    with np.load(path) as data:
        parameters = json.loads(str(data["calibration_parameters"]))
        force_parameters = (
            json.loads(str(data["force_calibration_parameters"]))
            if "force_calibration_parameters" in data
            else None
        )
        return LLPRPosterior(
            covariance_raw=data["covariance_raw"],
            ridge=float(data["ridge"]),
            calibration=CalibrationState("llpr", parameters),
            n_training_samples=int(data["n_training_samples"]),
            force_calibration=(
                CalibrationState("llpr_force_aware_force", force_parameters)
                if force_parameters is not None
                else None
            ),
        )


def _last_layer_hidden(model, output: dict) -> torch.Tensor:
    node_features = output["node_feats"]
    final_width = model.readouts[-1].linear_1.irreps_in.dim
    final = node_features[:, -final_width:]
    return model.readouts[-1].non_linearity(model.readouts[-1].linear_1(final))


def force_feature_matrix(model, batch) -> torch.Tensor:
    """Return d(last-layer energy features)/dR as [3 * n_atoms, feature_dim]."""
    batch.positions.requires_grad_(True)
    output = model(batch, training=False, compute_force=False)
    feature_total = _last_layer_hidden(model, output).sum(dim=0)
    columns = []
    for feature_index, feature in enumerate(feature_total):
        gradient = torch.autograd.grad(
            feature,
            batch.positions,
            retain_graph=feature_index + 1 < len(feature_total),
            create_graph=False,
        )[0]
        columns.append(-gradient.reshape(-1))
    return torch.stack(columns, dim=1)


def accumulate_force_gram_shard(
    config: BenchmarkConfig,
    split_path: str | Path,
    checkpoint: str | Path,
    output_path: str | Path,
    shard: int,
    total_shards: int,
    subset_size: int,
    device: str = "cuda",
) -> Path:
    if not 0 <= shard < total_shards:
        raise ValueError("invalid force-aware Gram shard")
    all_indices = load_split(split_path, "train")
    selected_indices = hashed_subset(all_indices, subset_size, "llpr-force-aware-train-v1")
    indices = selected_indices[shard::total_shards]
    model, state = load_checkpoint_model(checkpoint, device)
    if state["kind"] != "deterministic":
        raise ValueError("force-aware LLPR requires a deterministic MACE checkpoint")
    from torch.utils.data import Subset

    loader = make_data_loader(
        Subset(load_dataset(config), indices.tolist()), config, batch_size=1, shuffle=False, drop_last=False
    )
    gram = StreamingGram(config.model.num_channels)
    completed = 0
    started = time.perf_counter()
    progress_path = Path(output_path).with_suffix(".progress.json")
    progress_interval = max(1, len(indices) // 100)
    for structure_index, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        features = force_feature_matrix(model, batch)
        gram.update(features.detach().cpu().numpy())
        completed += 1
        if structure_index == 1 or structure_index == len(indices) or structure_index % progress_interval == 0:
            elapsed = time.perf_counter() - started
            payload = {
                "stage": "llpr_force_aware_gram",
                "shard": shard,
                "total_shards": total_shards,
                "completed_structures": structure_index,
                "total_structures": len(indices),
                "force_components": gram.n_samples,
                "percent": round(100 * structure_index / len(indices), 2),
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round(elapsed * (len(indices) - structure_index) / structure_index, 1),
            }
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = progress_path.with_suffix(progress_path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
            os.replace(temporary, progress_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        gram=gram.matrix,
        n_force_components=gram.n_samples,
        n_structures=completed,
        selected_subset_size=len(selected_indices),
        total_train_structures=len(all_indices),
        shard=shard,
        total_shards=total_shards,
    )
    return output_path


def aggregate_force_gram_shards(paths: list[str | Path], output_path: str | Path) -> Path:
    if not paths:
        raise ValueError("at least one force-aware Gram shard is required")
    records = [dict(np.load(path)) for path in paths]
    expected = int(records[0]["total_shards"])
    if sorted(int(record["shard"]) for record in records) != list(range(expected)):
        raise ValueError("force-aware Gram shards are incomplete")
    for field in ("selected_subset_size", "total_train_structures", "total_shards"):
        if any(int(record[field]) != int(records[0][field]) for record in records[1:]):
            raise ValueError(f"force-aware Gram metadata differs for {field}")
    if sum(int(record["n_structures"]) for record in records) != int(records[0]["selected_subset_size"]):
        raise ValueError("force-aware Gram structure count is incorrect")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        gram=np.sum([record["gram"] for record in records], axis=0),
        n_force_components=sum(int(record["n_force_components"]) for record in records),
        n_structures=sum(int(record["n_structures"]) for record in records),
        selected_subset_size=records[0]["selected_subset_size"],
        total_train_structures=records[0]["total_train_structures"],
        total_shards=records[0]["total_shards"],
    )
    return output_path


def collect_force_feature_shard(
    config: BenchmarkConfig,
    split_path: str | Path,
    checkpoint: str | Path,
    output_path: str | Path,
    shard: int,
    total_shards: int,
    subset_size: int,
    device: str = "cuda",
) -> Path:
    if not 0 <= shard < total_shards:
        raise ValueError("invalid force-aware feature shard")
    all_indices = load_split(split_path, "calibration")
    selected_indices = hashed_subset(all_indices, subset_size, "llpr-force-calibration-v1")
    indices = selected_indices[shard::total_shards]
    model, state = load_checkpoint_model(checkpoint, device)
    if state["kind"] != "deterministic":
        raise ValueError("force-aware LLPR requires a deterministic MACE checkpoint")
    from torch.utils.data import Subset

    loader = make_data_loader(
        Subset(load_dataset(config), indices.tolist()), config, batch_size=1, shuffle=False, drop_last=False
    )
    features, offsets = [], [0]
    started = time.perf_counter()
    progress_path = Path(output_path).with_suffix(".progress.json")
    progress_interval = max(1, len(indices) // 100)
    for structure_index, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        matrix = force_feature_matrix(model, batch).detach().cpu().numpy().astype(np.float32)
        features.append(matrix)
        offsets.append(offsets[-1] + len(matrix))
        if structure_index == 1 or structure_index == len(indices) or structure_index % progress_interval == 0:
            elapsed = time.perf_counter() - started
            payload = {
                "stage": "llpr_force_aware_calibration_features",
                "shard": shard,
                "total_shards": total_shards,
                "completed_structures": structure_index,
                "total_structures": len(indices),
                "percent": round(100 * structure_index / len(indices), 2),
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
        force_features=np.concatenate(features),
        force_offsets=np.asarray(offsets, dtype=np.int64),
        shard=shard,
        total_shards=total_shards,
        selected_subset_size=len(selected_indices),
    )
    return output_path


def aggregate_force_feature_shards(paths: list[str | Path], output_path: str | Path) -> Path:
    if not paths:
        raise ValueError("at least one force feature shard is required")
    records = [dict(np.load(path)) for path in paths]
    expected = int(records[0]["total_shards"])
    if sorted(int(record["shard"]) for record in records) != list(range(expected)):
        raise ValueError("force feature shards are incomplete")
    db_index = np.concatenate([record["db_index"] for record in records])
    if len(db_index) != int(records[0]["selected_subset_size"]) or len(np.unique(db_index)) != len(db_index):
        raise ValueError("force feature shards do not cover the selected subset")
    order = np.argsort(db_index, kind="stable")
    starts, lengths, cursor = [], [], 0
    for record in records:
        shard_lengths = np.diff(record["force_offsets"])
        starts.extend((record["force_offsets"][:-1] + cursor).tolist())
        lengths.extend(shard_lengths.tolist())
        cursor += len(record["force_features"])
    starts = np.asarray(starts, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    component_order = np.concatenate([np.arange(starts[i], starts[i] + lengths[i]) for i in order])
    combined_features = np.concatenate([record["force_features"] for record in records])[component_order]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        db_index=db_index[order],
        force_features=combined_features,
        force_offsets=np.concatenate(([0], np.cumsum(lengths[order]))).astype(np.int64),
        selected_subset_size=records[0]["selected_subset_size"],
        total_shards=records[0]["total_shards"],
    )
    return output_path


def fit_force_aware_posterior_from_files(
    energy_gram_path: str | Path,
    force_gram_path: str | Path,
    energy_calibration_path: str | Path,
    force_feature_path: str | Path,
    force_target_path: str | Path,
    ridge_multipliers: tuple[float, ...],
    output_path: str | Path,
) -> Path:
    from .calibration import apply_variance_map, fit_gaussian_variance_map
    from .distributions import gaussian_nll

    with np.load(energy_gram_path) as data:
        energy_gram = data["gram"].astype(np.float64)
        n_energy = int(data["n_samples"])
    with np.load(force_gram_path) as data:
        force_gram = data["gram"].astype(np.float64)
        force_subset_structures = int(data["n_structures"])
        total_train_structures = int(data["total_train_structures"])
        n_force_components = int(data["n_force_components"])
    with np.load(energy_calibration_path) as data:
        energy_features = data["llpr_features"].astype(np.float64)
        energy_target = data["energy_true_total"] / data["natoms"]
        energy_location = data["energy_pred_total"] / data["natoms"]
    with np.load(force_feature_path) as data:
        force_db = data["db_index"]
        force_features = data["force_features"].astype(np.float64)
    with np.load(force_target_path) as data:
        if not np.array_equal(force_db, data["db_index"]):
            raise ValueError("force calibration features and targets have different db_index")
        force_target = data["force_true"].reshape(-1)
        force_location = data["force_mean"].reshape(-1)
    if len(force_features) != len(force_target):
        raise ValueError("force calibration feature and target counts differ")

    energy_noise = max(float(np.sqrt(np.mean((energy_target - energy_location) ** 2))), 1e-8)
    force_noise = max(float(np.sqrt(np.mean((force_target - force_location) ** 2))), 1e-8)
    extrapolation = total_train_structures / force_subset_structures
    base_precision = energy_gram / energy_noise**2 + extrapolation * force_gram / force_noise**2
    trace_scale = max(float(np.trace(base_precision)) / len(base_precision), np.finfo(float).eps)
    candidates = []
    best = None
    for multiplier in ridge_multipliers:
        ridge = float(multiplier) * trace_scale
        covariance = np.linalg.inv(base_precision + ridge * np.eye(len(base_precision)))
        energy_raw = np.einsum("ni,ij,nj->n", energy_features, covariance, energy_features)
        force_raw = np.einsum("ni,ij,nj->n", force_features, covariance, force_features)
        energy_state = fit_gaussian_variance_map(
            energy_target, energy_location, np.maximum(energy_raw, 0), method="llpr_force_aware"
        )
        force_state = fit_gaussian_variance_map(
            force_target, force_location, np.maximum(force_raw, 0), method="llpr_force_aware_force"
        )
        energy_nll = float(np.mean(gaussian_nll(
            energy_target, energy_location, np.sqrt(apply_variance_map(energy_raw, energy_state))
        )))
        force_nll = float(np.mean(gaussian_nll(
            force_target, force_location, np.sqrt(apply_variance_map(force_raw, force_state))
        )))
        score = 0.5 * (energy_nll + force_nll)
        row = {
            "ridge_multiplier": float(multiplier),
            "ridge": ridge,
            "energy_calibration_nll": energy_nll,
            "force_calibration_nll": force_nll,
            "joint_task_balanced_nll": score,
            "energy_calibration_parameters": dict(energy_state.parameters),
            "force_calibration_parameters": dict(force_state.parameters),
        }
        candidates.append(row)
        if best is None or score < best[0]:
            best = (score, covariance, row)
    assert best is not None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        covariance_raw=best[1],
        ridge=best[2]["ridge"],
        n_training_samples=n_energy,
        calibration_parameters=json.dumps(best[2]["energy_calibration_parameters"], sort_keys=True),
        force_calibration_parameters=json.dumps(best[2]["force_calibration_parameters"], sort_keys=True),
        candidates=json.dumps(candidates, sort_keys=True),
        energy_noise_rmse=energy_noise,
        force_noise_rmse=force_noise,
        force_gram_extrapolation=extrapolation,
        force_gram_structures=force_subset_structures,
        force_gram_components=n_force_components,
        posterior_kind=np.asarray("force_aware_llpr"),
    )
    return output_path


def sample_force_subset(
    config: BenchmarkConfig,
    split_path: str | Path,
    split: str,
    checkpoint: str | Path,
    posterior_path: str | Path,
    output_path: str | Path,
    subset_size: int,
    device: str = "cuda",
    shard: int = 0,
    total_shards: int = 1,
    subset_namespace: str | None = None,
    n_samples: int | None = None,
    sample_seed: int | None = None,
) -> Path:
    if split not in ("calibration", "test"):
        raise ValueError("force uncertainty is only evaluated on calibration or test")
    if not 0 <= shard < total_shards:
        raise ValueError("invalid LLPR force shard")
    if subset_namespace is None:
        subset_namespace = f"llpr-force-{split}-v1"
    if not subset_namespace.strip():
        raise ValueError("LLPR force subset namespace must not be empty")
    n_samples = config.llpr.force_samples if n_samples is None else int(n_samples)
    sample_seed = config.llpr.sample_seed if sample_seed is None else int(sample_seed)
    if n_samples <= 0 or n_samples % 2:
        raise ValueError("LLPR force sampling requires a positive even sample count")
    all_indices = load_split(split_path, split, allow_test=split == "test")
    selected_indices = hashed_subset(all_indices, subset_size, subset_namespace)
    selected_indices_sha256 = hashlib.sha256(
        np.asarray(selected_indices, dtype="<i8").tobytes()
    ).hexdigest()
    indices = selected_indices[shard::total_shards]
    if len(indices) == 0:
        raise ValueError("LLPR force shard is empty; reduce total_shards or increase subset size")
    model, state = load_checkpoint_model(checkpoint, device)
    if state["kind"] != "deterministic":
        raise ValueError("LLPR force sampling requires deterministic MACE")
    posterior = load_posterior(posterior_path)
    mean_weight = model.readouts[-1].linear_2.weight.detach().reshape(-1).cpu().numpy()
    weight_samples = antithetic_weight_samples(
        mean_weight,
        posterior.covariance_raw,
        n_samples=n_samples,
        seed=sample_seed,
    )
    weight_samples = torch.as_tensor(weight_samples, device=device, dtype=next(model.parameters()).dtype)
    mean_weight_t = torch.as_tensor(mean_weight, device=device, dtype=weight_samples.dtype)
    from torch.utils.data import Subset

    loader = make_data_loader(
        Subset(load_dataset(config), indices.tolist()),
        config,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    means, scales, truths, offsets = [], [], [], [0]
    started = time.perf_counter()
    progress_path = Path(output_path).with_suffix(".progress.json")
    progress_interval = max(1, len(indices) // 100)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    for structure_index, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        batch.positions.requires_grad_(True)
        output = model(batch, training=False, compute_force=False)
        hidden = _last_layer_hidden(model, output)
        feature_total = hidden.sum(dim=0)
        energy_mean = output["energy"].sum()
        sampled_energies = energy_mean + (weight_samples - mean_weight_t) @ feature_total
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
        means.append(force_stack.mean(dim=0).cpu().numpy())
        scales.append(force_stack.std(dim=0, unbiased=True).cpu().numpy())
        truths.append(batch.forces.detach().cpu().numpy())
        offsets.append(offsets[-1] + int(batch.positions.shape[0]))
        if structure_index == 1 or structure_index == len(indices) or structure_index % progress_interval == 0:
            elapsed = time.perf_counter() - started
            payload = {
                "stage": "llpr_force",
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
        force_mean=np.concatenate(means),
        force_scale_raw=np.concatenate(scales),
        force_true=np.concatenate(truths),
        force_offsets=np.asarray(offsets, dtype=np.int64),
        sample_seed=np.asarray(sample_seed),
        n_samples=np.asarray(n_samples),
        sampling_method=np.asarray("llpr_last_layer_joint_weight_samples"),
        uncertainty_component=np.asarray("epistemic"),
        shared_sample_ids=np.asarray(True),
        split=np.asarray(split),
        subset_namespace=np.asarray(subset_namespace),
        selected_indices_sha256=np.asarray(selected_indices_sha256),
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


def aggregate_force_shards(paths: list[str | Path], output_path: str | Path) -> Path:
    if not paths:
        raise ValueError("at least one LLPR force shard is required")
    records = [dict(np.load(path)) for path in paths]
    metadata_fields = (
        "sample_seed",
        "n_samples",
        "sampling_method",
        "uncertainty_component",
        "shared_sample_ids",
        "split",
        "subset_namespace",
        "selected_indices_sha256",
        "total_shards",
        "selected_subset_size",
    )
    for field in metadata_fields:
        reference = records[0][field]
        if any(not np.array_equal(record[field], reference) for record in records[1:]):
            raise ValueError(f"LLPR force shard metadata differs for {field}")
    expected_shards = int(records[0]["total_shards"])
    observed_shards = sorted(int(record["shard"]) for record in records)
    if observed_shards != list(range(expected_shards)):
        raise ValueError(f"expected LLPR force shards 0..{expected_shards - 1}, got {observed_shards}")

    db_index = np.concatenate([record["db_index"] for record in records])
    if len(np.unique(db_index)) != len(db_index):
        raise ValueError("LLPR force shards contain duplicate db_index values")
    if len(db_index) != int(records[0]["selected_subset_size"]):
        raise ValueError("LLPR force shards do not cover the selected subset")
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
        "inference_seconds": np.asarray(sum(float(record["inference_seconds"]) for record in records)),
        "peak_memory_bytes": np.asarray(max(int(record["peak_memory_bytes"]) for record in records)),
    }
    for field in ("force_true", "force_mean", "force_scale_raw"):
        combined[field] = np.concatenate([record[field] for record in records], axis=0)[atom_order]
    for field in metadata_fields:
        combined[field] = records[0][field]
    combined["shard"] = np.asarray(-1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **combined)
    return output_path
