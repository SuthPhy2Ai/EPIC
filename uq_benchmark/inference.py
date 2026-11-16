from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .config import BenchmarkConfig, ModelConfig
from .data import load_dataset, make_data_loader
from .modeling import create_model
from .split import load_split

def load_checkpoint_model(path: str | Path, device: str = "cpu"):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in state:
        raise ValueError("benchmark inference only accepts versioned uq_benchmark checkpoints")
    config = ModelConfig(**state["config"]["model"])
    model = create_model(state["kind"], config, float(state["avg_num_neighbors"]))
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, state


def extract_last_layer_features(model, output: dict, batch) -> torch.Tensor:
    """Return structure-equal last-readout features used by LLPR."""
    node_features = output["node_feats"]
    final_width = model.readouts[-1].linear_1.irreps_in.dim
    final_node_features = node_features[:, -final_width:]
    hidden = model.readouts[-1].non_linearity(model.readouts[-1].linear_1(final_node_features))
    n_graphs = int(batch.ptr.numel() - 1)
    pooled = torch.zeros((n_graphs, hidden.shape[-1]), device=hidden.device, dtype=hidden.dtype)
    pooled.index_add_(0, batch.batch, hidden)
    natoms = (batch.ptr[1:] - batch.ptr[:-1]).to(hidden.dtype)
    return pooled / natoms[:, None]


def run_inference_shard(
    config: BenchmarkConfig,
    split_path: str | Path,
    split: Literal["train", "validation", "calibration", "test"],
    checkpoint: str | Path,
    output_path: str | Path,
    shard: int = 0,
    total_shards: int = 1,
    device: str = "cuda",
    include_llpr_features: bool = False,
) -> Path:
    if not 0 <= shard < total_shards:
        raise ValueError("invalid shard")
    selected_indices = load_split(split_path, split, allow_test=split == "test")
    selected_split_size = len(selected_indices)
    if total_shards > selected_split_size:
        raise ValueError(
            f"total_shards={total_shards} exceeds the selected split size "
            f"({selected_split_size})"
        )
    indices = selected_indices[shard::total_shards]
    model, state = load_checkpoint_model(checkpoint, device=device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    from torch.utils.data import Subset

    dataset = load_dataset(config)
    loader = make_data_loader(
        Subset(dataset, indices.tolist()),
        config,
        batch_size=config.training.per_gpu_batch_size,
        shuffle=False,
        drop_last=False,
    )
    energy_true, energy_pred, natoms, db_indices = [], [], [], []
    force_true, force_pred, atomic_numbers, offsets = [], [], [], [0]
    nu, alpha, beta, features = [], [], [], []
    graph_aleatoric, graph_epistemic = [], []
    atomic_gamma, atomic_nu, atomic_alpha, atomic_beta = [], [], [], []
    atomic_aleatoric_weight, atomic_epistemic_weight = [], []
    atomic_aleatoric_variance, atomic_epistemic_variance, atomic_total_variance = [], [], []
    cursor = 0
    started = time.perf_counter()
    progress_path = Path(output_path).with_suffix(".progress.json")
    progress_interval = max(1, len(indices) // 100)
    next_progress = 1
    for batch in loader:
        batch = batch.to(device)
        with torch.enable_grad():
            output = model(batch, training=False, compute_force=True)
        n_graphs = int(batch.ptr.numel() - 1)
        batch_indices = indices[cursor : cursor + n_graphs]
        cursor += n_graphs
        batch_natoms = (batch.ptr[1:] - batch.ptr[:-1]).detach().cpu().numpy().astype(np.int32)
        db_indices.extend(batch_indices)
        natoms.extend(batch_natoms)
        energy_true.extend(batch.energy.detach().cpu().numpy())
        energy_pred.extend(output["energy"].detach().cpu().numpy())
        true_f = batch.forces.detach().cpu().numpy()
        pred_f = output["forces"].detach().cpu().numpy()
        force_true.append(true_f)
        force_pred.append(pred_f)
        atomic_numbers.append(batch.node_attrs.detach().argmax(dim=1).cpu().numpy().astype(np.int16) + 1)
        offsets.extend(offsets[-1] + np.cumsum(batch_natoms))
        if state["kind"] in ("epic", "epic_hierarchical"):
            nu.extend(output["energy_nu"].detach().cpu().numpy())
            alpha.extend(output["energy_alpha"].detach().cpu().numpy())
            beta.extend(output["energy_beta"].detach().cpu().numpy())
        if state["kind"] in ("epic_atomwise", "epic_hierarchical"):
            graph_aleatoric.extend(
                output["energy_aleatoric_variance"].detach().cpu().numpy()
            )
            graph_epistemic.extend(
                output["energy_epistemic_variance"].detach().cpu().numpy()
            )
            atomic_gamma.append(output["atomic_energy_gamma"].detach().cpu().numpy())
            atomic_nu.append(output["atomic_energy_nu"].detach().cpu().numpy())
            atomic_alpha.append(output["atomic_energy_alpha"].detach().cpu().numpy())
            atomic_beta.append(output["atomic_energy_beta"].detach().cpu().numpy())
            if state["kind"] == "epic_hierarchical":
                atomic_aleatoric_weight.append(
                    output["atomic_aleatoric_weight"].detach().cpu().numpy()
                )
                atomic_epistemic_weight.append(
                    output["atomic_epistemic_weight"].detach().cpu().numpy()
                )
                atomic_aleatoric_variance.append(
                    output["atomic_aleatoric_variance"].detach().cpu().numpy()
                )
                atomic_epistemic_variance.append(
                    output["atomic_epistemic_variance"].detach().cpu().numpy()
                )
                atomic_total_variance.append(
                    output["atomic_total_variance"].detach().cpu().numpy()
                )
        if include_llpr_features:
            if state["kind"] != "deterministic":
                raise ValueError("LLPR features require a deterministic checkpoint")
            features.append(extract_last_layer_features(model, output, batch).detach().cpu().numpy())
        if cursor >= next_progress or cursor == len(indices):
            elapsed = time.perf_counter() - started
            payload = {
                "stage": "inference",
                "split": split,
                "shard": shard,
                "total_shards": total_shards,
                "completed_structures": cursor,
                "total_structures": len(indices),
                "percent": round(100.0 * cursor / len(indices), 2),
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round(elapsed * (len(indices) - cursor) / cursor, 1),
            }
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = progress_path.with_suffix(progress_path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
            os.replace(temporary, progress_path)
            next_progress = cursor + progress_interval
    if cursor != len(indices):
        raise RuntimeError(f"inference produced {cursor} rows for {len(indices)} indices")
    payload = {
        "db_index": np.asarray(db_indices, dtype=np.int64),
        "natoms": np.asarray(natoms, dtype=np.int32),
        "energy_true_total": np.asarray(energy_true),
        "energy_pred_total": np.asarray(energy_pred),
        "force_true": np.concatenate(force_true, axis=0),
        "force_pred": np.concatenate(force_pred, axis=0),
        "atomic_numbers": np.concatenate(atomic_numbers, axis=0),
        "force_offsets": np.asarray(offsets, dtype=np.int64),
        "inference_seconds": np.asarray(time.perf_counter() - started),
        "peak_memory_bytes": np.asarray(
            torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else 0,
            dtype=np.int64,
        ),
        "checkpoint_kind": np.asarray(state["kind"]),
        "split": np.asarray(split),
        "shard": np.asarray(shard, dtype=np.int64),
        "total_shards": np.asarray(total_shards, dtype=np.int64),
        "selected_split_size": np.asarray(selected_split_size, dtype=np.int64),
    }
    if nu:
        payload.update(nu=np.asarray(nu), alpha=np.asarray(alpha), beta=np.asarray(beta))
    if graph_aleatoric:
        assumption = (
            "hierarchical_shared_scale_latent_nig_marginals"
            if state["kind"] == "epic_hierarchical"
            else "conditionally_independent_gaussian_moment_match"
        )
        payload.update(
            energy_aleatoric_variance=np.asarray(graph_aleatoric),
            energy_epistemic_variance=np.asarray(graph_epistemic),
            atomic_energy_gamma=np.concatenate(atomic_gamma),
            atomic_energy_nu=np.concatenate(atomic_nu),
            atomic_energy_alpha=np.concatenate(atomic_alpha),
            atomic_energy_beta=np.concatenate(atomic_beta),
            atomwise_distribution_assumption=np.asarray(assumption),
        )
        if atomic_aleatoric_weight:
            payload.update(
                atomic_aleatoric_weight=np.concatenate(atomic_aleatoric_weight),
                atomic_epistemic_weight=np.concatenate(atomic_epistemic_weight),
                atomic_aleatoric_variance=np.concatenate(atomic_aleatoric_variance),
                atomic_epistemic_variance=np.concatenate(atomic_epistemic_variance),
                atomic_total_variance=np.concatenate(atomic_total_variance),
                epic_variant=np.asarray("hierarchical_atomwise_shared_scale"),
            )
    if features:
        payload["llpr_features"] = np.concatenate(features, axis=0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return output_path


def aggregate_shards(paths: list[str | Path], output_path: str | Path) -> Path:
    if not paths:
        raise ValueError("at least one inference shard is required")
    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            records.append(dict(archive))

    required_metadata = (
        "split",
        "shard",
        "total_shards",
        "selected_split_size",
        "checkpoint_kind",
    )
    for index, record in enumerate(records):
        missing = [field for field in required_metadata if field not in record]
        if missing:
            raise ValueError(f"inference shard {index} is missing metadata: {missing}")

    def scalar(record: dict, field: str):
        value = np.asarray(record[field])
        if value.ndim != 0:
            raise ValueError(f"inference shard metadata {field} must be scalar")
        return value.item()

    totals = {int(scalar(record, "total_shards")) for record in records}
    split_sizes = {int(scalar(record, "selected_split_size")) for record in records}
    splits = {str(scalar(record, "split")) for record in records}
    if len(totals) != 1 or next(iter(totals)) <= 0:
        raise ValueError("total_shards is invalid or differs across inference shards")
    if len(split_sizes) != 1 or next(iter(split_sizes)) < 0:
        raise ValueError("selected_split_size is invalid or differs across inference shards")
    if len(splits) != 1:
        raise ValueError("split differs across inference shards")

    total_shards = next(iter(totals))
    selected_split_size = next(iter(split_sizes))
    shard_ids = [int(scalar(record, "shard")) for record in records]
    if len(shard_ids) != len(set(shard_ids)):
        raise ValueError("duplicate inference shard id")
    expected_shards = set(range(total_shards))
    if set(shard_ids) != expected_shards:
        missing = sorted(expected_shards - set(shard_ids))
        unexpected = sorted(set(shard_ids) - expected_shards)
        raise ValueError(
            f"incomplete inference shard set: missing={missing}, unexpected={unexpected}"
        )
    structure_fields = (
        "db_index",
        "natoms",
        "energy_true_total",
        "energy_pred_total",
        "nu",
        "alpha",
        "beta",
        "energy_aleatoric_variance",
        "energy_epistemic_variance",
        "llpr_features",
    )
    combined = {}
    for field in structure_fields:
        values = [record[field] for record in records if field in record]
        if values:
            if len(values) != len(records):
                raise ValueError(f"field {field} is missing from some shards")
            combined[field] = np.concatenate(values, axis=0)
    atom_fields = (
        "force_true",
        "force_pred",
        "atomic_numbers",
        "atomic_energy_gamma",
        "atomic_energy_nu",
        "atomic_energy_alpha",
        "atomic_energy_beta",
        "atomic_aleatoric_weight",
        "atomic_epistemic_weight",
        "atomic_aleatoric_variance",
        "atomic_epistemic_variance",
        "atomic_total_variance",
    )
    required_arrays = {
        "db_index",
        "natoms",
        "energy_true_total",
        "energy_pred_total",
        "force_true",
        "force_pred",
        "atomic_numbers",
        "force_offsets",
        "inference_seconds",
    }
    for index, record in enumerate(records):
        missing = sorted(required_arrays - set(record))
        if missing:
            raise ValueError(f"inference shard {index} is missing arrays: {missing}")
        db_index = np.asarray(record["db_index"])
        natoms = np.asarray(record["natoms"])
        offsets = np.asarray(record["force_offsets"])
        if db_index.ndim != 1 or natoms.ndim != 1 or len(db_index) != len(natoms):
            raise ValueError(f"inference shard {index} has invalid structure arrays")
        if np.any(natoms <= 0):
            raise ValueError(f"inference shard {index} contains a non-positive atom count")
        if (
            offsets.ndim != 1
            or len(offsets) != len(db_index) + 1
            or offsets[0] != 0
            or not np.array_equal(np.diff(offsets), natoms)
        ):
            raise ValueError(f"inference shard {index} has invalid force_offsets")
        atom_count = int(offsets[-1])
        for field in structure_fields:
            if field in record and len(np.asarray(record[field])) != len(db_index):
                raise ValueError(
                    f"inference shard {index} structure field {field} has wrong length"
                )
        for field in atom_fields:
            if field in record and len(np.asarray(record[field])) != atom_count:
                raise ValueError(
                    f"inference shard {index} atom field {field} has wrong length"
                )

    checkpoint_kinds = {str(scalar(record, "checkpoint_kind")) for record in records}
    if len(checkpoint_kinds) != 1:
        raise ValueError("checkpoint kind differs across inference shards")
    for field in atom_fields:
        values = [record[field] for record in records if field in record]
        if values:
            if len(values) != len(records):
                raise ValueError(f"field {field} is missing from some shards")
            combined[field] = np.concatenate(values, axis=0)
    combined["inference_seconds"] = np.asarray(sum(float(r["inference_seconds"]) for r in records))
    combined["peak_memory_bytes"] = np.asarray(max(int(r.get("peak_memory_bytes", 0)) for r in records), dtype=np.int64)
    combined["split"] = np.asarray(next(iter(splits)))
    combined["total_shards"] = np.asarray(total_shards, dtype=np.int64)
    combined["selected_split_size"] = np.asarray(selected_split_size, dtype=np.int64)
    combined["aggregated_shards"] = np.arange(total_shards, dtype=np.int64)
    combined["checkpoint_kind"] = np.asarray(next(iter(checkpoint_kinds)))
    assumptions = [
        str(record["atomwise_distribution_assumption"])
        for record in records
        if "atomwise_distribution_assumption" in record
    ]
    if assumptions:
        if len(assumptions) != len(records) or len(set(assumptions)) != 1:
            raise ValueError("atomwise distribution assumption differs across shards")
        combined["atomwise_distribution_assumption"] = np.asarray(assumptions[0])
    variants = [
        str(record["epic_variant"])
        for record in records
        if "epic_variant" in record
    ]
    if variants:
        if len(variants) != len(records) or len(set(variants)) != 1:
            raise ValueError("EPIC variant differs across shards")
        combined["epic_variant"] = np.asarray(variants[0])
    if len(combined["db_index"]) != selected_split_size:
        raise ValueError(
            f"aggregated structure count {len(combined['db_index'])} does not match "
            f"selected_split_size={selected_split_size}"
        )
    if len(np.unique(combined["db_index"])) != len(combined["db_index"]):
        raise ValueError("duplicate db_index across inference shards")
    order = np.argsort(combined["db_index"], kind="stable")
    old_offsets = []
    start = 0
    for record in records:
        n = len(record["db_index"])
        old_offsets.extend((record["force_offsets"][:-1] + start).tolist())
        start += len(record["force_true"])
    old_offsets = np.asarray(old_offsets, dtype=np.int64)
    lengths = combined["natoms"]
    force_order = np.concatenate([np.arange(old_offsets[i], old_offsets[i] + lengths[i]) for i in order])
    for field in structure_fields:
        if field in combined:
            combined[field] = combined[field][order]
    for field in atom_fields:
        if field in combined:
            combined[field] = combined[field][force_order]
    combined["force_offsets"] = np.concatenate(([0], np.cumsum(combined["natoms"]))).astype(np.int64)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **combined)
    return output_path
