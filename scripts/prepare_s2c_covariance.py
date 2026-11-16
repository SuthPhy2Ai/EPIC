#!/usr/bin/env python3
"""Prepare the Scheme 2c (hepic_s2c_nig015) 64x64 feature-Jacobian covariance and calibration.

Supports two execution modes:
  1. `from-checkpoint`: Computes the calibration feature-Jacobian Gram matrix G_F
     directly from a trained MACE+EPIC checkpoint on the calibration split (Eq. 14),
     constructs Sigma_F = (G_F + lambda_cov * I_64)^(-1) = L @ L^T, and saves
     `s2c_covariance.npz` and `covariance_manifest.json`.
  2. `from-shards`: Reconstructs G_F, Sigma_F, L, and the post-hoc Gaussian affine
     calibration fit (tau_F, sigma_{0,F}) from precomputed calibration Jacobian shards
     and point/force prediction archives.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uq_benchmark.config import load_config
from uq_benchmark.s2c_covariance import (
    BASELINE_ID,
    DEFAULT_NIG_EXPONENT,
    DEFAULT_RIDGE_MULTIPLIER,
    DEFAULT_SIGMA0_F,
    DEFAULT_TAU_F,
    FEATURE_DIMENSION,
    accumulate_force_jacobian_gram,
    build_s2c_covariance,
    fit_s2c_affine_calibration,
)
from uq_benchmark.split import hashed_subset, load_split


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def run_from_checkpoint(args: argparse.Namespace) -> int:
    from uq_benchmark.data import load_dataset, make_data_loader
    from uq_benchmark.inference import load_checkpoint_model

    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    split_path = Path(args.split_path or (Path(config.data.split_dir) / "split_indices.npz"))
    cal_indices = load_split(split_path, "calibration")
    if args.subset_size is not None and int(args.subset_size) < len(cal_indices):
        cal_indices = hashed_subset(cal_indices, int(args.subset_size), "s2c-calibration-gram-v1")

    model, state = load_checkpoint_model(args.checkpoint, device=args.device)
    if state.get("kind") != "epic_hierarchical":
        raise ValueError(f"Expected epic_hierarchical checkpoint, got {state.get('kind')}")

    from torch.utils.data import Subset

    loader = make_data_loader(
        Subset(load_dataset(config), cal_indices.tolist()),
        config,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    gram_force, n_structures = accumulate_force_jacobian_gram(
        model,
        loader,
        device=args.device,
        feature_dim=args.feature_dim,
        normalize_by_structures=args.normalize_gram,
    )
    cov_info = build_s2c_covariance(gram_force, ridge_multiplier=args.ridge_multiplier)

    tau_f = float(args.tau_f)
    sigma0_f = float(args.sigma0_f)
    output_npz = output_dir / "s2c_covariance.npz"
    np.savez_compressed(
        output_npz,
        gram_force=cov_info["gram_force"],
        covariance=cov_info["covariance"],
        cholesky=cov_info["cholesky"],
        ridge=np.asarray(cov_info["ridge"]),
        ridge_multiplier=np.asarray(args.ridge_multiplier),
        nig_exponent=np.asarray(args.nig_exponent),
        calibration_db_index=cal_indices,
        a=np.asarray(tau_f ** 2),
        b=np.asarray(sigma0_f ** 2),
        tau=np.asarray(tau_f),
        sigma0=np.asarray(sigma0_f),
    )

    manifest = {
        "status": "completed",
        "mode": "from-checkpoint",
        "baseline_id": BASELINE_ID,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "n_calibration_structures": n_structures,
        "feature_dimension": int(args.feature_dim),
        "ridge_multiplier": float(args.ridge_multiplier),
        "nig_exponent": float(args.nig_exponent),
        "gram_trace": cov_info["gram_trace"],
        "ridge": cov_info["ridge"],
        "precision_condition_number": cov_info["precision_condition_number"],
        "d_eff_gram": cov_info["d_eff_gram"],
        "d_eff_covariance": cov_info["d_eff_covariance"],
        "cholesky_reconstruction_relative_error": cov_info["cholesky_reconstruction_relative_error"],
        "output_npz": str(output_npz),
        "output_npz_sha256": sha256_file(output_npz),
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = output_dir / "covariance_manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def run_from_shards(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = Path(args.shard_dir).resolve()

    with np.load(args.point_calibration, allow_pickle=False) as point_arch:
        point = {k: point_arch[k] for k in point_arch.files}
    with np.load(args.force_calibration, allow_pickle=False) as force_arch:
        force = {k: force_arch[k] for k in force_arch.files}

    cal_size = len(force["db_index"])
    selected = np.searchsorted(point["db_index"], force["db_index"])
    if np.any(selected >= len(point["db_index"])) or not np.array_equal(
        point["db_index"][selected], force["db_index"]
    ):
        raise ValueError("Force calibration db_index is not aligned with point calibration archive")

    direct_forces = np.concatenate(
        [point["force_pred"][point["force_offsets"][i] : point["force_offsets"][i + 1]] for i in selected]
    ).astype(np.float64)
    truth_forces = force["force_true"].astype(np.float64)
    component_offsets = force["force_offsets"].astype(np.int64) * 3
    dim = int(args.feature_dim)

    jacobian = np.empty((int(component_offsets[-1]), dim), dtype=np.float32)
    nig = np.empty(cal_size, dtype=np.float64)
    gram = np.zeros((dim, dim), dtype=np.float64)

    for rank in range(int(args.num_shards)):
        shard_path = shard_dir / f"shard_{rank}.pt"
        record = torch.load(shard_path, map_location="cpu", weights_only=False)
        cal = record["cal"]
        positions = np.arange(rank, cal_size, int(args.num_shards))
        local_gram = np.asarray(cal["G_F"], dtype=np.float64)
        gram += local_gram
        for local_idx, pos in enumerate(positions):
            begin, end = component_offsets[pos : pos + 2]
            jacobian[begin:end] = np.asarray(cal["J_unw"][local_idx], dtype=np.float32)
            nig[pos] = float(cal["nig_epi"][local_idx])
        del cal, record
        gc.collect()

    cov_info = build_s2c_covariance(gram, ridge_multiplier=args.ridge_multiplier)
    covariance = cov_info["covariance"]
    cholesky = cov_info["cholesky"]

    raw_variance = np.empty(len(jacobian), dtype=np.float64)
    for pos in range(cal_size):
        begin, end = component_offsets[pos : pos + 2]
        local = jacobian[begin:end].astype(np.float64)
        raw_variance[begin:end] = (nig[pos] ** float(args.nig_exponent)) * np.sum(
            (local @ covariance) * local, axis=1
        )
    del jacobian

    cal_fit = fit_s2c_affine_calibration(
        truth_forces, direct_forces, raw_variance, method=BASELINE_ID
    )
    f_fit = cal_fit["force"]

    output_npz = output_dir / "s2c_covariance.npz"
    np.savez_compressed(
        output_npz,
        gram_force=cov_info["gram_force"],
        covariance=covariance,
        cholesky=cholesky,
        ridge=np.asarray(cov_info["ridge"]),
        ridge_multiplier=np.asarray(args.ridge_multiplier),
        nig_exponent=np.asarray(args.nig_exponent),
        calibration_db_index=force["db_index"],
        a=np.asarray(f_fit["a"]),
        b=np.asarray(f_fit["b"]),
        tau=np.asarray(f_fit["tau_F"]),
        sigma0=np.asarray(f_fit["sigma0_F"]),
    )

    output_cal = output_dir / "s2c_calibration.json"
    atomic_json(output_cal, cal_fit)

    manifest = {
        "status": "completed",
        "mode": "from-shards",
        "baseline_id": BASELINE_ID,
        "n_calibration_structures": cal_size,
        "n_force_components": int(raw_variance.size),
        "gram_trace": cov_info["gram_trace"],
        "ridge": cov_info["ridge"],
        "precision_condition_number": cov_info["precision_condition_number"],
        "d_eff_gram": cov_info["d_eff_gram"],
        "d_eff_covariance": cov_info["d_eff_covariance"],
        "tau_F": f_fit["tau_F"],
        "sigma0_F": f_fit["sigma0_F"],
        "output_npz": str(output_npz),
        "output_npz_sha256": sha256_file(output_npz),
        "calibration_json": str(output_cal),
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = output_dir / "covariance_manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["from-checkpoint", "from-shards"],
        default="from-checkpoint",
        help="Compute G_F from a checkpoint or reconstruct from cached Jacobian shards.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "epic_single_gpu.example.yaml",
        help="Path to benchmark YAML configuration.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Path to trained MACE+EPIC checkpoint.")
    parser.add_argument("--split-path", type=Path, help="Optional override for split_indices.npz.")
    parser.add_argument("--subset-size", type=int, default=10000, help="Calibration subset size.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for Jacobian extraction.")
    parser.add_argument("--device", default="cuda", help="Torch device (e.g. cuda:0 or cpu).")
    parser.add_argument("--normalize-gram", action="store_true", help="Divide G_F by N_cal.")
    parser.add_argument("--shard-dir", type=Path, help="Directory with shard_*.pt for from-shards mode.")
    parser.add_argument("--num-shards", type=int, default=12, help="Number of calibration shards.")
    parser.add_argument("--point-calibration", type=Path, help="Point calibration aggregated.npz.")
    parser.add_argument("--force-calibration", type=Path, help="Force calibration aggregated.npz.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--feature-dim", type=int, default=FEATURE_DIMENSION)
    parser.add_argument("--ridge-multiplier", type=float, default=DEFAULT_RIDGE_MULTIPLIER)
    parser.add_argument("--nig-exponent", type=float, default=DEFAULT_NIG_EXPONENT)
    parser.add_argument("--tau-f", type=float, default=DEFAULT_TAU_F)
    parser.add_argument("--sigma0-f", type=float, default=DEFAULT_SIGMA0_F)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "from-checkpoint":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --mode=from-checkpoint")
        return run_from_checkpoint(args)
    if args.shard_dir is None or args.point_calibration is None or args.force_calibration is None:
        raise ValueError(
            "--shard-dir, --point-calibration, and --force-calibration are required when --mode=from-shards"
        )
    return run_from_shards(args)


if __name__ == "__main__":
    raise SystemExit(main())
