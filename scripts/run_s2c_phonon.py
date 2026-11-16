#!/usr/bin/env python3
"""Correlated weight-perturbation harmonic phonon UQ worker and stability screening CLI.

Implements Section 2.4, Section 4.6 (Phonon dispersions and dynamical stability
classification), and Supplementary Note 10:
  - Propagates Scheme 2c weight perturbations delta_w^(k) = tau_F * c_0(R_ref) * L @ z^(k)
    across 2x2x2 displaced Phonopy supercells with undisplaced supercell reference-force
    subtraction Delta F^(k) = F_disp^(k) - F_ref^(k).
  - Evaluates deterministic phonon dispersions omega^pt(q, nu) and K=32 Monte Carlo
    realizations omega^(k)(q, nu) along the standardized Seekpath Brillouin-zone path.
  - Classifies each structure into `robustly_stable`, `marginal` (high-suspicion
    soft-mode candidates), or `unstable` under omega_tol = -1.0 THz.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uq_benchmark.s2c_covariance import BASELINE_ID, DEFAULT_TAU_F, FEATURE_DIMENSION
from uq_benchmark.s2c_phonon import run_s2c_phonon_dispersion

LOG = logging.getLogger("s2c_phonon")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def load_cholesky_and_tau(covariance_path: Path, tau_override: float | None) -> tuple[np.ndarray, float]:
    with np.load(covariance_path, allow_pickle=False) as archive:
        chol = np.asarray(archive["cholesky"], dtype=np.float64)
        if tau_override is not None:
            tau_f = float(tau_override)
        elif "tau" in archive:
            tau_f = float(np.asarray(archive["tau"]).reshape(()))
        elif "a" in archive:
            tau_f = float(np.sqrt(np.asarray(archive["a"]).reshape(())))
        else:
            tau_f = DEFAULT_TAU_F
    if chol.shape != (FEATURE_DIMENSION, FEATURE_DIMENSION):
        raise ValueError(f"Expected {FEATURE_DIMENSION}x{FEATURE_DIMENSION} Cholesky factor, got {chol.shape}")
    return chol, tau_f


def save_case_outputs(case_dir: Path, db_index: int | None, result: dict) -> dict:
    started = time.monotonic()
    force_path = case_dir / "forces.npz"
    bands_path = case_dir / "bands.npz"
    atomic_npz(
        force_path,
        standard_normals=result["standard_normals"],
        latent_weights=result["latent_weights"],
        scaled_weights=result["scaled_weights"],
        reference_cell_point_forces=result["reference_cell_point_forces"],
        reference_cell_hidden_sum=result["reference_cell_hidden_sum"],
        reference_supercell_point_forces=result["reference_supercell_point_forces"],
        reference_supercell_force_samples=result["reference_supercell_force_samples"],
        phonopy_point_forces=result["phonopy_point_forces"],
        phonopy_force_samples=result["phonopy_force_samples"],
    )
    atomic_npz(
        bands_path,
        frequencies_point=result["frequencies_point"],
        frequencies_point_uncorrected=result["frequencies_point_uncorrected"],
        frequencies_samples=result["frequencies_samples"],
        frequencies_mc_mean=result["frequencies_mc_mean"],
        frequencies_mc_std=result["frequencies_mc_std"],
        frequencies_p05=result["frequencies_p05"],
        frequencies_p95=result["frequencies_p95"],
        sample_min_frequencies=result["sample_min_frequencies"],
        distances=result["distances"],
        qpoints=result["qpoints"],
    )
    summary = {
        "schema_version": 1,
        "completed": True,
        "baseline_id": BASELINE_ID,
        "db_index": -1 if db_index is None else int(db_index),
        "formula": result["formula"],
        "natoms": result["natoms"],
        "n_supercell_atoms": result["n_supercell_atoms"],
        "n_primitive_atoms": result["n_primitive_atoms"],
        "n_displacements": result["n_displacements"],
        "reference_energy": result["reference_energy"],
        "epistemic_std_eV_atom": result["reference_energy"]["epistemic_sigma_per_atom"],
        "aleatoric_std_eV_atom": result["reference_energy"]["aleatoric_sigma_per_atom"],
        "force_norm_max_eV_A": result["reference_energy"]["force_norm_max_eV_A"],
        "c0": result["c0"],
        "tau_f": result["tau_f"],
        "effective_amplitude": result["effective_amplitude"],
        "sampling_method": result["sampling_method"],
        "stability": result["stability"],
        "status": result["status"],
        "band_labels": result["band_labels"],
        "path_connections": result["path_connections"],
        "bands_path": "bands.npz",
        "forces_path": "forces.npz",
        "artifacts_sha256": {
            "bands.npz": sha256_file(bands_path),
            "forces.npz": sha256_file(force_path),
        },
        "save_seconds": time.monotonic() - started,
    }
    atomic_json(case_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained MACE+EPIC checkpoint.")
    parser.add_argument("--covariance-npz", type=Path, required=True, help="Path to s2c_covariance.npz.")
    parser.add_argument("--db-path", type=Path, help="Path to ASE database (e.g. mptrj.db).")
    parser.add_argument("--selection-csv", type=Path, help="CSV file with `db_index` column.")
    parser.add_argument("--db-index", type=int, help="Single 0-based ASE DB structure index to run.")
    parser.add_argument("--structure-file", type=Path, help="Single CIF/POSCAR structure file to run.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for phonon outputs.")
    parser.add_argument("--device", default="cuda", help="Torch device (e.g. cuda:0 or cpu).")
    parser.add_argument("--rank", type=int, default=0, help="Worker rank for sharded runs.")
    parser.add_argument("--world-size", type=int, default=1, help="Total workers for sharded runs.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of structures to process.")
    parser.add_argument("--supercell", type=int, nargs=3, default=[2, 2, 2], help="Supercell dimensions.")
    parser.add_argument("--displacement", type=float, default=0.01, help="Displacement amplitude in Angstrom.")
    parser.add_argument("--n-mc-samples", type=int, default=32, help="Number of MC realizations K.")
    parser.add_argument("--npoints", type=int, default=51, help="Points per Brillouin-zone segment.")
    parser.add_argument("--imag-tol", type=float, default=-1.0, help="Stability frequency threshold in THz.")
    parser.add_argument("--stable-threshold", type=float, default=0.95, help="Required stable MC fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for weight perturbations.")
    parser.add_argument("--tau-f", type=float, default=None, help="Optional override for tau_F.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.device))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_default_dtype(torch.float32)

    chol_f, tau_f = load_cholesky_and_tau(args.covariance_npz, args.tau_f)
    from uq_benchmark.inference import load_checkpoint_model

    model, state = load_checkpoint_model(args.checkpoint, device=args.device)
    if state.get("kind") != "epic_hierarchical":
        raise ValueError("Phonon UQ worker requires a hierarchical EPIC checkpoint")

    standard_normals = np.random.default_rng(args.seed).standard_normal(
        (args.n_mc_samples, FEATURE_DIMENSION)
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.structure_file is not None:
        from ase.io import read

        atoms = read(str(args.structure_file))
        case_dir = output_dir / args.structure_file.stem
        res = run_s2c_phonon_dispersion(
            model,
            atoms,
            chol_f,
            tau_f=tau_f,
            supercell=tuple(args.supercell),
            displacement=args.displacement,
            n_mc_samples=args.n_mc_samples,
            npoints=args.npoints,
            imag_tol=args.imag_tol,
            stable_threshold=args.stable_threshold,
            seed=args.seed,
            standard_normals=standard_normals,
            device=args.device,
        )
        summary = save_case_outputs(case_dir, None, res)
        print(json.dumps(summary["stability"], indent=2))
        return 0

    if args.db_path is None:
        raise ValueError("Either --structure-file or --db-path must be provided")
    from ase.db import connect

    database = connect(str(args.db_path))
    if args.db_index is not None:
        selected = [(0, int(args.db_index))]
        cases_root = output_dir / "probe"
    elif args.selection_csv is not None:
        with Path(args.selection_csv).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        indices = [int(r["db_index"]) for r in rows]
        selected = list(enumerate(indices))[args.rank :: args.world_size]
        if args.limit is not None:
            selected = selected[: int(args.limit)]
        cases_root = output_dir / "cases"
    else:
        raise ValueError("Provide either --db-index or --selection-csv when using --db-path")

    cases_root.mkdir(parents=True, exist_ok=True)
    failures = 0
    completed = 0
    for row_idx, db_idx in selected:
        case_dir = cases_root / f"db_{db_idx}"
        summary_path = case_dir / "summary.json"
        if summary_path.exists():
            LOG.info("db=%s already completed, skipping", db_idx)
            continue
        try:
            atoms = database.get(int(db_idx) + 1).toatoms()
            t0 = time.monotonic()
            res = run_s2c_phonon_dispersion(
                model,
                atoms,
                chol_f,
                tau_f=tau_f,
                supercell=tuple(args.supercell),
                displacement=args.displacement,
                n_mc_samples=args.n_mc_samples,
                npoints=args.npoints,
                imag_tol=args.imag_tol,
                stable_threshold=args.stable_threshold,
                seed=args.seed,
                standard_normals=standard_normals,
                device=args.device,
            )
            summary = save_case_outputs(case_dir, db_idx, res)
            completed += 1
            LOG.info(
                "db=%s (%s) completed: %s in %.1fs",
                db_idx,
                summary["formula"],
                summary["status"],
                time.monotonic() - t0,
            )
        except Exception as exc:
            failures += 1
            failure_log = output_dir / f"failures_rank{args.rank}.jsonl"
            with failure_log.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "db_index": db_idx,
                            "selection_row_index": row_idx,
                            "exception_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    + "\n"
                )
            LOG.exception("db=%s FAILED", db_idx)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    atomic_json(
        output_dir / f"worker_rank{args.rank}.json",
        {
            "rank": args.rank,
            "world_size": args.world_size,
            "assigned": len(selected),
            "completed": completed,
            "failures": failures,
            "status": "failed" if failures else "complete",
        },
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
