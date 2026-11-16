#!/usr/bin/env python3
"""Compute Scheme 2c (hepic_s2c_nig015) 3x3 atomic force uncertainty ellipsoids and VESTA CIFs.

Implements Section 2.3, Section 4.6 (Structural uncertainty ellipsoids), and
Supplementary Note 8:
  - Evaluates closed-form 3x3 atomic force covariance Sigma_{F_i}(R) = tau_F^2 * c_0^2 * J_{h,i} Sigma_F J_{h,i}^T
    and K-sample conservative Monte Carlo realizations.
  - Performs eigendecomposition into principal standard-deviation semi-axes
    sqrt(lambda_{i,1}) <= sqrt(lambda_{i,2}) <= sqrt(lambda_{i,3}) and orientation vectors.
  - Exports VESTA-compatible crystallographic CIF files with `_atom_site_aniso_U_*` records,
    NPZ archives, and CSV tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ase.db import connect
from ase.io import read
import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uq_benchmark.s2c_covariance import BASELINE_ID, DEFAULT_SIGMA0_F, DEFAULT_TAU_F
from uq_benchmark.s2c_ellipsoids import (
    evaluate_structure_ellipsoids,
    export_anisotropic_ellipsoid_cif,
)


def load_structures(args: argparse.Namespace) -> list[tuple[str, int | None, object]]:
    structures: list[tuple[str, int | None, object]] = []
    if args.structure_files:
        for path_str in args.structure_files:
            p = Path(path_str)
            atoms = read(str(p))
            structures.append((p.stem, None, atoms))
    if args.db_path is not None and args.db_indices:
        db = connect(str(args.db_path))
        for idx in args.db_indices:
            atoms = db.get(int(idx) + 1).toatoms()
            structures.append((f"db_{idx}_{atoms.get_chemical_formula()}", int(idx), atoms))
    if not structures:
        raise ValueError("Provide either --structure-files or (--db-path and --db-indices)")
    return structures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained MACE+EPIC checkpoint.")
    parser.add_argument("--covariance-npz", type=Path, required=True, help="Path to s2c_covariance.npz.")
    parser.add_argument("--db-path", type=Path, help="Optional ASE database path (e.g. mptrj.db).")
    parser.add_argument(
        "--db-indices",
        type=int,
        nargs="+",
        default=[0, 917, 1234, 1540354, 985361, 329314],
        help="0-based ASE DB structure indices to evaluate.",
    )
    parser.add_argument(
        "--structure-files",
        nargs="+",
        help="Optional CIF/POSCAR/XYZ files to evaluate directly instead of ASE DB indices.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--device", default="cuda", help="Torch device (e.g. cuda:0 or cpu).")
    parser.add_argument("--n-samples", type=int, default=32, help="MC realizations K (default: 32).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for MC realizations.")
    parser.add_argument("--tau-f", type=float, default=None, help="Optional override for tau_F.")
    parser.add_argument("--sigma0-f", type=float, default=None, help="Optional override for sigma_{0,F}.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.device))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    with np.load(args.covariance_npz, allow_pickle=False) as cov_arch:
        sigma_f = np.asarray(cov_arch["covariance"], dtype=np.float64)
        chol_f = np.asarray(cov_arch["cholesky"], dtype=np.float64)
        stored_tau = (
            float(np.asarray(cov_arch["tau"]).reshape(()))
            if "tau" in cov_arch
            else float(np.sqrt(np.asarray(cov_arch["a"]).reshape(())))
            if "a" in cov_arch
            else DEFAULT_TAU_F
        )
        stored_sigma0 = (
            float(np.asarray(cov_arch["sigma0"]).reshape(()))
            if "sigma0" in cov_arch
            else float(np.sqrt(np.asarray(cov_arch["b"]).reshape(())))
            if "b" in cov_arch
            else DEFAULT_SIGMA0_F
        )

    tau_f = float(args.tau_f) if args.tau_f is not None else stored_tau
    sigma0_f = float(args.sigma0_f) if args.sigma0_f is not None else stored_sigma0

    from uq_benchmark.inference import load_checkpoint_model

    model, _ = load_checkpoint_model(args.checkpoint, device=args.device)
    structures = load_structures(args)

    csv_rows = []
    manifest_cases = []
    for tag, db_idx, atoms in structures:
        res = evaluate_structure_ellipsoids(
            model,
            atoms,
            sigma_f,
            chol_f,
            tau_f=tau_f,
            sigma0_f=sigma0_f,
            n_samples=args.n_samples,
            seed=args.seed,
            device=args.device,
        )
        cif_analytic = output_dir / f"{tag}_analytic_s2c.cif"
        cif_mc = output_dir / f"{tag}_mc{args.n_samples}_s2c.cif"
        export_anisotropic_ellipsoid_cif(atoms, res["covariance_analytic"], cif_analytic)
        export_anisotropic_ellipsoid_cif(atoms, res["covariance_mc"], cif_mc)

        npz_path = output_dir / f"{tag}_s2c_ellipsoids.npz"
        np.savez_compressed(
            npz_path,
            db_index=np.asarray(-1 if db_idx is None else db_idx),
            atomic_numbers=atoms.numbers,
            positions_A=atoms.positions,
            cell_A=atoms.cell.array,
            nig_epi_per_atom=np.asarray(res["nig_epi_per_atom"]),
            c0=np.asarray(res["c0"]),
            tau_f=np.asarray(tau_f),
            sigma0_f=np.asarray(sigma0_f),
            point_force=res["point_force"],
            force_samples_mc=res["force_samples_mc"],
            covariance_raw=res["covariance_raw"],
            covariance_analytic=res["covariance_analytic"],
            covariance_mc=res["covariance_mc"],
            eigenvalues_analytic=res["eigendecomposition_analytic"]["eigenvalues"],
            eigenvectors_analytic=res["eigendecomposition_analytic"]["eigenvectors"],
        )

        decomp = res["eigendecomposition_analytic"]
        for atom_i, z_num in enumerate(atoms.numbers):
            csv_rows.append(
                {
                    "tag": tag,
                    "db_index": -1 if db_idx is None else int(db_idx),
                    "formula": res["formula"],
                    "atom_index": atom_i,
                    "atomic_number": int(z_num),
                    "rms_force_std": float(decomp["rms_force_std"][atom_i]),
                    "sigma_min": float(decomp["sigma_min"][atom_i]),
                    "sigma_mid": float(decomp["sigma_mid"][atom_i]),
                    "sigma_max": float(decomp["sigma_max"][atom_i]),
                    "anisotropy_ratio_max_min": float(decomp["anisotropy_ratio_max_min"][atom_i]),
                    "effective_rank_d_eff": float(decomp["effective_rank_d_eff"][atom_i]),
                    "fractional_anisotropy": float(decomp["fractional_anisotropy"][atom_i]),
                    "principal_axis_3_x": float(decomp["eigenvectors"][atom_i, 0, 2]),
                    "principal_axis_3_y": float(decomp["eigenvectors"][atom_i, 1, 2]),
                    "principal_axis_3_z": float(decomp["eigenvectors"][atom_i, 2, 2]),
                }
            )
        manifest_cases.append(
            {
                "tag": tag,
                "db_index": db_idx,
                "formula": res["formula"],
                "natoms": res["natoms"],
                "rms_sigma_f_eV_A": res["rms_sigma_f"],
                "max_semi_axis_eV_A": res["max_semi_axis_analytic"],
                "mc_vs_analytic_rel_frobenius_error": res[
                    "mean_relative_frobenius_error_mc_vs_analytic"
                ],
                "cif_analytic": cif_analytic.name,
                "cif_mc": cif_mc.name,
                "npz": npz_path.name,
            }
        )

    pd.DataFrame(csv_rows).to_csv(
        output_dir / "force_covariance_eigendecomposition.csv", index=False
    )
    manifest = {
        "baseline_id": BASELINE_ID,
        "tau_f": tau_f,
        "sigma0_f": sigma0_f,
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "cases": manifest_cases,
    }
    (output_dir / "ellipsoids_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
