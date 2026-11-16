"""Real-space 3x3 atomic force uncertainty ellipsoids and VESTA CIF export.

Implements Section 2.3, Section 4.6 (Structural uncertainty ellipsoids), and
Supplementary Note 8 (Wyckoff symmetry selection rules, Zingg triaxiality,
3D effective rank d_eff, fractional anisotropy FA, and Wishart K^{-1/2} MC convergence).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .s2c_covariance import (
    DEFAULT_SIGMA0_F,
    DEFAULT_TAU_F,
    build_ase_batch,
    compute_damped_evidence_scale,
    compute_feature_jacobian_batch,
    compute_s2c_atomic_force_covariances,
    draw_s2c_weight_perturbations,
)


def eigendecompose_atomic_force_covariances(
    force_covariances: np.ndarray,
) -> dict[str, np.ndarray]:
    """Eigendecompose 3x3 atomic force covariance matrices Sigma_{F_i} (Section 4.6 & SI Note 8).

    Sigma_{F_i} = U_i diag(lambda_{i,1}, lambda_{i,2}, lambda_{i,3}) U_i^T,
    with 0 <= lambda_{i,1} <= lambda_{i,2} <= lambda_{i,3} and principal standard-deviation
    semi-axes sqrt(lambda_{i,1}) <= sqrt(lambda_{i,2}) <= sqrt(lambda_{i,3}).
    """
    cov = np.asarray(force_covariances, dtype=np.float64)
    if cov.ndim != 3 or cov.shape[1:] != (3, 3):
        raise ValueError(f"force_covariances must have shape [N_atoms, 3, 3], got {cov.shape}")
    if not np.isfinite(cov).all():
        raise ValueError("force_covariances contains non-finite entries")

    cov_sym = 0.5 * (cov + cov.transpose(0, 2, 1))
    eigvals, eigvecs = np.linalg.eigh(cov_sym)
    eigvals = np.maximum(eigvals, 0.0)
    semi_axes = np.sqrt(eigvals)

    rms_std = np.sqrt(np.sum(eigvals, axis=1) / 3.0)
    anisotropy_ratio = semi_axes[:, 2] / np.maximum(semi_axes[:, 0], 1e-30)

    # Zingg triaxiality ratios: sigma_min / sigma_mid and sigma_mid / sigma_max
    zingg_b_over_a = np.where(
        semi_axes[:, 2] > 1e-15,
        semi_axes[:, 1] / np.maximum(semi_axes[:, 2], 1e-30),
        1.0,
    )
    zingg_c_over_b = np.where(
        semi_axes[:, 1] > 1e-15,
        semi_axes[:, 0] / np.maximum(semi_axes[:, 1], 1e-30),
        1.0,
    )

    # 3D effective participation rank d_eff = (sum_m lambda_m)^2 / sum_m lambda_m^2 (SI Note 8)
    trace_val = np.sum(eigvals, axis=1)
    sum_sq = np.sum(np.square(eigvals), axis=1)
    d_eff = np.where(sum_sq > 1e-30, np.square(trace_val) / np.maximum(sum_sq, 1e-30), 0.0)

    # Fractional anisotropy FA (SI Note 8)
    mean_semi = np.mean(semi_axes, axis=1, keepdims=True)
    num_fa = 1.5 * np.sum(np.square(semi_axes - mean_semi), axis=1)
    den_fa = np.sum(eigvals, axis=1)
    fractional_anisotropy = np.where(
        den_fa > 1e-30,
        np.sqrt(np.maximum(num_fa / np.maximum(den_fa, 1e-30), 0.0)),
        0.0,
    )

    return {
        "eigenvalues": eigvals,
        "eigenvectors": eigvecs,
        "semi_axes": semi_axes,
        "sigma_min": semi_axes[:, 0],
        "sigma_mid": semi_axes[:, 1],
        "sigma_max": semi_axes[:, 2],
        "rms_force_std": rms_std,
        "anisotropy_ratio_max_min": anisotropy_ratio,
        "zingg_mid_over_max": zingg_b_over_a,
        "zingg_min_over_mid": zingg_c_over_b,
        "effective_rank_d_eff": d_eff,
        "fractional_anisotropy": fractional_anisotropy,
    }


def export_anisotropic_ellipsoid_cif(
    atoms,
    force_covariances: np.ndarray,
    output_path: str | Path,
    *,
    display_scale: float | None = None,
    normalize_per_structure: bool = True,
) -> Path:
    """Export 3x3 atomic force uncertainty ellipsoids to a VESTA-compatible CIF file.

    Writes crystallographic `_atom_site_aniso_U_*` records and header metadata
    reporting exact physical magnitudes RMS(sigma_F) and max_i sqrt(lambda_{i,3}) in eV/A.
    """
    decomp = eigendecompose_atomic_force_covariances(force_covariances)
    cov = 0.5 * (
        np.asarray(force_covariances, dtype=np.float64)
        + np.asarray(force_covariances, dtype=np.float64).transpose(0, 2, 1)
    )
    n_atoms = len(atoms)
    if cov.shape[0] != n_atoms:
        raise ValueError(f"atoms count ({n_atoms}) does not match force_covariances ({cov.shape[0]})")

    rms_phys = float(np.sqrt(np.mean(np.trace(cov, axis1=1, axis2=2) / 3.0)))
    max_semi_phys = float(np.max(decomp["sigma_max"]))

    if display_scale is None:
        if normalize_per_structure and max_semi_phys > 0:
            cell_mean_len = float(np.linalg.norm(atoms.cell.array, axis=1).mean())
            typ_scale = float(np.quantile(decomp["semi_axes"][decomp["semi_axes"] > 0], 0.90))
            display_scale = 0.12 * cell_mean_len / max(typ_scale, 1e-15)
        else:
            display_scale = 1.0

    # Convert Cartesian standard-deviation ellipsoid scaling to U_cart in A^2 for CIF visualization
    u_cart = (float(display_scale) ** 2) * cov

    lengths = atoms.cell.lengths()
    angles = atoms.cell.angles()
    frac_coords = atoms.get_scaled_positions(wrap=False)
    symbols = atoms.get_chemical_symbols()
    formula = atoms.get_chemical_formula()

    lines = [
        f"data_{formula}_EPIC_S2c_ellipsoids",
        f"# EPIC Scheme 2c (hepic_s2c_nig015) 3x3 force uncertainty ellipsoids",
        f"# Physical RMS(sigma_F) = {rms_phys:.6e} eV/Angstrom",
        f"# Physical max_i sqrt(lambda_{{i,3}}) = {max_semi_phys:.6e} eV/Angstrom",
        f"# Visual display_scale = {float(display_scale):.6e} Angstrom / (eV/Angstrom)",
        "_symmetry_space_group_name_H-M   'P 1'",
        "_symmetry_Int_Tables_number      1",
        f"_cell_length_a                   {lengths[0]:.8f}",
        f"_cell_length_b                   {lengths[1]:.8f}",
        f"_cell_length_c                   {lengths[2]:.8f}",
        f"_cell_angle_alpha                {angles[0]:.6f}",
        f"_cell_angle_beta                 {angles[1]:.6f}",
        f"_cell_angle_gamma                {angles[2]:.6f}",
        "",
        "loop_",
        "  _atom_site_label",
        "  _atom_site_type_symbol",
        "  _atom_site_fract_x",
        "  _atom_site_fract_y",
        "  _atom_site_fract_z",
    ]
    labels = []
    for idx, (sym, frac) in enumerate(zip(symbols, frac_coords), start=1):
        label = f"{sym}{idx}"
        labels.append(label)
        lines.append(
            f"  {label:<8s} {sym:<4s} {frac[0]:12.8f} {frac[1]:12.8f} {frac[2]:12.8f}"
        )

    lines.extend(
        [
            "",
            "loop_",
            "  _atom_site_aniso_label",
            "  _atom_site_aniso_U_11",
            "  _atom_site_aniso_U_22",
            "  _atom_site_aniso_U_33",
            "  _atom_site_aniso_U_23",
            "  _atom_site_aniso_U_13",
            "  _atom_site_aniso_U_12",
        ]
    )
    for label, mat in zip(labels, u_cart):
        lines.append(
            f"  {label:<8s} {mat[0, 0]:13.6e} {mat[1, 1]:13.6e} {mat[2, 2]:13.6e} "
            f"{mat[1, 2]:13.6e} {mat[0, 2]:13.6e} {mat[0, 1]:13.6e}"
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def evaluate_structure_ellipsoids(
    model: torch.nn.Module,
    atoms,
    sigma_f: np.ndarray,
    chol_f: np.ndarray,
    *,
    tau_f: float = DEFAULT_TAU_F,
    sigma0_f: float = DEFAULT_SIGMA0_F,
    n_samples: int = 32,
    seed: int = 42,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Compute analytic and MC force uncertainty ellipsoids and Wishart convergence metrics."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    batch = build_ase_batch(model, atoms, device=device)
    jac_res = compute_feature_jacobian_batch(
        model, batch, compute_allocated=False, compute_point_force=True
    )
    j_mat = jac_res["jacobian_unweighted"].astype(np.float64)
    nig_epi_pa = float(jac_res["nig_epi_per_atom"][0])
    c0 = compute_damped_evidence_scale(nig_epi_pa * (len(atoms) ** 2), len(atoms))

    analytic = compute_s2c_atomic_force_covariances(
        j_mat, sigma_f, c0, tau_f=tau_f, sigma0_f=sigma0_f
    )
    cov_analytic = analytic["covariance_epistemic_calibrated"]
    decomp_analytic = eigendecompose_atomic_force_covariances(cov_analytic)

    scaled_weights, _, _ = draw_s2c_weight_perturbations(
        chol_f, float(c0), tau_f=tau_f, n_samples=n_samples, seed=seed
    )
    n_atoms = len(atoms)
    point_force = jac_res["point_force"]
    force_perturbations = (scaled_weights @ j_mat.T).reshape(n_samples, n_atoms, 3)
    force_samples = (point_force.astype(np.float64)[None, :, :] + force_perturbations).astype(
        np.float32
    )
    cov_mc = np.stack(
        [
            np.cov(force_samples[:, i, :].astype(np.float64), rowvar=False, ddof=1)
            for i in range(n_atoms)
        ],
        axis=0,
    )
    cov_mc = 0.5 * (cov_mc + cov_mc.transpose(0, 2, 1))
    decomp_mc = eigendecompose_atomic_force_covariances(cov_mc)

    frob_diff = np.linalg.norm(cov_mc - cov_analytic, axis=(1, 2))
    frob_ref = np.linalg.norm(cov_analytic, axis=(1, 2))
    active = frob_ref > 1e-12
    rel_frob = (
        float(np.mean(frob_diff[active] / frob_ref[active]))
        if np.any(active)
        else 0.0
    )

    return {
        "formula": atoms.get_chemical_formula(),
        "natoms": n_atoms,
        "nig_epi_per_atom": nig_epi_pa,
        "c0": float(c0),
        "tau_f": float(tau_f),
        "sigma0_f": float(sigma0_f),
        "point_force": point_force,
        "force_samples_mc": force_samples,
        "covariance_raw": analytic["covariance_raw"],
        "covariance_analytic": cov_analytic,
        "covariance_mc": cov_mc,
        "rms_sigma_f": analytic["rms_sigma_f"],
        "max_semi_axis_analytic": float(np.max(decomp_analytic["sigma_max"])),
        "eigendecomposition_analytic": decomp_analytic,
        "eigendecomposition_mc": decomp_mc,
        "mean_relative_frobenius_error_mc_vs_analytic": rel_frob,
    }
