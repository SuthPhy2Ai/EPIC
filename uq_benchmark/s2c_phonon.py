"""Correlated weight-perturbation harmonic phonon UQ propagation and stability screening.

Implements Section 2.4, Section 4.6 (Phonon dispersions and dynamical stability
classification), and Supplementary Note 10 (Eq. S7 reciprocal-space dynamical
matrix perturbation theory and soft-mode spectral Jacobian amplification).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .s2c_covariance import (
    DEFAULT_ETA,
    DEFAULT_TAU_F,
    SAMPLING_METHOD_MC,
    build_ase_batch,
    compute_damped_evidence_scale,
    draw_s2c_weight_perturbations,
    forward_pooled_hidden,
    sample_s2c_perturbed_pes_and_forces,
)


def classify_phonon_stability(
    point_frequencies: np.ndarray,
    sample_frequencies: np.ndarray,
    *,
    imag_tol: float = -1.0,
    stable_threshold: float = 0.95,
) -> tuple[dict[str, Any], np.ndarray]:
    """Classify harmonic phonon dynamical stability under EPIC uncertainty (Section 4.6).

    Categories:
      - `robustly_stable`: omega_min^pt > imag_tol, P_stable >= stable_threshold,
        and omega_{min,q05} > imag_tol.
      - `marginal` (high-suspicion soft-mode candidates): omega_min^pt > imag_tol,
        but P_stable < stable_threshold or omega_{min,q05} <= imag_tol.
      - `unstable`: omega_min^pt <= imag_tol.
    """
    pt_freq = np.asarray(point_frequencies, dtype=np.float64)
    mc_freq = np.asarray(sample_frequencies, dtype=np.float64)
    if mc_freq.ndim < 2 or mc_freq.shape[1:] != pt_freq.shape:
        raise ValueError(
            f"sample_frequencies shape {mc_freq.shape} does not match "
            f"[K, *point_frequencies.shape] with point shape {pt_freq.shape}"
        )
    if not np.isfinite(pt_freq).all() or not np.isfinite(mc_freq).all():
        raise ValueError("phonon frequencies must be finite")
    if not (0.0 < stable_threshold <= 1.0):
        raise ValueError("stable_threshold must lie in (0, 1]")

    reduce_axes = tuple(range(1, mc_freq.ndim))
    sample_minima = mc_freq.min(axis=reduce_axes)
    point_min = float(pt_freq.min())
    point_max = float(pt_freq.max())
    p05, p50 = np.percentile(sample_minima, [5, 50])
    stable_fraction = float(np.mean(sample_minima >= float(imag_tol)))

    if (
        point_min > float(imag_tol)
        and float(p05) > float(imag_tol)
        and stable_fraction >= float(stable_threshold)
    ):
        status = "robustly_stable"
    elif point_min > float(imag_tol):
        status = "marginal"
    else:
        status = "unstable"

    band_std = mc_freq.std(axis=0, ddof=0)
    count = len(sample_minima)
    z = 1.959963984540054
    denom = 1.0 + z * z / count
    center = (stable_fraction + z * z / (2.0 * count)) / denom
    radius = (
        z
        * math.sqrt(
            stable_fraction * (1.0 - stable_fraction) / count
            + z * z / (4.0 * count * count)
        )
        / denom
    )

    summary = {
        "status": status,
        "imag_tol": float(imag_tol),
        "stable_threshold": float(stable_threshold),
        "min_freq_mean_band": point_min,
        "min_freq_point_band": point_min,
        "omega_max_THz": point_max,
        "min_freq_p05": float(p05),
        "min_freq_p50": float(p50),
        "imaginary_depth_q05": float(max(0.0, -float(p05))),
        "worst_sample_min_freq": float(sample_minima.min()),
        "stable_sample_fraction": stable_fraction,
        "stable_sample_count": int(np.sum(sample_minima >= float(imag_tol))),
        "stable_fraction_wilson95": [
            float(max(0.0, center - radius)),
            float(min(1.0, center + radius)),
        ],
        "band_std_mean": float(band_std.mean()),
        "band_std_max": float(band_std.max()),
        "scope": "sample minimum over the computed high-symmetry path; not the full Brillouin zone",
    }
    return summary, sample_minima


def compute_spectral_jacobian_perturbation(
    point_frequencies: np.ndarray,
    sample_frequencies: np.ndarray,
    *,
    epsilon: float = 0.038,
) -> dict[str, Any]:
    """Analyze reciprocal-space dynamical matrix perturbation scaling (SI Note 10, Eq. S7).

    Verifies the universal inverse-frequency spectral Jacobian amplification law:
      sigma_omega(q, nu) / sigma_D(q, nu) approx 1 / (2 * sqrt(omega_pt(q, nu)^2 + epsilon^2)),
    where lambda_D = sign(omega) * omega^2 is the dynamical-matrix eigenvalue.
    """
    pt_freq = np.asarray(point_frequencies, dtype=np.float64)
    mc_freq = np.asarray(sample_frequencies, dtype=np.float64)
    if mc_freq.shape[1:] != pt_freq.shape:
        raise ValueError("sample_frequencies and point_frequencies shapes are incompatible")

    sigma_omega = mc_freq.std(axis=0, ddof=0)
    signed_lambda_d = np.sign(mc_freq) * np.square(mc_freq)
    sigma_d = signed_lambda_d.std(axis=0, ddof=0)

    active = sigma_d > 1e-12
    empirical_jacobian = np.where(active, sigma_omega / np.maximum(sigma_d, 1e-30), np.nan)
    theoretical_jacobian = 1.0 / (2.0 * np.sqrt(np.square(pt_freq) + float(epsilon) ** 2))

    valid = active & (np.abs(pt_freq) > 1e-3) & (empirical_jacobian > 0)
    if int(np.sum(valid)) >= 2:
        log_emp = np.log10(empirical_jacobian[valid])
        log_theo = np.log10(theoretical_jacobian[valid])
        pearson_r = float(np.corrcoef(log_emp, log_theo)[0, 1])
    else:
        pearson_r = float("nan")

    return {
        "epsilon_THz": float(epsilon),
        "sigma_omega": sigma_omega,
        "sigma_D": sigma_d,
        "empirical_spectral_jacobian": empirical_jacobian,
        "theoretical_spectral_jacobian": theoretical_jacobian,
        "log_pearson_r": pearson_r,
    }


def evaluate_reference_structure_properties(
    model: torch.nn.Module,
    atoms,
    device: str | torch.device = "cuda",
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate reference-cell NIG parameters, point forces, and pooled features H(R_ref)."""
    batch = build_ase_batch(model, atoms, device=device)
    output, _, hidden_sum = forward_pooled_hidden(model, batch)
    params = {
        name: float(output[f"energy_{name}"].detach().cpu().reshape(()))
        for name in ("nu", "alpha", "beta")
    }
    if params["nu"] <= 0 or params["alpha"] <= 1 or params["beta"] <= 0:
        raise ValueError("reference NIG parameters are outside their valid domain")
    energy = float(output["energy"].detach().cpu().reshape(()))
    aleatoric = params["beta"] / (params["alpha"] - 1.0)
    epistemic = aleatoric / params["nu"]
    natoms = len(atoms)
    info = {
        "energy": energy,
        **params,
        "epistemic_variance_total": epistemic,
        "aleatoric_variance_total": aleatoric,
        "total_variance_total": epistemic + aleatoric,
        "epistemic_sigma_per_atom": math.sqrt(epistemic) / natoms,
        "aleatoric_sigma_per_atom": math.sqrt(aleatoric) / natoms,
        "total_sigma_per_atom": math.sqrt(epistemic + aleatoric) / natoms,
    }
    force = -torch.autograd.grad(output["energy"].sum(), batch.positions)[0]
    force_np = force.detach().cpu().numpy()
    info["force_norm_max_eV_A"] = float(np.linalg.norm(force_np, axis=1).max())
    info["force_component_abs_max_eV_A"] = float(np.abs(force_np).max())
    return info, force_np, hidden_sum.detach().cpu().numpy()


def _phonopy_atoms_to_ase(cell):
    from ase import Atoms

    return Atoms(
        symbols=cell.symbols,
        cell=cell.cell,
        scaled_positions=cell.scaled_positions,
        pbc=True,
    )


def run_s2c_phonon_dispersion(
    model: torch.nn.Module,
    atoms,
    chol_f: np.ndarray,
    *,
    tau_f: float = DEFAULT_TAU_F,
    eta: float = DEFAULT_ETA,
    supercell: Sequence[int] = (2, 2, 2),
    displacement: float = 0.01,
    n_mc_samples: int = 32,
    npoints: int = 51,
    imag_tol: float = -1.0,
    stable_threshold: float = 0.95,
    seed: int = 42,
    standard_normals: np.ndarray | None = None,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Run Scheme 2c harmonic phonon dispersion and MC uncertainty propagation (Section 4.6).

    Computes deterministic force constants Phi_{ij} and K perturbed force constants
    Phi_{ij}^(k) on a supercell (default 2x2x2, displacement 0.01 A) with c_0(R_ref),
    H_0, and delta_w^(k) held fixed across all displaced supercells and subtracting
    the undisplaced supercell reference forces prior to force-constant fitting.
    """
    from phonopy import Phonopy
    from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath
    from phonopy.structure.atoms import PhonopyAtoms

    if len(atoms) == 0 or not np.all(atoms.pbc):
        raise ValueError("phonon dispersion requires a non-empty, 3D-periodic crystal structure")
    if len(supercell) != 3 or any(int(s) < 1 for s in supercell):
        raise ValueError("supercell must contain 3 positive integers")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ref_energy_info, ref_forces, ref_hidden = evaluate_reference_structure_properties(
        model, atoms, device=device
    )
    c0 = float(
        compute_damped_evidence_scale(
            ref_energy_info["epistemic_variance_total"], len(atoms), eta=eta
        )
    )
    scaled_weights, latent_weights, z_draws = draw_s2c_weight_perturbations(
        chol_f,
        c0,
        tau_f=tau_f,
        n_samples=n_mc_samples,
        seed=seed,
        standard_normals=standard_normals,
    )

    unitcell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
        masses=atoms.get_masses(),
    )
    phonon = Phonopy(unitcell, supercell_matrix=np.diag([int(x) for x in supercell]))
    phonon.generate_displacements(distance=float(displacement))
    supercells = phonon.supercells_with_displacements
    if not supercells or any(sc is None for sc in supercells):
        raise ValueError("Phonopy generated no displaced supercells")

    bands, labels, connections = get_band_qpoints_by_seekpath(
        phonon.primitive, int(npoints), is_const_interval=False
    )
    bands_array = np.stack(bands).astype(np.float64)

    equilibrium = sample_s2c_perturbed_pes_and_forces(
        model,
        _phonopy_atoms_to_ase(phonon.supercell),
        chol_f,
        c0,
        tau_f=tau_f,
        scaled_weights=scaled_weights,
        device=device,
    )
    displaced = []
    for sc in supercells:
        displaced.append(
            sample_s2c_perturbed_pes_and_forces(
                model,
                _phonopy_atoms_to_ase(sc),
                chol_f,
                c0,
                tau_f=tau_f,
                hidden_anchor=equilibrium["hidden_sum"],
                scaled_weights=scaled_weights,
                device=device,
            )
        )

    raw_point = np.stack([item["point_force"] for item in displaced])
    raw_samples = np.stack([item["force_samples"] for item in displaced])
    corrected_point = raw_point.astype(np.float64) - equilibrium["point_force"].astype(
        np.float64
    )
    corrected_samples = (
        raw_samples.astype(np.float64) - equilibrium["force_samples"].astype(np.float64)[None]
    )

    def _evaluate_bands(forces_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phonon.forces = forces_arr
        phonon.produce_force_constants()
        phonon.run_band_structure(
            bands,
            path_connections=connections,
            labels=labels,
            is_legacy_plot=False,
            with_eigenvectors=False,
            with_group_velocities=False,
        )
        res_dict = phonon.get_band_structure_dict()
        freqs = np.stack(res_dict["frequencies"]).astype(np.float64)
        dists = np.stack(res_dict["distances"]).astype(np.float64)
        return freqs, dists

    point_frequencies, distances = _evaluate_bands(corrected_point)
    point_uncorrected, _ = _evaluate_bands(raw_point)
    mc_freq_list = []
    for k in range(len(scaled_weights)):
        freqs_k, _ = _evaluate_bands(corrected_samples[:, k])
        mc_freq_list.append(freqs_k)
    sample_frequencies = np.stack(mc_freq_list, axis=0)

    stability, minima = classify_phonon_stability(
        point_frequencies,
        sample_frequencies,
        imag_tol=imag_tol,
        stable_threshold=stable_threshold,
    )

    return {
        "formula": atoms.get_chemical_formula(),
        "natoms": len(atoms),
        "n_supercell_atoms": len(phonon.supercell),
        "n_primitive_atoms": len(phonon.primitive),
        "n_displacements": len(supercells),
        "reference_energy": ref_energy_info,
        "c0": c0,
        "tau_f": float(tau_f),
        "effective_amplitude": float(tau_f) * c0,
        "sampling_method": SAMPLING_METHOD_MC,
        "stability": stability,
        "status": stability["status"],
        "band_labels": list(labels),
        "path_connections": [bool(x) for x in connections],
        "frequencies_point": point_frequencies,
        "frequencies_point_uncorrected": point_uncorrected,
        "frequencies_samples": sample_frequencies,
        "frequencies_mc_mean": sample_frequencies.mean(axis=0),
        "frequencies_mc_std": sample_frequencies.std(axis=0, ddof=0),
        "frequencies_p05": np.percentile(sample_frequencies, 5, axis=0),
        "frequencies_p95": np.percentile(sample_frequencies, 95, axis=0),
        "sample_min_frequencies": minima,
        "distances": distances,
        "qpoints": bands_array,
        "standard_normals": z_draws,
        "latent_weights": latent_weights,
        "scaled_weights": scaled_weights,
        "reference_cell_point_forces": ref_forces,
        "reference_cell_hidden_sum": ref_hidden,
        "reference_supercell_point_forces": equilibrium["point_force"],
        "reference_supercell_force_samples": equilibrium["force_samples"],
        "phonopy_point_forces": corrected_point,
        "phonopy_force_samples": corrected_samples,
    }
