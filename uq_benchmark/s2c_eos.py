"""Volumetric equation-of-state (EOS) uncertainty topography and sampling-bias diagnostics.

Implements Section 2.2 (Energy-volume topography exposes sampling biases),
Section 4.6 (Equation of state), and Supplementary Note 9 (Thermodynamic
equation-of-state curvature and asymmetric volumetric sampling bias across 54 crystals).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .s2c_covariance import build_ase_batch

DEFAULT_VOLUME_SCALES: tuple[float, ...] = tuple(
    float(round(x, 4)) for x in np.linspace(0.85, 1.15, 31)
)


def analyze_eos_uncertainty_profile(
    volumes_pa: np.ndarray,
    energies_pa: np.ndarray,
    epistemic_std_pa: np.ndarray,
    aleatoric_std_pa: np.ndarray | None = None,
) -> dict[str, Any]:
    """Analyze equilibrium volume V_0, minimum-uncertainty volume V_min^epist, and offset Delta V / V_0.

    Classifies each crystal into one of the three volumetric confidence regimes
    defined in Section 2.2 and Supplementary Note 9:
      - `compression_shifted`: Delta V / V_0 < -3% (-0.03)
      - `equilibrium_aligned`: |Delta V / V_0| <= 3% (0.03)
      - `expansion_shifted`: Delta V / V_0 > +3% (+0.03)
    """
    vol = np.asarray(volumes_pa, dtype=np.float64).reshape(-1)
    ene = np.asarray(energies_pa, dtype=np.float64).reshape(-1)
    epi = np.asarray(epistemic_std_pa, dtype=np.float64).reshape(-1)
    if not (vol.shape == ene.shape == epi.shape) or len(vol) < 3:
        raise ValueError("volumes_pa, energies_pa, and epistemic_std_pa must be matching arrays of length >= 3")
    if np.any(~np.isfinite(vol)) or np.any(~np.isfinite(ene)) or np.any(~np.isfinite(epi)):
        raise ValueError("EOS arrays must be finite")

    order = np.argsort(vol)
    vol = vol[order]
    ene = ene[order]
    epi = epi[order]
    if aleatoric_std_pa is not None:
        alea = np.asarray(aleatoric_std_pa, dtype=np.float64).reshape(-1)[order]
        tot = np.sqrt(np.square(epi) + np.square(alea))
    else:
        alea = None
        tot = epi.copy()

    idx_v0 = int(np.argmin(ene))
    idx_vmin_epi = int(np.argmin(epi))
    v0_pa = float(vol[idx_v0])
    e_eq_pa = float(ene[idx_v0])
    vmin_epi_pa = float(vol[idx_vmin_epi])
    delta_v_over_v0 = (vmin_epi_pa - v0_pa) / v0_pa

    if delta_v_over_v0 < -0.03:
        regime = "compression_shifted"
    elif delta_v_over_v0 > 0.03:
        regime = "expansion_shifted"
    else:
        regime = "equilibrium_aligned"

    rel_v = vol / v0_pa
    # Quadratic fit near equilibrium for stiffness curvature d^2(E/N)/d(V/V_0)^2|_{V_0}
    poly_coeffs = np.polyfit(rel_v - 1.0, ene, deg=2)
    stiffness_curvature = float(2.0 * poly_coeffs[0])

    # Tensile vs compressive epistemic uncertainty steepness <|d sigma_ep / d(V/V_0)|>
    d_epi_d_relv = np.gradient(epi, rel_v)
    comp_mask = rel_v <= 1.0
    exp_mask = rel_v >= 1.0
    steepness_comp = (
        float(np.mean(np.abs(d_epi_d_relv[comp_mask]))) if np.any(comp_mask) else float("nan")
    )
    steepness_exp = (
        float(np.mean(np.abs(d_epi_d_relv[exp_mask]))) if np.any(exp_mask) else float("nan")
    )
    modulation_ratio = float(np.max(epi) / max(float(np.min(epi)), 1e-30))

    return {
        "equilibrium_energy_pa_eV": e_eq_pa,
        "v0_pa_A3": v0_pa,
        "vmin_epist_pa_A3": vmin_epi_pa,
        "delta_v_over_v0": float(delta_v_over_v0),
        "delta_v_over_v0_percent": float(100.0 * delta_v_over_v0),
        "volumetric_confidence_regime": regime,
        "stiffness_curvature_eV_pa": stiffness_curvature,
        "epistemic_steepness_compressive": steepness_comp,
        "epistemic_steepness_tensile": steepness_exp,
        "tensile_steepened": bool(steepness_exp > steepness_comp),
        "epistemic_modulation_ratio": modulation_ratio,
        "volumes_pa_A3": vol,
        "relative_volumes": rel_v,
        "energies_pa_eV": ene,
        "epistemic_std_pa_eV": epi,
        "aleatoric_std_pa_eV": alea,
        "total_std_pa_eV": tot,
    }


def evaluate_eos_uncertainty_curve(
    model: torch.nn.Module,
    atoms,
    *,
    volume_scales: Sequence[float] | None = None,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Evaluate MACE+EPIC energy and epistemic/aleatoric uncertainty under isotropic volume scaling."""
    if volume_scales is None:
        volume_scales = DEFAULT_VOLUME_SCALES
    scales = [float(s) for s in volume_scales]
    if len(scales) < 3 or any(s <= 0 for s in scales):
        raise ValueError("volume_scales must contain at least 3 positive volume ratios")

    model.eval()
    natoms = len(atoms)
    base_cell = atoms.cell.array.copy()
    base_frac = atoms.get_scaled_positions(wrap=False)

    volumes_pa = []
    energies_pa = []
    epi_std_pa = []
    alea_std_pa = []

    for v_scale in scales:
        linear_scale = float(v_scale) ** (1.0 / 3.0)
        deformed = atoms.copy()
        deformed.set_cell(base_cell * linear_scale, scale_atoms=False)
        deformed.set_scaled_positions(base_frac)
        batch = build_ase_batch(model, deformed, device=device)
        with torch.no_grad():
            out = model(batch, training=False, compute_force=False)
        energy_tot = float(out["energy"].detach().cpu().reshape(()))
        epi_var_tot = float(out["energy_epistemic_variance"].detach().cpu().reshape(()))
        alea_var_tot = float(out["energy_aleatoric_variance"].detach().cpu().reshape(()))

        volumes_pa.append(float(deformed.get_volume()) / natoms)
        energies_pa.append(energy_tot / natoms)
        epi_std_pa.append(np.sqrt(max(epi_var_tot, 0.0)) / natoms)
        alea_std_pa.append(np.sqrt(max(alea_var_tot, 0.0)) / natoms)

    profile = analyze_eos_uncertainty_profile(
        np.asarray(volumes_pa),
        np.asarray(energies_pa),
        np.asarray(epi_std_pa),
        np.asarray(alea_std_pa),
    )
    profile["formula"] = atoms.get_chemical_formula()
    profile["natoms"] = natoms
    profile["volume_scales"] = np.asarray(scales, dtype=np.float64)
    return profile
