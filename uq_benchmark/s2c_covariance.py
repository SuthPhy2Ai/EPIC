"""Scheme 2c (hepic_s2c_nig015) feature-Jacobian covariance and force UQ propagation.

Implements Section 4.6 (Eqs. 14--20) of the EPIC manuscript:
  - Eq. (14): Last-layer feature-Jacobian Gram matrix G_F and ridge-regularized
    inverse covariance Sigma_F = (G_F + lambda_cov * I_D)^(-1) = L @ L^T
    with lambda_cov = 1e-3 * tr(G_F) / D (D = 64).
  - Eq. (15): Damped structure-level energy evidence scale
    c_0(R_ref) = (sigma_{ep,G}^2(R_ref) / N_ref^2)^eta with eta = 0.075 (2*eta = 0.15).
  - Eq. (16): Perturbed scalar potential energy surface realizations
    delta_w^(k) = tau_F * c_0 * L @ z^(k),
    E_G^(k)(R) = gamma_G(R) + [H(R) - H_0]^T delta_w^(k).
  - Eq. (17): Conservative Monte Carlo force realizations
    F_i^(k)(R) = -grad_{R_i} E_G^(k)(R) = F_i(R) - J_{h,i}(R) delta_w^(k).
  - Eq. (18): Closed-form 3x3 atomic force covariance tensor
    Sigma_{F_i}(R) = tau_F^2 * c_0^2 * J_{h,i}(R) Sigma_F J_{h,i}(R)^T.
  - Eq. (19): Post-hoc affine variance recalibration for marginal intervals
    (sigma_{F,i,alpha}^cal(R))^2 = (Sigma_{F_i}(R))_{alpha,alpha} + sigma_{0,F}^2.
  - Eq. (20): Normalized candidate prescreen score S_pre and active-learning
    DFT review prioritization trigger.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from scipy.stats import norm, rankdata

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .calibration import apply_variance_map, fit_gaussian_variance_map
from .distributions import gaussian_nll

BASELINE_ID = "hepic_s2c_nig015"
SAMPLING_METHOD_ANALYTIC = "hepic_s2c_nig015_analytic_jacobian_covariance"
SAMPLING_METHOD_MC = "s2c_local_frozen_reference_scale_gaussian_weights"

FEATURE_DIMENSION = 64
DEFAULT_RIDGE_MULTIPLIER = 1e-3
DEFAULT_NIG_EXPONENT = 0.15
DEFAULT_ETA = 0.075  # 2 * eta = 0.15 on variance

# Post-hoc affine calibration parameters fitted on the disjoint calibration split
# (Section 4.6 Eq. 19, Table 2, and Supplementary Note 3 Table S2)
DEFAULT_TAU_F = 47.42317194970634
DEFAULT_SIGMA0_F = 0.010240335903317568
DEFAULT_TAU_F_10K = 48.2697106123643
DEFAULT_SIGMA0_F_10K = 0.003373804686343892

DEFAULT_TAU_E = 1.007753
DEFAULT_SIGMA0_E = 0.02228
DEFAULT_TAU_E_EPI = 0.5698
DEFAULT_SIGMA0_E_EPI = 0.008023


def compute_damped_evidence_scale(
    epistemic_variance_total: np.ndarray | float,
    natoms: np.ndarray | int,
    eta: float = DEFAULT_ETA,
) -> np.ndarray | float:
    """Compute the damped structure-level energy-evidence scale c_0 (Eq. 15).

    c_0(R_ref) = (sigma_{ep,G}^2(R_ref) / N_ref^2)^eta, with eta = 0.075
    corresponding to a power-law damping exponent 2*eta = 0.15 on the
    per-atom epistemic energy variance.
    """
    if eta <= 0 or not math.isfinite(eta):
        raise ValueError("eta must be finite and positive")
    if isinstance(epistemic_variance_total, (int, float)) and isinstance(natoms, (int, float)):
        if epistemic_variance_total <= 0 or natoms <= 0:
            raise ValueError("epistemic_variance_total and natoms must be positive")
        per_atom_var = float(epistemic_variance_total) / (float(natoms) ** 2)
        return float(per_atom_var ** eta)

    epi_var = np.asarray(epistemic_variance_total, dtype=np.float64)
    counts = np.asarray(natoms, dtype=np.float64)
    if np.any(~np.isfinite(epi_var)) or np.any(epi_var <= 0):
        raise ValueError("epistemic_variance_total must be finite and positive")
    if np.any(~np.isfinite(counts)) or np.any(counts <= 0):
        raise ValueError("natoms must be finite and positive")
    per_atom_var = epi_var / np.square(counts)
    return np.power(per_atom_var, eta)


def build_ase_batch(model: torch.nn.Module, atoms, device: str | torch.device = "cuda"):
    """Construct a single-structure PyG Batch with grad-enabled positions from ASE Atoms."""
    from mace import data as mace_data
    from mace.tools import torch_geometric, utils

    base = getattr(model, "base_model", model)
    z_table = utils.AtomicNumberTable([int(z) for z in base.atomic_numbers])
    config = mace_data.config_from_atoms(atoms, head_name="Default")
    record = mace_data.AtomicData.from_config(
        config, z_table=z_table, cutoff=float(base.r_max), heads=["Default"]
    )
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[record], batch_size=1, shuffle=False, drop_last=False
    )
    batch = next(iter(loader)).to(device)
    batch.positions.requires_grad_(True)
    return batch


def forward_pooled_hidden(
    model: torch.nn.Module,
    batch,
    feature_dim: int = FEATURE_DIMENSION,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Run forward pass and extract live last-layer invariant readout features (Eq. 5).

    Returns:
        output: Model output dictionary.
        hidden: Atom-wise shared invariant features tilde{h}_i of shape [N_atoms, D].
        hidden_sum: Pooled structure feature vector H(R) = sum_i tilde{h}_i of shape [D].
    """
    if not batch.positions.requires_grad:
        batch.positions.requires_grad_(True)
    output = model(batch, training=False, compute_force=False)
    base = getattr(model, "base_model", model)
    if hasattr(base, "edl_readout") and hasattr(base.edl_readout, "_hidden"):
        hidden = base.edl_readout._hidden(output["node_feats"])
    else:
        final_width = base.readouts[-1].linear_1.irreps_in.dim
        final_node = output["node_feats"][:, -final_width:]
        hidden = base.readouts[-1].non_linearity(base.readouts[-1].linear_1(final_node))
    if hidden.ndim != 2 or hidden.shape[1] != feature_dim or not hidden.requires_grad:
        raise ValueError(
            f"expected live invariant hidden features of shape [natoms, {feature_dim}], "
            f"got {tuple(hidden.shape)} (requires_grad={hidden.requires_grad})"
        )
    if not torch.isfinite(hidden).all() or not torch.isfinite(output["energy"]).all():
        raise FloatingPointError("non-finite hidden features or energy during forward pass")
    return output, hidden, hidden.sum(dim=0)


def compute_feature_jacobian_batch(
    model: torch.nn.Module,
    batch,
    *,
    compute_allocated: bool = False,
    compute_point_force: bool = False,
    feature_dim: int = FEATURE_DIMENSION,
) -> dict[str, Any]:
    """Compute last-layer feature-Jacobian matrix J_h and NIG variances for a batch.

    By convention matching the benchmark shards, column d of `jacobian_unweighted`
    stores -dH_d/dR in R^{3*N_atoms x D} (which has the exact same quadratic form
    J_h @ Sigma_F @ J_h^T as +dH_d/dR).
    """
    output, hidden, phi_unw = forward_pooled_hidden(model, batch, feature_dim=feature_dim)
    dim = int(hidden.shape[1])
    counts = (batch.ptr[1:] - batch.ptr[:-1]).to(hidden.dtype)
    w_epi = output["atomic_epistemic_weight"]
    rho = counts[batch.batch] * w_epi

    phi_alloc = (torch.sqrt(rho)[:, None] * hidden).sum(dim=0) if compute_allocated else None

    point_force = None
    if compute_point_force:
        grad_e = torch.autograd.grad(
            output["energy"].sum(), batch.positions, retain_graph=True, create_graph=False
        )[0]
        point_force = (-grad_e).detach().cpu().numpy().astype(np.float32)

    cols_unw: list[torch.Tensor] = []
    cols_alloc: list[torch.Tensor] = []
    for d in range(dim):
        retain_unw = True if compute_allocated else (d + 1 < dim)
        g1 = torch.autograd.grad(
            phi_unw[d], batch.positions, retain_graph=retain_unw, create_graph=False
        )[0]
        cols_unw.append(-g1.detach().reshape(-1, 1))
        if compute_allocated and phi_alloc is not None:
            g2 = torch.autograd.grad(
                phi_alloc[d], batch.positions, retain_graph=(d + 1 < dim), create_graph=False
            )[0]
            cols_alloc.append(-g2.detach().reshape(-1, 1))

    j_unw = torch.cat(cols_unw, dim=1).to(torch.float32).cpu().numpy()
    nig_epi_pa = (
        (output["energy_epistemic_variance"] / (counts ** 2))
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    nig_alea_pa = (
        (output["energy_aleatoric_variance"] / (counts ** 2))
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    result: dict[str, Any] = {
        "jacobian_unweighted": j_unw,
        "hidden_sum": phi_unw.detach().cpu().numpy().astype(np.float32),
        "nig_epi_per_atom": nig_epi_pa,
        "nig_alea_per_atom": nig_alea_pa,
        "rho": rho.detach().cpu().numpy().astype(np.float32),
        "ptr": batch.ptr.detach().cpu().numpy().astype(np.int64),
        "energy_pred_total": output["energy"].detach().cpu().numpy().astype(np.float64),
    }
    if compute_allocated:
        result["jacobian_allocated"] = (
            torch.cat(cols_alloc, dim=1).to(torch.float32).cpu().numpy()
        )
    if point_force is not None:
        result["point_force"] = point_force
    return result


def accumulate_force_jacobian_gram(
    model: torch.nn.Module,
    dataloader,
    *,
    device: str | torch.device = "cuda",
    feature_dim: int = FEATURE_DIMENSION,
    normalize_by_structures: bool = False,
) -> tuple[np.ndarray, int]:
    """Accumulate the 64x64 calibration feature-Jacobian Gram matrix G_F (Eq. 14).

    G_F = sum_{c in D_cal} sum_{i=1}^{N_c} J_{h,i}(R_c)^T J_{h,i}(R_c).
    If `normalize_by_structures` is True, divides G_F by N_cal (which is equivalent
    up to a scalar constant absorbed into tau_F^2).
    """
    gram = np.zeros((feature_dim, feature_dim), dtype=np.float64)
    n_structures = 0
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for batch in dataloader:
        batch = batch.to(device)
        res = compute_feature_jacobian_batch(
            model, batch, compute_allocated=False, feature_dim=feature_dim
        )
        j_mat = res["jacobian_unweighted"].astype(np.float64)
        gram += j_mat.T @ j_mat
        n_structures += int(len(res["ptr"]) - 1)
    if n_structures <= 0:
        raise ValueError("dataloader yielded zero structures for Gram matrix accumulation")
    if normalize_by_structures:
        gram = gram / float(n_structures)
    return gram, n_structures


def build_s2c_covariance(
    gram_force: np.ndarray,
    ridge_multiplier: float = DEFAULT_RIDGE_MULTIPLIER,
) -> dict[str, Any]:
    """Build ridge-regularized inverse covariance Sigma_F and Cholesky factor L (Eq. 14).

    Sigma_F = (G_F + lambda_cov * I_D)^(-1) = L @ L^T,
    where lambda_cov = ridge_multiplier * tr(G_F) / D.
    """
    gram = np.asarray(gram_force, dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"gram_force must be a square 2D matrix, got shape {gram.shape}")
    dim = int(gram.shape[0])
    if not np.isfinite(gram).all():
        raise ValueError("gram_force contains non-finite values")
    max_abs = max(1.0, float(np.abs(gram).max()))
    if not np.allclose(gram, gram.T, rtol=1e-10, atol=1e-12 * max_abs):
        raise ValueError("gram_force must be symmetric")
    if ridge_multiplier <= 0 or not math.isfinite(ridge_multiplier):
        raise ValueError("ridge_multiplier must be finite and positive")

    gram = 0.5 * (gram + gram.T)
    trace_val = float(np.trace(gram))
    trace_scale = trace_val / float(dim)
    if not math.isfinite(trace_scale) or trace_scale <= 0:
        raise ValueError("gram_force trace must be finite and positive")

    ridge = float(ridge_multiplier) * trace_scale
    precision = gram + ridge * np.eye(dim, dtype=np.float64)
    precision_eigvals = np.linalg.eigvalsh(precision)
    if not np.isfinite(precision_eigvals).all() or np.any(precision_eigvals <= 0):
        raise ValueError("regularized precision matrix G_F + lambda_cov * I is not SPD")

    covariance = np.linalg.inv(precision)
    covariance = 0.5 * (covariance + covariance.T)
    cholesky = np.linalg.cholesky(covariance)
    recon_err = float(
        np.linalg.norm(cholesky @ cholesky.T - covariance) / np.linalg.norm(covariance)
    )
    if recon_err > 1e-9:
        raise ValueError(f"Cholesky factor failed to reconstruct covariance: error={recon_err}")

    gram_eigvals = np.maximum(np.linalg.eigvalsh(gram)[::-1], 0.0)
    cov_eigvals = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0.0)
    d_eff_gram = float((gram_eigvals.sum() ** 2) / np.square(gram_eigvals).sum())
    d_eff_cov = float((cov_eigvals.sum() ** 2) / np.square(cov_eigvals).sum())

    return {
        "gram_force": gram,
        "covariance": covariance,
        "cholesky": cholesky,
        "ridge": ridge,
        "ridge_multiplier": float(ridge_multiplier),
        "gram_trace": trace_val,
        "precision_condition_number": float(precision_eigvals[-1] / precision_eigvals[0]),
        "gram_condition_number": float(
            gram_eigvals[0] / max(float(gram_eigvals[-1]), np.finfo(np.float64).tiny)
        ),
        "d_eff_gram": d_eff_gram,
        "d_eff_covariance": d_eff_cov,
        "cholesky_reconstruction_relative_error": recon_err,
    }


def compute_s2c_atomic_force_covariances(
    jacobian: np.ndarray,
    sigma_f: np.ndarray,
    c0: float | np.ndarray,
    *,
    tau_f: float = DEFAULT_TAU_F,
    sigma0_f: float = DEFAULT_SIGMA0_F,
) -> dict[str, Any]:
    """Compute closed-form 3x3 atomic force covariance tensors and calibrated variances (Eqs. 18--19).

    Args:
        jacobian: Coordinate Jacobian J_h of shape [3*N_atoms, D] or [N_atoms, 3, D].
        sigma_f: Regularized inverse Gram matrix Sigma_F of shape [D, D].
        c0: Damped energy-evidence scale c_0(R) = (sigma_{ep,G}^2 / N^2)^{0.075}
            (scalar for a single structure, or array of shape [N_atoms]).
        tau_f: Multiplicative force calibration scale tau_F (default: 47.4232).
        sigma0_f: Additive force noise floor sigma_{0,F} in eV/A (default: 0.010240).

    Returns:
        Dictionary with:
          - `covariance_geometry`: J_{h,i} Sigma_F J_{h,i}^T [N_atoms, 3, 3]
          - `covariance_raw`: c_0^2 J_{h,i} Sigma_F J_{h,i}^T [N_atoms, 3, 3]
          - `covariance_epistemic_calibrated`: tau_F^2 c_0^2 J_{h,i} Sigma_F J_{h,i}^T [N_atoms, 3, 3] (Eq. 18)
          - `covariance_total_calibrated`: Sigma_{F_i} + sigma_{0,F}^2 I_3 [N_atoms, 3, 3]
          - `variance_raw`: diagonal of `covariance_raw` [N_atoms, 3]
          - `variance_calibrated`: (Sigma_{F_i})_{alpha,alpha} + sigma_{0,F}^2 [N_atoms, 3] (Eq. 19)
          - `std_calibrated`: sqrt(variance_calibrated) [N_atoms, 3]
          - `rms_sigma_f_raw`: structure-level RMS(sigma_F) uncalibrated
          - `rms_sigma_f`: structure-level RMS(sigma_F) = sqrt(1/(3N) sum_i tr(Sigma_{F_i}))
    """
    sigma_f_arr = np.asarray(sigma_f, dtype=np.float64)
    dim = int(sigma_f_arr.shape[0])
    j_arr = np.asarray(jacobian, dtype=np.float64)
    if j_arr.ndim == 2:
        if j_arr.shape[0] % 3 != 0 or j_arr.shape[1] != dim:
            raise ValueError(f"invalid 2D jacobian shape {j_arr.shape} for D={dim}")
        n_atoms = j_arr.shape[0] // 3
        j_blocks = j_arr.reshape(n_atoms, 3, dim)
    elif j_arr.ndim == 3:
        if j_arr.shape[1] != 3 or j_arr.shape[2] != dim:
            raise ValueError(f"invalid 3D jacobian shape {j_arr.shape} for D={dim}")
        n_atoms = j_arr.shape[0]
        j_blocks = j_arr
    else:
        raise ValueError("jacobian must have 2 or 3 dimensions")

    cov_geom = np.einsum("nai,ij,nbj->nab", j_blocks, sigma_f_arr, j_blocks, optimize=True)
    cov_geom = 0.5 * (cov_geom + cov_geom.transpose(0, 2, 1))

    c0_arr = np.asarray(c0, dtype=np.float64)
    if c0_arr.ndim == 0:
        c0_sq = float(c0_arr) ** 2
        cov_raw = c0_sq * cov_geom
    elif c0_arr.shape == (n_atoms,):
        cov_raw = (c0_arr[:, None, None] ** 2) * cov_geom
    else:
        raise ValueError(f"c0 must be a scalar or shape ({n_atoms},), got {c0_arr.shape}")

    a_scale = float(tau_f) ** 2
    b_floor = float(sigma0_f) ** 2
    cov_epi_cal = a_scale * cov_raw
    cov_tot_cal = cov_epi_cal + b_floor * np.eye(3, dtype=np.float64)[None, :, :]

    var_raw = np.maximum(np.diagonal(cov_raw, axis1=1, axis2=2), 0.0)
    var_epi_cal = np.maximum(np.diagonal(cov_epi_cal, axis1=1, axis2=2), 0.0)
    var_tot_cal = var_epi_cal + b_floor

    rms_raw = float(np.sqrt(np.mean(var_raw)))
    rms_epi_cal = float(np.sqrt(np.mean(var_epi_cal)))
    rms_tot_cal = float(np.sqrt(np.mean(var_tot_cal)))

    return {
        "covariance_geometry": cov_geom,
        "covariance_raw": cov_raw,
        "covariance_epistemic_calibrated": cov_epi_cal,
        "covariance_total_calibrated": cov_tot_cal,
        "variance_raw": var_raw,
        "variance_epistemic_calibrated": var_epi_cal,
        "variance_calibrated": var_tot_cal,
        "std_calibrated": np.sqrt(var_tot_cal),
        "rms_sigma_f_raw": rms_raw,
        "rms_sigma_f": rms_epi_cal,
        "rms_sigma_f_total": rms_tot_cal,
        "tau_f": float(tau_f),
        "sigma0_f": float(sigma0_f),
    }


def draw_s2c_weight_perturbations(
    cholesky_l: np.ndarray,
    c0: float,
    *,
    tau_f: float = DEFAULT_TAU_F,
    n_samples: int = 32,
    seed: int = 42,
    standard_normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw correlated weight perturbations delta_w^(k) = tau_F * c_0 * L @ z^(k) (Eq. 16).

    Returns:
        scaled_weights: Array of shape [K, D] where row k is delta_w^(k).
        latent_weights: Unscaled L @ z^(k) of shape [K, D].
        standard_normals: Standard normal draws z^(k) of shape [K, D].
    """
    chol = np.asarray(cholesky_l, dtype=np.float64)
    dim = int(chol.shape[0])
    if chol.shape != (dim, dim):
        raise ValueError("cholesky_l must be square")
    if standard_normals is None:
        if n_samples < 2:
            raise ValueError("n_samples must be >= 2")
        rng = np.random.default_rng(int(seed))
        z_draws = rng.standard_normal((int(n_samples), dim))
    else:
        z_draws = np.asarray(standard_normals, dtype=np.float64)
        if z_draws.ndim != 2 or z_draws.shape[1] != dim:
            raise ValueError(f"standard_normals must have shape [K, {dim}]")
    amplitude = float(tau_f) * float(c0)
    latent_weights = z_draws @ chol.T
    scaled_weights = amplitude * latent_weights
    return scaled_weights, latent_weights, z_draws


def sample_s2c_perturbed_pes_and_forces(
    model: torch.nn.Module,
    atoms,
    cholesky_l: np.ndarray,
    c0: float,
    *,
    tau_f: float = DEFAULT_TAU_F,
    n_samples: int = 32,
    seed: int = 42,
    hidden_anchor: np.ndarray | None = None,
    standard_normals: np.ndarray | None = None,
    scaled_weights: np.ndarray | None = None,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Evaluate perturbed scalar PES E_G^(k)(R) and conservative forces F_i^(k)(R) (Eqs. 16--17).

    E_G^(k)(R) = gamma_G(R) + [H(R) - H_0]^T delta_w^(k),
    F_i^(k)(R) = -grad_{R_i} E_G^(k)(R) = F_i(R) - J_{h,i}(R) delta_w^(k).
    """
    if scaled_weights is None:
        scaled_weights, latent_weights, z_draws = draw_s2c_weight_perturbations(
            cholesky_l,
            c0,
            tau_f=tau_f,
            n_samples=n_samples,
            seed=seed,
            standard_normals=standard_normals,
        )
    else:
        scaled_weights = np.asarray(scaled_weights, dtype=np.float64)
        latent_weights = None
        z_draws = standard_normals

    batch = build_ase_batch(model, atoms, device=device)
    output, _, hidden_sum = forward_pooled_hidden(
        model, batch, feature_dim=int(scaled_weights.shape[1])
    )
    anchor = (
        hidden_sum.detach()
        if hidden_anchor is None
        else torch.as_tensor(hidden_anchor, device=hidden_sum.device, dtype=hidden_sum.dtype)
    )
    if anchor.shape != hidden_sum.shape or anchor.requires_grad:
        raise ValueError("hidden_anchor must be a frozen 1D tensor matching feature_dim")

    weights_t = torch.as_tensor(
        scaled_weights, device=hidden_sum.device, dtype=hidden_sum.dtype
    )
    if weights_t.ndim != 2 or weights_t.shape[1] != hidden_sum.numel() or weights_t.requires_grad:
        raise ValueError("scaled_weights must be a frozen 2D tensor [K, D]")

    mean_energy = output["energy"].sum()
    perturbations = (hidden_sum - anchor) @ weights_t.T
    sampled_energies = (mean_energy.detach() + perturbations.detach()).cpu().numpy()

    mean_force = -torch.autograd.grad(mean_energy, batch.positions, retain_graph=True)[0]
    mean_force_detached = mean_force.detach()

    force_realizations = []
    for idx, pert in enumerate(perturbations.unbind()):
        grad_pert = torch.autograd.grad(
            pert,
            batch.positions,
            retain_graph=(idx + 1 < len(weights_t)),
            create_graph=False,
            allow_unused=False,
        )[0]
        force_realizations.append((mean_force_detached - grad_pert.detach()).cpu().numpy())

    result: dict[str, Any] = {
        "point_energy": float(mean_energy.detach().cpu().item()),
        "energy_samples": sampled_energies,
        "point_force": mean_force_detached.cpu().numpy(),
        "force_samples": np.stack(force_realizations, axis=0),
        "hidden_sum": hidden_sum.detach().cpu().numpy(),
        "scaled_weights": scaled_weights,
    }
    if latent_weights is not None:
        result["latent_weights"] = latent_weights
    if z_draws is not None:
        result["standard_normals"] = z_draws
    return result


def fit_s2c_affine_calibration(
    force_true: np.ndarray,
    force_mean: np.ndarray,
    raw_force_variance: np.ndarray,
    *,
    energy_true_pa: np.ndarray | None = None,
    energy_pred_pa: np.ndarray | None = None,
    raw_energy_variance_pa: np.ndarray | None = None,
    method: str = BASELINE_ID,
) -> dict[str, Any]:
    """Fit post-hoc affine Gaussian variance recalibration on calibration data (Eq. 19).

    Minimizes Gaussian NLL over (tau, sigma_0) where:
      sigma_cal^2 = tau^2 * sigma_raw^2 + sigma_0^2.
    """
    f_true = np.asarray(force_true, dtype=np.float64).reshape(-1)
    f_mean = np.asarray(force_mean, dtype=np.float64).reshape(-1)
    f_var = np.asarray(raw_force_variance, dtype=np.float64).reshape(-1)
    if f_true.shape != f_mean.shape or f_true.shape != f_var.shape:
        raise ValueError("force_true, force_mean, and raw_force_variance must have matching size")

    force_state = fit_gaussian_variance_map(f_true, f_mean, f_var, method=f"{method}_force")
    tau_f = float(force_state.parameters["tau"])
    sigma0_f = float(force_state.parameters["sigma0"])
    a_f = tau_f ** 2
    b_f = sigma0_f ** 2
    cal_var_f = apply_variance_map(f_var, force_state)
    cal_std_f = np.sqrt(cal_var_f)
    f_err = np.abs(f_mean - f_true)
    f_nll = float(np.mean(gaussian_nll(f_true, f_mean, cal_std_f)))
    f_cov95 = float(np.mean(f_err <= norm.ppf(0.975) * cal_std_f))

    report: dict[str, Any] = {
        "method": method,
        "force": {
            "tau_F": tau_f,
            "sigma0_F": sigma0_f,
            "a": a_f,
            "b": b_f,
            "n_components": int(f_true.size),
            "mae": float(np.mean(f_err)),
            "rmse": float(np.sqrt(np.mean(np.square(f_err)))),
            "calibrated_nll": f_nll,
            "coverage_95": f_cov95,
        },
    }

    if (
        energy_true_pa is not None
        and energy_pred_pa is not None
        and raw_energy_variance_pa is not None
    ):
        e_true = np.asarray(energy_true_pa, dtype=np.float64).reshape(-1)
        e_pred = np.asarray(energy_pred_pa, dtype=np.float64).reshape(-1)
        e_var = np.asarray(raw_energy_variance_pa, dtype=np.float64).reshape(-1)
        e_state = fit_gaussian_variance_map(e_true, e_pred, e_var, method=f"{method}_energy")
        tau_e = float(e_state.parameters["tau"])
        sigma0_e = float(e_state.parameters["sigma0"])
        cal_var_e = apply_variance_map(e_var, e_state)
        cal_std_e = np.sqrt(cal_var_e)
        e_err = np.abs(e_pred - e_true)
        e_nll = float(np.mean(gaussian_nll(e_true, e_pred, cal_std_e)))
        e_cov95 = float(np.mean(e_err <= norm.ppf(0.975) * cal_std_e))
        report["energy"] = {
            "tau_E": tau_e,
            "sigma0_E": sigma0_e,
            "n_structures": int(e_true.size),
            "mae": float(np.mean(e_err)),
            "rmse": float(np.sqrt(np.mean(np.square(e_err)))),
            "calibrated_nll": e_nll,
            "coverage_95": e_cov95,
        }

    return report


def compute_prescreen_score(
    features: dict[str, np.ndarray] | np.ndarray,
    weights: dict[str, float] | np.ndarray | None = None,
    *,
    normalization: Literal["minmax", "percentile"] = "minmax",
) -> np.ndarray:
    """Compute normalized multi-criteria prescreen score S_pre (Eq. 20).

    S_pre = sum_j w_j * x_j', where x_j' = (x_j - min(x_j)) / (max(x_j) - min(x_j))
    for `normalization="minmax"`, or empirical percentile rank in [0, 1] for
    `normalization="percentile"`.
    """
    if isinstance(features, dict):
        keys = list(features.keys())
        if not keys:
            raise ValueError("features dictionary must not be empty")
        matrix = np.column_stack([np.asarray(features[k], dtype=np.float64) for k in keys])
        if weights is None:
            w_vec = np.full(len(keys), 1.0 / len(keys), dtype=np.float64)
        elif isinstance(weights, dict):
            w_vec = np.asarray([float(weights[k]) for k in keys], dtype=np.float64)
        else:
            w_vec = np.asarray(weights, dtype=np.float64)
    else:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if weights is None:
            w_vec = np.full(matrix.shape[1], 1.0 / matrix.shape[1], dtype=np.float64)
        else:
            w_vec = np.asarray(weights, dtype=np.float64)

    if matrix.ndim != 2 or len(w_vec) != matrix.shape[1]:
        raise ValueError("features and weights dimensions do not match")
    if np.any(~np.isfinite(matrix)) or np.any(~np.isfinite(w_vec)):
        raise ValueError("features and weights must be finite")

    normed = np.empty_like(matrix, dtype=np.float64)
    n_rows = matrix.shape[0]
    for col in range(matrix.shape[1]):
        x = matrix[:, col]
        if normalization == "minmax":
            xmin, xmax = float(np.min(x)), float(np.max(x))
            span = xmax - xmin
            normed[:, col] = (x - xmin) / span if span > 0 else 0.0
        elif normalization == "percentile":
            if n_rows <= 1:
                normed[:, col] = 0.0
            else:
                normed[:, col] = (rankdata(x, method="average") - 1.0) / float(n_rows - 1)
        else:
            raise ValueError(f"unsupported normalization mode: {normalization}")

    return normed @ w_vec


def compute_dft_review_trigger_score(
    force_rms_std: np.ndarray,
    energy_epi_std_pa: np.ndarray,
    max_pred_force: np.ndarray,
    *,
    weights: tuple[float, float, float] = (0.5, 0.2, 0.3),
) -> np.ndarray:
    """Compute the EPIC combined energy-force active-learning DFT review trigger (Section 2.1).

    Combines percentile ranks of structure-level RMS(sigma_F), per-atom energy epistemic
    uncertainty sigma_{ep,G}/N, and maximum predicted atomic force magnitude max_i ||F_i||_2
    with weights (0.5, 0.2, 0.3).
    """
    f_rms = np.asarray(force_rms_std, dtype=np.float64).reshape(-1)
    e_epi = np.asarray(energy_epi_std_pa, dtype=np.float64).reshape(-1)
    f_max = np.asarray(max_pred_force, dtype=np.float64).reshape(-1)
    if not (f_rms.shape == e_epi.shape == f_max.shape):
        raise ValueError("force_rms_std, energy_epi_std_pa, and max_pred_force must have identical length")
    stacked = np.column_stack([f_rms, e_epi, f_max])
    return compute_prescreen_score(stacked, np.asarray(weights, dtype=np.float64), normalization="percentile")
