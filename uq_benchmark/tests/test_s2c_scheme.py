"""Unit tests for Scheme 2c (hepic_s2c_nig015) covariance, ellipsoids, phonon UQ, and EOS."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from ase import Atoms
import numpy as np
import pytest

from uq_benchmark.s2c_covariance import (
    DEFAULT_ETA,
    DEFAULT_NIG_EXPONENT,
    DEFAULT_RIDGE_MULTIPLIER,
    DEFAULT_SIGMA0_F,
    DEFAULT_TAU_F,
    FEATURE_DIMENSION,
    build_s2c_covariance,
    compute_damped_evidence_scale,
    compute_dft_review_trigger_score,
    compute_prescreen_score,
    compute_s2c_atomic_force_covariances,
    draw_s2c_weight_perturbations,
    fit_s2c_affine_calibration,
)
from uq_benchmark.s2c_ellipsoids import (
    eigendecompose_atomic_force_covariances,
    export_anisotropic_ellipsoid_cif,
)
from uq_benchmark.s2c_eos import analyze_eos_uncertainty_profile
from uq_benchmark.s2c_phonon import (
    classify_phonon_stability,
    compute_spectral_jacobian_perturbation,
)


def test_eq14_build_s2c_covariance_and_cholesky() -> None:
    rng = np.random.default_rng(42)
    q, _ = np.linalg.qr(rng.standard_normal((FEATURE_DIMENSION, FEATURE_DIMENSION)))
    stiff_eigs = np.logspace(7.0, 2.5, FEATURE_DIMENSION)
    gram = (q * stiff_eigs) @ q.T
    gram = 0.5 * (gram + gram.T)
    info = build_s2c_covariance(gram, ridge_multiplier=DEFAULT_RIDGE_MULTIPLIER)

    expected_ridge = DEFAULT_RIDGE_MULTIPLIER * np.trace(gram) / FEATURE_DIMENSION
    assert np.isclose(info["ridge"], expected_ridge, rtol=1e-12)

    cov = info["covariance"]
    chol = info["cholesky"]
    assert cov.shape == (FEATURE_DIMENSION, FEATURE_DIMENSION)
    assert chol.shape == (FEATURE_DIMENSION, FEATURE_DIMENSION)
    assert np.allclose(chol @ chol.T, cov, rtol=1e-10, atol=1e-12)
    assert np.allclose(np.linalg.inv(gram + expected_ridge * np.eye(FEATURE_DIMENSION)), cov, rtol=1e-10)
    assert info["d_eff_covariance"] > info["d_eff_gram"]


def test_eq15_damped_evidence_scale() -> None:
    epi_var_total = 0.64
    natoms = 8
    c0 = compute_damped_evidence_scale(epi_var_total, natoms, eta=DEFAULT_ETA)
    expected_pa_var = epi_var_total / (natoms ** 2)
    assert np.isclose(c0, expected_pa_var ** 0.075, rtol=1e-12)
    assert np.isclose(c0 ** 2, expected_pa_var ** DEFAULT_NIG_EXPONENT, rtol=1e-12)


def test_eq16_to_eq19_force_covariance_and_mc_wishart_convergence() -> None:
    rng = np.random.default_rng(7)
    a_mat = rng.standard_normal((200, FEATURE_DIMENSION))
    cov_info = build_s2c_covariance(a_mat.T @ a_mat)
    sigma_f = cov_info["covariance"]
    chol_f = cov_info["cholesky"]

    n_atoms = 4
    jacobian = rng.standard_normal((3 * n_atoms, FEATURE_DIMENSION))
    c0 = 0.82
    tau_f = DEFAULT_TAU_F
    sigma0_f = DEFAULT_SIGMA0_F

    analytic = compute_s2c_atomic_force_covariances(
        jacobian, sigma_f, c0, tau_f=tau_f, sigma0_f=sigma0_f
    )
    cov_epi = analytic["covariance_epistemic_calibrated"]
    cov_tot = analytic["covariance_total_calibrated"]
    assert cov_epi.shape == (n_atoms, 3, 3)
    assert np.allclose(
        cov_tot - cov_epi,
        (sigma0_f ** 2) * np.eye(3)[None, :, :],
        rtol=1e-7,
        atol=1e-10,
    )

    scaled_w, _, _ = draw_s2c_weight_perturbations(
        chol_f, c0, tau_f=tau_f, n_samples=4096, seed=123
    )
    force_pert = (scaled_w @ jacobian.T).reshape(4096, n_atoms, 3)
    mc_cov = np.stack(
        [np.cov(force_pert[:, i, :], rowvar=False, ddof=1) for i in range(n_atoms)],
        axis=0,
    )
    rel_frob = np.linalg.norm(mc_cov - cov_epi) / np.linalg.norm(cov_epi)
    assert rel_frob < 0.05


def test_eq19_affine_calibration_fit() -> None:
    rng = np.random.default_rng(99)
    n = 4000
    raw_var = rng.uniform(1e-5, 1e-2, size=n)
    true_tau = 25.0
    true_sigma0 = 0.02
    true_std = np.sqrt((true_tau ** 2) * raw_var + true_sigma0 ** 2)
    location = rng.normal(0.0, 1.0, size=n)
    target = location + rng.normal(0.0, true_std)

    report = fit_s2c_affine_calibration(target, location, raw_var)
    f_res = report["force"]
    assert 0.90 <= f_res["coverage_95"] <= 0.99
    assert np.isfinite(f_res["calibrated_nll"])


def test_eq20_prescreen_and_dft_review_trigger() -> None:
    f_rms = np.array([0.01, 0.05, 0.20, 0.40])
    e_epi = np.array([0.005, 0.015, 0.010, 0.050])
    f_max = np.array([0.1, 0.5, 1.2, 3.0])

    s_pre = compute_prescreen_score({"f": f_rms, "e": e_epi}, {"f": 0.6, "e": 0.4}, normalization="minmax")
    assert s_pre.shape == (4,)
    assert np.isclose(s_pre[0], 0.0)
    assert np.isclose(s_pre[-1], 1.0)

    trigger = compute_dft_review_trigger_score(f_rms, e_epi, f_max)
    assert trigger.shape == (4,)
    assert trigger[-1] > trigger[0]


def test_ellipsoid_symmetry_and_cif_export(tmp_path: Path) -> None:
    cov_rank0 = np.zeros((1, 3, 3))
    cov_rank1 = np.diag([0.0, 0.0, 0.25])[None, :, :]
    cov_rank2 = np.diag([0.0, 0.16, 0.25])[None, :, :]
    cov_rank3 = np.diag([0.09, 0.16, 0.25])[None, :, :]
    covs = np.concatenate([cov_rank0, cov_rank1, cov_rank2, cov_rank3], axis=0)

    decomp = eigendecompose_atomic_force_covariances(covs)
    assert np.isclose(decomp["effective_rank_d_eff"][0], 0.0)
    assert np.isclose(decomp["effective_rank_d_eff"][1], 1.0)
    assert np.isclose(decomp["fractional_anisotropy"][1], 1.0)
    assert 1.5 < decomp["effective_rank_d_eff"][2] <= 2.0
    assert 2.0 < decomp["effective_rank_d_eff"][3] <= 3.0

    atoms = Atoms(
        symbols="AgNi3",
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        cell=[3.8, 3.8, 3.8],
        pbc=True,
    )
    cif_path = export_anisotropic_ellipsoid_cif(atoms, covs, tmp_path / "AgNi3.cif")
    text = cif_path.read_text()
    assert "_atom_site_aniso_U_11" in text
    assert "Physical RMS(sigma_F)" in text


def test_phonon_stability_classification_and_spectral_jacobian() -> None:
    pt = np.linspace(0.05, 8.0, 60).reshape(2, 10, 3)
    rng = np.random.default_rng(11)

    mc_stable = pt[None, ...] + rng.normal(0.0, 0.1, size=(32, 2, 10, 3))
    stab_1, _ = classify_phonon_stability(pt, mc_stable, imag_tol=-1.0, stable_threshold=0.95)
    assert stab_1["status"] == "robustly_stable"

    mc_marginal = pt[None, ...] + rng.normal(0.0, 1.5, size=(32, 2, 10, 3))
    mc_marginal[:10, 0, 0, 0] = -2.5
    stab_2, _ = classify_phonon_stability(pt, mc_marginal, imag_tol=-1.0, stable_threshold=0.95)
    assert stab_2["status"] == "marginal"

    pt_unstable = pt.copy()
    pt_unstable[0, 0, 0] = -3.0
    stab_3, _ = classify_phonon_stability(pt_unstable, mc_marginal, imag_tol=-1.0, stable_threshold=0.95)
    assert stab_3["status"] == "unstable"

    # Verify Eq. S7 spectral Jacobian law on synthetic dynamical matrix perturbations
    lambda_d_pt = np.square(pt)
    delta_lambda = rng.normal(0.0, 0.05, size=(64, *pt.shape))
    lambda_d_mc = np.maximum(lambda_d_pt[None, ...] + delta_lambda, 1e-6)
    omega_mc = np.sqrt(lambda_d_mc)
    spec = compute_spectral_jacobian_perturbation(pt, omega_mc, epsilon=0.038)
    assert spec["log_pearson_r"] > 0.99


def test_eos_uncertainty_profile_regimes() -> None:
    vols = np.linspace(12.0, 18.0, 25)
    enes = 0.2 * np.square(vols - 15.0) - 8.0
    epi_comp = 0.01 + 0.005 * np.square(vols - 13.5)
    res_comp = analyze_eos_uncertainty_profile(vols, enes, epi_comp)
    assert res_comp["volumetric_confidence_regime"] == "compression_shifted"
    assert res_comp["delta_v_over_v0_percent"] < -3.0

    epi_exp = 0.01 + 0.005 * np.square(vols - 16.5)
    res_exp = analyze_eos_uncertainty_profile(vols, enes, epi_exp)
    assert res_exp["volumetric_confidence_regime"] == "expansion_shifted"
    assert res_exp["delta_v_over_v0_percent"] > 3.0


@pytest.mark.parametrize(
    "script_name",
    [
        "prepare_s2c_covariance.py",
        "run_s2c_force_uq.py",
        "run_s2c_calibration.py",
        "run_s2c_ellipsoids.py",
        "run_s2c_phonon.py",
    ],
)
def test_s2c_cli_scripts_help(script_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / script_name
    proc = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
