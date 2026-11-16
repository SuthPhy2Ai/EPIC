#!/usr/bin/env python3
"""Scheme 2c (hepic_s2c_nig015) Force UQ Sharded Worker, Full-Set Evaluator, and Active-Learning Trigger.

Implements Section 2.1 and Section 4.6 (Eqs. 14--20):
  - `worker`: Multi-GPU / sharded extraction of last-layer feature Jacobians J_h,
    3x3 atomic force covariance eigenvalues, and NIG variances across a split.
  - `eval`: Reassembles calibration and test shards, computes Scheme 2c raw and
    calibrated force variances, evaluates component- and structure-level UQ metrics
    (Spearman rho, Top-5% AUROC/AUPRC, AtomNDCG, Gaussian NLL, 95% coverage),
    and evaluates active-learning DFT review prioritization recall @ 5%, 10%, 15%, 20%.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import spearmanr

try:
    import torch
    import torch.multiprocessing as mp
    from mace.tools import torch_geometric
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    mp = None  # type: ignore[assignment]
    torch_geometric = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uq_benchmark.calibration import fit_gaussian_variance_map
from uq_benchmark.config import load_config
from uq_benchmark.method_comparison import _evaluate, _statistics
from uq_benchmark.s2c_covariance import (
    BASELINE_ID,
    DEFAULT_NIG_EXPONENT,
    DEFAULT_SIGMA0_F,
    DEFAULT_TAU_F,
    SAMPLING_METHOD_ANALYTIC,
    compute_dft_review_trigger_score,
)
from uq_benchmark.supplement import (
    _material_ids_for_predictions,
    _ndcg,
    _ranking_groups,
    _weighted_auroc_auprc,
)

LEVELS = (0.50, 0.68, 0.80, 0.90, 0.95, 0.99)


def build_batches(
    indices: np.ndarray,
    natoms: np.ndarray,
    max_structures: int = 32,
    max_atoms: int = 850,
) -> list[list[int]]:
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_atoms = 0
    for idx, n in zip(indices, natoms):
        n_int = int(n)
        if cur and (len(cur) >= max_structures or cur_atoms + n_int > max_atoms):
            batches.append(cur)
            cur = []
            cur_atoms = 0
        cur.append(int(idx))
        cur_atoms += n_int
    if cur:
        batches.append(cur)
    return batches


def load_covariances(covariance_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(covariance_path, allow_pickle=False) as z:
        if "cov_F" in z and "cov_Fa" in z:
            return np.asarray(z["cov_F"], dtype=np.float64), np.asarray(z["cov_Fa"], dtype=np.float64)
        if "covariance" in z:
            cov_f = np.asarray(z["covariance"], dtype=np.float64)
            return cov_f, cov_f
    raise ValueError(f"Covariance archive {covariance_path} must contain 'covariance' or 'cov_F'")


def worker(
    local_rank: int,
    rank_offset: int,
    world_size: int,
    indices: np.ndarray,
    natoms: np.ndarray,
    config_path: Path,
    checkpoint_path: Path,
    covariance_path: Path,
    shard_dir: Path,
    max_structures: int = 32,
    max_atoms: int = 850,
) -> None:
    global_rank = rank_offset + local_rank
    final_path = shard_dir / f"shard_{global_rank}.pt"
    if final_path.exists():
        print(f"[Global Rank {global_rank}] {final_path} already exists, skipping.", flush=True)
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        device = "cpu"
    torch.set_num_threads(4)

    cov_F, cov_Fa = load_covariances(covariance_path)
    cov_F_t = torch.as_tensor(cov_F, dtype=torch.float64, device=device)
    cov_Fa_t = torch.as_tensor(cov_Fa, dtype=torch.float64, device=device)

    from uq_benchmark.data import load_dataset
    from uq_benchmark.inference import load_checkpoint_model

    cfg = load_config(config_path)
    model, _ = load_checkpoint_model(checkpoint_path, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dataset = load_dataset(cfg)

    my_idx = indices[global_rank::world_size]
    my_natoms = natoms[global_rank::world_size]
    batches = build_batches(my_idx, my_natoms, max_structures=max_structures, max_atoms=max_atoms)

    v_s2_unw_all, v_s2_alloc_all = [], []
    v_s1_unw_all, v_s1_alloc_all = [], []
    eigvals_s2_unw_all, rho_all = [], []
    nig_epi_all, nig_alea_all = [], []

    t_start = time.perf_counter()
    done_structures = 0
    for b_idx, sub in enumerate(batches):
        items = [dataset[int(i)] for i in sub]
        batch = torch_geometric.Batch.from_data_list(items).to(device)
        batch.positions.requires_grad_(True)
        out = model(batch, training=False, compute_force=False)
        hidden = model.base_model.edl_readout._hidden(out["node_feats"])
        dim = hidden.shape[1]
        counts = (batch.ptr[1:] - batch.ptr[:-1]).to(hidden.dtype)
        w_epi = out["atomic_epistemic_weight"]
        rho = counts[batch.batch] * w_epi

        phi_unw = hidden.sum(dim=0)
        phi_alloc = (torch.sqrt(rho)[:, None] * hidden).sum(dim=0)

        cols_unw, cols_alloc = [], []
        for d in range(dim):
            g1 = torch.autograd.grad(phi_unw[d], batch.positions, retain_graph=True, create_graph=False)[0]
            g2 = torch.autograd.grad(
                phi_alloc[d], batch.positions, retain_graph=(d + 1 < dim), create_graph=False
            )[0]
            cols_unw.append(-g1.reshape(-1, 1))
            cols_alloc.append(-g2.reshape(-1, 1))

        Ju64 = torch.cat(cols_unw, dim=1).detach().to(torch.float32).to(torch.float64)
        Ja64 = torch.cat(cols_alloc, dim=1).detach().to(torch.float32).to(torch.float64)

        Ju_cov = Ju64 @ cov_F_t
        v_s2_unw = torch.sum(Ju_cov * Ju64, dim=1).cpu().numpy()
        v_s2_alloc = torch.sum((Ja64 @ cov_Fa_t) * Ja64, dim=1).cpu().numpy()
        v_s1_unw = torch.sum(Ju64 * Ju64, dim=1).cpu().numpy()
        v_s1_alloc = torch.sum(Ja64 * Ja64, dim=1).cpu().numpy()

        n_atoms_batch = Ju64.shape[0] // 3
        Ju_atom = Ju64.reshape(n_atoms_batch, 3, dim)
        Ju_cov_atom = Ju_cov.reshape(n_atoms_batch, 3, dim)
        C_atom = torch.bmm(Ju_atom, Ju_cov_atom.transpose(1, 2))
        C_atom = 0.5 * (C_atom + C_atom.transpose(1, 2))
        eigvals_atom = torch.linalg.eigvalsh(C_atom).cpu().numpy().astype(np.float32)

        nig_epi = (out["energy_epistemic_variance"] / (counts ** 2)).detach().cpu().numpy().astype(np.float64)
        nig_alea = (out["energy_aleatoric_variance"] / (counts ** 2)).detach().cpu().numpy().astype(np.float64)
        ptr = batch.ptr.detach().cpu().numpy()
        rho_np = rho.detach().cpu().numpy().astype(np.float32)

        for g in range(len(sub)):
            a_s, a_e = int(ptr[g]), int(ptr[g + 1])
            c_s, c_e = a_s * 3, a_e * 3
            v_s2_unw_all.append(v_s2_unw[c_s:c_e].astype(np.float32))
            v_s2_alloc_all.append(v_s2_alloc[c_s:c_e].astype(np.float32))
            v_s1_unw_all.append(v_s1_unw[c_s:c_e].astype(np.float32))
            v_s1_alloc_all.append(v_s1_alloc[c_s:c_e].astype(np.float32))
            eigvals_s2_unw_all.append(eigvals_atom[a_s:a_e])
            rho_all.append(rho_np[a_s:a_e])
            nig_epi_all.append(float(nig_epi[g]))
            nig_alea_all.append(float(nig_alea[g]))

        done_structures += len(sub)
        if b_idx % 10 == 0 or done_structures == len(my_idx):
            elapsed = time.perf_counter() - t_start
            rate = done_structures / max(elapsed, 1e-6)
            eta = (len(my_idx) - done_structures) / max(rate, 1e-6)
            print(
                f"[Global Rank {global_rank}] {done_structures}/{len(my_idx)} structures "
                f"({elapsed:.1f}s elapsed, {rate:.1f} struct/s, ETA {eta:.1f}s)",
                flush=True,
            )

    shard_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = shard_dir / f"shard_{global_rank}.pt.tmp.{os.getpid()}"
    torch.save(
        {
            "db_index": my_idx,
            "v_s2_unw": v_s2_unw_all,
            "v_s2_alloc": v_s2_alloc_all,
            "v_s1_unw": v_s1_unw_all,
            "v_s1_alloc": v_s1_alloc_all,
            "eigvals_s2_unw": eigvals_s2_unw_all,
            "rho": rho_all,
            "nig_epi": np.asarray(nig_epi_all, dtype=np.float64),
            "nig_alea": np.asarray(nig_alea_all, dtype=np.float64),
        },
        tmp_path,
    )
    os.replace(tmp_path, final_path)
    print(f"[Global Rank {global_rank}] Saved {final_path} ({time.perf_counter() - t_start:.1f}s)", flush=True)


def evaluate_dft_review_prioritization(
    struct_force_mae: np.ndarray,
    struct_s2c_rms: np.ndarray,
    energy_epi_std_pa: np.ndarray,
    struct_max_pred_force: np.ndarray,
    budgets: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
) -> dict[str, dict[str, float]]:
    """Evaluate active-learning DFT review recall for top-5% dangerous force errors (Table 2)."""
    p95 = float(np.quantile(struct_force_mae, 0.95, method="linear"))
    danger = struct_force_mae >= p95
    n_danger = int(np.sum(danger))
    n_total = len(struct_force_mae)
    weights = np.ones(n_total, dtype=np.float64)

    combined_trigger = compute_dft_review_trigger_score(
        struct_s2c_rms, energy_epi_std_pa, struct_max_pred_force, weights=(0.5, 0.2, 0.3)
    )
    triggers = {
        "energy_epi_only": energy_epi_std_pa,
        "force_rms_only": struct_s2c_rms,
        "epic_combined_trigger": combined_trigger,
    }
    report: dict[str, dict[str, float]] = {}
    for name, score in triggers.items():
        order, starts = _ranking_groups(score)
        auroc, auprc = _weighted_auroc_auprc(danger, weights, order, starts)
        recalls = {}
        for b in budgets:
            k = max(1, int(round(b * n_total)))
            top_k = order[:k]
            recalls[f"recall_at_{int(b * 100):02d}pct"] = float(np.sum(danger[top_k]) / max(n_danger, 1))
        report[name] = {
            "danger_threshold_eV_A": p95,
            "n_danger_structures": n_danger,
            "spearman_vs_struct_mae": float(spearmanr(score, struct_force_mae).statistic),
            "top5_auroc": float(auroc),
            "top5_auprc": float(auprc),
            **recalls,
        }
    return report


def evaluate_shards(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    test_shard_dir = Path(args.test_shard_dir).resolve()
    world_size = int(args.world_size)

    test_f = dict(np.load(args.test_force_npz, allow_pickle=True))
    test_idx = test_f["db_index"]
    n_test = len(test_idx)
    test_offsets = test_f["force_offsets"]
    test_natoms = np.diff(test_offsets)

    v_test_s2_unw = [None] * n_test
    eigvals_test_s2_unw = [None] * n_test
    test_rho = [None] * n_test
    test_nig_epi = np.zeros(n_test, dtype=np.float64)
    test_nig_alea = np.zeros(n_test, dtype=np.float64)

    for r in range(world_size):
        d = torch.load(test_shard_dir / f"shard_{r}.pt", map_location="cpu", weights_only=False)
        positions = list(range(r, n_test, world_size))
        if not np.array_equal(d["db_index"], test_idx[positions]):
            raise ValueError(f"Shard {r} db_index mismatch")
        for k, pos in enumerate(positions):
            v_test_s2_unw[pos] = d["v_s2_unw"][k]
            eigvals_test_s2_unw[pos] = d["eigvals_s2_unw"][k]
            test_rho[pos] = d["rho"][k]
            test_nig_epi[pos] = d["nig_epi"][k]
            test_nig_alea[pos] = d["nig_alea"][k]

    v_test_s2_unw_flat = np.concatenate(v_test_s2_unw, axis=0).astype(np.float64)
    test_nig_comp = np.repeat(test_nig_epi, test_natoms * 3)
    v_test_s2c = (test_nig_comp ** float(args.nig_exponent)) * v_test_s2_unw_flat

    test_out_dir = output_dir / "force_test" / "hierarchical_epic_s2c"
    test_out_dir.mkdir(parents=True, exist_ok=True)
    test_out = dict(test_f)
    test_out["force_scale_raw"] = np.sqrt(np.maximum(v_test_s2c, 0.0)).reshape(-1, 3).astype(np.float32)
    test_out["sampling_method"] = np.asarray(SAMPLING_METHOD_ANALYTIC)
    np.savez_compressed(test_out_dir / "aggregated.npz", **test_out)

    tau_f = float(args.tau_f)
    sigma0_f = float(args.sigma0_f)
    if args.calibration_force_npz is not None and Path(args.calibration_force_npz).exists():
        cal_f = dict(np.load(args.calibration_force_npz, allow_pickle=True))
        cal_var = np.square(cal_f["force_scale_raw"].astype(np.float64)).reshape(-1)
        cal_state = fit_gaussian_variance_map(
            cal_f["force_true"].reshape(-1),
            cal_f["force_mean"].reshape(-1),
            cal_var,
            method=BASELINE_ID,
        )
        tau_f = float(cal_state.parameters["tau"])
        sigma0_f = float(cal_state.parameters["sigma0"])

    a_cal = tau_f ** 2
    b_cal = sigma0_f ** 2
    test_true = test_f["force_true"].astype(np.float64)
    test_mean = test_f["force_mean"].astype(np.float64)
    atom_err = np.linalg.norm(test_true - test_mean, axis=1)
    v_test_s2c_3d = v_test_s2c.reshape(-1, 3)
    v_test_s2c_cal_3d = a_cal * v_test_s2c_3d + b_cal
    atom_s2c_raw_std = np.sqrt(np.mean(v_test_s2c_3d, axis=1))
    atom_s2c_cal_std = np.sqrt(np.mean(v_test_s2c_cal_3d, axis=1))

    struct_force_rmse = np.zeros(n_test, dtype=np.float64)
    struct_force_mae = np.zeros(n_test, dtype=np.float64)
    struct_max_atom_force_err = np.zeros(n_test, dtype=np.float64)
    struct_max_pred_force = np.zeros(n_test, dtype=np.float64)
    struct_s2c_raw_rms = np.zeros(n_test, dtype=np.float64)
    struct_s2c_cal_rms = np.zeros(n_test, dtype=np.float64)
    struct_s2c_raw_max_atom = np.zeros(n_test, dtype=np.float64)
    struct_s2c_cal_max_atom = np.zeros(n_test, dtype=np.float64)

    comp_diff = test_true - test_mean
    pred_atom_norm = np.linalg.norm(test_mean, axis=1)
    for i, (s, e) in enumerate(zip(test_offsets[:-1], test_offsets[1:])):
        d_sub = comp_diff[s:e]
        struct_force_rmse[i] = float(np.sqrt(np.mean(d_sub ** 2)))
        struct_force_mae[i] = float(np.mean(np.abs(d_sub)))
        struct_max_atom_force_err[i] = float(np.max(atom_err[s:e]))
        struct_max_pred_force[i] = float(np.max(pred_atom_norm[s:e]))
        struct_s2c_raw_rms[i] = float(np.sqrt(np.mean(v_test_s2c_3d[s:e])))
        struct_s2c_cal_rms[i] = float(np.sqrt(np.mean(v_test_s2c_cal_3d[s:e])))
        struct_s2c_raw_max_atom[i] = float(np.max(atom_s2c_raw_std[s:e]))
        struct_s2c_cal_max_atom[i] = float(np.max(atom_s2c_cal_std[s:e]))

    energy_epi_std_pa = np.sqrt(np.maximum(test_nig_epi, 0.0))
    energy_alea_std_pa = np.sqrt(np.maximum(test_nig_alea, 0.0))

    np.savez_compressed(
        output_dir / "s2c_full_structure_summary.npz",
        db_index=test_idx,
        natoms=test_natoms,
        nig_epi_per_atom=test_nig_epi,
        nig_alea_per_atom=test_nig_alea,
        energy_epi_std_per_atom=energy_epi_std_pa,
        energy_alea_std_per_atom=energy_alea_std_pa,
        struct_force_rmse=struct_force_rmse,
        struct_force_mae=struct_force_mae,
        struct_max_atom_force_err=struct_max_atom_force_err,
        struct_max_pred_force=struct_max_pred_force,
        struct_s2c_raw_rms=struct_s2c_raw_rms,
        struct_s2c_cal_rms=struct_s2c_cal_rms,
        struct_s2c_raw_max_atom=struct_s2c_raw_max_atom,
        struct_s2c_cal_max_atom=struct_s2c_cal_max_atom,
        tau_F=np.asarray(tau_f),
        sigma0_F=np.asarray(sigma0_f),
    )

    if args.store_path is not None and Path(args.store_path).exists():
        mat_ids = _material_ids_for_predictions(args.store_path, test_idx)
        _, cluster = np.unique(mat_ids, return_inverse=True)
        weights = np.ones(cluster.max() + 1, dtype=np.float64)
    else:
        cluster = np.arange(n_test, dtype=np.int64)
        weights = np.ones(n_test, dtype=np.float64)

    stats = _statistics(
        test_true,
        test_mean,
        v_test_s2c_3d,
        {"tau": tau_f, "sigma0": sigma0_f},
        LEVELS,
        test_offsets,
    )
    res = _evaluate(stats, weights, cluster, LEVELS)
    atom_unc = np.linalg.norm(np.sqrt(np.maximum(v_test_s2c_3d, 0.0)), axis=1)
    ndcgs = [
        _ndcg(atom_err[s:e], atom_unc[s:e])
        for s, e in zip(test_offsets[:-1], test_offsets[1:])
        if e - s > 1
    ]
    ndcg_val = float(np.nanmean(ndcgs))

    trigger_report = evaluate_dft_review_prioritization(
        struct_force_mae, struct_s2c_raw_rms, energy_epi_std_pa, struct_max_pred_force
    )

    summary_payload = {
        "method_id": BASELINE_ID,
        "n_test_structures": int(n_test),
        "n_test_components": int(test_true.size),
        "point_mae": float(res["point_mae"]),
        "point_rmse": float(res["point_rmse"]),
        "risk_spearman": float(res["risk_spearman"]),
        "top5_auroc": float(res["top5_auroc"]),
        "top5_auprc": float(res["top5_auprc"]),
        "gaussian_affine_nll": float(res["gaussian_affine_nll"]),
        "atom_ndcg": ndcg_val,
        "cal_tau_F": tau_f,
        "cal_sigma0_F": sigma0_f,
        "coverage_90": float(res["coverage_90"]),
        "coverage_95": float(res["coverage_95"]),
        "dft_review_prioritization": trigger_report,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    out_json = output_dir / "s2c_force_uq_evaluation.json"
    out_json.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary_payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["worker", "eval"], required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "epic_mptrj_8gpu.yaml",
        help="Path to benchmark YAML configuration.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Path to trained MACE+EPIC checkpoint.")
    parser.add_argument("--covariance-npz", type=Path, help="Path to s2c_covariance.npz.")
    parser.add_argument("--test-force-npz", type=Path, required=True, help="Test force aggregated.npz.")
    parser.add_argument("--calibration-force-npz", type=Path, help="Optional calibration force aggregated.npz.")
    parser.add_argument("--store-path", type=Path, help="Optional HDF5 benchmark store for material_id clusters.")
    parser.add_argument("--test-shard-dir", type=Path, required=True, help="Directory for test Jacobian shards.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for evaluation results.")
    parser.add_argument("--rank-offset", type=int, default=0)
    parser.add_argument("--local-gpus", type=int, default=1)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-structures", type=int, default=32)
    parser.add_argument("--max-atoms", type=int, default=850)
    parser.add_argument("--nig-exponent", type=float, default=DEFAULT_NIG_EXPONENT)
    parser.add_argument("--tau-f", type=float, default=DEFAULT_TAU_F)
    parser.add_argument("--sigma0-f", type=float, default=DEFAULT_SIGMA0_F)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "worker":
        if args.checkpoint is None or args.covariance_npz is None:
            raise ValueError("--checkpoint and --covariance-npz are required for --mode=worker")
        with np.load(args.test_force_npz, allow_pickle=True) as z:
            test_idx = z["db_index"]
            test_natoms = np.diff(z["force_offsets"])
        if args.local_gpus > 1:
            mp.set_start_method("spawn", force=True)
            mp.spawn(
                worker,
                args=(
                    args.rank_offset,
                    args.world_size,
                    test_idx,
                    test_natoms,
                    args.config,
                    args.checkpoint,
                    args.covariance_npz,
                    args.test_shard_dir,
                    args.max_structures,
                    args.max_atoms,
                ),
                nprocs=args.local_gpus,
                join=True,
            )
        else:
            worker(
                0,
                args.rank_offset,
                args.world_size,
                test_idx,
                test_natoms,
                args.config,
                args.checkpoint,
                args.covariance_npz,
                args.test_shard_dir,
                args.max_structures,
                args.max_atoms,
            )
        return 0
    return evaluate_shards(args)


if __name__ == "__main__":
    raise SystemExit(main())
