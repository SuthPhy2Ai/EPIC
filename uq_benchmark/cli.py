from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .diagnostics import diagnose_nig_predictions
from .ensemble_pipeline import build_ensemble_records
from .inference import aggregate_shards, run_inference_shard
from .downstream import create_downstream_manifests
from .epic_pipeline import aggregate_epic_force_shards, sample_epic_force_subset
from .external_ood import (
    audit_external_element_support,
    audit_external_ood_leakage,
    build_external_ood_database,
    prepare_elementa_ood,
)
from .external_ood_analysis import analyze_external_ood
from .llpr_pipeline import (
    accumulate_gram_shard,
    accumulate_force_gram_shard,
    aggregate_force_shards,
    aggregate_force_feature_shards,
    aggregate_force_gram_shards,
    aggregate_gram_shards,
    collect_force_feature_shard,
    fit_force_aware_posterior_from_files,
    fit_posterior_from_files,
    sample_force_subset,
)
from .method_comparison import (
    MethodFiles,
    compare_methods_with_ensemble,
    compare_nonensemble_methods,
)
from .report import generate_report
from .pipeline import (
    build_store,
    build_two_model_store,
    calibration_state_from_selection,
    fit_epic_calibration_file,
    fit_epic_force_calibration_file,
    select_epic_lambda,
)
from .cost import write_cost_tables
from .hdf5_dataset import convert_ase_db_to_sharded_hdf5, verify_sharded_hdf5
from .split import create_split
from .training import run_training
from .training_ablation_comparison import compare_training_ablations
from .training_continuity import (
    audit_reference_training_continuity,
    audit_training_ablation_preflight,
)
from .supplement_release import finalize_supplement_release
from .two_model_analysis import analyze_force_convergence, analyze_two_model_predictions
from .two_model_summary import write_two_model_summary
from .checkpoint_audit import audit_hierarchical_checkpoint
from .supplement import (
    analyze_atomic_attribution,
    analyze_calibration_sensitivity,
    bootstrap_uq_metrics,
    compare_force_allocations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m uq_benchmark.cli")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config") / "hierarchical_epic.example.yaml"),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    split = commands.add_parser("make-split")
    split.add_argument("--no-db-hash", action="store_true")

    hdf5 = commands.add_parser("make-hdf5")
    hdf5.add_argument("--output-dir")
    hdf5.add_argument("--shard-size", type=int, default=25_000)
    hdf5.add_argument("--progress-every", type=int, default=1_000)
    hdf5.add_argument("--verify-samples", type=int, default=1_000)
    hdf5.add_argument("--hash-source", action="store_true")
    hdf5.add_argument("--overwrite", action="store_true")

    external_ood = commands.add_parser("prepare-elementa-ood")
    external_ood.add_argument("--archive", required=True)
    external_ood.add_argument("--member", default="ELEMENTA_open.extxyz")
    external_ood.add_argument("--output-extxyz", required=True)
    external_ood.add_argument("--manifest", required=True)
    external_ood.add_argument("--subset-size", type=int, default=10_000)
    external_ood.add_argument("--seed", type=int, default=42)
    external_ood.add_argument("--max-atoms", type=int, default=256)
    external_ood.add_argument("--supported-elements", default="1-83,89-94")
    external_ood.add_argument("--source-url")
    external_ood.add_argument("--source-license")
    external_ood.add_argument("--dft-protocol")

    external_db = commands.add_parser("build-external-ood-db")
    external_db.add_argument("--extxyz", required=True)
    external_db.add_argument("--selection-manifest", required=True)
    external_db.add_argument("--output-db", required=True)
    external_db.add_argument("--split-dir", required=True)
    external_db.add_argument("--base-config", required=True)
    external_db.add_argument("--output-config", required=True)

    external_leakage = commands.add_parser("audit-external-ood-leakage")
    external_leakage.add_argument("--external-db", required=True)
    external_leakage.add_argument("--reference-db", required=True)
    external_leakage.add_argument("--selection-manifest", required=True)
    external_leakage.add_argument("--output", required=True)
    external_leakage.add_argument("--decimals", type=int, default=6)

    external_elements = commands.add_parser("audit-external-element-support")
    external_elements.add_argument("--external-db", required=True)
    external_elements.add_argument("--reference-db", required=True)
    external_elements.add_argument("--split-indices", required=True)
    external_elements.add_argument("--output", required=True)

    external_analysis = commands.add_parser("analyze-external-ood")
    external_analysis.add_argument("--deterministic-point", required=True)
    external_analysis.add_argument("--hierarchical-point", required=True)
    external_analysis.add_argument("--hierarchical-force", required=True)
    external_analysis.add_argument("--in-domain-point", required=True)
    external_analysis.add_argument("--in-domain-force", required=True)
    external_analysis.add_argument("--in-domain-store", required=True)
    external_analysis.add_argument("--in-domain-calibration-parameters", required=True)
    external_analysis.add_argument("--selection-manifest", required=True)
    external_analysis.add_argument("--leakage-audit", required=True)
    external_analysis.add_argument("--output-dir", required=True)
    external_analysis.add_argument("--bootstrap-replicates", type=int, default=1000)
    external_analysis.add_argument("--seed", type=int, default=42)

    training_comparison = commands.add_parser("compare-training-ablations")
    for prefix in ("seed42-reference", "seed43", "seed44", "allocation-weight-0-seed42"):
        training_comparison.add_argument(f"--{prefix}-energy-calibration", required=True)
        training_comparison.add_argument(f"--{prefix}-energy-test", required=True)
        training_comparison.add_argument(f"--{prefix}-force-calibration", required=True)
        training_comparison.add_argument(f"--{prefix}-force-test", required=True)
        training_comparison.add_argument(f"--{prefix}-atomic-summary", required=True)
    training_comparison.add_argument("--store", required=True)
    training_comparison.add_argument("--output-dir", required=True)
    training_comparison.add_argument("--bootstrap-replicates", type=int, default=1000)
    training_comparison.add_argument("--seed", type=int, default=42)

    training_continuity = commands.add_parser("audit-reference-training-continuity")
    training_continuity.add_argument("--probe-run-dir", required=True)
    training_continuity.add_argument("--formal-run-dir", required=True)
    training_continuity.add_argument("--output", required=True)

    training_preflight = commands.add_parser("audit-training-ablation-preflight")
    training_preflight.add_argument("--reference-run-config", required=True)
    training_preflight.add_argument("--reference-split", required=True)
    training_preflight.add_argument("--seed43-run-config", required=True)
    training_preflight.add_argument("--ablation-split", required=True)
    training_preflight.add_argument("--learned-config", required=True)
    training_preflight.add_argument("--allocation-weight-zero-config", required=True)
    training_preflight.add_argument("--output", required=True)

    supplement_release = commands.add_parser("finalize-supplement-release")
    supplement_release.add_argument("--supplement-root", required=True)
    supplement_release.add_argument("--source-root", required=True)
    supplement_release.add_argument("--output-dir", required=True)

    train = commands.add_parser("train")
    train.add_argument(
        "--kind",
        choices=[
            "deterministic",
            "epic",
            "epic_atomwise",
            "epic_hierarchical",
        ],
        required=True,
    )
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--lambda-max", type=float, default=0.0)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--resume")
    train.add_argument("--reset-training-state", action="store_true")

    infer = commands.add_parser("infer-shard")
    infer.add_argument("--split", choices=["train", "validation", "calibration", "test"], required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--shard", type=int, default=0)
    infer.add_argument("--total-shards", type=int, default=1)
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--include-llpr-features", action="store_true")

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--inputs", nargs="+", required=True)
    aggregate.add_argument("--output", required=True)

    diagnose = commands.add_parser("diagnose-nig")
    diagnose.add_argument("--predictions", required=True)
    diagnose.add_argument("--output")

    gram = commands.add_parser("llpr-gram-shard")
    gram.add_argument("--checkpoint", required=True)
    gram.add_argument("--output", required=True)
    gram.add_argument("--shard", type=int, required=True)
    gram.add_argument("--total-shards", type=int, default=8)
    gram.add_argument("--device", default="cuda")

    gram_aggregate = commands.add_parser("llpr-gram-aggregate")
    gram_aggregate.add_argument("--inputs", nargs="+", required=True)
    gram_aggregate.add_argument("--output", required=True)

    force_gram = commands.add_parser("llpr-force-gram-shard")
    force_gram.add_argument("--checkpoint", required=True)
    force_gram.add_argument("--output", required=True)
    force_gram.add_argument("--shard", type=int, required=True)
    force_gram.add_argument("--total-shards", type=int, default=8)
    force_gram.add_argument("--subset-size", type=int, default=10_000)
    force_gram.add_argument("--device", default="cuda")

    force_gram_aggregate = commands.add_parser("llpr-force-gram-aggregate")
    force_gram_aggregate.add_argument("--inputs", nargs="+", required=True)
    force_gram_aggregate.add_argument("--output", required=True)

    force_features = commands.add_parser("llpr-force-feature-shard")
    force_features.add_argument("--checkpoint", required=True)
    force_features.add_argument("--output", required=True)
    force_features.add_argument("--shard", type=int, required=True)
    force_features.add_argument("--total-shards", type=int, default=8)
    force_features.add_argument("--subset-size", type=int, default=10_000)
    force_features.add_argument("--device", default="cuda")

    force_features_aggregate = commands.add_parser("llpr-force-feature-aggregate")
    force_features_aggregate.add_argument("--inputs", nargs="+", required=True)
    force_features_aggregate.add_argument("--output", required=True)

    force_aware_fit = commands.add_parser("llpr-force-aware-fit")
    force_aware_fit.add_argument("--energy-gram", required=True)
    force_aware_fit.add_argument("--force-gram", required=True)
    force_aware_fit.add_argument("--energy-calibration", required=True)
    force_aware_fit.add_argument("--force-features", required=True)
    force_aware_fit.add_argument("--force-targets", required=True)
    force_aware_fit.add_argument("--output", required=True)

    posterior = commands.add_parser("llpr-fit")
    posterior.add_argument("--gram", required=True)
    posterior.add_argument("--calibration-predictions", required=True)
    posterior.add_argument("--output", required=True)

    force = commands.add_parser("llpr-force")
    force.add_argument("--split", choices=["calibration", "test"], required=True)
    force.add_argument("--checkpoint", required=True)
    force.add_argument("--posterior", required=True)
    force.add_argument("--output", required=True)
    force.add_argument("--device", default="cuda")
    force.add_argument("--shard", type=int, default=0)
    force.add_argument("--total-shards", type=int, default=1)
    force.add_argument("--n-samples", type=int)
    force.add_argument("--sample-seed", type=int)
    force.add_argument(
        "--subset-namespace",
        help="hashed subset namespace; use the EPIC namespace for exact cross-method alignment",
    )

    force_aggregate = commands.add_parser("llpr-force-aggregate")
    force_aggregate.add_argument("--inputs", nargs="+", required=True)
    force_aggregate.add_argument("--output", required=True)

    method_comparison = commands.add_parser("compare-nonensemble-methods")
    for prefix in ("hierarchical", "graph-nig", "llpr", "llpr-force-aware"):
        method_comparison.add_argument(f"--{prefix}-energy-calibration", required=True)
        method_comparison.add_argument(f"--{prefix}-energy-test", required=True)
        method_comparison.add_argument(f"--{prefix}-force-calibration", required=True)
        method_comparison.add_argument(f"--{prefix}-force-test", required=True)
    method_comparison.add_argument("--llpr-posterior", required=True)
    method_comparison.add_argument("--llpr-force-aware-posterior", required=True)
    method_comparison.add_argument("--store", required=True)
    method_comparison.add_argument("--output-dir", required=True)
    method_comparison.add_argument("--bootstrap-replicates", type=int, default=1000)
    method_comparison.add_argument("--seed", type=int, default=42)

    ensemble_records = commands.add_parser("build-ensemble-records")
    ensemble_records.add_argument("--member", action="append", required=True)
    ensemble_records.add_argument("--force-reference", required=True)
    ensemble_records.add_argument("--point-output", required=True)
    ensemble_records.add_argument("--force-output", required=True)

    ensemble_comparison = commands.add_parser("compare-methods-with-ensemble")
    for prefix in (
        "hierarchical",
        "graph-nig",
        "llpr",
        "llpr-force-aware",
        "ensemble5",
    ):
        ensemble_comparison.add_argument(f"--{prefix}-energy-calibration", required=True)
        ensemble_comparison.add_argument(f"--{prefix}-energy-test", required=True)
        ensemble_comparison.add_argument(f"--{prefix}-force-calibration", required=True)
        ensemble_comparison.add_argument(f"--{prefix}-force-test", required=True)
    ensemble_comparison.add_argument("--llpr-posterior", required=True)
    ensemble_comparison.add_argument("--llpr-force-aware-posterior", required=True)
    ensemble_comparison.add_argument("--store", required=True)
    ensemble_comparison.add_argument("--output-dir", required=True)
    ensemble_comparison.add_argument("--bootstrap-replicates", type=int, default=1000)
    ensemble_comparison.add_argument("--seed", type=int, default=42)

    epic_force = commands.add_parser("epic-force")
    epic_force.add_argument("--split", choices=["calibration", "test"], required=True)
    epic_force.add_argument("--checkpoint", required=True)
    epic_force.add_argument("--output", required=True)
    epic_force.add_argument("--device", default="cuda")
    epic_force.add_argument("--shard", type=int, default=0)
    epic_force.add_argument("--total-shards", type=int, default=1)
    epic_force.add_argument("--subset-size", type=int)
    epic_force.add_argument("--n-samples", type=int)
    epic_force.add_argument("--sample-seed", type=int)
    epic_force.add_argument(
        "--allocation-mode", choices=["learned", "uniform"], default="learned"
    )

    epic_force_aggregate = commands.add_parser("epic-force-aggregate")
    epic_force_aggregate.add_argument("--inputs", nargs="+", required=True)
    epic_force_aggregate.add_argument("--output", required=True)

    report = commands.add_parser("report")
    report.add_argument("--store", required=True)
    report.add_argument("--output-dir", required=True)

    downstream = commands.add_parser("downstream-manifests")
    downstream.add_argument("--store", required=True)
    downstream.add_argument("--output-dir", required=True)

    select_epic = commands.add_parser("select-epic")
    select_epic.add_argument("--candidate", action="append", required=True, help="LAMBDA=calibration.npz")
    select_epic.add_argument("--output", required=True)

    calibrate_epic = commands.add_parser("calibrate-epic")
    calibrate_epic.add_argument("--predictions", required=True)
    calibrate_epic.add_argument("--output", required=True)
    calibrate_epic.add_argument("--method", default="hierarchical_epic")

    calibrate_epic_force = commands.add_parser("calibrate-epic-force")
    calibrate_epic_force.add_argument("--predictions", required=True)
    calibrate_epic_force.add_argument("--output", required=True)
    calibrate_epic_force.add_argument("--method", default="hierarchical_epic_force")

    two_model = commands.add_parser("build-two-model-store")
    two_model.add_argument("--mace-test", required=True)
    two_model.add_argument("--hierarchical-calibration", required=True)
    two_model.add_argument("--hierarchical-test", required=True)
    two_model.add_argument("--hierarchical-calibration-state", required=True)
    two_model.add_argument("--hierarchical-force-calibration")
    two_model.add_argument("--hierarchical-force-test")
    two_model.add_argument("--output", required=True)

    two_model_analysis = commands.add_parser("analyze-two-model")
    two_model_analysis.add_argument("--mace-test", required=True)
    two_model_analysis.add_argument("--hierarchical-test", required=True)
    two_model_analysis.add_argument("--hierarchical-calibration-state", required=True)
    two_model_analysis.add_argument("--output-dir", required=True)

    force_convergence = commands.add_parser("analyze-force-convergence")
    force_convergence.add_argument("--inputs", nargs="+", required=True)
    force_convergence.add_argument("--output-dir", required=True)
    force_convergence.add_argument("--hierarchical-predictions")

    atomic_attribution = commands.add_parser("analyze-atomic-attribution")
    atomic_attribution.add_argument("--predictions", required=True)
    atomic_attribution.add_argument("--store", required=True)
    atomic_attribution.add_argument("--output-dir", required=True)
    atomic_attribution.add_argument("--force-error-floor", type=float, default=1e-6)
    atomic_attribution.add_argument("--bootstrap-replicates", type=int, default=1000)
    atomic_attribution.add_argument("--seed", type=int, default=42)

    uq_bootstrap = commands.add_parser("bootstrap-uq")
    uq_bootstrap.add_argument("--store", required=True)
    uq_bootstrap.add_argument("--output-dir", required=True)
    uq_bootstrap.add_argument("--bootstrap-replicates", type=int, default=1000)
    uq_bootstrap.add_argument("--seed", type=int, default=42)

    calibration_sensitivity = commands.add_parser("analyze-calibration-sensitivity")
    calibration_sensitivity.add_argument("--energy-calibration", required=True)
    calibration_sensitivity.add_argument("--energy-test", required=True)
    calibration_sensitivity.add_argument("--force-calibration", required=True)
    calibration_sensitivity.add_argument("--force-test", required=True)
    calibration_sensitivity.add_argument("--output-dir", required=True)

    allocation_comparison = commands.add_parser("compare-force-allocations")
    allocation_comparison.add_argument("--learned-calibration", required=True)
    allocation_comparison.add_argument("--learned-test", required=True)
    allocation_comparison.add_argument("--uniform-calibration", required=True)
    allocation_comparison.add_argument("--uniform-test", required=True)
    allocation_comparison.add_argument("--store", required=True)
    allocation_comparison.add_argument("--learned-sensitivity-dir", required=True)
    allocation_comparison.add_argument("--uniform-sensitivity-dir", required=True)
    allocation_comparison.add_argument("--output-dir", required=True)
    allocation_comparison.add_argument("--bootstrap-replicates", type=int, default=1000)
    allocation_comparison.add_argument("--seed", type=int, default=42)

    two_model_summary = commands.add_parser("summarize-two-model")
    two_model_summary.add_argument("--report-dir", required=True)
    two_model_summary.add_argument("--analysis-dir", required=True)
    two_model_summary.add_argument("--force-convergence", required=True)
    two_model_summary.add_argument("--checkpoint-audit", required=True)
    two_model_summary.add_argument("--force-calibration", required=True)
    two_model_summary.add_argument("--cost-dir", required=True)
    two_model_summary.add_argument("--output-dir", required=True)

    checkpoint_audit = commands.add_parser("audit-hierarchical-checkpoint")
    checkpoint_audit.add_argument("--checkpoint", required=True)
    checkpoint_audit.add_argument("--output", required=True)
    checkpoint_audit.add_argument("--split", choices=["validation", "calibration"], default="calibration")
    checkpoint_audit.add_argument("--subset-size", type=int, default=8)
    checkpoint_audit.add_argument("--n-samples", type=int, default=8)
    checkpoint_audit.add_argument("--sample-seed", type=int, default=42)
    checkpoint_audit.add_argument("--finite-difference-step", type=float, default=1e-2)
    checkpoint_audit.add_argument("--device", default="cuda")

    build = commands.add_parser("build-store")
    build.add_argument("--det-test", nargs=5, required=True)
    build.add_argument("--det-calibration", nargs=5, required=True)
    build.add_argument("--epic-test", required=True)
    build.add_argument("--epic-selection", required=True)
    build.add_argument("--llpr-posterior", required=True)
    build.add_argument("--llpr-force-calibration")
    build.add_argument("--llpr-force-test")
    build.add_argument("--epic-force-calibration")
    build.add_argument("--epic-force-test")
    build.add_argument("--output", required=True)

    cost = commands.add_parser("cost")
    cost.add_argument("--training-run", action="append", default=[], help="METHOD=run_dir")
    cost.add_argument("--inference-run", action="append", default=[], help="METHOD=predictions.npz")
    cost.add_argument("--output-dir", required=True)

    s2c_cov = commands.add_parser("s2c-prepare-covariance")
    s2c_cov.add_argument("--mode", choices=["from-checkpoint", "from-shards"], default="from-checkpoint")
    s2c_cov.add_argument("--checkpoint")
    s2c_cov.add_argument("--split-path")
    s2c_cov.add_argument("--subset-size", type=int, default=10_000)
    s2c_cov.add_argument("--batch-size", type=int, default=1)
    s2c_cov.add_argument("--device", default="cuda")
    s2c_cov.add_argument("--normalize-gram", action="store_true")
    s2c_cov.add_argument("--shard-dir")
    s2c_cov.add_argument("--num-shards", type=int, default=12)
    s2c_cov.add_argument("--point-calibration")
    s2c_cov.add_argument("--force-calibration")
    s2c_cov.add_argument("--output-dir", required=True)
    s2c_cov.add_argument("--feature-dim", type=int, default=64)
    s2c_cov.add_argument("--ridge-multiplier", type=float, default=1e-3)
    s2c_cov.add_argument("--nig-exponent", type=float, default=0.15)
    s2c_cov.add_argument("--tau-f", type=float, default=47.42317194970634)
    s2c_cov.add_argument("--sigma0-f", type=float, default=0.010240335903317568)

    s2c_force = commands.add_parser("s2c-force-uq")
    s2c_force.add_argument("--mode", choices=["worker", "eval"], required=True)
    s2c_force.add_argument("--checkpoint")
    s2c_force.add_argument("--covariance-npz")
    s2c_force.add_argument("--test-force-npz", required=True)
    s2c_force.add_argument("--calibration-force-npz")
    s2c_force.add_argument("--store-path")
    s2c_force.add_argument("--test-shard-dir", required=True)
    s2c_force.add_argument("--output-dir", required=True)
    s2c_force.add_argument("--rank-offset", type=int, default=0)
    s2c_force.add_argument("--local-gpus", type=int, default=1)
    s2c_force.add_argument("--world-size", type=int, default=1)
    s2c_force.add_argument("--max-structures", type=int, default=32)
    s2c_force.add_argument("--max-atoms", type=int, default=850)
    s2c_force.add_argument("--nig-exponent", type=float, default=0.15)
    s2c_force.add_argument("--tau-f", type=float, default=47.42317194970634)
    s2c_force.add_argument("--sigma0-f", type=float, default=0.010240335903317568)

    s2c_cal = commands.add_parser("s2c-calibrate")
    s2c_cal.add_argument("--force-calibration", required=True)
    s2c_cal.add_argument("--force-test")
    s2c_cal.add_argument("--energy-calibration")
    s2c_cal.add_argument("--energy-test")
    s2c_cal.add_argument("--energy-component", choices=["total", "epistemic", "aleatoric"], default="total")
    s2c_cal.add_argument("--output", required=True)

    s2c_ellip = commands.add_parser("s2c-ellipsoids")
    s2c_ellip.add_argument("--checkpoint", required=True)
    s2c_ellip.add_argument("--covariance-npz", required=True)
    s2c_ellip.add_argument("--db-path")
    s2c_ellip.add_argument("--db-indices", type=int, nargs="+", default=[0, 917, 1234, 1540354, 985361, 329314])
    s2c_ellip.add_argument("--structure-files", nargs="+")
    s2c_ellip.add_argument("--output-dir", required=True)
    s2c_ellip.add_argument("--device", default="cuda")
    s2c_ellip.add_argument("--n-samples", type=int, default=32)
    s2c_ellip.add_argument("--seed", type=int, default=42)
    s2c_ellip.add_argument("--tau-f", type=float, default=None)
    s2c_ellip.add_argument("--sigma0-f", type=float, default=None)

    s2c_ph = commands.add_parser("s2c-phonon")
    s2c_ph.add_argument("--checkpoint", required=True)
    s2c_ph.add_argument("--covariance-npz", required=True)
    s2c_ph.add_argument("--db-path")
    s2c_ph.add_argument("--selection-csv")
    s2c_ph.add_argument("--db-index", type=int)
    s2c_ph.add_argument("--structure-file")
    s2c_ph.add_argument("--output-dir", required=True)
    s2c_ph.add_argument("--device", default="cuda")
    s2c_ph.add_argument("--rank", type=int, default=0)
    s2c_ph.add_argument("--world-size", type=int, default=1)
    s2c_ph.add_argument("--limit", type=int)
    s2c_ph.add_argument("--supercell", type=int, nargs=3, default=[2, 2, 2])
    s2c_ph.add_argument("--displacement", type=float, default=0.01)
    s2c_ph.add_argument("--n-mc-samples", type=int, default=32)
    s2c_ph.add_argument("--npoints", type=int, default=51)
    s2c_ph.add_argument("--imag-tol", type=float, default=-1.0)
    s2c_ph.add_argument("--stable-threshold", type=float, default=0.95)
    s2c_ph.add_argument("--seed", type=int, default=42)
    s2c_ph.add_argument("--tau-f", type=float, default=None)
    return parser


def _key_values(values: list[str], value_type=str) -> dict:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected KEY=VALUE, got {value}")
        key, item = value.split("=", 1)
        result[value_type(key)] = item
    return result


def _grouped_key_values(values: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected KEY=VALUE, got {value}")
        key, item = value.split("=", 1)
        result.setdefault(key, []).append(item)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    split_path = Path(config.data.split_dir) / "split_indices.npz"
    if args.command == "make-split":
        artifacts = create_split(
            config.data.db_path,
            config.data.split_dir,
            config.data.ratios,
            config.data.seed,
            hash_database=not args.no_db_hash,
        )
        print(json.dumps({key: str(value) for key, value in artifacts.__dict__.items()}, indent=2))
    elif args.command == "make-hdf5":
        output_dir = args.output_dir or config.data.hdf5_path
        if not output_dir:
            output_dir = str(Path(config.data.db_path).with_suffix("")) + "_hdf5"
        manifest = convert_ase_db_to_sharded_hdf5(
            config.data.db_path,
            output_dir,
            shard_size=args.shard_size,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
            hash_source=args.hash_source,
        )
        verification = verify_sharded_hdf5(
            config.data.db_path,
            output_dir,
            samples=args.verify_samples,
            seed=config.data.seed,
        )
        print(json.dumps({"manifest": str(manifest), "verification": verification}, indent=2))
    elif args.command == "prepare-elementa-ood":
        print(
            prepare_elementa_ood(
                args.archive,
                args.member,
                args.output_extxyz,
                args.manifest,
                subset_size=args.subset_size,
                seed=args.seed,
                max_atoms=args.max_atoms,
                supported_atomic_numbers=args.supported_elements,
                source_url=args.source_url,
                source_license=args.source_license,
                dft_protocol=args.dft_protocol,
            )
        )
    elif args.command == "build-external-ood-db":
        print(
            build_external_ood_database(
                args.extxyz,
                args.selection_manifest,
                args.output_db,
                args.split_dir,
                args.base_config,
                args.output_config,
            )
        )
    elif args.command == "audit-external-ood-leakage":
        print(
            audit_external_ood_leakage(
                args.external_db,
                args.reference_db,
                args.selection_manifest,
                args.output,
                decimals=args.decimals,
            )
        )
    elif args.command == "audit-external-element-support":
        print(
            audit_external_element_support(
                args.external_db,
                args.reference_db,
                args.split_indices,
                args.output,
            )
        )
    elif args.command == "analyze-external-ood":
        print(
            analyze_external_ood(
                args.deterministic_point,
                args.hierarchical_point,
                args.hierarchical_force,
                args.in_domain_point,
                args.in_domain_force,
                args.in_domain_store,
                args.in_domain_calibration_parameters,
                args.selection_manifest,
                args.leakage_audit,
                args.output_dir,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
            )
        )
    elif args.command == "compare-training-ablations":
        prefix_to_run = {
            "seed42_reference": "seed42_reference",
            "seed43": "seed43",
            "seed44": "seed44",
            "allocation_weight_0_seed42": "allocation_weight_0_seed42",
        }
        files = {}
        atomic = {}
        for prefix, run in prefix_to_run.items():
            files[run] = MethodFiles(
                Path(getattr(args, f"{prefix}_energy_calibration")),
                Path(getattr(args, f"{prefix}_energy_test")),
                Path(getattr(args, f"{prefix}_force_calibration")),
                Path(getattr(args, f"{prefix}_force_test")),
            )
            atomic[run] = Path(getattr(args, f"{prefix}_atomic_summary"))
        print(
            compare_training_ablations(
                files,
                atomic,
                args.store,
                args.output_dir,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
            )
        )
    elif args.command == "audit-reference-training-continuity":
        print(
            audit_reference_training_continuity(
                args.probe_run_dir,
                args.formal_run_dir,
                args.output,
            )
        )
    elif args.command == "audit-training-ablation-preflight":
        print(
            audit_training_ablation_preflight(
                args.reference_run_config,
                args.reference_split,
                args.seed43_run_config,
                args.ablation_split,
                args.learned_config,
                args.allocation_weight_zero_config,
                args.output,
            )
        )
    elif args.command == "finalize-supplement-release":
        print(
            finalize_supplement_release(
                args.supplement_root,
                args.source_root,
                args.output_dir,
            )
        )
    elif args.command == "train":
        path = run_training(
            config,
            split_path,
            args.output_dir,
            args.kind,
            args.seed,
            args.lambda_max,
            args.resume,
            args.reset_training_state,
        )
        print(path)
    elif args.command == "infer-shard":
        print(
            run_inference_shard(
                config,
                split_path,
                args.split,
                args.checkpoint,
                args.output,
                args.shard,
                args.total_shards,
                args.device,
                args.include_llpr_features,
            )
        )
    elif args.command == "aggregate":
        print(aggregate_shards(args.inputs, args.output))
    elif args.command == "diagnose-nig":
        result = diagnose_nig_predictions(args.predictions, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "llpr-gram-shard":
        print(
            accumulate_gram_shard(
                config, split_path, args.checkpoint, args.output, args.shard, args.total_shards, args.device
            )
        )
    elif args.command == "llpr-gram-aggregate":
        print(aggregate_gram_shards(args.inputs, args.output))
    elif args.command == "llpr-force-gram-shard":
        print(accumulate_force_gram_shard(
            config, split_path, args.checkpoint, args.output, args.shard, args.total_shards,
            args.subset_size, args.device,
        ))
    elif args.command == "llpr-force-gram-aggregate":
        print(aggregate_force_gram_shards(args.inputs, args.output))
    elif args.command == "llpr-force-feature-shard":
        print(collect_force_feature_shard(
            config, split_path, args.checkpoint, args.output, args.shard, args.total_shards,
            args.subset_size, args.device,
        ))
    elif args.command == "llpr-force-feature-aggregate":
        print(aggregate_force_feature_shards(args.inputs, args.output))
    elif args.command == "llpr-force-aware-fit":
        print(fit_force_aware_posterior_from_files(
            args.energy_gram, args.force_gram, args.energy_calibration, args.force_features,
            args.force_targets, config.llpr.ridge_multipliers, args.output,
        ))
    elif args.command == "llpr-fit":
        print(
            fit_posterior_from_files(
                args.gram,
                args.calibration_predictions,
                config.llpr.ridge_multipliers,
                args.output,
            )
        )
    elif args.command == "llpr-force":
        subset_size = (
            config.data.force_calibration_size if args.split == "calibration" else config.data.force_test_size
        )
        print(
            sample_force_subset(
                config,
                split_path,
                args.split,
                args.checkpoint,
                args.posterior,
                args.output,
                subset_size,
                args.device,
                args.shard,
                args.total_shards,
                subset_namespace=args.subset_namespace,
                n_samples=args.n_samples,
                sample_seed=args.sample_seed,
            )
        )
    elif args.command == "llpr-force-aggregate":
        print(aggregate_force_shards(args.inputs, args.output))
    elif args.command == "build-ensemble-records":
        print(
            build_ensemble_records(
                args.member,
                args.force_reference,
                args.point_output,
                args.force_output,
            )
        )
    elif args.command == "compare-nonensemble-methods":
        method_files = {
            "hierarchical_epic": MethodFiles(
                Path(args.hierarchical_energy_calibration),
                Path(args.hierarchical_energy_test),
                Path(args.hierarchical_force_calibration),
                Path(args.hierarchical_force_test),
            ),
            "graph_nig": MethodFiles(
                Path(args.graph_nig_energy_calibration),
                Path(args.graph_nig_energy_test),
                Path(args.graph_nig_force_calibration),
                Path(args.graph_nig_force_test),
            ),
            "llpr": MethodFiles(
                Path(args.llpr_energy_calibration),
                Path(args.llpr_energy_test),
                Path(args.llpr_force_calibration),
                Path(args.llpr_force_test),
                Path(args.llpr_posterior),
            ),
            "llpr_force_aware": MethodFiles(
                Path(args.llpr_force_aware_energy_calibration),
                Path(args.llpr_force_aware_energy_test),
                Path(args.llpr_force_aware_force_calibration),
                Path(args.llpr_force_aware_force_test),
                Path(args.llpr_force_aware_posterior),
            ),
        }
        print(
            compare_nonensemble_methods(
                method_files,
                args.store,
                args.output_dir,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
            )
        )
    elif args.command == "compare-methods-with-ensemble":
        method_files = {
            "hierarchical_epic": MethodFiles(
                Path(args.hierarchical_energy_calibration),
                Path(args.hierarchical_energy_test),
                Path(args.hierarchical_force_calibration),
                Path(args.hierarchical_force_test),
            ),
            "graph_nig": MethodFiles(
                Path(args.graph_nig_energy_calibration),
                Path(args.graph_nig_energy_test),
                Path(args.graph_nig_force_calibration),
                Path(args.graph_nig_force_test),
            ),
            "llpr": MethodFiles(
                Path(args.llpr_energy_calibration),
                Path(args.llpr_energy_test),
                Path(args.llpr_force_calibration),
                Path(args.llpr_force_test),
                Path(args.llpr_posterior),
            ),
            "llpr_force_aware": MethodFiles(
                Path(args.llpr_force_aware_energy_calibration),
                Path(args.llpr_force_aware_energy_test),
                Path(args.llpr_force_aware_force_calibration),
                Path(args.llpr_force_aware_force_test),
                Path(args.llpr_force_aware_posterior),
            ),
            "ensemble5": MethodFiles(
                Path(args.ensemble5_energy_calibration),
                Path(args.ensemble5_energy_test),
                Path(args.ensemble5_force_calibration),
                Path(args.ensemble5_force_test),
            ),
        }
        print(
            compare_methods_with_ensemble(
                method_files,
                args.store,
                args.output_dir,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
            )
        )
    elif args.command == "epic-force-aggregate":
        print(aggregate_epic_force_shards(args.inputs, args.output))
    elif args.command == "epic-force":
        subset_size = args.subset_size or (
            config.data.force_calibration_size if args.split == "calibration" else config.data.force_test_size
        )
        print(
            sample_epic_force_subset(
                config,
                split_path,
                args.split,
                args.checkpoint,
                args.output,
                subset_size,
                args.device,
                args.shard,
                args.total_shards,
                args.n_samples,
                args.sample_seed,
                args.allocation_mode,
            )
        )
    elif args.command == "report":
        print(
            generate_report(
                args.store,
                args.output_dir,
                config.evaluation.coverage_levels,
                config.evaluation.budget_fractions,
                config.evaluation.bootstrap_replicates,
                config.evaluation.bootstrap_seed,
            )
        )
    elif args.command == "downstream-manifests":
        print(create_downstream_manifests(args.store, config.data.db_path, args.output_dir))
    elif args.command == "select-epic":
        selected, _ = select_epic_lambda(_key_values(args.candidate, float), args.output)
        print(selected)
    elif args.command == "calibrate-epic":
        print(fit_epic_calibration_file(args.predictions, args.output, args.method))
    elif args.command == "calibrate-epic-force":
        print(
            fit_epic_force_calibration_file(
                args.predictions,
                args.output,
                args.method,
            )
        )
    elif args.command == "build-two-model-store":
        manifest = {
            "schema_version": 2,
            "config": config.to_dict(),
            "protocol": "two_model_hierarchical_epic_v1",
            "no_dft_validation": True,
        }
        print(
            build_two_model_store(
                args.output,
                config.data.db_path,
                args.mace_test,
                args.hierarchical_calibration,
                args.hierarchical_test,
                args.hierarchical_calibration_state,
                manifest,
                args.hierarchical_force_calibration,
                args.hierarchical_force_test,
                config.data.force_calibration_size,
                config.data.force_test_size,
            )
        )
    elif args.command == "analyze-two-model":
        print(
            analyze_two_model_predictions(
                args.mace_test,
                args.hierarchical_test,
                args.hierarchical_calibration_state,
                args.output_dir,
            )
        )
    elif args.command == "analyze-force-convergence":
        print(
            analyze_force_convergence(
                args.inputs,
                args.output_dir,
                args.hierarchical_predictions,
            )
        )
    elif args.command == "analyze-atomic-attribution":
        print(
            analyze_atomic_attribution(
                args.predictions,
                args.store,
                args.output_dir,
                force_error_floor=args.force_error_floor,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
            )
        )
    elif args.command == "bootstrap-uq":
        print(
            bootstrap_uq_metrics(
                args.store,
                args.output_dir,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
            )
        )
    elif args.command == "analyze-calibration-sensitivity":
        print(
            analyze_calibration_sensitivity(
                args.energy_calibration,
                args.energy_test,
                args.force_calibration,
                args.force_test,
                args.output_dir,
                coverage_levels=config.evaluation.coverage_levels,
            )
        )
    elif args.command == "compare-force-allocations":
        print(
            compare_force_allocations(
                args.learned_calibration,
                args.learned_test,
                args.uniform_calibration,
                args.uniform_test,
                args.store,
                args.learned_sensitivity_dir,
                args.uniform_sensitivity_dir,
                args.output_dir,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
                coverage_levels=config.evaluation.coverage_levels,
            )
        )
    elif args.command == "summarize-two-model":
        print(
            write_two_model_summary(
                args.report_dir,
                args.analysis_dir,
                args.force_convergence,
                args.checkpoint_audit,
                args.force_calibration,
                args.output_dir,
                args.cost_dir,
            )
        )
    elif args.command == "audit-hierarchical-checkpoint":
        print(
            audit_hierarchical_checkpoint(
                config,
                split_path,
                args.checkpoint,
                args.output,
                split=args.split,
                subset_size=args.subset_size,
                n_samples=args.n_samples,
                sample_seed=args.sample_seed,
                finite_difference_step=args.finite_difference_step,
                device=args.device,
            )
        )
    elif args.command == "build-store":
        selected_lambda, epic_state = calibration_state_from_selection(args.epic_selection)
        manifest = {
            "schema_version": 1,
            "config": config.to_dict(),
            "selected_epic_lambda_max": selected_lambda,
            "no_dft_validation": True,
        }
        print(
            build_store(
                args.output,
                config.data.db_path,
                args.det_test[0],
                args.det_test,
                args.det_calibration,
                args.epic_test,
                epic_state,
                args.llpr_posterior,
                manifest,
                args.llpr_force_calibration,
                args.llpr_force_test,
                args.epic_force_calibration,
                args.epic_force_test,
                config.data.force_calibration_size,
                config.data.force_test_size,
            )
        )
    elif args.command == "cost":
        print(
            write_cost_tables(
                _grouped_key_values(args.training_run),
                _key_values(args.inference_run),
                args.output_dir,
            )
        )
    elif args.command == "s2c-prepare-covariance":
        import sys as _sys

        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in _sys.path:
            _sys.path.insert(0, str(repo_root))
        from scripts.prepare_s2c_covariance import run_from_checkpoint, run_from_shards

        if args.mode == "from-checkpoint":
            return run_from_checkpoint(args)
        return run_from_shards(args)
    elif args.command == "s2c-force-uq":
        import sys as _sys

        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in _sys.path:
            _sys.path.insert(0, str(repo_root))
        from scripts.run_s2c_force_uq import main as _s2c_force_main

        return _s2c_force_main(argv[1:] if argv is not None else None)
    elif args.command == "s2c-calibrate":
        import sys as _sys

        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in _sys.path:
            _sys.path.insert(0, str(repo_root))
        from scripts.run_s2c_calibration import main as _s2c_cal_main

        return _s2c_cal_main(argv[1:] if argv is not None else None)
    elif args.command == "s2c-ellipsoids":
        import sys as _sys

        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in _sys.path:
            _sys.path.insert(0, str(repo_root))
        from scripts.run_s2c_ellipsoids import main as _s2c_ellip_main

        return _s2c_ellip_main(argv[1:] if argv is not None else None)
    elif args.command == "s2c-phonon":
        import sys as _sys

        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in _sys.path:
            _sys.path.insert(0, str(repo_root))
        from scripts.run_s2c_phonon import main as _s2c_ph_main

        return _s2c_ph_main(argv[1:] if argv is not None else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
