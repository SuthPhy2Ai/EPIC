from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

from .calibration import apply_variance_map, fit_gaussian_variance_map
from .distributions import student_t_nll
from .llpr_pipeline import load_posterior
from .hierarchical_epic import (
    SAMPLING_METHOD as EPIC_HIERARCHICAL_FORCE_SAMPLING_METHOD,
    UNIFORM_SAMPLING_METHOD as EPIC_HIERARCHICAL_UNIFORM_FORCE_SAMPLING_METHOD,
)
from .s2c_covariance import SAMPLING_METHOD_ANALYTIC as EPIC_S2C_ANALYTIC_FORCE_SAMPLING_METHOD
from .methods import calibrate_ensemble, calibrate_epic, combine_ensemble, epic_prediction_from_nig
from .schema import BenchmarkStore, validate_method_alignment
from .split import hashed_subset
from .types import CalibrationState, PredictionBatch


EPIC_FORCE_SAMPLING_METHOD = "graph_nig_latent_mean_rsample"
EPIC_ATOMWISE_FORCE_SAMPLING_METHOD = "atomwise_nig_latent_mean_rsample"
EPIC_FORCE_SAMPLING_METHODS = {
    EPIC_FORCE_SAMPLING_METHOD,
    EPIC_ATOMWISE_FORCE_SAMPLING_METHOD,
    EPIC_HIERARCHICAL_FORCE_SAMPLING_METHOD,
    EPIC_HIERARCHICAL_UNIFORM_FORCE_SAMPLING_METHOD,
    EPIC_S2C_ANALYTIC_FORCE_SAMPLING_METHOD,
}


def validate_epic_force_record(record: dict, label: str) -> None:
    required = {
        "db_index",
        "force_true",
        "force_mean",
        "force_scale_raw",
        "force_offsets",
        "sample_seed",
        "n_samples",
        "sampling_method",
        "uncertainty_component",
        "rng_seed_replayed_per_structure",
        "cross_structure_base_random_variable_alignment_verified",
        "force_samples_energy_gradient_derived",
        "global_random_pes_verified",
        "force_posterior_identified",
        "samplewise_permutation_equivariance_verified",
        "aleatoric_force_noise",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"EPIC {label} force file is missing fields: {sorted(missing)}")
    if str(record["sampling_method"]) not in EPIC_FORCE_SAMPLING_METHODS:
        raise ValueError(f"EPIC {label} force file was not produced by energy-first NIG sampling")
    if str(record["uncertainty_component"]) != "epistemic":
        raise ValueError(f"EPIC {label} force file must contain epistemic-only samples")
    if (
        "split" in record
        and label in ("calibration", "test")
        and str(record["split"]) != label
    ):
        raise ValueError(f"EPIC {label} force file declares split {str(record['split'])}")
    if not bool(record["rng_seed_replayed_per_structure"]):
        raise ValueError(f"EPIC {label} force file must declare per-structure seed replay")
    if bool(record["cross_structure_base_random_variable_alignment_verified"]):
        raise ValueError(
            f"EPIC {label} force file makes an unsupported cross-structure "
            "random-variable alignment claim"
        )
    if not bool(record["force_samples_energy_gradient_derived"]):
        raise ValueError(f"EPIC {label} force samples are not energy-gradient-derived")
    if bool(record["global_random_pes_verified"]):
        raise ValueError(f"EPIC {label} force file makes an unsupported global random-PES claim")
    if bool(record["force_posterior_identified"]):
        raise ValueError(f"EPIC {label} force file makes an unsupported posterior-identification claim")
    if bool(record["samplewise_permutation_equivariance_verified"]):
        raise ValueError(
            f"EPIC {label} force file makes an unsupported sample-wise permutation claim"
        )
    if bool(record["aleatoric_force_noise"]):
        raise ValueError(f"EPIC {label} force file includes forbidden aleatoric force noise")
    if int(record["n_samples"]) < 2:
        raise ValueError(f"EPIC {label} force file needs at least two samples")
    db_index = np.asarray(record["db_index"])
    offsets = np.asarray(record["force_offsets"])
    force_true = np.asarray(record["force_true"])
    force_mean = np.asarray(record["force_mean"])
    force_scale = np.asarray(record["force_scale_raw"])
    if len(offsets) != len(db_index) + 1 or offsets[0] != 0 or offsets[-1] != len(force_true):
        raise ValueError(f"EPIC {label} force offsets do not match structures and atoms")
    if force_true.shape != force_mean.shape or force_true.shape != force_scale.shape:
        raise ValueError(f"EPIC {label} force arrays have inconsistent shapes")
    if force_true.ndim != 2 or force_true.shape[1] != 3:
        raise ValueError(f"EPIC {label} force arrays must have shape [n_atoms, 3]")
    if np.any(~np.isfinite(force_mean)) or np.any(~np.isfinite(force_scale)) or np.any(force_scale < 0):
        raise ValueError(f"EPIC {label} force moments must be finite with nonnegative scales")


def deterministic_prediction(path: str | Path) -> PredictionBatch:
    with np.load(path) as data:
        prediction = PredictionBatch(
            db_index=data["db_index"],
            energy_location=data["energy_pred_total"] / data["natoms"],
            force_location=data["force_pred"],
            distribution_family="none",
            inference_seconds=float(data["inference_seconds"]),
            not_applicable={"energy_scale": "deterministic control has no UQ distribution", "force_scale": "deterministic control has no UQ distribution"},
        )
        prediction.peak_memory_bytes = int(data.get("peak_memory_bytes", 0))
    prediction.validate()
    return prediction


def epic_prediction(path: str | Path, calibration=None) -> PredictionBatch:
    with np.load(path) as data:
        prediction = epic_prediction_from_nig(
            data["db_index"], data["energy_pred_total"], data["nu"], data["alpha"], data["beta"], data["natoms"], calibration, data["force_pred"]
        )
        prediction.inference_seconds = float(data["inference_seconds"])
        prediction.peak_memory_bytes = int(data.get("peak_memory_bytes", 0))
        prediction.metadata = {
            **dict(prediction.metadata),
            "checkpoint_kind": str(data.get("checkpoint_kind", "unknown")),
            "epic_variant": str(data.get("epic_variant", "unknown")),
        }
        return prediction


def select_epic_lambda(calibration_files: dict[float, str | Path], output_path: str | Path) -> tuple[float, object]:
    candidates = []
    for lambda_max, path in sorted(calibration_files.items()):
        raw = epic_prediction(path)
        with np.load(path) as data:
            target = data["energy_true_total"] / data["natoms"]
        state = calibrate_epic(target, raw)
        score = float(np.mean(student_t_nll(target, raw.energy_location, raw.energy_scale_raw * state.parameters["tau"], raw.energy_df)))
        candidates.append({"lambda_max": lambda_max, "calibrated_nll": score, "calibration": dict(state.parameters), "path": str(path), "state": state})
    best = min(candidates, key=lambda row: row["calibrated_nll"])
    serializable = [{key: value for key, value in row.items() if key != "state"} for row in candidates]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"selected_lambda_max": best["lambda_max"], "candidates": serializable}, indent=2, sort_keys=True) + "\n")
    return float(best["lambda_max"]), best["state"]


def calibration_state_from_selection(path: str | Path):
    selection = json.loads(Path(path).read_text())
    selected = float(selection["selected_lambda_max"])
    row = next(item for item in selection["candidates"] if float(item["lambda_max"]) == selected)
    return selected, CalibrationState("epic", row["calibration"])


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_epic_calibration_file(
    predictions_path: str | Path,
    output_path: str | Path,
    method: str = "hierarchical_epic",
) -> Path:
    """Fit one Student-t scale using calibration predictions only."""
    with np.load(predictions_path) as data:
        if "split" in data and str(data["split"]) != "calibration":
            raise ValueError("EPIC calibration input must come from the calibration split")
        target = data["energy_true_total"] / data["natoms"]
        n_structures = len(data["db_index"])
    raw = epic_prediction(predictions_path)
    state = calibrate_epic(target, raw, method=method)
    source = Path(predictions_path).resolve()
    payload = {
        "schema_version": 1,
        "method": state.method,
        "objective": state.objective,
        "split": state.split,
        "parameters": dict(state.parameters),
        "n_structures": n_structures,
        "source_predictions": str(source),
        "source_predictions_sha256": _sha256_file(source),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def fit_epic_force_calibration_file(
    predictions_path: str | Path,
    output_path: str | Path,
    method: str = "hierarchical_epic_force",
) -> Path:
    """Fit the force variance map from one calibration force record."""
    with np.load(predictions_path) as source:
        record = {name: source[name] for name in source.files}
    validate_epic_force_record(record, "calibration")
    if "split" in record and str(record["split"]) != "calibration":
        raise ValueError("EPIC force calibration input must come from calibration")
    state = fit_gaussian_variance_map(
        record["force_true"].reshape(-1),
        record["force_mean"].reshape(-1),
        record["force_scale_raw"].reshape(-1) ** 2,
        method=method,
    )
    source_path = Path(predictions_path).resolve()
    payload = {
        "schema_version": 1,
        "method": state.method,
        "objective": state.objective,
        "split": state.split,
        "parameters": dict(state.parameters),
        "n_structures": int(len(record["db_index"])),
        "n_force_components": int(record["force_true"].size),
        "n_samples": int(record["n_samples"]),
        "sample_seed": int(record["sample_seed"]),
        "source_predictions": str(source_path),
        "source_predictions_sha256": _sha256_file(source_path),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def load_calibration_state_file(
    path: str | Path,
    expected_method: str | None = None,
) -> CalibrationState:
    payload = json.loads(Path(path).read_text())
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported calibration-state schema")
    if payload.get("split") != "calibration":
        raise ValueError("calibration state was not fitted on the calibration split")
    method = str(payload["method"])
    if expected_method is not None and method != expected_method:
        raise ValueError(f"expected calibration method {expected_method}, got {method}")
    parameters = payload.get("parameters", {})
    tau = float(parameters.get("tau", 0.0))
    sigma0 = float(parameters.get("sigma0", 0.0))
    if not np.isfinite(tau) or tau <= 0 or not np.isfinite(sigma0) or sigma0 < 0:
        raise ValueError("calibration state contains invalid scale parameters")
    return CalibrationState(
        method=method,
        parameters={"tau": tau, "sigma0": sigma0},
        objective=str(payload.get("objective", "negative_log_likelihood")),
        split="calibration",
    )


def llpr_energy_prediction(deterministic_path: str | Path, posterior_path: str | Path) -> PredictionBatch:
    posterior = load_posterior(posterior_path)
    with np.load(deterministic_path) as data:
        raw_variance = posterior.energy_variance(data["llpr_features"], calibrated=False)
        calibrated = posterior.energy_variance(data["llpr_features"], calibrated=True)
        prediction = PredictionBatch(
            db_index=data["db_index"],
            energy_location=data["energy_pred_total"] / data["natoms"],
            energy_scale_raw=np.sqrt(np.maximum(raw_variance, 1e-16)),
            energy_scale_calibrated=np.sqrt(np.maximum(calibrated, 1e-16)),
            energy_risk=raw_variance,
            force_location=data["force_pred"],
            distribution_family="gaussian",
            metadata={"ridge": posterior.ridge, "n_training_samples": posterior.n_training_samples},
        )
    prediction.validate()
    return prediction


def material_ids_for_indices(db_path: str | Path, indices: np.ndarray) -> np.ndarray:
    connection = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    try:
        lookup = dict(connection.execute("SELECT id - 1, value FROM text_key_values WHERE key='material_id'"))
    finally:
        connection.close()
    return np.asarray([lookup[int(index)] for index in indices], dtype=object)


def _force_subset_from_members(
    paths: list[str | Path], subset_size: int, namespace: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    records = [dict(np.load(path)) for path in paths]
    reference = records[0]
    if any(not np.array_equal(record["db_index"], reference["db_index"]) for record in records[1:]):
        raise ValueError("force ensemble member indices differ")
    selected_db = hashed_subset(reference["db_index"], subset_size, namespace)
    positions = np.searchsorted(reference["db_index"], selected_db)
    if not np.array_equal(reference["db_index"][positions], selected_db):
        raise ValueError("force subset selection failed because db_index is not sorted")
    atom_indices, offsets = [], [0]
    for position in positions:
        start, stop = reference["force_offsets"][position : position + 2]
        atom_indices.append(np.arange(start, stop))
        offsets.append(offsets[-1] + int(stop - start))
    atom_indices = np.concatenate(atom_indices)
    member_forces = np.stack([record["force_pred"][atom_indices] for record in records])
    return (
        selected_db,
        reference["force_true"][atom_indices],
        member_forces.mean(axis=0),
        member_forces.var(axis=0, ddof=1),
        np.asarray(offsets, dtype=np.int64),
    )


def _validate_force_subset_against_predictions(
    force_record: dict,
    predictions: dict,
    label: str,
) -> None:
    reference_index = np.asarray(predictions["db_index"], dtype=np.int64)
    reference_offsets = np.asarray(predictions["force_offsets"], dtype=np.int64)
    reference_force = np.asarray(predictions["force_true"])
    positions = {int(value): index for index, value in enumerate(reference_index)}
    subset_index = np.asarray(force_record["db_index"], dtype=np.int64)
    subset_offsets = np.asarray(force_record["force_offsets"], dtype=np.int64)
    subset_force = np.asarray(force_record["force_true"])
    for subset_position, db_index in enumerate(subset_index):
        if int(db_index) not in positions:
            raise ValueError(f"{label} force subset contains db_index outside its split")
        reference_position = positions[int(db_index)]
        reference_start, reference_stop = reference_offsets[reference_position : reference_position + 2]
        subset_start, subset_stop = subset_offsets[subset_position : subset_position + 2]
        expected = reference_force[reference_start:reference_stop]
        observed = subset_force[subset_start:subset_stop]
        if expected.shape != observed.shape or not np.allclose(expected, observed, rtol=0.0, atol=1e-7):
            raise ValueError(f"{label} force truth is not aligned with point predictions")


def build_two_model_store(
    output_path: str | Path,
    db_path: str | Path,
    mace_test_path: str | Path,
    hierarchical_calibration_path: str | Path,
    hierarchical_test_path: str | Path,
    hierarchical_calibration_state_path: str | Path,
    manifest: dict,
    hierarchical_force_calibration_path: str | Path | None = None,
    hierarchical_force_test_path: str | Path | None = None,
    expected_force_calibration_size: int | None = 10_000,
    expected_force_test_size: int | None = 20_000,
) -> Path:
    """Build the paper store containing only MACE and Hierarchical EPIC."""
    mace = deterministic_prediction(mace_test_path)
    calibration_state = load_calibration_state_file(
        hierarchical_calibration_state_path,
        expected_method="hierarchical_epic",
    )
    calibration_payload = json.loads(Path(hierarchical_calibration_state_path).read_text())
    if calibration_payload.get("source_predictions_sha256") != _sha256_file(
        hierarchical_calibration_path
    ):
        raise ValueError("calibration state does not match Hierarchical EPIC calibration predictions")
    hierarchical = epic_prediction(hierarchical_test_path, calibration_state)

    with np.load(mace_test_path) as source:
        mace_record = {name: source[name] for name in source.files}
    with np.load(hierarchical_calibration_path) as source:
        calibration_record = {name: source[name] for name in source.files}
    with np.load(hierarchical_test_path) as source:
        hierarchical_record = {name: source[name] for name in source.files}
    if "split" in mace_record and str(mace_record["split"]) != "test":
        raise ValueError("MACE point predictions must come from the test split")
    if "split" in calibration_record and str(calibration_record["split"]) != "calibration":
        raise ValueError("Hierarchical EPIC calibration predictions use the wrong split")
    if "split" in hierarchical_record and str(hierarchical_record["split"]) != "test":
        raise ValueError("Hierarchical EPIC test predictions use the wrong split")
    if str(hierarchical_record.get("checkpoint_kind", "")) != "epic_hierarchical":
        raise ValueError("Hierarchical EPIC test predictions have the wrong checkpoint kind")
    if not np.array_equal(mace.db_index, hierarchical.db_index):
        raise ValueError("MACE and Hierarchical EPIC test db_index arrays differ")
    for field in ("natoms", "energy_true_total", "force_true", "force_offsets"):
        if field not in mace_record or field not in hierarchical_record:
            raise ValueError(f"two-model input is missing {field}")
        if not np.allclose(mace_record[field], hierarchical_record[field], rtol=0.0, atol=1e-7):
            raise ValueError(f"MACE and Hierarchical EPIC disagree on {field}")

    force_state = None
    force_test = None
    if (hierarchical_force_calibration_path is None) != (hierarchical_force_test_path is None):
        raise ValueError("both Hierarchical EPIC force calibration and test files are required together")
    if hierarchical_force_calibration_path is not None:
        with np.load(hierarchical_force_calibration_path) as source:
            force_calibration = {name: source[name] for name in source.files}
        with np.load(hierarchical_force_test_path) as source:
            force_test = {name: source[name] for name in source.files}
        validate_epic_force_record(force_calibration, "calibration")
        validate_epic_force_record(force_test, "test")
        _validate_force_subset_against_predictions(force_calibration, calibration_record, "calibration")
        _validate_force_subset_against_predictions(force_test, hierarchical_record, "test")
        if expected_force_calibration_size is not None and len(force_calibration["db_index"]) != expected_force_calibration_size:
            raise ValueError("Hierarchical EPIC calibration force file has the wrong subset size")
        if expected_force_test_size is not None and len(force_test["db_index"]) != expected_force_test_size:
            raise ValueError("Hierarchical EPIC test force file has the wrong subset size")
        if int(force_calibration["sample_seed"]) != int(force_test["sample_seed"]):
            raise ValueError("Hierarchical EPIC force files use different sample seeds")
        if int(force_calibration["n_samples"]) != int(force_test["n_samples"]):
            raise ValueError("Hierarchical EPIC force files use different sample counts")
        force_state = fit_gaussian_variance_map(
            force_calibration["force_true"].reshape(-1),
            force_calibration["force_mean"].reshape(-1),
            force_calibration["force_scale_raw"].reshape(-1) ** 2,
            method="hierarchical_epic_force",
        )

    artifact_paths = {
        "mace_test": mace_test_path,
        "hierarchical_calibration": hierarchical_calibration_path,
        "hierarchical_test": hierarchical_test_path,
        "hierarchical_calibration_state": hierarchical_calibration_state_path,
    }
    if hierarchical_force_calibration_path is not None:
        artifact_paths["hierarchical_force_calibration"] = hierarchical_force_calibration_path
        artifact_paths["hierarchical_force_test"] = hierarchical_force_test_path
    complete_manifest = dict(manifest)
    complete_manifest.update(
        {
            "schema_version": 2,
            "benchmark_scope": ["mace", "hierarchical_epic"],
            "deferred_methods": ["ensemble5", "llpr", "graph_nig_epic"],
            "input_artifacts": {
                name: {
                    "path": str(Path(path).resolve()),
                    "sha256": _sha256_file(path),
                }
                for name, path in artifact_paths.items()
            },
        }
    )

    indices = np.asarray(mace_record["db_index"])
    natoms = np.asarray(mace_record["natoms"])
    energy_true_pa = np.asarray(mace_record["energy_true_total"]) / natoms
    force_true = np.asarray(mace_record["force_true"])
    force_offsets = np.asarray(mace_record["force_offsets"])
    material_id = material_ids_for_indices(db_path, indices)
    with BenchmarkStore(output_path, "w") as store:
        store.write_manifest(complete_manifest)
        store.write_structures(
            indices,
            material_id,
            natoms,
            energy_true_pa,
            force_true,
            force_offsets,
        )
        store.write_method("mace", mace)
        store.write_method("hierarchical_epic", hierarchical)
        store.write_calibration(calibration_state)
        if force_state is not None and force_test is not None:
            calibrated_variance = apply_variance_map(
                force_test["force_scale_raw"] ** 2,
                force_state,
            )
            store.write_force_subset(
                "hierarchical_epic",
                force_test["db_index"],
                force_test["force_true"],
                force_test["force_mean"],
                force_test["force_scale_raw"],
                np.sqrt(calibrated_variance),
                force_test["force_offsets"],
                distribution_family="gaussian",
                metadata={
                    "distribution_approximation": "gaussian_moment_matched_mc",
                    "sampling_method": str(force_test["sampling_method"]),
                    "uncertainty_component": "epistemic",
                    "force_samples_energy_gradient_derived": True,
                    "global_random_pes_verified": False,
                    "force_posterior_identified": False,
                    "n_samples": int(force_test["n_samples"]),
                    "sample_seed": int(force_test["sample_seed"]),
                    "calibration_n_structures": int(
                        len(force_calibration["db_index"])
                    ),
                    "calibration_inference_seconds_summed_shards": float(
                        force_calibration.get("inference_seconds", 0.0)
                    ),
                    "calibration_peak_memory_bytes_per_device": int(
                        force_calibration.get("peak_memory_bytes", 0)
                    ),
                    "test_inference_seconds_summed_shards": float(
                        force_test.get("inference_seconds", 0.0)
                    ),
                    "test_peak_memory_bytes_per_device": int(
                        force_test.get("peak_memory_bytes", 0)
                    ),
                },
            )
            store.write_calibration(force_state)
    validate_method_alignment(output_path)
    return Path(output_path)


def build_store(
    output_path: str | Path,
    db_path: str | Path,
    test_reference_path: str | Path,
    deterministic_test_paths: list[str | Path],
    deterministic_calibration_paths: list[str | Path],
    epic_test_path: str | Path,
    epic_state,
    llpr_posterior_path: str | Path,
    manifest: dict,
    llpr_force_calibration_path: str | Path | None = None,
    llpr_force_test_path: str | Path | None = None,
    epic_force_calibration_path: str | Path | None = None,
    epic_force_test_path: str | Path | None = None,
    ensemble_force_calibration_size: int = 10_000,
    ensemble_force_test_size: int = 20_000,
) -> Path:
    test_members = [deterministic_prediction(path) for path in deterministic_test_paths]
    calibration_members = [deterministic_prediction(path) for path in deterministic_calibration_paths]
    with np.load(deterministic_calibration_paths[0]) as calibration_data:
        calibration_target = calibration_data["energy_true_total"] / calibration_data["natoms"]
    ensemble_state = calibrate_ensemble(calibration_target, calibration_members)
    ensemble = combine_ensemble(test_members, ensemble_state)
    _, ensemble_force_cal_true, ensemble_force_cal_mean, ensemble_force_cal_var, _ = _force_subset_from_members(
        deterministic_calibration_paths, ensemble_force_calibration_size, "ensemble-force-calibration-v1"
    )
    ensemble_force_state = fit_gaussian_variance_map(
        ensemble_force_cal_true.reshape(-1),
        ensemble_force_cal_mean.reshape(-1),
        ensemble_force_cal_var.reshape(-1),
        method="ensemble5_force",
    )
    ensemble.force_scale_calibrated = np.sqrt(
        apply_variance_map(ensemble.force_scale_raw**2, ensemble_force_state)
    )
    (
        ensemble_force_db,
        ensemble_force_true,
        ensemble_force_mean,
        ensemble_force_var,
        ensemble_force_offsets,
    ) = _force_subset_from_members(
        deterministic_test_paths, ensemble_force_test_size, "ensemble-force-test-v1"
    )
    deterministic = test_members[0]
    epic = epic_prediction(epic_test_path, epic_state)
    llpr = llpr_energy_prediction(deterministic_test_paths[0], llpr_posterior_path)
    llpr_force_state = None
    llpr_force_test = None
    if (llpr_force_calibration_path is None) != (llpr_force_test_path is None):
        raise ValueError("both LLPR force calibration and test subset files are required together")
    if llpr_force_calibration_path is not None:
        with np.load(llpr_force_calibration_path) as force_calibration:
            llpr_force_state = fit_gaussian_variance_map(
                force_calibration["force_true"].reshape(-1),
                force_calibration["force_mean"].reshape(-1),
                force_calibration["force_scale_raw"].reshape(-1) ** 2,
                method="llpr_force",
            )
        llpr_force_test = dict(np.load(llpr_force_test_path))
    epic_force_state = None
    epic_force_test = None
    if (epic_force_calibration_path is None) != (epic_force_test_path is None):
        raise ValueError("both EPIC force calibration and test subset files are required together")
    if epic_force_calibration_path is not None:
        epic_force_calibration = dict(np.load(epic_force_calibration_path))
        epic_force_test = dict(np.load(epic_force_test_path))
        for label, record in (("calibration", epic_force_calibration), ("test", epic_force_test)):
            validate_epic_force_record(record, label)
        if len(epic_force_calibration["db_index"]) != ensemble_force_calibration_size:
            raise ValueError("EPIC calibration force file has the wrong subset size")
        if len(epic_force_test["db_index"]) != ensemble_force_test_size:
            raise ValueError("EPIC test force file has the wrong subset size")
        if int(epic_force_calibration["sample_seed"]) != int(epic_force_test["sample_seed"]):
            raise ValueError("EPIC calibration and test force files use different sample seeds")
        if int(epic_force_calibration["n_samples"]) != int(epic_force_test["n_samples"]):
            raise ValueError("EPIC calibration and test force files use different sample counts")
        epic_force_state = fit_gaussian_variance_map(
            epic_force_calibration["force_true"].reshape(-1),
            epic_force_calibration["force_mean"].reshape(-1),
            epic_force_calibration["force_scale_raw"].reshape(-1) ** 2,
            method="epic_force",
        )
    with np.load(test_reference_path) as reference:
        indices = reference["db_index"]
        natoms = reference["natoms"]
        energy_true_pa = reference["energy_true_total"] / natoms
        force_true = reference["force_true"]
        force_offsets = reference["force_offsets"]
    material_id = material_ids_for_indices(db_path, indices)
    with BenchmarkStore(output_path, "w") as store:
        store.write_manifest(manifest)
        store.write_structures(indices, material_id, natoms, energy_true_pa, force_true, force_offsets)
        store.write_method("deterministic", deterministic)
        store.write_method("ensemble5", ensemble)
        store.write_method("epic", epic)
        store.write_method("llpr", llpr)
        store.write_calibration(ensemble_state)
        store.write_calibration(ensemble_force_state)
        store.write_calibration(epic_state)
        store.write_calibration(load_posterior(llpr_posterior_path).calibration)
        store.write_force_subset(
            "ensemble5",
            ensemble_force_db,
            ensemble_force_true,
            ensemble_force_mean,
            np.sqrt(np.maximum(ensemble_force_var, 0.0)),
            np.sqrt(apply_variance_map(ensemble_force_var, ensemble_force_state)),
            ensemble_force_offsets,
        )
        if llpr_force_state is not None and llpr_force_test is not None:
            calibrated_variance = apply_variance_map(
                llpr_force_test["force_scale_raw"] ** 2, llpr_force_state
            )
            store.write_force_subset(
                "llpr",
                llpr_force_test["db_index"],
                llpr_force_test["force_true"],
                llpr_force_test["force_mean"],
                llpr_force_test["force_scale_raw"],
                np.sqrt(calibrated_variance),
                llpr_force_test["force_offsets"],
            )
            store.write_calibration(llpr_force_state)
        if epic_force_state is not None and epic_force_test is not None:
            calibrated_variance = apply_variance_map(
                epic_force_test["force_scale_raw"] ** 2, epic_force_state
            )
            store.write_force_subset(
                "epic",
                epic_force_test["db_index"],
                epic_force_test["force_true"],
                epic_force_test["force_mean"],
                epic_force_test["force_scale_raw"],
                np.sqrt(calibrated_variance),
                epic_force_test["force_offsets"],
                distribution_family="gaussian",
                metadata={
                    "distribution_approximation": "gaussian_moment_matched_mc",
                    "sampling_method": str(epic_force_test["sampling_method"]),
                    "uncertainty_component": "epistemic",
                    "rng_seed_replayed_per_structure": True,
                    "cross_structure_base_random_variable_alignment_verified": False,
                    "force_samples_energy_gradient_derived": True,
                    "global_random_pes_verified": False,
                    "force_posterior_identified": False,
                    "samplewise_permutation_equivariance_verified": False,
                    "aleatoric_force_noise": False,
                    "n_samples": int(epic_force_test["n_samples"]),
                    "sample_seed": int(epic_force_test["sample_seed"]),
                },
            )
            store.write_calibration(epic_force_state)
    validate_method_alignment(output_path)
    return Path(output_path)
