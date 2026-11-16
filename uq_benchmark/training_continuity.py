from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch

from .config import load_config
from .distributions import annealed_regularization


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain non-empty JSON objects")
    return rows


def _checkpoint(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a checkpoint dictionary")
    return payload


def _state_present(checkpoint: Mapping[str, object], key: str) -> bool:
    state = checkpoint.get(key)
    return isinstance(state, Mapping) and bool(state)


def _rng_state_keys(checkpoint: Mapping[str, object]) -> list[str]:
    return sorted(
        str(key)
        for key in checkpoint
        if "rng" in str(key).lower() or "random" in str(key).lower()
    )


def _resume_evidence_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and Path(fields[1]).name == "checkpoint_latest.pt":
            digest = fields[0].lower()
            if len(digest) == 64 and all(character in "0123456789abcdef" for character in digest):
                return digest
    raise ValueError(f"{path} does not contain a checkpoint_latest.pt SHA-256 record")


def _normalized_training_config(
    run_config: Mapping[str, object],
) -> dict[str, object]:
    config = copy.deepcopy(run_config.get("config"))
    if not isinstance(config, dict):
        raise ValueError("run config lacks a configuration dictionary")
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("run config lacks a training configuration dictionary")
    training.pop("max_epochs", None)
    training.pop("early_stopping_patience", None)
    return config


def _canonical_json_sha256(payload: object) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _differences(left: object, right: object, prefix: str = "config") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}"
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(_differences(left[key], right[key], path))
        return differences
    return [] if left == right else [prefix]


def audit_training_ablation_preflight(
    reference_run_config_path: str | Path,
    reference_split_path: str | Path,
    seed43_run_config_path: str | Path,
    ablation_split_path: str | Path,
    learned_config_path: str | Path,
    allocation_weight_zero_config_path: str | Path,
    output_path: str | Path,
) -> Path:
    reference_run_config_path = Path(reference_run_config_path)
    reference_split_path = Path(reference_split_path)
    seed43_run_config_path = Path(seed43_run_config_path)
    ablation_split_path = Path(ablation_split_path)
    learned_config_path = Path(learned_config_path)
    allocation_weight_zero_config_path = Path(allocation_weight_zero_config_path)
    output_path = Path(output_path)

    reference_run = _load_json(reference_run_config_path)
    seed43_run = _load_json(seed43_run_config_path)
    reference_config = reference_run.get("config")
    seed43_config = seed43_run.get("config")
    if not isinstance(reference_config, dict) or not isinstance(seed43_config, dict):
        raise ValueError("training run config lacks a configuration dictionary")
    learned_config = load_config(learned_config_path).to_dict()
    zero_config = load_config(allocation_weight_zero_config_path).to_dict()
    if seed43_config != learned_config:
        raise ValueError("seed43 run config differs from the frozen learned-allocation config")

    expected_reference_differences = {
        "config.data.split_dir",
        "config.epic.force_samples",
        "config.output_root",
    }
    observed_reference_differences = set(
        _differences(reference_config, learned_config)
    )
    if observed_reference_differences != expected_reference_differences:
        raise ValueError(
            "reference and seed-ablation configs differ outside the frozen protocol: "
            f"{sorted(observed_reference_differences)}"
        )
    observed_zero_differences = _differences(learned_config, zero_config)
    if observed_zero_differences != ["config.epic.hierarchical_allocation_weight"]:
        raise ValueError(
            "allocation-loss-zero config has unintended changes: "
            f"{observed_zero_differences}"
        )
    if float(learned_config["epic"]["hierarchical_allocation_weight"]) != 0.1:
        raise ValueError("learned allocation weight differs from 0.1")
    if float(zero_config["epic"]["hierarchical_allocation_weight"]) != 0.0:
        raise ValueError("allocation-loss-zero weight differs from 0.0")
    if (
        seed43_run.get("kind") != "epic_hierarchical"
        or int(seed43_run.get("seed", -1)) != 43
        or float(seed43_run.get("lambda_max", -1.0)) != 0.001
    ):
        raise ValueError("seed43 run identity differs from the frozen protocol")

    reference_split_sha256 = _sha256_file(reference_split_path)
    ablation_split_sha256 = _sha256_file(ablation_split_path)
    if reference_split_sha256 != ablation_split_sha256:
        raise ValueError("seed ablations do not use the formal reference split")

    learned_hash = _canonical_json_sha256(learned_config)
    zero_hash = _canonical_json_sha256(zero_config)
    payload = {
        "schema_version": 1,
        "audit_scope": "training_seed_and_allocation_ablation_preflight",
        "training_ablation_protocol_preflight_verified": True,
        "reference_split_sha256": reference_split_sha256,
        "ablation_split_sha256": ablation_split_sha256,
        "reference_split_byte_identity_verified": True,
        "reference_vs_learned_allowed_differences": sorted(
            expected_reference_differences
        ),
        "reference_vs_learned_protocol_alignment_verified": True,
        "learned_vs_zero_only_intended_difference": True,
        "intended_ablation_field": "config.epic.hierarchical_allocation_weight",
        "learned_allocation_weight": 0.1,
        "allocation_weight_zero": 0.0,
        "expected_run_config_sha256": {
            "hierarchical_seed43": learned_hash,
            "hierarchical_seed44": learned_hash,
            "allocation_weight_0_seed42": zero_hash,
        },
        "expected_run_seeds": {
            "hierarchical_seed43": 43,
            "hierarchical_seed44": 44,
            "allocation_weight_0_seed42": 42,
        },
        "input_file_sha256": {
            "reference_run_config.json": _sha256_file(reference_run_config_path),
            "seed43_run_config.json": _sha256_file(seed43_run_config_path),
            "training_learned_allocation.yaml": _sha256_file(learned_config_path),
            "training_allocation_weight_0.yaml": _sha256_file(
                allocation_weight_zero_config_path
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def audit_reference_training_continuity(
    probe_run_dir: str | Path,
    formal_run_dir: str | Path,
    output_path: str | Path,
) -> Path:
    probe_run_dir = Path(probe_run_dir)
    formal_run_dir = Path(formal_run_dir)
    output_path = Path(output_path)

    probe_latest_path = probe_run_dir / "checkpoint_latest.pt"
    resume_epoch4_path = formal_run_dir / "resume_epoch4.pt"
    final_latest_path = formal_run_dir / "checkpoint_latest.pt"
    final_best_path = formal_run_dir / "checkpoint_best.pt"
    probe_config_path = probe_run_dir / "run_config.json"
    copied_probe_config_path = formal_run_dir / "probe_run_config.json"

    probe_latest = _checkpoint(probe_latest_path)
    resume_epoch4 = _checkpoint(resume_epoch4_path)
    final_latest = _checkpoint(final_latest_path)
    final_best = _checkpoint(final_best_path)
    probe_config = _load_json(probe_config_path)
    copied_probe_config = _load_json(copied_probe_config_path)
    formal_config = _load_json(formal_run_dir / "run_config.json")
    probe_rows = _load_jsonl(probe_run_dir / "epochs.jsonl")
    formal_rows = _load_jsonl(formal_run_dir / "epochs.jsonl")

    probe_epochs = [int(row["epoch"]) for row in probe_rows]
    formal_epochs = [int(row["epoch"]) for row in formal_rows]
    combined_epochs = probe_epochs + formal_epochs
    expected_epochs = list(range(combined_epochs[0], combined_epochs[-1] + 1))
    if probe_epochs != list(range(0, 5)):
        raise ValueError(f"probe epochs differ from the archived 0-4 protocol: {probe_epochs}")
    if formal_epochs != list(range(5, 51)):
        raise ValueError("formal epochs differ from the archived 5-50 protocol")
    if combined_epochs != expected_epochs:
        raise ValueError("combined reference-training epoch sequence is not contiguous")

    probe_hash = _sha256_file(probe_latest_path)
    resume_epoch4_hash = _sha256_file(resume_epoch4_path)
    if probe_hash != resume_epoch4_hash:
        raise ValueError("probe checkpoint and formal epoch-4 resume checkpoint differ")
    probe_config_hash = _sha256_file(probe_config_path)
    copied_probe_config_hash = _sha256_file(copied_probe_config_path)
    if probe_config_hash != copied_probe_config_hash or probe_config != copied_probe_config:
        raise ValueError("copied probe run configuration differs from the original")

    identity_fields = ("kind", "seed", "lambda_max", "epic_variant")
    if any(probe_latest.get(key) != resume_epoch4.get(key) for key in identity_fields):
        raise ValueError("checkpoint identity changed at the probe-to-formal handoff")
    if int(probe_latest.get("epoch", -1)) != 4 or int(resume_epoch4.get("epoch", -1)) != 4:
        raise ValueError("probe-to-formal handoff is not an epoch-4 checkpoint")
    if _normalized_training_config(probe_config) != _normalized_training_config(formal_config):
        raise ValueError("probe and formal configurations differ beyond allowed run-length controls")
    for key in ("kind", "seed", "lambda_max", "epic_variant"):
        if probe_config.get(key) != formal_config.get(key):
            raise ValueError(f"probe and formal run configs disagree on {key}")

    training_config = formal_config["config"]["training"]
    lambda_max = float(formal_config["lambda_max"])
    anneal_epochs = int(training_config["lambda_anneal_epochs"])
    schedule_errors = []
    for row in probe_rows + formal_rows:
        epoch = int(row["epoch"])
        expected = annealed_regularization(lambda_max, epoch, anneal_epochs)
        observed = float(row["regularization"])
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15):
            schedule_errors.append({"epoch": epoch, "expected": expected, "observed": observed})
    if schedule_errors:
        raise ValueError(f"regularization schedule is discontinuous: {schedule_errors[:3]}")

    all_rng_keys = sorted(
        set(
            _rng_state_keys(probe_latest)
            + _rng_state_keys(resume_epoch4)
            + _rng_state_keys(final_latest)
            + _rng_state_keys(final_best)
        )
    )
    initial_optimizer_present = _state_present(resume_epoch4, "optimizer_state_dict")
    initial_scheduler_present = _state_present(resume_epoch4, "scheduler_state_dict")
    if not initial_optimizer_present or not initial_scheduler_present:
        raise ValueError("epoch-4 resume checkpoint lacks optimizer or scheduler state")

    best_row = min(probe_rows + formal_rows, key=lambda row: float(row["validation_loss"]))
    if int(final_best.get("epoch", -1)) != int(best_row["epoch"]):
        raise ValueError("best checkpoint epoch does not match the epoch history")
    if not math.isclose(
        float(final_best.get("best_validation_loss", math.nan)),
        float(best_row["validation_loss"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("best checkpoint loss does not match the epoch history")
    if int(final_latest.get("epoch", -1)) != formal_epochs[-1]:
        raise ValueError("latest checkpoint does not match the final logged epoch")

    intermediate_evidence_path = formal_run_dir / "resume_epoch14_20260804.sha256"
    intermediate_hash = _resume_evidence_hash(intermediate_evidence_path)
    payload = {
        "schema_version": 1,
        "audit_scope": "seed42_reference_training_continuity",
        "reference_training_epoch_and_state_continuity_verified": True,
        "probe_to_formal_handoff": {
            "probe_latest_sha256": probe_hash,
            "formal_resume_epoch4_sha256": resume_epoch4_hash,
            "checkpoint_byte_identity_verified": True,
            "checkpoint_epoch": 4,
            "checkpoint_kind": str(resume_epoch4["kind"]),
            "checkpoint_seed": int(resume_epoch4["seed"]),
            "checkpoint_lambda_max": float(resume_epoch4["lambda_max"]),
            "optimizer_state_present": initial_optimizer_present,
            "scheduler_state_present": initial_scheduler_present,
        },
        "configuration_handoff": {
            "probe_run_config_sha256": probe_config_hash,
            "copied_probe_run_config_sha256": copied_probe_config_hash,
            "probe_run_config_byte_identity_verified": True,
            "training_configuration_match_except_run_length_controls": True,
            "allowed_run_length_control_changes": [
                "training.max_epochs",
                "training.early_stopping_patience",
            ],
        },
        "epoch_history": {
            "probe_first_epoch": probe_epochs[0],
            "probe_last_epoch": probe_epochs[-1],
            "formal_first_epoch": formal_epochs[0],
            "formal_last_epoch": formal_epochs[-1],
            "combined_epoch_count": len(combined_epochs),
            "combined_epoch_sequence_contiguous": True,
            "regularization_schedule_continuous": True,
            "lambda_max": lambda_max,
            "lambda_anneal_epochs": anneal_epochs,
        },
        "final_checkpoints": {
            "best_checkpoint_sha256": _sha256_file(final_best_path),
            "best_checkpoint_epoch": int(final_best["epoch"]),
            "best_validation_loss": float(final_best["best_validation_loss"]),
            "latest_checkpoint_sha256": _sha256_file(final_latest_path),
            "latest_checkpoint_epoch": int(final_latest["epoch"]),
            "latest_stale_epochs": int(final_latest["stale_epochs"]),
        },
        "intermediate_epoch14_resume": {
            "hash_record_present": intermediate_hash is not None,
            "recorded_checkpoint_sha256": intermediate_hash,
            "historical_checkpoint_object_available": False,
            "historical_checkpoint_state_verified": False,
            "interpretation": (
                "The contemporaneous hash record documents an intermediate resume, but the "
                "historical checkpoint object is unavailable for state-level reinspection."
            ),
        },
        "reproducibility_claims": {
            "optimizer_scheduler_state_restoration_supported": True,
            "rng_state_keys_observed": all_rng_keys,
            "rng_state_continuity_verified": False,
            "sampler_data_order_continuity_verified": False,
            "bitwise_lossless_resume_verified": False,
            "permitted_wording": (
                "Model, optimizer, and learning-rate scheduler state were restored; the logged "
                "epoch and regularization schedules are contiguous."
            ),
            "prohibited_wording": (
                "Bitwise-lossless continuation or identical sampler/data order was verified."
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path
