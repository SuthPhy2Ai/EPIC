from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from uq_benchmark.config import load_config
from uq_benchmark.training_continuity import (
    audit_reference_training_continuity,
    audit_training_ablation_preflight,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(max_epochs: int, patience: int, resume: str | None) -> dict:
    return {
        "kind": "epic_hierarchical",
        "seed": 42,
        "lambda_max": 0.001,
        "epic_variant": "hierarchical_atomwise_shared_scale",
        "resume_checkpoint": resume,
        "started_at_utc": "ignored",
        "config": {
            "data": {"seed": 42},
            "training": {
                "max_epochs": max_epochs,
                "early_stopping_patience": patience,
                "lambda_anneal_epochs": 5,
            },
        },
    }


def _checkpoint(epoch: int, best_loss: float, stale_epochs: int = 0) -> dict:
    return {
        "kind": "epic_hierarchical",
        "seed": 42,
        "lambda_max": 0.001,
        "epic_variant": "hierarchical_atomwise_shared_scale",
        "epoch": epoch,
        "best_validation_loss": best_loss,
        "stale_epochs": stale_epochs,
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "optimizer_state_dict": {"state": {0: {"step": torch.tensor(1.0)}}},
        "scheduler_state_dict": {"best": best_loss},
    }


def _epoch_row(epoch: int) -> dict:
    return {
        "epoch": epoch,
        "validation_loss": 2.0 - 0.01 * epoch if epoch <= 30 else 1.7 + 0.01 * (epoch - 30),
        "regularization": 0.001 * min((epoch + 1) / 5, 1.0),
    }


def test_reference_training_continuity_audit_is_conservative(tmp_path):
    probe = tmp_path / "probe"
    formal = tmp_path / "formal"
    probe.mkdir()
    formal.mkdir()

    probe_config = _config(5, 5, None)
    (probe / "run_config.json").write_text(json.dumps(probe_config, sort_keys=True))
    (formal / "probe_run_config.json").write_bytes((probe / "run_config.json").read_bytes())
    (formal / "run_config.json").write_text(
        json.dumps(_config(300, 20, "checkpoint_latest.pt"), sort_keys=True)
    )
    (probe / "epochs.jsonl").write_text(
        "\n".join(json.dumps(_epoch_row(epoch)) for epoch in range(5)) + "\n"
    )
    (formal / "epochs.jsonl").write_text(
        "\n".join(json.dumps(_epoch_row(epoch)) for epoch in range(5, 51)) + "\n"
    )

    handoff = _checkpoint(4, 1.96)
    torch.save(handoff, probe / "checkpoint_latest.pt")
    (formal / "resume_epoch4.pt").write_bytes((probe / "checkpoint_latest.pt").read_bytes())
    torch.save(_checkpoint(30, 1.7), formal / "checkpoint_best.pt")
    torch.save(_checkpoint(50, 1.7, stale_epochs=20), formal / "checkpoint_latest.pt")
    historical_hash = "a" * 64
    (formal / "resume_epoch14_20260804.sha256").write_text(
        f"{historical_hash}  /private/run/checkpoint_latest.pt\n"
    )

    output = audit_reference_training_continuity(probe, formal, tmp_path / "audit.json")
    result = json.loads(output.read_text())
    assert result["reference_training_epoch_and_state_continuity_verified"] is True
    assert result["probe_to_formal_handoff"]["checkpoint_byte_identity_verified"] is True
    assert result["epoch_history"]["combined_epoch_sequence_contiguous"] is True
    assert result["final_checkpoints"]["best_checkpoint_epoch"] == 30
    assert result["intermediate_epoch14_resume"]["recorded_checkpoint_sha256"] == historical_hash
    assert result["reproducibility_claims"]["rng_state_continuity_verified"] is False
    assert result["reproducibility_claims"]["bitwise_lossless_resume_verified"] is False
    assert str(tmp_path) not in output.read_text()
    assert result["probe_to_formal_handoff"]["probe_latest_sha256"] == _sha256(
        probe / "checkpoint_latest.pt"
    )


def test_reference_training_continuity_rejects_epoch_gap(tmp_path):
    probe = tmp_path / "probe"
    formal = tmp_path / "formal"
    probe.mkdir()
    formal.mkdir()
    config = _config(5, 5, None)
    (probe / "run_config.json").write_text(json.dumps(config))
    (formal / "probe_run_config.json").write_text(json.dumps(config))
    (formal / "run_config.json").write_text(json.dumps(_config(300, 20, None)))
    (probe / "epochs.jsonl").write_text(
        "\n".join(json.dumps(_epoch_row(epoch)) for epoch in range(5))
    )
    (formal / "epochs.jsonl").write_text(
        "\n".join(json.dumps(_epoch_row(epoch)) for epoch in range(6, 51))
    )
    torch.save(_checkpoint(4, 1.96), probe / "checkpoint_latest.pt")
    (formal / "resume_epoch4.pt").write_bytes((probe / "checkpoint_latest.pt").read_bytes())
    torch.save(_checkpoint(30, 1.7), formal / "checkpoint_best.pt")
    torch.save(_checkpoint(50, 1.7, 20), formal / "checkpoint_latest.pt")

    try:
        audit_reference_training_continuity(probe, formal, tmp_path / "audit.json")
    except ValueError as error:
        assert "formal epochs" in str(error)
    else:
        raise AssertionError("epoch gap was accepted")


def test_training_ablation_preflight_locks_config_and_split(tmp_path):
    source_config = Path(__file__).parents[1] / "config/hierarchical_epic_8gpu.selected.yaml"
    learned = load_config(source_config).to_dict()
    learned["data"]["split_dir"] = "/public/shared_split"
    learned["output_root"] = "/public/supplement"
    learned["epic"]["force_samples"] = 256
    learned["epic"]["hierarchical_allocation_weight"] = 0.1
    zero = deepcopy(learned)
    zero["epic"]["hierarchical_allocation_weight"] = 0.0
    reference = deepcopy(learned)
    reference["data"]["split_dir"] = "uq_benchmark_runs/shared_split"
    reference["output_root"] = "uq_benchmark_runs/epic_hierarchical"
    reference["epic"]["force_samples"] = 32

    learned_path = tmp_path / "learned.yaml"
    zero_path = tmp_path / "zero.yaml"
    learned_path.write_text(yaml.safe_dump(learned, sort_keys=False))
    zero_path.write_text(yaml.safe_dump(zero, sort_keys=False))
    reference_run = tmp_path / "reference_run_config.json"
    seed43_run = tmp_path / "seed43_run_config.json"
    reference_run.write_text(json.dumps({"config": reference}))
    seed43_run.write_text(
        json.dumps(
            {
                "kind": "epic_hierarchical",
                "seed": 43,
                "lambda_max": 0.001,
                "config": learned,
            }
        )
    )
    reference_split = tmp_path / "reference_split.npz"
    ablation_split = tmp_path / "ablation_split.npz"
    reference_split.write_bytes(b"frozen split")
    ablation_split.write_bytes(reference_split.read_bytes())

    output = audit_training_ablation_preflight(
        reference_run,
        reference_split,
        seed43_run,
        ablation_split,
        learned_path,
        zero_path,
        tmp_path / "preflight.json",
    )
    result = json.loads(output.read_text())
    assert result["training_ablation_protocol_preflight_verified"] is True
    assert result["reference_split_byte_identity_verified"] is True
    assert result["learned_vs_zero_only_intended_difference"] is True
    assert result["expected_run_seeds"] == {
        "hierarchical_seed43": 43,
        "hierarchical_seed44": 44,
        "allocation_weight_0_seed42": 42,
    }
    assert str(tmp_path) not in output.read_text()
