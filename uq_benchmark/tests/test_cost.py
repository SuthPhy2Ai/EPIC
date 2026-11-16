from __future__ import annotations

import json

import torch

from uq_benchmark.cost import training_cost


def _write_training_segment(path, epochs):
    path.mkdir()
    (path / "run_config.json").write_text(json.dumps({"world_size": 8}))
    (path / "epochs.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "epoch": epoch,
                    "epoch_seconds": seconds,
                    "world_size": 8,
                }
            )
            + "\n"
            for epoch, seconds in epochs
        )
    )
    torch.save(
        {"model_state_dict": {"weight": torch.ones(3), "output_mask": torch.ones(2)}},
        path / "checkpoint_best.pt",
    )


def test_training_cost_combines_resumed_segments(tmp_path):
    probe = tmp_path / "probe"
    continuation = tmp_path / "continuation"
    _write_training_segment(probe, [(0, 10.0), (1, 20.0)])
    _write_training_segment(continuation, [(2, 30.0), (3, 40.0)])

    cost = training_cost([probe, continuation], "hierarchical_epic")

    assert cost["training_segments"] == 2
    assert cost["epochs_completed"] == 4
    assert cost["epoch_first"] == 0
    assert cost["epoch_last"] == 3
    assert cost["training_wall_hours"] == 100.0 / 3600.0
    assert cost["training_gpu_hours"] == 800.0 / 3600.0
    assert cost["parameter_count_state_tensors"] == 3
