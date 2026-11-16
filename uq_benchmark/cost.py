from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def training_cost(run_dir: str | Path | list[str | Path], method: str) -> dict:
    run_dirs = [Path(path) for path in run_dir] if isinstance(run_dir, list) else [Path(run_dir)]
    configs = [json.loads((path / "run_config.json").read_text()) for path in run_dirs]
    epoch_segments = [
        [json.loads(line) for line in (path / "epochs.jsonl").read_text().splitlines() if line]
        for path in run_dirs
    ]
    epochs = [row for segment in epoch_segments for row in segment]
    checkpoint_path = run_dirs[-1] / "checkpoint_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    trainable_parameters = sum(value.numel() for key, value in state.items() if not key.endswith("output_mask"))
    wall_seconds = float(sum(row.get("epoch_seconds", 0.0) for row in epochs))
    world_sizes = {
        int(config.get("world_size", 1))
        for config in configs
    } | {int(row.get("world_size", 1)) for row in epochs}
    if len(world_sizes) != 1:
        raise ValueError(f"training segments for {method} use different world sizes")
    world_size = world_sizes.pop()
    return {
        "method": method,
        "training_wall_hours": wall_seconds / 3600.0,
        "training_gpu_hours": wall_seconds * world_size / 3600.0,
        "world_size": world_size,
        "epochs_completed": len(epochs),
        "epoch_first": min(int(row["epoch"]) for row in epochs),
        "epoch_last": max(int(row["epoch"]) for row in epochs),
        "training_segments": len(run_dirs),
        "parameter_count_state_tensors": int(trainable_parameters),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "run_dirs": json.dumps([str(path.resolve()) for path in run_dirs]),
    }


def inference_cost(prediction_path: str | Path, method: str) -> dict:
    with np.load(prediction_path) as data:
        n = len(data["db_index"])
        seconds = float(data["inference_seconds"])
        peak_memory = int(data.get("peak_memory_bytes", 0))
    return {
        "method": method,
        "n_structures": n,
        "inference_seconds": seconds,
        "structures_per_second": n / seconds,
        "milliseconds_per_structure": 1000.0 * seconds / n,
        "peak_memory_bytes": peak_memory,
        "prediction_path": str(Path(prediction_path).resolve()),
    }


def write_cost_tables(
    training_runs: dict[str, str | Path | list[str | Path]],
    inference_runs: dict[str, str | Path],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_path = output_dir / "training_cost.csv"
    inference_path = output_dir / "inference_cost.csv"
    pd.DataFrame([training_cost(path, method) for method, path in training_runs.items()]).to_csv(training_path, index=False)
    pd.DataFrame([inference_cost(path, method) for method, path in inference_runs.items()]).to_csv(inference_path, index=False)
    return training_path, inference_path
