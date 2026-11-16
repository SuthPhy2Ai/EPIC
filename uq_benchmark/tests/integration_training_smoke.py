#!/usr/bin/env python3
"""Run one end-to-end Hierarchical EPIC training epoch on a tiny ASE DB."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uq_benchmark.config import (
    BenchmarkConfig,
    DataConfig,
    EPICConfig,
    ModelConfig,
    TrainingConfig,
)
from uq_benchmark.data import load_dataset, make_data_loader
from uq_benchmark.inference import load_checkpoint_model
from uq_benchmark.training import run_training


os.environ.setdefault("ACCELERATE_USE_CPU", "true")


def _write_database(path: Path) -> None:
    database = connect(path)
    for index in range(6):
        distance = 0.65 + 0.02 * index
        atoms = Atoms(
            "H2",
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, distance]],
            cell=[6.0, 6.0, 6.0],
            pbc=True,
        )
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=-1.0 + 0.01 * index,
            forces=np.zeros((2, 3)),
        )
        database.write(atoms, material_id=f"smoke-{index}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="hierarchical-epic-training-") as directory:
        root = Path(directory)
        database_path = root / "smoke.db"
        split_dir = root / "split"
        split_dir.mkdir()
        split_path = split_dir / "split_indices.npz"
        _write_database(database_path)
        np.savez(
            split_path,
            train=np.arange(4, dtype=np.int64),
            validation=np.arange(4, 6, dtype=np.int64),
            calibration=np.asarray([], dtype=np.int64),
            test=np.asarray([], dtype=np.int64),
        )

        config = BenchmarkConfig(
            data=DataConfig(db_path=str(database_path), split_dir=str(split_dir)),
            model=ModelConfig(
                r_max=2.5,
                num_bessel=3,
                num_polynomial_cutoff=3,
                max_ell=1,
                num_interactions=1,
                num_channels=4,
                correlation=1,
                num_elements=1,
            ),
            training=TrainingConfig(
                num_gpus=1,
                global_batch_size=2,
                per_gpu_batch_size=2,
                gradient_accumulation_steps=1,
                num_workers=0,
                pin_memory=False,
                learning_rate=1.0e-3,
                max_epochs=1,
                early_stopping_patience=1,
                scheduler_patience=1,
                force_weight=1.0,
                force_loss="mse",
                lambda_anneal_epochs=1,
                epic_lambda_max=(1.0e-3,),
                ensemble_seeds=(42,),
            ),
            epic=EPICConfig(
                force_samples=4,
                sample_seed=42,
                hierarchical_allocation_weight=0.1,
                hierarchical_force_error_floor=1.0e-6,
            ),
            output_root=str(root / "runs" / "epic_hierarchical"),
        )
        config.validate()
        run_dir = root / "runs" / "epic_hierarchical" / "smoke"
        best = run_training(
            config,
            split_path,
            run_dir,
            "epic_hierarchical",
            seed=42,
            lambda_max=1.0e-3,
        )
        latest = run_dir / "checkpoint_latest.pt"
        if not best.is_file() or not latest.is_file():
            raise AssertionError("training did not write best and latest checkpoints")

        model, state = load_checkpoint_model(best, device="cpu")
        if state["kind"] != "epic_hierarchical" or state["epic_variant"] != (
            "hierarchical_atomwise_shared_scale"
        ):
            raise AssertionError("checkpoint identity metadata is incorrect")
        loader = make_data_loader(
            load_dataset(config),
            config,
            batch_size=2,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(loader))
        with torch.enable_grad():
            output = model(batch, training=False, compute_force=True)
        required = {
            "energy",
            "forces",
            "energy_nu",
            "energy_alpha",
            "energy_beta",
            "atomic_energy_gamma",
            "atomic_aleatoric_variance",
            "atomic_epistemic_variance",
            "atomic_total_variance",
        }
        if missing := required - set(output):
            raise AssertionError(f"strictly loaded checkpoint is missing outputs: {sorted(missing)}")
        if not all(torch.all(torch.isfinite(output[key])) for key in required):
            raise AssertionError("strictly loaded checkpoint produced non-finite outputs")

    print("end-to-end hierarchical training smoke test passed on cpu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
