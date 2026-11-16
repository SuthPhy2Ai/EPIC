from __future__ import annotations

import gc
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect

from uq_benchmark.hdf5_dataset import (
    _stress_voigt,
    convert_ase_db_to_sharded_hdf5,
    load_sharded_hdf5,
    verify_sharded_hdf5,
)


def make_database(path, n_structures: int = 7) -> None:
    database = connect(path)
    for index in range(n_structures):
        n_atoms = 2 + index % 3
        atoms = Atoms(
            "H" * n_atoms,
            positions=np.arange(n_atoms * 3, dtype=float).reshape(n_atoms, 3) * 0.1,
            cell=np.eye(3) * (5.0 + index),
            pbc=(True, index % 2 == 0, True),
        )
        results = {
            "energy": -float(index + 1),
            "forces": np.arange(n_atoms * 3, dtype=float).reshape(n_atoms, 3) + index,
        }
        if index % 2 == 0:
            results["stress"] = np.arange(6, dtype=float) + index
        atoms.calc = SinglePointCalculator(atoms, **results)
        database.write(
            atoms,
            material_id=f"material-{index // 2}",
            config_type="bulk",
            config_weight=2.0,
            config_energy_weight=3.0,
            config_forces_weight=4.0,
        )


def test_flat_nine_component_stress_is_converted_to_voigt():
    stress = np.arange(9, dtype=float)
    np.testing.assert_allclose(_stress_voigt(stress), [0.0, 4.0, 8.0, 6.0, 4.0, 2.0])


def test_convert_verify_and_load_sharded_hdf5(tmp_path):
    db_path = tmp_path / "source.db"
    output_dir = tmp_path / "hdf5"
    make_database(db_path)

    manifest_path = convert_ase_db_to_sharded_hdf5(
        db_path, output_dir, shard_size=3, progress_every=2
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["complete"] is True
    assert manifest["n_structures"] == 7
    assert manifest["n_shards"] == 3
    assert [(item["start"], item["stop"]) for item in manifest["shards"]] == [
        (0, 3),
        (3, 6),
        (6, 7),
    ]

    verification = verify_sharded_hdf5(db_path, output_dir, samples=20, seed=7)
    assert verification["checked_structures"] == 7
    assert verification["checked_atoms"] == 20

    with h5py.File(output_dir / "part-00000.h5", "r") as handle:
        assert handle.attrs["complete"]
        assert handle["source_db_index"][:].tolist() == [0, 1, 2]
        assert handle["atom_offsets"][:].tolist() == [0, 2, 5, 9]
        assert handle["material_id"].asstr()[:].tolist() == [
            "material-0",
            "material-0",
            "material-1",
        ]
        assert handle["stress_present"][:].tolist() == [True, False, True]

    from mace.tools import AtomicNumberTable

    dataset = load_sharded_hdf5(
        output_dir, r_max=2.0, z_table=AtomicNumberTable([1]), heads=["Default"]
    )
    assert len(dataset) == 7
    first = dataset[0]
    last = dataset[6]
    assert first.positions.shape == (2, 3)
    assert last.positions.shape == (2, 3)
    np.testing.assert_allclose(first.energy.numpy(), -1.0)
    np.testing.assert_allclose(last.energy.numpy(), -7.0)
    np.testing.assert_allclose(first.forces.numpy(), np.arange(6).reshape(2, 3))
    assert float(first.weight) == 2.0
    assert float(first.energy_weight) == 3.0
    assert float(first.forces_weight) == 4.0

    mtimes = {path.name: path.stat().st_mtime_ns for path in output_dir.glob("*.h5")}
    assert convert_ase_db_to_sharded_hdf5(db_path, output_dir, shard_size=3) == manifest_path
    assert {path.name: path.stat().st_mtime_ns for path in output_dir.glob("*.h5")} == mtimes


@pytest.mark.skipif(
    sys.platform == "darwin" and sys.version_info >= (3, 13),
    reason="PyTorch worker resource_tracker can hang during Python 3.13 macOS shutdown",
)
def test_packed_hdf5_supports_worker_processes(tmp_path, monkeypatch):
    import mace

    from uq_benchmark.config import BenchmarkConfig, DataConfig, ModelConfig, TrainingConfig
    from uq_benchmark.data import load_dataset, make_data_loader

    db_path = tmp_path / "source.db"
    output_dir = tmp_path / "hdf5"
    make_database(db_path)
    convert_ase_db_to_sharded_hdf5(db_path, output_dir, shard_size=3)
    monkeypatch.setenv("MACE_SOURCE_ROOT", str(Path(mace.__file__).resolve().parent.parent))
    config = BenchmarkConfig(
        data=DataConfig(db_path=str(db_path), hdf5_path=str(output_dir)),
        model=ModelConfig(num_elements=1),
        training=TrainingConfig(
            global_batch_size=2,
            per_gpu_batch_size=2,
            num_workers=2,
            prefetch_factor=2,
        ),
    )
    config.validate()
    loader = make_data_loader(
        load_dataset(config), config, batch_size=2, shuffle=False, drop_last=False
    )
    iterator = iter(loader)
    batch = next(iterator)
    assert batch.energy.tolist() == [-1.0, -2.0]
    assert batch.positions.shape == (5, 3)
    iterator._shutdown_workers()
    loader._iterator = None
    del batch, iterator, loader
    gc.collect()
