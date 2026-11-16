from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect

from uq_benchmark.split import create_split, load_split, read_structure_metadata, validate_split


def make_database(path, n_materials: int = 40, rows_per_material: int = 3):
    db = connect(path)
    for material in range(n_materials):
        for frame in range(rows_per_material):
            atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7 + frame * 0.01]])
            atoms.calc = SinglePointCalculator(atoms, energy=float(material + frame), forces=np.zeros((2, 3)))
            db.write(atoms, material_id=f"m-{material:03d}")


def test_material_disjoint_split_is_reproducible(tmp_path):
    db_path = tmp_path / "tiny.db"
    make_database(db_path)
    ratios = {"train": 0.8, "validation": 0.1, "calibration": 0.05, "test": 0.05}
    first = create_split(db_path, tmp_path / "first", ratios, seed=42, hash_database=True)
    second = create_split(db_path, tmp_path / "second", ratios, seed=42, hash_database=True)
    metadata = read_structure_metadata(db_path)
    with np.load(first.indices_path) as left, np.load(second.indices_path) as right:
        split = {name: left[name] for name in left.files}
        validate_split(split, metadata)
        assert all(np.array_equal(left[name], right[name]) for name in left.files)
        material_sets = [set(metadata.material_id.iloc[left[name]]) for name in left.files]
        assert all(not (a & b) for i, a in enumerate(material_sets) for b in material_sets[i + 1 :])
    manifest = json.loads(first.manifest_path.read_text())
    assert manifest["n_structures"] == 120
    assert manifest["n_materials"] == 40
    assert manifest["material_overlap"] == 0
    assert len(manifest["db_sha256"]) == 64


def test_test_access_requires_explicit_permission(tmp_path):
    path = tmp_path / "split.npz"
    np.savez(path, train=[0], validation=[1], calibration=[2], test=[3])
    with pytest.raises(PermissionError):
        load_split(path, "test")
    assert load_split(path, "test", allow_test=True).tolist() == [3]

