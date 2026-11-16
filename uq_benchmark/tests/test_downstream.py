from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd

from uq_benchmark import downstream


def test_downstream_manifests_use_only_available_two_model_risk(monkeypatch, tmp_path):
    n_structures = 16
    store_path = tmp_path / "two_model.h5"
    with h5py.File(store_path, "w") as handle:
        handle.create_dataset("structures/db_index", data=np.arange(n_structures))
        handle.create_group("methods/mace")
        hierarchical = handle.create_group("methods/hierarchical_epic")
        hierarchical.create_dataset("energy_risk", data=np.arange(n_structures, dtype=float))

    metadata = pd.DataFrame(
        {
            "db_index": np.arange(n_structures),
            "natoms": np.ones(n_structures, dtype=int),
            "chemistry_class": np.tile(
                ["oxide", "halide", "nitride", "other"], 4
            ),
        }
    )
    monkeypatch.setattr(downstream, "_metadata", lambda _path, _indices: metadata.copy())

    _, phonon_path, manifest_path = downstream.create_downstream_manifests(
        store_path, tmp_path / "unused.db", tmp_path / "manifests"
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 2
    assert manifest["risk_consensus_methods"] == ["hierarchical_epic"]
    assert manifest["joint_energy_model_methods"] == []
    assert manifest["marginal_energy_only_methods"] == ["hierarchical_epic"]
    assert manifest["deferred_methods"] == ["ensemble5", "llpr", "graph_nig_epic"]

    phonon = pd.read_csv(phonon_path, keep_default_na=False)
    assert set(phonon["joint_uq_methods"]) == {""}
    assert set(phonon["marginal_energy_methods"]) == {"hierarchical_epic"}
