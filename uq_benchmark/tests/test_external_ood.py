from __future__ import annotations

import io
import json

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from uq_benchmark.external_ood import (
    ELEMENTA_SELECTION_NAMESPACE,
    _sha256_file,
    audit_external_element_support,
    audit_external_ood_leakage,
    build_external_ood_database,
    geometry_fingerprint,
    invariant_geometry_fingerprint,
    parse_atomic_number_spec,
    select_elementa_records,
)


def _record(material: str, ionic_step: int, symbol: str = "H", natoms: int = 1) -> str:
    atoms = Atoms(
        symbols=[symbol] * natoms,
        positions=np.arange(3 * natoms, dtype=float).reshape(natoms, 3) * 0.01,
        cell=np.eye(3) * 5.0,
        pbc=True,
    )
    atoms.info.update(material=material, structure=f"structure_{material}", ionic_step=ionic_step)
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=float(natoms),
        forces=np.zeros((natoms, 3)),
    )
    destination = io.StringIO()
    write(destination, atoms, format="extxyz")
    return destination.getvalue()


def test_parse_atomic_number_ranges():
    assert parse_atomic_number_spec("1-3,8,10-11") == {1, 2, 3, 8, 10, 11}


def test_elementa_selection_is_grouped_deterministic_and_prediction_free():
    text = "".join(
        [
            _record("material_a", 1),
            _record("material_a", 0),
            _record("material_a", 0),
            _record("material_b", 0),
            _record("material_c", 0),
            _record("unsupported", 0, symbol="He"),
            _record("too_large", 0, natoms=3),
        ]
    )
    arguments = {
        "subset_size": 2,
        "seed": 42,
        "supported_atomic_numbers": {1},
        "max_atoms": 2,
        "namespace": ELEMENTA_SELECTION_NAMESPACE,
    }
    first, counts = select_elementa_records(io.StringIO(text), **arguments)
    second, _ = select_elementa_records(io.StringIO(text), **arguments)
    assert [record.material for record in first] == [record.material for record in second]
    assert len({record.material for record in first}) == 2
    assert {record.material for record in first}.issubset(
        {"material_a", "material_b", "material_c"}
    )
    assert counts["not_ionic_step_zero"] == 1
    assert counts["duplicate_material"] == 1
    assert counts["unsupported_element"] == 1
    assert counts["too_many_atoms"] == 1


def test_external_ood_database_contains_only_frozen_test_structures(tmp_path):
    extxyz = tmp_path / "subset.extxyz"
    records = [_record("material_a", 0), _record("material_b", 0)]
    extxyz.write_text("".join(records))
    selection_manifest = tmp_path / "selection.json"
    selection_manifest.write_text(
        json.dumps(
            {
                "subset_size": 2,
                "output_sha256": _sha256_file(extxyz),
                "selected_materials": ["material_a", "material_b"],
                "selected_source_indices": [10, 20],
                "formal_external_ood_claim_permitted": False,
                "remaining_leakage_checks": ["exact_geometry_overlap_against_MPtrj"],
            }
        )
    )
    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        "data:\n  db_path: old.db\n  hdf5_path: null\n  split_dir: old_split\n  force_test_size: 20\n"
    )
    split_dir = tmp_path / "split"
    output_db = tmp_path / "external.db"
    output_config = tmp_path / "external.yaml"
    manifest_path = build_external_ood_database(
        extxyz,
        selection_manifest,
        output_db,
        split_dir,
        base_config,
        output_config,
    )
    with np.load(split_dir / "split_indices.npz") as split:
        np.testing.assert_array_equal(split["test"], [0, 1])
        assert len(split["calibration"]) == 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_structures"] == 2
    assert manifest["n_materials"] == 2
    assert manifest["selection_uses_model_predictions"] is False
    assert manifest["formal_external_ood_claim_permitted"] is False
    assert str(output_db.resolve()) in output_config.read_text()


def test_external_element_support_audit_uses_train_split_only(tmp_path):
    from ase.db import connect

    reference = tmp_path / "reference.db"
    external = tmp_path / "external.db"
    connect(reference).write(Atoms("H", cell=np.eye(3) * 5, pbc=True))
    connect(reference).write(Atoms("He", cell=np.eye(3) * 5, pbc=True))
    connect(reference).write(Atoms("Li", cell=np.eye(3) * 5, pbc=True))
    connect(external).write(Atoms("HLi", cell=np.eye(3) * 5, pbc=True))
    split = tmp_path / "split.npz"
    np.savez(
        split,
        train=np.asarray([0, 1]),
        validation=np.asarray([], dtype=np.int64),
        calibration=np.asarray([], dtype=np.int64),
        test=np.asarray([2]),
    )
    output = audit_external_element_support(
        external, reference, split, tmp_path / "element_support.json"
    )
    audit = json.loads(output.read_text())
    assert audit["reference_train_elements"] == [1, 2]
    assert audit["external_elements"] == [1, 3]
    assert audit["unsupported_external_elements"] == [3]
    assert audit["external_structures_with_unsupported_elements"] == 1
    assert audit["all_external_elements_seen_in_reference_train"] is False
    assert audit["external_unique_compositions"] == 1
    assert audit["external_maximum_composition_multiplicity"] == 1
    assert audit["external_natoms_distribution"] == {"2": 1}
    assert audit["external_chemical_arity_distribution"] == {"2": 1}
    assert audit["formal_external_ood_claim_permitted"] is False


def test_geometry_fingerprint_is_atom_order_and_periodic_image_invariant():
    numbers = np.array([8, 1, 1])
    positions = np.array([[0.1, 0.2, 0.3], [1.0, 1.2, 1.4], [2.1, 2.2, 2.3]])
    cell = np.eye(3) * 5.0
    reference = geometry_fingerprint(numbers, positions, cell)
    order = np.array([2, 0, 1])
    shifted = positions[order].copy()
    shifted[0] += cell[1]
    assert geometry_fingerprint(numbers[order], shifted, cell) == reference


def test_invariant_geometry_fingerprint_handles_translation_and_cell_basis_change():
    numbers = np.array([8, 1, 1])
    fractional = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.8, 0.2, 0.7]])
    cell = np.array([[4.0, 0.0, 0.0], [0.2, 5.0, 0.0], [0.1, 0.3, 6.0]])
    positions = fractional @ cell
    reference = invariant_geometry_fingerprint(numbers, positions, cell)
    basis_change = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    changed_cell = basis_change @ cell
    order = np.array([2, 0, 1])
    translated = positions + np.array([0.17, 0.31, 0.23]) @ cell
    assert (
        invariant_geometry_fingerprint(
            numbers[order], translated[order], changed_cell
        )
        == reference
    )


def test_external_leakage_audit_finds_strict_geometry_overlap(tmp_path):
    from ase.db import connect

    external = tmp_path / "external.db"
    reference = tmp_path / "reference.db"
    first = Atoms("H", positions=[[0.2, 0.3, 0.4]], cell=np.eye(3) * 5, pbc=True)
    second = Atoms("He", positions=[[1.2, 1.3, 1.4]], cell=np.eye(3) * 6, pbc=True)
    connect(external).write(first, material_id="external-a")
    connect(external).write(second, material_id="external-b")
    connect(reference).write(first, material_id="reference-a")
    connect(reference).write(
        Atoms("H", positions=[[1.0, 1.0, 1.0]], cell=np.eye(3) * 5, pbc=True),
        material_id="reference-b",
    )
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"subset_size": 2}))
    output = audit_external_ood_leakage(
        external, reference, selection, tmp_path / "audit.json"
    )
    audit = json.loads(output.read_text())
    assert audit["strict_geometry_overlap_count"] == 1
    # Both one-atom cells are equivalent after a global periodic translation.
    assert audit["translation_cell_invariant_geometry_overlap_count"] == 2
    assert audit["material_string_overlap_count"] == 0
    assert audit["formal_external_ood_claim_permitted"] is False
