from __future__ import annotations

import hashlib
import heapq
import io
import json
import re
import sqlite3
import subprocess
import time
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

import numpy as np
import spglib
import yaml
from ase.db import connect
from ase.io import read
from ase.io import iread
from ase.io.extxyz import key_val_str_to_dict


ELEMENTA_SELECTION_NAMESPACE = "elementa-open-material-ionic-step0-v1"
_MATERIAL_PATTERN = re.compile(r"(?:^|\s)material=(?:\"([^\"]*)\"|'([^']*)'|([^\s]+))")
_IONIC_STEP_PATTERN = re.compile(r"(?:^|\s)ionic_step=(?:\"([^\"]*)\"|'([^']*)'|([^\s]+))")


@dataclass(frozen=True)
class ExternalRecord:
    score: int
    source_index: int
    material: str
    structure: str
    natoms: int
    atomic_numbers: tuple[int, ...]
    energy: float
    record: str


def parse_atomic_number_spec(specification: str) -> set[int]:
    result: set[int] = set()
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lower_text, upper_text = item.split("-", 1)
            lower, upper = int(lower_text), int(upper_text)
            if lower > upper:
                raise ValueError(f"invalid decreasing atomic-number range: {item}")
            result.update(range(lower, upper + 1))
        else:
            result.add(int(item))
    if not result or min(result) < 1 or max(result) > 118:
        raise ValueError("supported atomic numbers must be a non-empty subset of 1..118")
    return result


def iter_extxyz_records(stream: TextIO):
    source_index = 0
    while True:
        first = stream.readline()
        while first and not first.strip():
            first = stream.readline()
        if not first:
            return
        try:
            natoms = int(first.strip())
        except ValueError as error:
            raise ValueError(
                f"invalid atom count at external record {source_index}: {first.strip()!r}"
            ) from error
        if natoms <= 0:
            raise ValueError(f"external record {source_index} has no atoms")
        comment = stream.readline()
        if not comment:
            raise EOFError(f"external record {source_index} is missing its comment line")
        atom_lines = [stream.readline() for _ in range(natoms)]
        if any(not line for line in atom_lines):
            raise EOFError(f"external record {source_index} is truncated")
        yield source_index, natoms, comment, first + comment + "".join(atom_lines)
        source_index += 1


def _selection_score(material: str, seed: int, namespace: str) -> int:
    payload = f"{namespace}|seed={seed}|material={material}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _header_value(pattern: re.Pattern[str], comment: str) -> str | None:
    match = pattern.search(comment)
    if match is None:
        return None
    return next((value for value in match.groups() if value is not None), None)


def _validated_record(
    source_index: int,
    natoms: int,
    metadata: dict[str, object],
    raw_record: str,
    supported_atomic_numbers: set[int],
    max_atoms: int,
    seed: int,
    namespace: str,
) -> ExternalRecord:
    atoms = read(io.StringIO(raw_record), format="extxyz", index=0)
    atomic_numbers = tuple(int(value) for value in atoms.numbers)
    if len(atoms) != natoms:
        raise ValueError("parsed atom count differs from the extended-XYZ header")
    if natoms > max_atoms:
        raise ValueError("too_many_atoms")
    if not set(atomic_numbers).issubset(supported_atomic_numbers):
        raise ValueError("unsupported_element")
    if not np.all(atoms.pbc):
        raise ValueError("nonperiodic_structure")
    positions = np.asarray(atoms.positions, dtype=np.float64)
    cell = np.asarray(atoms.cell, dtype=np.float64)
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(cell)):
        raise ValueError("nonfinite_geometry")
    try:
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    except Exception as error:
        raise ValueError("missing_energy_or_forces") from error
    if not np.isfinite(energy) or forces.shape != (natoms, 3) or not np.all(np.isfinite(forces)):
        raise ValueError("nonfinite_energy_or_forces")
    material = str(metadata["material"])
    structure = str(metadata.get("structure", ""))
    return ExternalRecord(
        score=_selection_score(material, seed, namespace),
        source_index=source_index,
        material=material,
        structure=structure,
        natoms=natoms,
        atomic_numbers=atomic_numbers,
        energy=energy,
        record=raw_record,
    )


def select_elementa_records(
    stream: TextIO,
    *,
    subset_size: int,
    seed: int,
    supported_atomic_numbers: set[int],
    max_atoms: int,
    namespace: str = ELEMENTA_SELECTION_NAMESPACE,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> tuple[list[ExternalRecord], dict[str, int]]:
    if subset_size <= 0:
        raise ValueError("external OOD subset size must be positive")
    if max_atoms <= 0:
        raise ValueError("maximum atom count must be positive")
    seen_materials: set[str] = set()
    rejection_counts: Counter[str] = Counter()
    selected_heap: list[tuple[int, int, ExternalRecord]] = []
    scanned = 0
    eligible = 0
    for source_index, natoms, comment, raw_record in iter_extxyz_records(stream):
        scanned += 1
        if progress_callback is not None and scanned % 10_000 == 0:
            progress_callback(scanned, eligible, len(selected_heap))
        material = (_header_value(_MATERIAL_PATTERN, comment) or "").strip()
        if not material:
            rejection_counts["missing_material"] += 1
            continue
        try:
            ionic_step = int(_header_value(_IONIC_STEP_PATTERN, comment) or "-1")
        except (TypeError, ValueError):
            rejection_counts["invalid_ionic_step"] += 1
            continue
        if ionic_step != 0:
            rejection_counts["not_ionic_step_zero"] += 1
            continue
        if material in seen_materials:
            rejection_counts["duplicate_material"] += 1
            continue
        seen_materials.add(material)
        try:
            metadata = key_val_str_to_dict(comment.strip())
        except Exception:
            rejection_counts["invalid_metadata"] += 1
            continue
        metadata["material"] = material
        try:
            candidate = _validated_record(
                source_index,
                natoms,
                metadata,
                raw_record,
                supported_atomic_numbers,
                max_atoms,
                seed,
                namespace,
            )
        except ValueError as error:
            rejection_counts[str(error)] += 1
            continue
        eligible += 1
        item = (-candidate.score, candidate.source_index, candidate)
        if len(selected_heap) < subset_size:
            heapq.heappush(selected_heap, item)
        elif candidate.score < -selected_heap[0][0]:
            heapq.heapreplace(selected_heap, item)
            rejection_counts["hash_rank_outside_subset"] += 1
        else:
            rejection_counts["hash_rank_outside_subset"] += 1
    selected = sorted((item[2] for item in selected_heap), key=lambda item: item.score)
    if len(selected) != subset_size:
        raise RuntimeError(
            f"only {len(selected)} eligible materials were found for a requested subset of {subset_size}"
        )
    if len({record.material for record in selected}) != len(selected):
        raise RuntimeError("external OOD selection contains duplicate material groups")
    counts = {
        "scanned_records": scanned,
        "unique_step_zero_materials": len(seen_materials),
        "eligible_materials": eligible,
    }
    counts.update({key: int(value) for key, value in sorted(rejection_counts.items())})
    return selected, counts


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_elementa_ood(
    archive_path: str | Path,
    archive_member: str,
    output_extxyz: str | Path,
    manifest_path: str | Path,
    *,
    subset_size: int = 10_000,
    seed: int = 42,
    max_atoms: int = 256,
    supported_atomic_numbers: str = "1-83,89-94",
    namespace: str = ELEMENTA_SELECTION_NAMESPACE,
    source_url: str | None = None,
    source_license: str | None = None,
    dft_protocol: str | None = None,
) -> Path:
    archive_path = Path(archive_path)
    output_extxyz = Path(output_extxyz)
    manifest_path = Path(manifest_path)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    for output in (output_extxyz, manifest_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing external OOD artifact: {output}")
    supported = parse_atomic_number_spec(supported_atomic_numbers)
    command = [
        "tar",
        "--use-compress-program=unzstd",
        "-xOf",
        str(archive_path),
        archive_member,
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if process.stdout is None:
        raise RuntimeError("failed to open the ELEMENTA archive stream")
    progress_path = manifest_path.with_suffix(".progress.json")
    started = time.monotonic()

    def write_progress(scanned: int, eligible: int, selected: int) -> None:
        payload = {
            "stage": "select_external_records",
            "scanned_records": scanned,
            "eligible_materials": eligible,
            "selected_heap_size": selected,
            "requested_subset_size": subset_size,
            "elapsed_seconds": round(time.monotonic() - started, 1),
        }
        temporary = progress_path.with_suffix(progress_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
        temporary.replace(progress_path)

    try:
        try:
            selected, counts = select_elementa_records(
                process.stdout,
                subset_size=subset_size,
                seed=seed,
                supported_atomic_numbers=supported,
                max_atoms=max_atoms,
                namespace=namespace,
                progress_callback=write_progress,
            )
        finally:
            process.stdout.close()
    except BaseException:
        process.terminate()
        process.wait()
        raise
    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"archive extraction failed with code {return_code}: {stderr.strip()}")

    write_progress(counts["scanned_records"], counts["eligible_materials"], len(selected))
    output_extxyz.parent.mkdir(parents=True, exist_ok=True)
    with output_extxyz.open("w", encoding="utf-8") as destination:
        for record in selected:
            destination.write(record.record)
    output_sha256 = _sha256_file(output_extxyz)
    write_progress(counts["scanned_records"], counts["eligible_materials"], len(selected))
    progress = json.loads(progress_path.read_text())
    progress["stage"] = "hash_archive"
    progress_path.write_text(json.dumps(progress, sort_keys=True) + "\n")
    archive_sha256 = _sha256_file(archive_path)
    manifest = {
        "schema_version": 1,
        "dataset": "ELEMENTA_open",
        "archive": str(archive_path.resolve()),
        "archive_member": archive_member,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "output_extxyz": str(output_extxyz.resolve()),
        "output_sha256": output_sha256,
        "selection_namespace": namespace,
        "selection_seed": seed,
        "selection_uses_model_predictions": False,
        "selection_uses_targets_for_ranking": False,
        "selection_rule": (
            "periodic finite ionic_step=0 structures; at most one frame per material; "
            "supported elements and natoms<=max_atoms; smallest SHA-256 material ranks"
        ),
        "subset_size": subset_size,
        "max_atoms": max_atoms,
        "supported_atomic_numbers": sorted(supported),
        "selected_source_indices": [record.source_index for record in selected],
        "selected_materials": [record.material for record in selected],
        "selected_structures": [record.structure for record in selected],
        "selected_natoms": [record.natoms for record in selected],
        "counts": counts,
        "provenance": {
            "source_url": source_url,
            "license": source_license,
            "dft_protocol": dft_protocol,
            "verified": False,
            "verification_status": "pending_manual_verification",
        },
        "formal_external_ood_claim_permitted": False,
        "remaining_leakage_checks": [
            "exact_geometry_overlap_against_MPtrj",
            "cross_dataset_material_identity_overlap",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    progress["stage"] = "complete"
    progress["elapsed_seconds"] = round(time.monotonic() - started, 1)
    progress_path.write_text(json.dumps(progress, sort_keys=True) + "\n")
    return manifest_path


def build_external_ood_database(
    extxyz_path: str | Path,
    selection_manifest_path: str | Path,
    output_database: str | Path,
    split_directory: str | Path,
    base_config_path: str | Path,
    output_config_path: str | Path,
) -> Path:
    extxyz_path = Path(extxyz_path)
    selection_manifest_path = Path(selection_manifest_path)
    output_database = Path(output_database)
    split_directory = Path(split_directory)
    base_config_path = Path(base_config_path)
    output_config_path = Path(output_config_path)
    for source in (extxyz_path, selection_manifest_path, base_config_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    for output in (
        output_database,
        split_directory / "split_indices.npz",
        split_directory / "external_ood_database_manifest.json",
        output_config_path,
    ):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing external OOD artifact: {output}")

    selection = json.loads(selection_manifest_path.read_text())
    if _sha256_file(extxyz_path) != selection["output_sha256"]:
        raise ValueError("selected external extxyz hash differs from its manifest")
    expected_materials = [str(value) for value in selection["selected_materials"]]
    expected_indices = [int(value) for value in selection["selected_source_indices"]]
    if len(expected_materials) != int(selection["subset_size"]):
        raise ValueError("selection manifest material count differs from subset size")

    output_database.parent.mkdir(parents=True, exist_ok=True)
    connection = connect(output_database)
    observed_materials = []
    for output_index, atoms in enumerate(iread(extxyz_path, format="extxyz")):
        if output_index >= len(expected_materials):
            raise ValueError("selected extxyz contains more structures than its manifest")
        material = str(atoms.info.get("material", ""))
        if material != expected_materials[output_index]:
            raise ValueError(
                f"selected extxyz material order differs at output index {output_index}"
            )
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
            raise ValueError(f"non-finite labels at external output index {output_index}")
        connection.write(
            atoms,
            material_id=material,
            external_dataset="ELEMENTA_open",
            external_source_index=expected_indices[output_index],
        )
        observed_materials.append(material)
    if observed_materials != expected_materials:
        raise ValueError("selected extxyz contains fewer structures than its manifest")

    n_structures = len(observed_materials)
    split_directory.mkdir(parents=True, exist_ok=True)
    split_path = split_directory / "split_indices.npz"
    np.savez_compressed(
        split_path,
        train=np.asarray([], dtype=np.int64),
        validation=np.asarray([], dtype=np.int64),
        calibration=np.asarray([], dtype=np.int64),
        test=np.arange(n_structures, dtype=np.int64),
    )
    config = yaml.safe_load(base_config_path.read_text()) or {}
    config.setdefault("data", {})
    config["data"].update(
        {
            "db_path": str(output_database.resolve()),
            "hdf5_path": None,
            "split_dir": str(split_directory.resolve()),
            "force_test_size": n_structures,
        }
    )
    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    database_manifest = {
        "schema_version": 1,
        "dataset": "ELEMENTA_open",
        "selection_manifest": str(selection_manifest_path.resolve()),
        "selection_manifest_sha256": _sha256_file(selection_manifest_path),
        "database": str(output_database.resolve()),
        "database_sha256": _sha256_file(output_database),
        "database_size_bytes": output_database.stat().st_size,
        "n_structures": n_structures,
        "n_materials": len(set(observed_materials)),
        "split_indices": str(split_path.resolve()),
        "split_indices_sha256": _sha256_file(split_path),
        "test_contains_all_selected_structures": True,
        "calibration_contains_external_structures": False,
        "selection_uses_model_predictions": False,
        "formal_external_ood_claim_permitted": bool(
            selection.get("formal_external_ood_claim_permitted", False)
        ),
        "remaining_leakage_checks": selection.get("remaining_leakage_checks", []),
    }
    manifest_path = split_directory / "external_ood_database_manifest.json"
    manifest_path.write_text(json.dumps(database_manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def geometry_fingerprint(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray,
    *,
    decimals: int = 6,
) -> str:
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int16)
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    if len(atomic_numbers) != len(positions) or len(atomic_numbers) == 0:
        raise ValueError("geometry fingerprint requires matching non-empty atom arrays")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(cell)):
        raise ValueError("geometry fingerprint inputs must be finite")
    try:
        fractional = np.linalg.solve(cell.T, positions.T).T
    except np.linalg.LinAlgError as error:
        raise ValueError("geometry fingerprint requires a nonsingular periodic cell") from error
    fractional = np.mod(fractional, 1.0)
    tolerance = 10.0**-decimals
    fractional[np.isclose(fractional, 1.0, atol=tolerance)] = 0.0
    rounded_fractional = np.round(fractional, decimals=decimals)
    order = np.lexsort(
        (
            rounded_fractional[:, 2],
            rounded_fractional[:, 1],
            rounded_fractional[:, 0],
            atomic_numbers,
        )
    )
    cell_metric = np.round(cell @ cell.T, decimals=decimals)
    digest = hashlib.sha256()
    digest.update(np.asarray(atomic_numbers[order], dtype="<i2").tobytes())
    digest.update(np.asarray(rounded_fractional[order], dtype="<f8").tobytes())
    digest.update(np.asarray(cell_metric, dtype="<f8").tobytes())
    return digest.hexdigest()


def invariant_geometry_fingerprint(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray,
    *,
    decimals: int = 6,
) -> str:
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int16)
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    if len(atomic_numbers) != len(positions) or len(atomic_numbers) == 0:
        raise ValueError("geometry fingerprint requires matching non-empty atom arrays")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(cell)):
        raise ValueError("geometry fingerprint inputs must be finite")
    reduced_cell = spglib.niggli_reduce(cell, eps=10.0 ** (-(decimals + 1)))
    if reduced_cell is None:
        raise ValueError("Niggli reduction failed for periodic cell")
    try:
        fractional = np.linalg.solve(reduced_cell.T, positions.T).T
    except np.linalg.LinAlgError as error:
        raise ValueError("geometry fingerprint requires a nonsingular periodic cell") from error
    fractional = np.mod(fractional, 1.0)
    tolerance = 10.0**-decimals
    counts = Counter(atomic_numbers.tolist())
    anchor_species = min(counts, key=lambda number: (counts[number], number))
    anchor_indices = np.flatnonzero(atomic_numbers == anchor_species)
    cell_metric = np.round(reduced_cell @ reduced_cell.T, decimals=decimals)
    candidates = []
    for anchor in anchor_indices:
        shifted = np.mod(fractional - fractional[anchor], 1.0)
        shifted[np.isclose(shifted, 1.0, atol=tolerance)] = 0.0
        shifted = np.round(shifted, decimals=decimals)
        order = np.lexsort(
            (shifted[:, 2], shifted[:, 1], shifted[:, 0], atomic_numbers)
        )
        candidates.append(
            np.asarray(atomic_numbers[order], dtype="<i2").tobytes()
            + np.asarray(shifted[order], dtype="<f8").tobytes()
        )
    digest = hashlib.sha256()
    digest.update(min(candidates))
    digest.update(np.asarray(cell_metric, dtype="<f8").tobytes())
    return digest.hexdigest()


def _composition_key(atomic_numbers: np.ndarray) -> tuple[tuple[int, int], ...]:
    values, counts = np.unique(np.asarray(atomic_numbers, dtype=np.int16), return_counts=True)
    return tuple(zip(values.astype(int).tolist(), counts.astype(int).tolist(), strict=True))


def _database_geometry_rows(database_path: Path, batch_size: int = 10_000):
    connection = sqlite3.connect(f"file:{database_path.resolve()}?mode=ro", uri=True)
    try:
        cursor = connection.execute(
            "SELECT id - 1, natoms, numbers, positions, cell FROM systems ORDER BY id"
        )
        while rows := cursor.fetchmany(batch_size):
            for db_index, natoms, numbers_blob, positions_blob, cell_blob in rows:
                numbers = np.frombuffer(numbers_blob, dtype="<i4", count=natoms)
                positions = np.frombuffer(positions_blob, dtype="<f8", count=3 * natoms).reshape(
                    natoms, 3
                )
                cell = np.frombuffer(cell_blob, dtype="<f8", count=9).reshape(3, 3)
                yield int(db_index), numbers, positions, cell
    finally:
        connection.close()


def _database_material_ids(database_path: Path) -> set[str]:
    connection = sqlite3.connect(f"file:{database_path.resolve()}?mode=ro", uri=True)
    try:
        return {
            str(value)
            for (value,) in connection.execute(
                "SELECT value FROM text_key_values WHERE key='material_id'"
            )
        }
    finally:
        connection.close()


def audit_external_ood_leakage(
    external_database: str | Path,
    reference_database: str | Path,
    selection_manifest_path: str | Path,
    output_path: str | Path,
    *,
    decimals: int = 6,
) -> Path:
    external_database = Path(external_database)
    reference_database = Path(reference_database)
    selection_manifest_path = Path(selection_manifest_path)
    output_path = Path(output_path)
    for source in (external_database, reference_database, selection_manifest_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite external OOD leakage audit: {output_path}")
    selection = json.loads(selection_manifest_path.read_text())
    expected_size = int(selection["subset_size"])
    candidate_hashes: dict[str, list[int]] = defaultdict(list)
    candidate_invariant_hashes: dict[str, list[int]] = defaultdict(list)
    candidate_compositions: set[tuple[tuple[int, int], ...]] = set()
    external_count = 0
    for db_index, numbers, positions, cell in _database_geometry_rows(external_database):
        candidate_hashes[
            geometry_fingerprint(numbers, positions, cell, decimals=decimals)
        ].append(db_index)
        candidate_invariant_hashes[
            invariant_geometry_fingerprint(numbers, positions, cell, decimals=decimals)
        ].append(db_index)
        candidate_compositions.add(_composition_key(numbers))
        external_count += 1
    if external_count != expected_size:
        raise ValueError("external database size differs from selection manifest")

    progress_path = output_path.with_suffix(".progress.json")
    started = time.monotonic()
    reference_count = 0
    composition_candidates = 0
    overlaps = []
    invariant_overlaps = []
    for db_index, numbers, positions, cell in _database_geometry_rows(reference_database):
        reference_count += 1
        if _composition_key(numbers) in candidate_compositions:
            composition_candidates += 1
            fingerprint = geometry_fingerprint(numbers, positions, cell, decimals=decimals)
            for external_index in candidate_hashes.get(fingerprint, []):
                overlaps.append(
                    {
                        "external_db_index": external_index,
                        "reference_db_index": db_index,
                        "fingerprint": fingerprint,
                    }
                )
            invariant_fingerprint = invariant_geometry_fingerprint(
                numbers, positions, cell, decimals=decimals
            )
            for external_index in candidate_invariant_hashes.get(
                invariant_fingerprint, []
            ):
                invariant_overlaps.append(
                    {
                        "external_db_index": external_index,
                        "reference_db_index": db_index,
                        "fingerprint": invariant_fingerprint,
                    }
                )
        if reference_count % 10_000 == 0:
            payload = {
                "stage": "scan_reference_database",
                "reference_structures_scanned": reference_count,
                "composition_candidates": composition_candidates,
                "strict_geometry_overlaps": len(overlaps),
                "translation_cell_invariant_geometry_overlaps": len(
                    invariant_overlaps
                ),
                "elapsed_seconds": round(time.monotonic() - started, 1),
            }
            temporary = progress_path.with_suffix(progress_path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
            temporary.replace(progress_path)

    external_materials = _database_material_ids(external_database)
    reference_materials = _database_material_ids(reference_database)
    material_string_overlap = sorted(external_materials & reference_materials)
    audit = {
        "schema_version": 1,
        "external_database": str(external_database.resolve()),
        "external_database_sha256": _sha256_file(external_database),
        "reference_database": str(reference_database.resolve()),
        "reference_database_sha256": _sha256_file(reference_database),
        "selection_manifest": str(selection_manifest_path.resolve()),
        "selection_manifest_sha256": _sha256_file(selection_manifest_path),
        "external_structures": external_count,
        "reference_structures": reference_count,
        "candidate_composition_reference_structures": composition_candidates,
        "geometry_fingerprint": {
            "kind": "periodic_wrapped_fractional_sorted_cell_metric_sha256",
            "decimals": decimals,
            "atom_order_invariant": True,
            "periodic_image_invariant": True,
            "rigid_rotation_invariant": True,
            "global_fractional_translation_invariant": False,
            "cell_basis_change_invariant": False,
        },
        "strict_geometry_overlap_count": len(overlaps),
        "strict_geometry_overlaps": overlaps,
        "invariant_geometry_fingerprint": {
            "kind": "niggli_reduced_translation_canonical_fractional_sha256",
            "decimals": decimals,
            "atom_order_invariant": True,
            "periodic_image_invariant": True,
            "rigid_rotation_invariant": True,
            "global_fractional_translation_invariant": True,
            "cell_basis_change_invariant": True,
            "primitive_supercell_invariant": False,
        },
        "translation_cell_invariant_geometry_overlap_count": len(
            invariant_overlaps
        ),
        "translation_cell_invariant_geometry_overlaps": invariant_overlaps,
        "external_material_strings": len(external_materials),
        "reference_material_strings": len(reference_materials),
        "material_string_overlap_count": len(material_string_overlap),
        "material_string_overlap": material_string_overlap,
        "cross_dataset_material_identity_mapping_available": False,
        "selection_uses_model_predictions": False,
        "formal_external_ood_claim_permitted": False,
        "remaining_gates": [
            "provenance_license_and_dft_protocol_verification",
            "cross_dataset_material_identity_mapping",
            "primitive_supercell_and_cross_dataset_material_identity_mapping",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    progress_path.write_text(
        json.dumps(
            {
                "stage": "complete",
                "reference_structures_scanned": reference_count,
                "composition_candidates": composition_candidates,
                "strict_geometry_overlaps": len(overlaps),
                "elapsed_seconds": round(time.monotonic() - started, 1),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return output_path


def audit_external_element_support(
    external_database: str | Path,
    reference_database: str | Path,
    split_indices: str | Path,
    output_path: str | Path,
) -> Path:
    external_database = Path(external_database)
    reference_database = Path(reference_database)
    split_indices = Path(split_indices)
    output_path = Path(output_path)
    for path in (external_database, reference_database, split_indices):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(split_indices) as split:
        if "train" not in split:
            raise ValueError("split indices do not contain a train split")
        train_indices = np.asarray(split["train"], dtype=np.int64)
    if len(train_indices) == 0 or len(np.unique(train_indices)) != len(train_indices):
        raise ValueError("train indices must be non-empty and unique")

    reference = sqlite3.connect(
        f"file:{reference_database.resolve()}?mode=ro", uri=True
    )
    try:
        n_reference = int(reference.execute("SELECT COUNT(*) FROM systems").fetchone()[0])
        if train_indices.min() < 0 or train_indices.max() >= n_reference:
            raise ValueError("train indices fall outside the reference database")
        train_mask = np.zeros(n_reference, dtype=bool)
        train_mask[train_indices] = True
        train_structure_counts: Counter[int] = Counter()
        train_atom_counts: Counter[int] = Counter()
        cursor = reference.execute("SELECT id,Z,n FROM species")
        while rows := cursor.fetchmany(200_000):
            values = np.asarray(rows, dtype=np.int64)
            values = values[train_mask[values[:, 0] - 1]]
            if not len(values):
                continue
            for atomic_number, count in zip(
                *np.unique(values[:, 1], return_counts=True), strict=True
            ):
                train_structure_counts[int(atomic_number)] += int(count)
            for atomic_number, atom_count in values[:, 1:3]:
                train_atom_counts[int(atomic_number)] += int(atom_count)
    finally:
        reference.close()

    external = sqlite3.connect(
        f"file:{external_database.resolve()}?mode=ro", uri=True
    )
    try:
        n_external = int(external.execute("SELECT COUNT(*) FROM systems").fetchone()[0])
        external_structure_counts: Counter[int] = Counter()
        external_atom_counts: Counter[int] = Counter()
        external_elements_by_structure: defaultdict[int, set[int]] = defaultdict(set)
        external_composition_by_structure: defaultdict[int, dict[int, int]] = defaultdict(dict)
        for row_id, atomic_number, atom_count in external.execute(
            "SELECT id,Z,n FROM species"
        ):
            atomic_number = int(atomic_number)
            external_structure_counts[atomic_number] += 1
            external_atom_counts[atomic_number] += int(atom_count)
            external_elements_by_structure[int(row_id)].add(atomic_number)
            external_composition_by_structure[int(row_id)][atomic_number] = int(atom_count)
    finally:
        external.close()
    if len(external_elements_by_structure) != n_external:
        raise ValueError("external database contains structures without species metadata")

    train_elements = set(train_structure_counts)
    external_elements = set(external_structure_counts)
    unsupported = sorted(external_elements - train_elements)
    unsupported_structures = sum(
        bool(elements.intersection(unsupported))
        for elements in external_elements_by_structure.values()
    )
    external_training_counts = {
        atomic_number: train_structure_counts[atomic_number]
        for atomic_number in sorted(external_elements)
    }
    composition_counts = Counter(
        tuple(sorted(composition.items()))
        for composition in external_composition_by_structure.values()
    )
    natoms_counts = Counter(
        sum(composition.values())
        for composition in external_composition_by_structure.values()
    )
    chemical_arity_counts = Counter(
        len(composition) for composition in external_composition_by_structure.values()
    )
    audit = {
        "schema_version": 1,
        "audit": "external_element_support_against_reference_train_split",
        "reference_database_sha256": _sha256_file(reference_database),
        "split_indices_sha256": _sha256_file(split_indices),
        "external_database_sha256": _sha256_file(external_database),
        "reference_structures": n_reference,
        "reference_train_structures": len(train_indices),
        "external_structures": n_external,
        "reference_train_elements": sorted(train_elements),
        "external_elements": sorted(external_elements),
        "unsupported_external_elements": unsupported,
        "external_structures_with_unsupported_elements": unsupported_structures,
        "all_external_elements_seen_in_reference_train": not unsupported,
        "minimum_reference_train_structure_count_for_external_elements": min(
            external_training_counts.values()
        ),
        "reference_train_structure_counts_by_element": external_training_counts,
        "reference_train_atom_counts_by_element": {
            atomic_number: train_atom_counts[atomic_number]
            for atomic_number in sorted(external_elements)
        },
        "external_structure_counts_by_element": dict(
            sorted(external_structure_counts.items())
        ),
        "external_atom_counts_by_element": dict(sorted(external_atom_counts.items())),
        "external_unique_compositions": len(composition_counts),
        "external_maximum_composition_multiplicity": max(composition_counts.values()),
        "external_natoms_distribution": dict(sorted(natoms_counts.items())),
        "external_chemical_arity_distribution": dict(
            sorted(chemical_arity_counts.items())
        ),
        "formal_external_ood_claim_permitted": False,
        "remaining_gates": [
            "provenance_license_and_dft_protocol_verification",
            "cross_dataset_material_identity_mapping",
            "primitive_supercell_and_cross_dataset_material_identity_mapping",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return output_path
