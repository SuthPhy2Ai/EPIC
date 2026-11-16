from __future__ import annotations

import bisect
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from ase.db import connect
from ase.stress import full_3x3_to_voigt_6_stress
from torch.utils.data import ConcatDataset, Dataset


SCHEMA_NAME = "uq_benchmark_packed_mace"
SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _source_fingerprint(path: Path, n_structures: int) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "n_structures": n_structures,
    }


def _same_source(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("path", "size_bytes", "mtime_ns", "n_structures")
    return all(left.get(key) == right.get(key) for key in keys)


def _database_layout(path: Path) -> tuple[np.ndarray, np.ndarray]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT id, natoms FROM systems ORDER BY id").fetchall()
    finally:
        connection.close()
    ids = np.fromiter((row[0] for row in rows), dtype=np.int64, count=len(rows))
    natoms = np.fromiter((row[1] for row in rows), dtype=np.int64, count=len(rows))
    if not np.array_equal(ids, np.arange(1, len(ids) + 1, dtype=np.int64)):
        raise ValueError("ASE DB row IDs must be contiguous and start at 1")
    if np.any(natoms <= 0):
        raise ValueError("every structure must contain at least one atom")
    return ids, natoms


def _row_value(row, atoms, name: str, ref_name: str | None = None):
    candidates = [name]
    if ref_name is not None:
        candidates.append(ref_name)
    for key in candidates:
        value = row.get(key)
        if value is not None:
            return value
        if key in atoms.info:
            return atoms.info[key]
        if key in atoms.arrays:
            return atoms.arrays[key]
        data = row.get("data")
        if data is not None and key in data:
            return data[key]
    if atoms.calc is not None:
        return atoms.calc.results.get(name)
    return None


def _scalar_row_value(row, name: str, default):
    value = row.get(name)
    if value is None:
        data = row.get("data")
        value = data.get(name) if data is not None else None
    return default if value is None else value


def _stress_voigt(value) -> np.ndarray:
    stress = np.asarray(value, dtype=np.float64)
    if stress.shape == (9,):
        stress = stress.reshape(3, 3)
    if stress.shape == (3, 3):
        return np.asarray(full_3x3_to_voigt_6_stress(stress), dtype=np.float64)
    if stress.shape == (6,):
        return stress
    raise ValueError(f"stress must have shape (6,), (9,), or (3, 3), got {stress.shape}")


def _create_shard(
    db,
    output_path: Path,
    source: dict[str, Any],
    natoms: np.ndarray,
    start: int,
    stop: int,
    progress_every: int,
) -> dict[str, Any]:
    shard_natoms = natoms[start:stop]
    offsets = np.empty(len(shard_natoms) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(shard_natoms, out=offsets[1:])
    n_atoms = int(offsets[-1])
    n_structures = stop - start
    atom_chunk = (min(max(n_atoms, 1), 8192),)
    vector_chunk = (atom_chunk[0], 3)
    structure_chunk = (min(max(n_structures, 1), 1024),)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    with h5py.File(temporary, "w") as handle:
        handle.attrs.update(
            schema_name=SCHEMA_NAME,
            schema_version=SCHEMA_VERSION,
            complete=False,
            source_path=source["path"],
            source_size_bytes=source["size_bytes"],
            source_mtime_ns=source["mtime_ns"],
            source_n_structures=source["n_structures"],
            source_start_index=start,
            source_stop_index=stop,
        )
        handle.create_dataset("atom_offsets", data=offsets)
        handle.create_dataset(
            "source_db_index", data=np.arange(start, stop, dtype=np.int64), chunks=structure_chunk
        )
        atomic_numbers = handle.create_dataset(
            "atomic_numbers", shape=(n_atoms,), dtype=np.int16, chunks=atom_chunk
        )
        positions = handle.create_dataset(
            "positions", shape=(n_atoms, 3), dtype=np.float64, chunks=vector_chunk
        )
        forces = handle.create_dataset(
            "forces", shape=(n_atoms, 3), dtype=np.float64, chunks=vector_chunk
        )
        cells = handle.create_dataset(
            "cell", shape=(n_structures, 3, 3), dtype=np.float64, chunks=(structure_chunk[0], 3, 3)
        )
        pbc = handle.create_dataset(
            "pbc", shape=(n_structures, 3), dtype=np.bool_, chunks=(structure_chunk[0], 3)
        )
        energy = handle.create_dataset(
            "energy", shape=(n_structures,), dtype=np.float64, chunks=structure_chunk
        )
        stress = handle.create_dataset(
            "stress", shape=(n_structures, 6), dtype=np.float64, chunks=(structure_chunk[0], 6)
        )
        stress_present = handle.create_dataset(
            "stress_present", shape=(n_structures,), dtype=np.bool_, chunks=structure_chunk
        )
        config_weight = handle.create_dataset(
            "config_weight", shape=(n_structures,), dtype=np.float64, chunks=structure_chunk
        )
        energy_weight = handle.create_dataset(
            "energy_weight", shape=(n_structures,), dtype=np.float64, chunks=structure_chunk
        )
        forces_weight = handle.create_dataset(
            "forces_weight", shape=(n_structures,), dtype=np.float64, chunks=structure_chunk
        )
        stress_weight = handle.create_dataset(
            "stress_weight", shape=(n_structures,), dtype=np.float64, chunks=structure_chunk
        )
        material_id = handle.create_dataset("material_id", shape=(n_structures,), dtype=string_dtype)
        config_type = handle.create_dataset("config_type", shape=(n_structures,), dtype=string_dtype)
        head = handle.create_dataset("head", shape=(n_structures,), dtype=string_dtype)

        for local_index, db_index in enumerate(range(start, stop)):
            row = db.get(id=db_index + 1)
            atoms = row.toatoms()
            first, last = int(offsets[local_index]), int(offsets[local_index + 1])
            row_energy = _row_value(row, atoms, "energy", "REF_energy")
            row_forces = _row_value(row, atoms, "forces", "REF_forces")
            if row_energy is None or row_forces is None:
                missing = [
                    label
                    for label, value in (("energy", row_energy), ("forces", row_forces))
                    if value is None
                ]
                raise ValueError(f"DB row {db_index + 1} is missing required labels: {missing}")
            row_forces = np.asarray(row_forces, dtype=np.float64)
            if row_forces.shape != (len(atoms), 3):
                raise ValueError(
                    f"DB row {db_index + 1} forces have shape {row_forces.shape}, "
                    f"expected {(len(atoms), 3)}"
                )
            row_stress = _row_value(row, atoms, "stress", "REF_stress")

            atomic_numbers[first:last] = atoms.numbers
            positions[first:last] = atoms.positions
            forces[first:last] = row_forces
            cells[local_index] = np.asarray(atoms.cell, dtype=np.float64)
            pbc[local_index] = atoms.pbc
            energy[local_index] = float(row_energy)
            if row_stress is None:
                stress[local_index] = 0.0
                stress_present[local_index] = False
            else:
                stress[local_index] = _stress_voigt(row_stress)
                stress_present[local_index] = True
            config_weight[local_index] = float(_scalar_row_value(row, "config_weight", 1.0))
            energy_weight[local_index] = float(_scalar_row_value(row, "config_energy_weight", 1.0))
            forces_weight[local_index] = float(_scalar_row_value(row, "config_forces_weight", 1.0))
            stress_weight[local_index] = float(
                _scalar_row_value(row, "config_stress_weight", 1.0 if row_stress is not None else 0.0)
            )
            material_id[local_index] = str(_scalar_row_value(row, "material_id", ""))
            config_type[local_index] = str(_scalar_row_value(row, "config_type", "Default"))
            head[local_index] = str(_scalar_row_value(row, "head", "Default"))

            completed = local_index + 1
            if completed == n_structures or completed % progress_every == 0:
                print(
                    f"[hdf5] {output_path.name}: {completed:,}/{n_structures:,} structures",
                    flush=True,
                )
        handle.attrs["complete"] = True
        handle.flush()
    os.replace(temporary, output_path)
    return {
        "file": output_path.name,
        "start": start,
        "stop": stop,
        "n_structures": n_structures,
        "n_atoms": n_atoms,
        "size_bytes": output_path.stat().st_size,
    }


def _valid_existing_shard(path: Path, source: dict[str, Any], start: int, stop: int) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as handle:
            return bool(handle.attrs.get("complete", False)) and all(
                (
                    handle.attrs.get("schema_name") == SCHEMA_NAME,
                    int(handle.attrs.get("schema_version", -1)) == SCHEMA_VERSION,
                    int(handle.attrs.get("source_size_bytes", -1)) == source["size_bytes"],
                    int(handle.attrs.get("source_mtime_ns", -1)) == source["mtime_ns"],
                    int(handle.attrs.get("source_start_index", -1)) == start,
                    int(handle.attrs.get("source_stop_index", -1)) == stop,
                )
            )
    except (OSError, KeyError, ValueError):
        return False


def _shard_record(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        return {
            "file": path.name,
            "start": int(handle.attrs["source_start_index"]),
            "stop": int(handle.attrs["source_stop_index"]),
            "n_structures": int(handle.attrs["source_stop_index"])
            - int(handle.attrs["source_start_index"]),
            "n_atoms": int(handle["atom_offsets"][-1]),
            "size_bytes": path.stat().st_size,
        }


def convert_ase_db_to_sharded_hdf5(
    db_path: str | Path,
    output_dir: str | Path,
    *,
    shard_size: int = 25_000,
    overwrite: bool = False,
    progress_every: int = 1_000,
    hash_source: bool = False,
) -> Path:
    if shard_size <= 0 or progress_every <= 0:
        raise ValueError("shard_size and progress_every must be positive")
    db_path = Path(db_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    _, natoms = _database_layout(db_path)
    source = _source_fingerprint(db_path, len(natoms))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME

    if overwrite:
        for path in output_dir.glob("part-*.h5"):
            path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()
    elif manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if not _same_source(existing.get("source", {}), source) or existing.get("shard_size") != shard_size:
            raise ValueError("existing HDF5 manifest does not match source DB or shard size")
        if existing.get("complete"):
            valid = all(
                _valid_existing_shard(
                    output_dir / item["file"], source, int(item["start"]), int(item["stop"])
                )
                for item in existing["shards"]
            )
            if valid:
                return manifest_path

    started = time.monotonic()
    shard_ranges = [
        (start, min(start + shard_size, len(natoms))) for start in range(0, len(natoms), shard_size)
    ]
    manifest: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "complete": False,
        "source": source,
        "shard_size": shard_size,
        "n_shards": len(shard_ranges),
        "shards": [],
    }
    _atomic_json_write(manifest_path, manifest)
    database = connect(db_path)
    for shard_index, (start, stop) in enumerate(shard_ranges):
        path = output_dir / f"part-{shard_index:05d}.h5"
        if _valid_existing_shard(path, source, start, stop):
            record = _shard_record(path)
            print(f"[hdf5] reusing complete shard {path.name}", flush=True)
        else:
            record = _create_shard(
                database, path, source, natoms, start, stop, progress_every
            )
        manifest["shards"].append(record)
        manifest["completed_shards"] = shard_index + 1
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 1)
        _atomic_json_write(manifest_path, manifest)

    if hash_source:
        digest = hashlib.sha256()
        with db_path.open("rb") as source_file:
            while block := source_file.read(8 << 20):
                digest.update(block)
        manifest["source"]["sha256"] = digest.hexdigest()
    manifest["complete"] = True
    manifest["n_structures"] = len(natoms)
    manifest["n_atoms"] = int(natoms.sum())
    manifest["size_bytes"] = sum(item["size_bytes"] for item in manifest["shards"])
    manifest["elapsed_seconds"] = round(time.monotonic() - started, 1)
    _atomic_json_write(manifest_path, manifest)
    return manifest_path


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


class PackedHDF5ShardDataset(Dataset):
    def __init__(self, path: str | Path, r_max: float, z_table, heads: list[str] | None = None):
        self.path = str(Path(path).resolve())
        self.r_max = float(r_max)
        self.z_table = z_table
        self.heads = heads or ["Default"]
        self._file = None
        self._pid = None
        with h5py.File(self.path, "r") as handle:
            if handle.attrs.get("schema_name") != SCHEMA_NAME or not handle.attrs.get("complete", False):
                raise ValueError(f"invalid or incomplete packed MACE HDF5 shard: {self.path}")
            self.length = len(handle["energy"])

    def __len__(self) -> int:
        return self.length

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        state["_pid"] = None
        return state

    def _handle(self):
        pid = os.getpid()
        if self._file is None or self._pid != pid:
            if self._file is not None:
                self._file.close()
            self._file = h5py.File(self.path, "r")
            self._pid = pid
        return self._file

    def __getitem__(self, index: int):
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        from mace.data.atomic_data import AtomicData
        from mace.data.utils import Configuration

        handle = self._handle()
        first, last = map(int, handle["atom_offsets"][index : index + 2])
        properties = {
            "energy": np.asarray(handle["energy"][index]),
            "forces": np.asarray(handle["forces"][first:last]),
        }
        if bool(handle["stress_present"][index]):
            properties["stress"] = np.asarray(handle["stress"][index])
        config = Configuration(
            atomic_numbers=np.asarray(handle["atomic_numbers"][first:last]),
            positions=np.asarray(handle["positions"][first:last]),
            properties=properties,
            property_weights={
                "energy": float(handle["energy_weight"][index]),
                "forces": float(handle["forces_weight"][index]),
                "stress": float(handle["stress_weight"][index]),
            },
            cell=np.asarray(handle["cell"][index]),
            pbc=tuple(bool(value) for value in handle["pbc"][index]),
            weight=float(handle["config_weight"][index]),
            config_type=_decode(handle["config_type"][index]),
        )
        config.head = _decode(handle["head"][index])
        return AtomicData.from_config(
            config,
            z_table=self.z_table,
            cutoff=self.r_max,
            heads=self.heads,
        )

    def __del__(self):
        file = getattr(self, "_file", None)
        if file is not None:
            file.close()


def load_sharded_hdf5(path: str | Path, r_max: float, z_table, heads: list[str] | None = None):
    root = Path(path).resolve()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_name") != SCHEMA_NAME or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported packed HDF5 manifest: {manifest_path}")
    if not manifest.get("complete"):
        raise ValueError(f"packed HDF5 conversion is incomplete: {manifest_path}")
    shards = sorted(manifest["shards"], key=lambda item: item["start"])
    expected = 0
    datasets = []
    for item in shards:
        if item["start"] != expected:
            raise ValueError("HDF5 shards do not form a contiguous dataset")
        datasets.append(PackedHDF5ShardDataset(root / item["file"], r_max, z_table, heads))
        expected = item["stop"]
    if expected != manifest["n_structures"]:
        raise ValueError("HDF5 manifest length does not match its shards")
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def verify_sharded_hdf5(
    db_path: str | Path,
    hdf5_dir: str | Path,
    *,
    samples: int = 1_000,
    seed: int = 42,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    db_path = Path(db_path).resolve()
    root = Path(hdf5_dir).resolve()
    manifest = json.loads((root / MANIFEST_NAME).read_text())
    n_structures = int(manifest["n_structures"])
    sample_indices = np.sort(
        np.random.default_rng(seed).choice(n_structures, size=min(samples, n_structures), replace=False)
    )
    shard_starts = [item["start"] for item in manifest["shards"]]
    handles: dict[int, h5py.File] = {}
    database = connect(db_path)
    checked_atoms = 0
    try:
        for db_index in sample_indices:
            shard_index = bisect.bisect_right(shard_starts, int(db_index)) - 1
            item = manifest["shards"][shard_index]
            if shard_index not in handles:
                handles[shard_index] = h5py.File(root / item["file"], "r")
            handle = handles[shard_index]
            local_index = int(db_index) - int(item["start"])
            first, last = map(int, handle["atom_offsets"][local_index : local_index + 2])
            row = database.get(id=int(db_index) + 1)
            atoms = row.toatoms()
            row_energy = _row_value(row, atoms, "energy", "REF_energy")
            row_forces = _row_value(row, atoms, "forces", "REF_forces")
            np.testing.assert_array_equal(handle["source_db_index"][local_index], db_index)
            np.testing.assert_array_equal(handle["atomic_numbers"][first:last], atoms.numbers)
            np.testing.assert_allclose(handle["positions"][first:last], atoms.positions, rtol=0, atol=0)
            np.testing.assert_allclose(handle["cell"][local_index], atoms.cell, rtol=0, atol=0)
            np.testing.assert_array_equal(handle["pbc"][local_index], atoms.pbc)
            np.testing.assert_allclose(handle["energy"][local_index], row_energy, rtol=0, atol=0)
            np.testing.assert_allclose(handle["forces"][first:last], row_forces, rtol=0, atol=0)
            checked_atoms += len(atoms)
    finally:
        for handle in handles.values():
            handle.close()
    return {"checked_structures": len(sample_indices), "checked_atoms": checked_atoms, "seed": seed}
