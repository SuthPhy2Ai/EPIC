from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "validation", "calibration", "test")
ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class SplitArtifacts:
    indices_path: Path
    groups_path: Path
    manifest_path: Path


def sha256_file(
    path: str | Path,
    block_size: int = 8 << 20,
    progress: ProgressCallback | None = None,
) -> str:
    digest = hashlib.sha256()
    total = Path(path).stat().st_size
    completed = 0
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
            completed += len(block)
            if progress is not None:
                progress("database_sha256", completed, total)
    return digest.hexdigest()


def read_structure_metadata(
    db_path: str | Path,
    progress: ProgressCallback | None = None,
    batch_size: int = 100_000,
) -> pd.DataFrame:
    """Read ASE metadata using linear table scans rather than an unindexed join."""
    connection = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    try:
        n_structures = int(connection.execute("SELECT COUNT(*) FROM systems").fetchone()[0])
        db_indices = np.empty(n_structures, dtype=np.int64)
        natoms = np.empty(n_structures, dtype=np.int32)
        systems_cursor = connection.execute("SELECT id - 1, natoms FROM systems ORDER BY id")
        completed = 0
        while rows := systems_cursor.fetchmany(batch_size):
            ids = np.fromiter((row[0] for row in rows), dtype=np.int64, count=len(rows))
            db_indices[completed : completed + len(rows)] = ids
            natoms[completed : completed + len(rows)] = np.fromiter(
                (row[1] for row in rows), dtype=np.int32, count=len(rows)
            )
            completed += len(rows)
            if progress is not None:
                progress("systems", completed, n_structures)

        material_ids = np.empty(n_structures, dtype=object)
        material_ids.fill(None)
        material_cursor = connection.execute(
            "SELECT id - 1, value FROM text_key_values WHERE key = 'material_id'"
        )
        completed = 0
        while rows := material_cursor.fetchmany(batch_size):
            for db_index, material_id in rows:
                if 0 <= db_index < n_structures:
                    material_ids[db_index] = material_id
            completed += len(rows)
            if progress is not None:
                progress("material_ids", min(completed, n_structures), n_structures)

        if any(value is None for value in material_ids):
            raise RuntimeError("every structure must have material_id")
        material_codes, unique_materials = pd.factorize(material_ids, sort=False)
        material_elements = [set() for _ in range(len(unique_materials))]
        n_species = int(connection.execute("SELECT COUNT(*) FROM species").fetchone()[0])
        species_cursor = connection.execute("SELECT id - 1, Z FROM species")
        completed = 0
        while rows := species_cursor.fetchmany(batch_size):
            for db_index, atomic_number in rows:
                if 0 <= db_index < n_structures:
                    material_elements[material_codes[db_index]].add(int(atomic_number))
            completed += len(rows)
            if progress is not None:
                progress("elements", completed, n_species)
    finally:
        connection.close()

    element_by_material = {
        str(material_id): ",".join(map(str, sorted(elements)))
        for material_id, elements in zip(unique_materials, material_elements, strict=True)
    }
    frame = pd.DataFrame(
        {
            "db_index": db_indices,
            "material_id": material_ids,
            "natoms": natoms,
            "elements": [element_by_material[str(material_id)] for material_id in material_ids],
        }
    )
    if frame.empty or frame["db_index"].duplicated().any():
        raise RuntimeError("database metadata is empty or contains duplicate row IDs")
    if not np.array_equal(frame.db_index.to_numpy(), np.arange(len(frame))):
        raise RuntimeError("material_id metadata does not cover every database row")
    return frame


def assign_material_groups(
    metadata: pd.DataFrame, ratios: dict[str, float], seed: int
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    if tuple(ratios) != SPLIT_NAMES or abs(sum(ratios.values()) - 1.0) > 1e-12:
        raise ValueError("ratios must define ordered train/validation/calibration/test splits summing to one")
    counts = (
        metadata.groupby("material_id", sort=True)
        .agg(
            n_structures=("db_index", "size"),
            elements=(
                "elements",
                lambda values: ",".join(
                    map(str, sorted({int(z) for value in values for z in str(value).split(",")}))
                ),
            ),
        )
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(counts))
    counts = counts.iloc[order].reset_index(drop=True)

    total = int(counts.n_structures.sum())
    targets = np.array([ratios[name] * total for name in SPLIT_NAMES], dtype=float)
    assigned = np.zeros(len(counts), dtype=np.int8)
    running = np.zeros(len(SPLIT_NAMES), dtype=np.int64)
    split_idx = 0
    for i, n_structures in enumerate(counts.n_structures.to_numpy(dtype=np.int64)):
        while split_idx < len(SPLIT_NAMES) - 1:
            current_error = abs(running[split_idx] - targets[split_idx])
            next_error = abs(running[split_idx] + n_structures - targets[split_idx])
            if running[split_idx] > 0 and current_error <= next_error:
                split_idx += 1
            else:
                break
        assigned[i] = split_idx
        running[split_idx] += n_structures
    counts["split"] = [SPLIT_NAMES[i] for i in assigned]

    mapping = dict(zip(counts.material_id.astype(str), counts.split, strict=True))
    row_split = metadata.material_id.astype(str).map(mapping)
    result = {
        name: np.sort(metadata.loc[row_split == name, "db_index"].to_numpy(dtype=np.int64))
        for name in SPLIT_NAMES
    }
    validate_split(result, metadata)
    return result, counts.sort_values("material_id").reset_index(drop=True)


def validate_split(indices: dict[str, np.ndarray], metadata: pd.DataFrame) -> None:
    if tuple(indices) != SPLIT_NAMES:
        raise ValueError(f"split keys must be {SPLIT_NAMES}")
    arrays = [np.asarray(indices[name], dtype=np.int64) for name in SPLIT_NAMES]
    merged = np.concatenate(arrays)
    if len(np.unique(merged)) != len(merged):
        raise ValueError("split indices overlap")
    expected = np.arange(len(metadata), dtype=np.int64)
    if not np.array_equal(np.sort(merged), expected):
        raise ValueError("split indices do not cover the database exactly")
    material_sets = []
    mids = metadata.material_id.astype(str).to_numpy()
    for name, array in zip(SPLIT_NAMES, arrays, strict=True):
        if len(np.unique(array)) != len(array) or np.any(array < 0) or np.any(array >= len(metadata)):
            raise ValueError(f"invalid indices in {name}")
        material_sets.append(set(mids[array]))
    for i, left in enumerate(material_sets):
        for right in material_sets[i + 1 :]:
            if left & right:
                raise ValueError("material_id leakage detected")


def create_split(
    db_path: str | Path,
    output_dir: str | Path,
    ratios: dict[str, float],
    seed: int = 42,
    hash_database: bool = True,
) -> SplitArtifacts:
    db_path = Path(db_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "split_progress.json"
    started = time.monotonic()

    def report_progress(stage: str, completed: int, total: int) -> None:
        percent = 100.0 if total == 0 else 100.0 * completed / total
        payload = {
            "stage": stage,
            "completed": int(completed),
            "total": int(total),
            "percent": round(percent, 2),
            "elapsed_seconds": round(time.monotonic() - started, 1),
        }
        temporary = progress_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
        temporary.replace(progress_path)
        print(
            f"[split] {stage}: {completed:,}/{total:,} ({percent:.1f}%)",
            flush=True,
        )

    metadata = read_structure_metadata(db_path, progress=report_progress)
    indices, groups = assign_material_groups(metadata, ratios, seed)
    report_progress("assign_and_validate", 1, 1)

    indices_path = output_dir / "split_indices.npz"
    groups_path = output_dir / "material_groups.csv"
    manifest_path = output_dir / "split_manifest.json"
    np.savez_compressed(indices_path, **indices)
    groups.to_csv(groups_path, index=False)
    manifest = {
        "schema_version": 1,
        "db_path": str(db_path),
        "db_size_bytes": db_path.stat().st_size,
        "db_sha256": sha256_file(db_path, progress=report_progress) if hash_database else None,
        "seed": seed,
        "ratios": ratios,
        "n_structures": len(metadata),
        "n_materials": int(metadata.material_id.nunique()),
        "splits": {
            name: {
                "n_structures": int(len(indices[name])),
                "n_materials": int(groups.split.eq(name).sum()),
            }
            for name in SPLIT_NAMES
        },
        "material_overlap": 0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report_progress("complete", 1, 1)
    return SplitArtifacts(indices_path, groups_path, manifest_path)


def load_split(path: str | Path, split: str, allow_test: bool = False) -> np.ndarray:
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split: {split}")
    if split == "test" and not allow_test:
        raise PermissionError("test split access requires allow_test=True and is forbidden during tuning")
    with np.load(path) as data:
        return np.asarray(data[split], dtype=np.int64)


def hashed_subset(indices: np.ndarray, size: int, namespace: str) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if size >= len(indices):
        return np.sort(indices.copy())
    scores = np.fromiter(
        (
            int.from_bytes(hashlib.blake2b(f"{namespace}:{int(i)}".encode(), digest_size=8).digest(), "big")
            for i in indices
        ),
        dtype=np.uint64,
        count=len(indices),
    )
    return np.sort(indices[np.argpartition(scores, size)[:size]])
