from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .types import CalibrationState, PredictionBatch


class BenchmarkStore:
    def __init__(self, path: str | Path, mode: str = "a"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.path, mode)
        self.handle.attrs.setdefault("schema_version", 1)

    def __enter__(self) -> "BenchmarkStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.handle.close()

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self.handle.attrs["manifest_json"] = json.dumps(manifest, sort_keys=True)

    def write_structures(
        self,
        db_index: np.ndarray,
        material_id: np.ndarray,
        natoms: np.ndarray,
        energy_true_pa: np.ndarray,
        force_true: np.ndarray | None = None,
        force_offsets: np.ndarray | None = None,
    ) -> None:
        group = self.handle.require_group("structures")
        self._replace(group, "db_index", np.asarray(db_index, dtype=np.int64))
        self._replace(group, "material_id", np.asarray(material_id, dtype=h5py.string_dtype("utf-8")))
        self._replace(group, "natoms", np.asarray(natoms, dtype=np.int32))
        self._replace(group, "energy_true_pa", np.asarray(energy_true_pa, dtype=np.float64))
        if force_true is not None:
            if force_offsets is None:
                raise ValueError("force_offsets are required with force_true")
            self._replace(group, "force_true", np.asarray(force_true, dtype=np.float32), compression="gzip")
            self._replace(group, "force_offsets", np.asarray(force_offsets, dtype=np.int64))

    def write_method(self, name: str, prediction: PredictionBatch) -> None:
        prediction.validate()
        group = self.handle.require_group(f"methods/{name}")
        group.attrs["distribution_family"] = prediction.distribution_family
        group.attrs["inference_seconds"] = prediction.inference_seconds
        group.attrs["peak_memory_bytes"] = prediction.peak_memory_bytes
        group.attrs["not_applicable_json"] = json.dumps(dict(prediction.not_applicable), sort_keys=True)
        group.attrs["metadata_json"] = json.dumps(dict(prediction.metadata), sort_keys=True)
        for field in (
            "db_index",
            "energy_location",
            "energy_scale_raw",
            "energy_scale_calibrated",
            "energy_df",
            "energy_risk",
            "force_location",
            "force_scale_raw",
            "force_scale_calibrated",
            "force_risk",
        ):
            value = getattr(prediction, field)
            if value is not None:
                self._replace(group, field, np.asarray(value), compression="gzip")

    def write_calibration(self, state: CalibrationState) -> None:
        group = self.handle.require_group(f"calibration/{state.method}")
        group.attrs["objective"] = state.objective
        group.attrs["split"] = state.split
        group.attrs["parameters_json"] = json.dumps(dict(state.parameters), sort_keys=True)

    def write_force_subset(
        self,
        method: str,
        db_index: np.ndarray,
        force_true: np.ndarray,
        force_location: np.ndarray,
        force_scale_raw: np.ndarray,
        force_scale_calibrated: np.ndarray,
        force_offsets: np.ndarray,
        distribution_family: str = "gaussian",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        group = self.handle.require_group(f"methods/{method}/force_uq_subset")
        group.attrs["distribution_family"] = distribution_family
        group.attrs["metadata_json"] = json.dumps(metadata or {}, sort_keys=True)
        values = {
            "db_index": np.asarray(db_index, dtype=np.int64),
            "force_true": np.asarray(force_true, dtype=np.float32),
            "force_location": np.asarray(force_location, dtype=np.float32),
            "force_scale_raw": np.asarray(force_scale_raw, dtype=np.float32),
            "force_scale_calibrated": np.asarray(force_scale_calibrated, dtype=np.float32),
            "force_offsets": np.asarray(force_offsets, dtype=np.int64),
        }
        if values["force_true"].shape != values["force_location"].shape or values["force_true"].shape != values["force_scale_raw"].shape:
            raise ValueError("force subset arrays must have matching shapes")
        if values["force_true"].ndim != 2 or values["force_true"].shape[1] != 3:
            raise ValueError("force subset arrays must have shape [n_atoms, 3]")
        if len(values["force_offsets"]) != len(values["db_index"]) + 1:
            raise ValueError("force subset needs one offset interval per structure")
        if values["force_offsets"][-1] != len(values["force_true"]):
            raise ValueError("force subset offsets do not terminate at the atom array length")
        if np.any(~np.isfinite(values["force_scale_raw"])) or np.any(values["force_scale_raw"] < 0):
            raise ValueError("raw force scales must be finite and nonnegative")
        if np.any(~np.isfinite(values["force_scale_calibrated"])) or np.any(values["force_scale_calibrated"] <= 0):
            raise ValueError("calibrated force scales must be finite and positive")
        for name, value in values.items():
            self._replace(group, name, value, compression="gzip" if value.ndim > 1 else None)

    @staticmethod
    def _replace(group: h5py.Group, name: str, value: np.ndarray, **kwargs) -> None:
        if name in group:
            del group[name]
        group.create_dataset(name, data=value, **kwargs)


def validate_method_alignment(path: str | Path) -> None:
    with h5py.File(path, "r") as handle:
        expected = handle["structures/db_index"][:]
        for name, group in handle["methods"].items():
            observed = group["db_index"][:]
            if not np.array_equal(expected, observed):
                raise ValueError(f"method {name} does not align with structures/db_index")
