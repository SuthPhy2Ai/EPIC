from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class CalibrationState:
    method: str
    parameters: Mapping[str, float]
    objective: str = "negative_log_likelihood"
    split: str = "calibration"


@dataclass
class PredictionBatch:
    db_index: np.ndarray
    energy_location: np.ndarray
    energy_scale_raw: np.ndarray | None = None
    energy_scale_calibrated: np.ndarray | None = None
    energy_df: np.ndarray | None = None
    energy_risk: np.ndarray | None = None
    force_location: np.ndarray | None = None
    force_scale_raw: np.ndarray | None = None
    force_scale_calibrated: np.ndarray | None = None
    force_risk: np.ndarray | None = None
    distribution_family: str = "none"
    inference_seconds: float = 0.0
    peak_memory_bytes: int = 0
    not_applicable: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.db_index = np.asarray(self.db_index, dtype=np.int64)
        self.energy_location = np.asarray(self.energy_location, dtype=np.float64)
        if self.db_index.ndim != 1 or self.energy_location.shape != self.db_index.shape:
            raise ValueError("db_index and energy_location must be matching 1D arrays")
        if len(np.unique(self.db_index)) != len(self.db_index):
            raise ValueError("db_index contains duplicates")
        for name in (
            "energy_scale_raw",
            "energy_scale_calibrated",
            "energy_df",
            "energy_risk",
            "force_risk",
        ):
            value = getattr(self, name)
            if value is not None and np.asarray(value).shape != self.db_index.shape:
                raise ValueError(f"{name} must have one value per structure")
        for name in ("energy_scale_raw", "energy_scale_calibrated"):
            value = getattr(self, name)
            if value is not None and np.any(np.asarray(value) <= 0):
                raise ValueError(f"{name} must be strictly positive")

