from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np

from .calibration import apply_variance_map, fit_gaussian_variance_map, fit_student_t_scale
from .distributions import nig_predictive_parameters, nig_variances
from .types import CalibrationState, PredictionBatch


class UQMethod(ABC):
    name: str

    @abstractmethod
    def predict(self, batch) -> PredictionBatch:
        raise NotImplementedError

    @abstractmethod
    def calibrate(self, calibration_data) -> CalibrationState:
        raise NotImplementedError

    def sample_energy_models(self, n_samples: int, seed: int):
        raise NotImplementedError(f"{self.name} does not define a joint stochastic energy model")


def epic_prediction_from_nig(
    db_index: np.ndarray,
    gamma_total: np.ndarray,
    nu: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    natoms: np.ndarray,
    calibration: CalibrationState | None = None,
    force_location: np.ndarray | None = None,
) -> PredictionBatch:
    location, scale, df = nig_predictive_parameters(gamma_total, nu, alpha, beta)
    _, epistemic, _ = nig_variances(nu, alpha, beta)
    natoms = np.asarray(natoms, dtype=float)
    raw_scale_pa = scale / natoms
    calibrated = raw_scale_pa
    if calibration is not None:
        calibrated = raw_scale_pa * float(calibration.parameters["tau"])
    prediction = PredictionBatch(
        db_index=np.asarray(db_index),
        energy_location=location / natoms,
        energy_scale_raw=raw_scale_pa,
        energy_scale_calibrated=calibrated,
        energy_df=df,
        energy_risk=epistemic / np.square(natoms),
        force_location=force_location,
        distribution_family="student_t",
        metadata={
            "force_uq": "energy-first latent-mean NIG Monte Carlo on a fixed force subset"
        },
    )
    prediction.validate()
    return prediction


def calibrate_epic(
    target_pa: np.ndarray,
    prediction: PredictionBatch,
    method: str = "epic",
) -> CalibrationState:
    if prediction.energy_scale_raw is None or prediction.energy_df is None:
        raise ValueError("EPIC calibration requires raw scale and degrees of freedom")
    return fit_student_t_scale(
        target_pa,
        prediction.energy_location,
        prediction.energy_scale_raw,
        prediction.energy_df,
        method=method,
    )


def combine_ensemble(members: Iterable[PredictionBatch], state: CalibrationState | None = None) -> PredictionBatch:
    members = list(members)
    if len(members) < 2:
        raise ValueError("an ensemble requires at least two members")
    for member in members:
        member.validate()
    reference = members[0].db_index
    if any(not np.array_equal(m.db_index, reference) for m in members[1:]):
        raise ValueError("ensemble members have different db_index order")
    energies = np.stack([m.energy_location for m in members])
    raw_variance = energies.var(axis=0, ddof=1)
    calibrated_variance = apply_variance_map(raw_variance, state) if state else np.maximum(raw_variance, 1e-16)
    forces = None
    force_raw = None
    if all(m.force_location is not None for m in members):
        force_stack = np.stack([m.force_location for m in members])
        forces = force_stack.mean(axis=0)
        force_raw = force_stack.var(axis=0, ddof=1)
    result = PredictionBatch(
        db_index=reference.copy(),
        energy_location=energies.mean(axis=0),
        energy_scale_raw=np.sqrt(np.maximum(raw_variance, 1e-16)),
        energy_scale_calibrated=np.sqrt(calibrated_variance),
        energy_risk=raw_variance,
        force_location=forces,
        force_scale_raw=None if force_raw is None else np.sqrt(np.maximum(force_raw, 0.0)),
        distribution_family="gaussian",
        metadata={"n_members": len(members)},
    )
    result.validate()
    return result


def calibrate_ensemble(target_pa: np.ndarray, members: Iterable[PredictionBatch]) -> CalibrationState:
    members = list(members)
    energies = np.stack([m.energy_location for m in members])
    return fit_gaussian_variance_map(
        target_pa, energies.mean(axis=0), energies.var(axis=0, ddof=1), method=f"ensemble{len(members)}"
    )
