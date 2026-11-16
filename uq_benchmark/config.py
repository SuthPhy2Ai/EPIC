from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(frozen=True)
class DataConfig:
    db_path: str = "data/dataset.db"
    hdf5_path: str | None = None
    split_dir: str = "runs/shared_split"
    seed: int = 42
    ratios: dict[str, float] = field(
        default_factory=lambda: {"train": 0.80, "validation": 0.10, "calibration": 0.05, "test": 0.05}
    )
    force_calibration_size: int = 10_000
    force_test_size: int = 20_000


@dataclass(frozen=True)
class ModelConfig:
    r_max: float = 5.5
    num_bessel: int = 8
    num_polynomial_cutoff: int = 5
    max_ell: int = 3
    num_interactions: int = 2
    num_channels: int = 64
    correlation: int = 3
    num_elements: int = 118


@dataclass(frozen=True)
class TrainingConfig:
    num_gpus: int = 1
    global_batch_size: int = 8
    per_gpu_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_workers: int = 6
    prefetch_factor: int = 2
    pin_memory: bool = True
    learning_rate: float = 0.005
    max_epochs: int = 300
    early_stopping_patience: int = 20
    scheduler_patience: int = 5
    energy_loss: Literal["mse_total", "huber_per_atom"] = "mse_total"
    force_weight: float = 100.0
    force_loss: Literal["mse", "conditional_huber"] = "mse"
    force_huber_delta: float = 0.01
    lambda_anneal_epochs: int = 20
    epic_lambda_max: tuple[float, ...] = (1e-4, 1e-3, 1e-2)
    ensemble_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)


@dataclass(frozen=True)
class LLPRConfig:
    ridge_multipliers: tuple[float, ...] = (1e-8, 1e-6, 1e-4, 1e-2)
    force_samples: int = 32
    sample_seed: int = 42


@dataclass(frozen=True)
class EPICConfig:
    force_samples: int = 32
    sample_seed: int = 42
    hierarchical_allocation_weight: float = 0.1
    hierarchical_force_error_floor: float = 1e-6


@dataclass(frozen=True)
class EvaluationConfig:
    coverage_levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)
    bootstrap_replicates: int = 1000
    bootstrap_seed: int = 42
    danger_quantile: float = 0.95
    budget_fractions: tuple[float, ...] = (0.01, 0.05, 0.10)
    benchmark_structures: int = 10_000


@dataclass(frozen=True)
class BenchmarkConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    epic: EPICConfig = field(default_factory=EPICConfig)
    llpr: LLPRConfig = field(default_factory=LLPRConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output_root: str = "uq_benchmark_runs"

    def validate(self) -> None:
        ratios = self.data.ratios
        if tuple(ratios) != ("train", "validation", "calibration", "test"):
            raise ValueError("split keys must be train, validation, calibration, test in that order")
        if abs(sum(ratios.values()) - 1.0) > 1e-12 or any(v <= 0 for v in ratios.values()):
            raise ValueError("split ratios must be positive and sum to one")
        expected_global = (
            self.training.per_gpu_batch_size
            * self.training.num_gpus
            * self.training.gradient_accumulation_steps
        )
        if expected_global != self.training.global_batch_size:
            raise ValueError("GPU count, per-GPU batch and accumulation do not match global_batch_size")
        if self.training.num_workers < 0 or self.training.prefetch_factor <= 0:
            raise ValueError("num_workers must be non-negative and prefetch_factor must be positive")
        if self.training.force_loss not in ("mse", "conditional_huber"):
            raise ValueError("force_loss must be mse or conditional_huber")
        if self.training.energy_loss not in ("mse_total", "huber_per_atom"):
            raise ValueError("energy_loss must be mse_total or huber_per_atom")
        if self.training.force_weight <= 0 or self.training.force_huber_delta <= 0:
            raise ValueError("force weight and Huber delta must be positive")
        if self.llpr.force_samples <= 0 or self.llpr.force_samples % 2:
            raise ValueError("LLPR force_samples must be a positive even number")
        if self.epic.force_samples < 2:
            raise ValueError("EPIC force_samples must be at least two")
        if self.epic.hierarchical_allocation_weight < 0:
            raise ValueError("hierarchical allocation weight must be nonnegative")
        if self.epic.hierarchical_force_error_floor <= 0:
            raise ValueError("hierarchical force-error floor must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type, values: dict[str, Any]) -> Any:
    fields = {f.name: f for f in __import__("dataclasses").fields(cls)}
    unknown = set(values) - set(fields)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path | None = None) -> BenchmarkConfig:
    if path is None:
        config = BenchmarkConfig()
    else:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        config = BenchmarkConfig(
            data=_construct(DataConfig, raw.get("data", {})),
            model=_construct(ModelConfig, raw.get("model", {})),
            training=_construct(TrainingConfig, raw.get("training", {})),
            epic=_construct(EPICConfig, raw.get("epic", {})),
            llpr=_construct(LLPRConfig, raw.get("llpr", {})),
            evaluation=_construct(EvaluationConfig, raw.get("evaluation", {})),
            output_root=raw.get("output_root", "uq_benchmark_runs"),
        )
    config.validate()
    return config
