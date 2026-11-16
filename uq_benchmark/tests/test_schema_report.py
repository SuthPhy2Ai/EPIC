from __future__ import annotations

import numpy as np
import pandas as pd

from uq_benchmark.report import generate_report
from uq_benchmark.schema import BenchmarkStore, validate_method_alignment
from uq_benchmark.types import PredictionBatch


def test_hdf5_alignment_and_report_fields(tmp_path):
    rng = np.random.default_rng(4)
    n = 80
    index = np.arange(n)
    truth = rng.normal(size=n)
    force_true = rng.normal(size=(n, 3))
    offsets = np.arange(n + 1)
    path = tmp_path / "benchmark.h5"
    with BenchmarkStore(path, "w") as store:
        store.write_manifest({"test": True})
        store.write_structures(index, np.array([f"m-{i//2}" for i in index]), np.ones(n), truth, force_true, offsets)
        for method, family, offset in (
            ("deterministic", "none", 0.10),
            ("ensemble5", "gaussian", 0.06),
            ("epic", "student_t", 0.08),
            ("llpr", "gaussian", 0.07),
        ):
            prediction = PredictionBatch(
                db_index=index,
                energy_location=truth + rng.normal(scale=offset, size=n),
                energy_scale_raw=None if family == "none" else np.full(n, offset),
                energy_scale_calibrated=None if family == "none" else np.full(n, offset * 1.1),
                energy_df=np.full(n, 6.0) if family == "student_t" else None,
                energy_risk=None if family == "none" else rng.uniform(0.01, 1.0, n),
                force_location=force_true + rng.normal(scale=offset, size=(n, 3)),
                distribution_family=family,
            )
            store.write_method(method, prediction)
            if method in ("ensemble5", "llpr"):
                store.write_force_subset(
                    method,
                    index,
                    force_true,
                    prediction.force_location,
                    np.full((n, 3), offset),
                    np.full((n, 3), offset * 1.1),
                    offsets,
                )
    validate_method_alignment(path)
    output = generate_report(path, tmp_path / "report", bootstrap_replicates=20)
    metrics = pd.read_csv(output / "tables/metrics.csv")
    required = {"method", "target", "metric", "estimate", "unit", "n", "distribution_family", "calibration"}
    assert required <= set(metrics.columns)
    assert {"raw", "calibrated"} <= set(metrics.calibration)
    confidence = pd.read_csv(output / "tables/paired_cluster_confidence_intervals.csv")
    assert {"estimate", "ci_low_95", "ci_high_95", "bootstrap_unit", "bootstrap_replicates", "seed", "n"} <= set(confidence.columns)

