from __future__ import annotations

import json

import numpy as np
import pytest

from uq_benchmark.diagnostics import diagnose_nig_predictions


def test_diagnose_nig_predictions_reports_heavy_tail_fraction(tmp_path):
    prediction = tmp_path / "predictions.npz"
    np.savez(
        prediction,
        nu=np.array([1.0, 2.0, 3.0]),
        alpha=np.array([1.25, 2.0, 5.0]),
        beta=np.array([0.5, 0.5, 0.5]),
        natoms=np.array([2, 4, 8]),
    )
    output = tmp_path / "diagnostics.json"
    result = diagnose_nig_predictions(prediction, output)
    assert result["domain_valid"] is True
    assert result["fraction_alpha_le_2"] == pytest.approx(2.0 / 3.0)
    assert result["requires_mc_convergence_study"] is True
    assert json.loads(output.read_text())["force_posterior_identified"] is False


def test_diagnose_nig_predictions_rejects_invalid_domain(tmp_path):
    prediction = tmp_path / "invalid.npz"
    np.savez(
        prediction,
        nu=np.array([1.0]),
        alpha=np.array([1.0]),
        beta=np.array([0.5]),
        natoms=np.array([2]),
    )
    with pytest.raises(ValueError, match="domain"):
        diagnose_nig_predictions(prediction)
