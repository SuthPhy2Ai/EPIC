from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_importing_mace_does_not_disable_weights_only_loading_globally():
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; import mace; "
                "assert 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD' not in os.environ"
            ),
        ],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_direct_model_creation_safely_loads_e3nn_constants():
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    script = """
import os
from uq_benchmark.config import ModelConfig
from uq_benchmark.modeling import create_model

config = ModelConfig(
    num_channels=4,
    num_interactions=1,
    num_elements=2,
    r_max=3.0,
    num_bessel=4,
    num_polynomial_cutoff=3,
    max_ell=1,
    correlation=2,
)
model = create_model("epic_hierarchical", config, 4.0)
assert model.__class__.__name__ == "HierarchicalAtomwiseEPIC"
assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in os.environ
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
