from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HALOGENS = {9, 17, 35, 53, 85}


def chemistry_class(atomic_numbers: set[int]) -> str:
    if 8 in atomic_numbers:
        return "oxide"
    if atomic_numbers & HALOGENS:
        return "halide"
    if 7 in atomic_numbers:
        return "nitride"
    return "other"


def _metadata(db_path: str | Path, indices: np.ndarray) -> pd.DataFrame:
    connection = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    try:
        wanted = {int(index) for index in indices}
        rows = {}
        for row_id, natoms in connection.execute("SELECT id - 1, natoms FROM systems"):
            if row_id in wanted:
                rows[int(row_id)] = {"natoms": int(natoms), "elements": set()}
        for row_id, atomic_number in connection.execute("SELECT id - 1, Z FROM species"):
            if row_id in wanted:
                rows[int(row_id)]["elements"].add(int(atomic_number))
    finally:
        connection.close()
    return pd.DataFrame(
        [
            {
                "db_index": int(index),
                "natoms": rows[int(index)]["natoms"],
                "chemistry_class": chemistry_class(rows[int(index)]["elements"]),
            }
            for index in indices
        ]
    )


def _stable_tie(index: int, namespace: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"{namespace}:{index}".encode(), digest_size=8).digest(), "big")


def create_downstream_manifests(
    store_path: str | Path, db_path: str | Path, output_dir: str | Path
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(store_path, "r") as handle:
        indices = handle["structures/db_index"][:]
        risks = []
        method_names = sorted(handle["methods"].keys())
        risk_methods = [
            method
            for method in method_names
            if "energy_risk" in handle[f"methods/{method}"]
        ]
        if not risk_methods:
            raise ValueError("benchmark store has no method with an energy-risk score")
        for method in risk_methods:
            value = handle[f"methods/{method}/energy_risk"][:]
            order = np.argsort(np.argsort(value, kind="stable"), kind="stable")
            risks.append(order / max(len(value) - 1, 1))
    joint_methods = [method for method in risk_methods if method in {"ensemble5", "llpr"}]
    marginal_methods = [method for method in risk_methods if method not in joint_methods]
    frame = _metadata(db_path, indices)
    frame["consensus_risk_percentile"] = np.mean(risks, axis=0)
    frame["risk_quartile"] = pd.qcut(
        frame.consensus_risk_percentile, 4, labels=["Q1", "Q2", "Q3", "Q4"]
    ).astype(str)
    frame["tie"] = [_stable_tie(index, "downstream-v1") for index in frame.db_index]

    eos_rows = []
    for quartile, group in frame.groupby("risk_quartile", sort=True):
        eligible = group[group.natoms <= 16].sort_values("tie").head(3)
        if len(eligible) != 3:
            raise RuntimeError(f"not enough EOS candidates in {quartile}")
        eos_rows.append(eligible)
    eos = pd.concat(eos_rows).drop(columns="tie").sort_values(["risk_quartile", "db_index"])
    eos["volume_scales"] = "0.94,0.96,0.98,1.00,1.02,1.04,1.06"
    eos["reference_type"] = "MLIP-only diagnostic; no DFT validation"
    eos_path = output_dir / "eos_representatives.csv"
    eos.to_csv(eos_path, index=False)

    phonon_rows = []
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        quartile_frame = frame[(frame.risk_quartile == quartile) & (frame.natoms <= 12)]
        for family in ("oxide", "halide", "nitride", "other"):
            eligible = quartile_frame[quartile_frame.chemistry_class == family].sort_values("tie")
            if eligible.empty:
                raise RuntimeError(f"no phonon candidate for {quartile}/{family}")
            phonon_rows.append(eligible.iloc[0])
    phonon = pd.DataFrame(phonon_rows).drop(columns="tie").sort_values(["risk_quartile", "chemistry_class"])
    phonon["reference_type"] = "MLIP-predicted; no DFT validation"
    phonon["joint_uq_methods"] = ",".join(joint_methods)
    phonon["marginal_energy_methods"] = ",".join(marginal_methods)
    phonon["marginal_energy_treatment"] = "structure risk only; no joint phonon band"
    phonon_path = output_dir / "phonon_representatives.csv"
    phonon.to_csv(phonon_path, index=False)

    manifest_path = output_dir / "downstream_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "store": str(Path(store_path).resolve()),
                "n_eos": len(eos),
                "n_phonon": len(phonon),
                "volume_scales": [0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06],
                "allowed_claim": "MLIP-only cross-method diagnostic",
                "forbidden_claims": ["DFT verified", "physical stability accuracy", "demonstrated screening utility"],
                "risk_consensus_methods": risk_methods,
                "joint_energy_model_methods": joint_methods,
                "marginal_energy_only_methods": marginal_methods,
                "deferred_methods": ["ensemble5", "llpr", "graph_nig_epic"]
                if risk_methods == ["hierarchical_epic"]
                else [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return eos_path, phonon_path, manifest_path
