###########################################################################################
# Script for evaluating pretrained MACEEDL models with EDL outputs
# Authors: Codex agent
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

"""Evaluate structures with a pretrained MACEEDL model while saving EDL parameters."""

from __future__ import annotations

import argparse
from typing import List

import ase.io
import numpy as np
import torch

from mace import data
from mace.tools import torch_geometric, torch_tools, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a pretrained MACEEDL model and store energy, forces, and EDL"
            " parameters in an output XYZ file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--configs", required=True, help="Path to input XYZ configs")
    parser.add_argument("--model", required=True, help="Path to pretrained model")
    parser.add_argument("--output", required=True, help="Destination XYZ path")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--default_dtype",
        choices=["float32", "float64"],
        default="float64",
        help="Default dtype used during inference",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--compute_stress",
        action="store_true",
        help="Whether to compute stress tensors",
    )
    parser.add_argument(
        "--info_prefix",
        type=str,
        default="MACEEDL_",
        help="Prefix used when writing results to Atoms.info/arrays",
    )
    parser.add_argument(
        "--head",
        type=str,
        default=None,
        help="Optional model head to use during evaluation",
    )
    return parser.parse_args()


def _split_per_structure(array: np.ndarray, ptr: torch.Tensor) -> List[np.ndarray]:
    """Split flattened per-atom arrays into per-structure segments."""

    segments = np.split(array, indices_or_sections=ptr[1:], axis=0)
    return segments[:-1]  # drop final empty entry produced by torch_geometric ptr


def run(args: argparse.Namespace) -> None:
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    model = torch.load(args.model, map_location=args.device, weights_only=False)
    model = model.to(device)
    model.eval()

    if not hasattr(model, "edl_readout"):
        raise ValueError(
            "Loaded model does not expose EDL outputs. Please provide a MACEEDL model."
        )

    atoms_list = ase.io.read(args.configs, index=":")
    if args.head is not None:
        for atoms in atoms_list:
            atoms.info["head"] = args.head

    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    try:
        heads = model.heads
    except AttributeError:
        heads = None

    dataset = [
        data.AtomicData.from_config(
            config, z_table=z_table, cutoff=float(model.r_max), heads=heads
        )
        for config in configs
    ]
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    energy_batches: List[np.ndarray] = []
    nu_batches: List[np.ndarray] = []
    alpha_batches: List[np.ndarray] = []
    beta_batches: List[np.ndarray] = []
    stress_batches: List[np.ndarray] = []
    forces_segments: List[List[np.ndarray]] = []
    gamma_node_segments: List[List[np.ndarray]] = []
    nu_node_segments: List[List[np.ndarray]] = []
    alpha_node_segments: List[List[np.ndarray]] = []
    beta_node_segments: List[List[np.ndarray]] = []

    for batch in data_loader:
        batch = batch.to(device)
        node_heads_tensor = getattr(batch, "head", None)
        node_assignments = batch.batch
        if node_heads_tensor is not None:
            node_heads = node_heads_tensor[node_assignments]
        else:
            node_heads = None
        with torch.no_grad():
            output = model(
                batch.to_dict(),
                compute_stress=args.compute_stress,
            )

        required_keys = ["energy", "energy_nu", "energy_alpha", "energy_beta", "forces"]
        for key in required_keys:
            if key not in output:
                raise KeyError(
                    f"Model output does not contain '{key}', cannot compute EDL metrics."
                )

        energy_batches.append(torch_tools.to_numpy(output["energy"]))
        nu_batches.append(torch_tools.to_numpy(output["energy_nu"]))
        alpha_batches.append(torch_tools.to_numpy(output["energy_alpha"]))
        beta_batches.append(torch_tools.to_numpy(output["energy_beta"]))

        if args.compute_stress:
            stress_batches.append(torch_tools.to_numpy(output["stress"]))

        node_feats = output.get("node_feats")
        if node_feats is None:
            raise KeyError(
                "Model output does not contain 'node_feats', cannot compute per-atom EDL."
            )

        gamma_node, nu_node, alpha_node, beta_node = model.edl_readout(
            node_feats, node_heads
        )

        forces_np = torch_tools.to_numpy(output["forces"])
        forces_segments.append(_split_per_structure(forces_np, batch.ptr))
        gamma_node_segments.append(
            _split_per_structure(torch_tools.to_numpy(gamma_node), batch.ptr)
        )
        nu_node_segments.append(
            _split_per_structure(torch_tools.to_numpy(nu_node), batch.ptr)
        )
        alpha_node_segments.append(
            _split_per_structure(torch_tools.to_numpy(alpha_node), batch.ptr)
        )
        beta_node_segments.append(
            _split_per_structure(torch_tools.to_numpy(beta_node), batch.ptr)
        )

    energies = np.concatenate(energy_batches, axis=0)
    nus = np.concatenate(nu_batches, axis=0)
    alphas = np.concatenate(alpha_batches, axis=0)
    betas = np.concatenate(beta_batches, axis=0)
    forces_list = [forces for sub in forces_segments for forces in sub]
    gamma_nodes = [arr for sub in gamma_node_segments for arr in sub]
    nu_nodes = [arr for sub in nu_node_segments for arr in sub]
    alpha_nodes = [arr for sub in alpha_node_segments for arr in sub]
    beta_nodes = [arr for sub in beta_node_segments for arr in sub]

    if args.compute_stress:
        stresses = np.concatenate(stress_batches, axis=0)
    else:
        stresses = None

    assert len(atoms_list) == len(energies) == len(forces_list)
    assert len(gamma_nodes) == len(atoms_list)

    info_prefix = args.info_prefix
    for idx, atoms in enumerate(atoms_list):
        atoms.calc = None
        atoms.info[f"{info_prefix}energy"] = float(energies[idx])
        atoms.arrays[f"{info_prefix}forces"] = forces_list[idx]
        atoms.info[f"{info_prefix}edl_nu"] = float(nus[idx])
        atoms.info[f"{info_prefix}edl_alpha"] = float(alphas[idx])
        atoms.info[f"{info_prefix}edl_beta"] = float(betas[idx])
        atoms.arrays[f"{info_prefix}edl_gamma_node"] = gamma_nodes[idx]
        atoms.arrays[f"{info_prefix}edl_nu_node"] = nu_nodes[idx]
        atoms.arrays[f"{info_prefix}edl_alpha_node"] = alpha_nodes[idx]
        atoms.arrays[f"{info_prefix}edl_beta_node"] = beta_nodes[idx]
        if stresses is not None:
            atoms.info[f"{info_prefix}stress"] = stresses[idx]

    ase.io.write(args.output, images=atoms_list, format="extxyz")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
