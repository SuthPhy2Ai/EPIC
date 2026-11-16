from __future__ import annotations

import json
import math
import os
import random
import signal
import time
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from .config import BenchmarkConfig
from .data import load_dataset, make_data_loader
from .distributions import (
    annealed_regularization,
    full_nig_loss_torch,
    independent_atomic_moment_loss_torch,
)
from .hierarchical_epic import hierarchical_epic_energy_loss
from .modeling import create_model
from .split import load_split


class TerminationFlag:
    def __init__(self) -> None:
        self.requested = False

    def handler(self, signum, frame) -> None:  # pragma: no cover - exercised by Slurm
        del signum, frame
        self.requested = True


def _uses_cuda(device: torch.device | str) -> bool:
    return torch.device(device).type == "cuda"


def _prepare_cuda_device(device: torch.device | str) -> None:
    if _uses_cuda(device):
        torch.cuda.set_device(device)


def _reset_peak_memory_stats(device: torch.device | str) -> None:
    if _uses_cuda(device):
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_bytes(device: torch.device | str) -> int:
    if not _uses_cuda(device):
        return 0
    return int(torch.cuda.max_memory_allocated(device))


def seed_everything(seed: int, device: torch.device | str = "cpu") -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if _uses_cuda(device):
        torch.cuda.manual_seed(seed)


def _write_progress(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _make_loaders(config: BenchmarkConfig, split_path: str | Path):
    from torch.utils.data import Subset

    dataset = load_dataset(config)
    train_indices = load_split(split_path, "train")
    validation_indices = load_split(split_path, "validation")
    train_loader = make_data_loader(
        Subset(dataset, train_indices.tolist()),
        config,
        batch_size=config.training.per_gpu_batch_size,
        shuffle=True,
        drop_last=True,
    )
    validation_loader = make_data_loader(
        Subset(dataset, validation_indices.tolist()),
        config,
        batch_size=config.training.per_gpu_batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, validation_loader


def _average_neighbors(loader, max_batches: int = 10) -> float:
    edges = 0
    atoms = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        edges += int(batch.edge_index.shape[1])
        atoms += int(batch.positions.shape[0])
    return edges / atoms if atoms else 50.0


def _loss(
    kind: Literal[
        "deterministic",
        "epic",
        "epic_atomwise",
        "epic_hierarchical",
    ],
    output: dict,
    batch,
    force_weight: float,
    regularization: float,
    force_loss_kind: Literal["mse", "conditional_huber"] = "mse",
    force_huber_delta: float = 0.01,
    energy_loss_kind: Literal["mse_total", "huber_per_atom"] = "mse_total",
    hierarchical_allocation_weight: float = 0.1,
    hierarchical_force_error_floor: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if kind == "deterministic":
        if energy_loss_kind == "mse_total":
            energy_loss = F.mse_loss(output["energy"], batch.energy)
        elif energy_loss_kind == "huber_per_atom":
            atoms_per_graph = (batch.ptr[1:] - batch.ptr[:-1]).to(output["energy"].dtype)
            energy_loss = F.huber_loss(
                output["energy"] / atoms_per_graph,
                batch.energy / atoms_per_graph,
                delta=force_huber_delta,
            )
        else:  # pragma: no cover - validated by TrainingConfig
            raise ValueError(f"unsupported deterministic energy loss: {energy_loss_kind}")
    elif kind == "epic":
        energy_loss = full_nig_loss_torch(
            batch.energy,
            output["energy"],
            output["energy_nu"],
            output["energy_alpha"],
            output["energy_beta"],
            regularization,
        )
    elif kind == "epic_atomwise":
        atoms_per_graph = (batch.ptr[1:] - batch.ptr[:-1]).to(output["energy"].dtype)
        energy_loss = independent_atomic_moment_loss_torch(
            batch.energy,
            output["energy"],
            output["energy_aleatoric_variance"],
            output["energy_epistemic_variance"],
            output["atomic_energy_nu"],
            output["atomic_energy_alpha"],
            batch.batch,
            atoms_per_graph,
            regularization,
        )
    elif kind == "epic_hierarchical":
        energy_loss, _, _ = hierarchical_epic_energy_loss(
            batch.energy,
            output,
            batch.forces,
            batch.batch,
            regularization,
            hierarchical_allocation_weight,
            hierarchical_force_error_floor,
        )
    else:  # pragma: no cover - validated by the CLI
        raise ValueError(f"unsupported model kind: {kind}")
    if force_loss_kind == "mse":
        force_loss = F.mse_loss(output["forces"], batch.forces)
    elif force_loss_kind == "conditional_huber":
        from mace.modules.loss import conditional_huber_forces

        force_loss = conditional_huber_forces(
            batch.forces,
            output["forces"],
            huber_delta=force_huber_delta,
            ddp=True,
        )
    else:  # pragma: no cover - validated by TrainingConfig
        raise ValueError(f"unsupported force loss: {force_loss_kind}")
    # Force loss must remain attached; this is a scientific control, not a monitor.
    return energy_loss + float(force_weight) * force_loss, energy_loss, force_loss


def _validate_hierarchical_output_dir(
    output_dir: Path,
    resume: str | Path | None,
    *,
    check_occupancy: bool = True,
) -> None:
    """Prevent hierarchical runs from sharing legacy EPIC result directories."""
    if "epic_hierarchical" not in output_dir.parts:
        raise ValueError(
            "hierarchical EPIC output_dir must contain an 'epic_hierarchical' "
            "path component"
        )
    forbidden = {"epic_graph_nig", "epic_atomwise"}
    overlap = forbidden.intersection(output_dir.parts)
    if overlap:
        raise ValueError(
            f"hierarchical EPIC output_dir overlaps legacy directories: {sorted(overlap)}"
        )
    if (
        check_occupancy
        and output_dir.exists()
        and any(output_dir.iterdir())
        and resume is None
    ):
        raise FileExistsError(
            f"refusing to reuse non-empty hierarchical run directory: {output_dir}"
        )
    if resume is not None:
        resume_path = Path(resume).resolve()
        if resume_path.parent != output_dir.resolve():
            raise ValueError(
                "hierarchical resume checkpoint must be inside the same isolated run directory"
            )


def _prediction_error_sums(output: dict, batch) -> torch.Tensor:
    """Return additive point-error statistics in dataset-native eV units."""
    energy_prediction = output["energy"].detach().reshape(-1)
    energy_target = batch.energy.detach().reshape(-1)
    atoms_per_graph = (batch.ptr[1:] - batch.ptr[:-1]).to(energy_prediction.dtype)
    if energy_prediction.numel() != atoms_per_graph.numel():
        raise ValueError("energy prediction count does not match graph count")
    energy_error_per_atom = (energy_prediction - energy_target) / atoms_per_graph
    force_error = (output["forces"].detach() - batch.forces.detach()).reshape(-1)
    return torch.stack(
        (
            energy_error_per_atom.abs().sum(),
            energy_error_per_atom.square().sum(),
            energy_prediction.new_tensor(float(energy_error_per_atom.numel())),
            force_error.abs().sum(),
            force_error.square().sum(),
            energy_prediction.new_tensor(float(force_error.numel())),
        )
    ).to(torch.float64)


def _prediction_metrics(error_sums: torch.Tensor) -> tuple[float, float, float, float]:
    energy_count = float(error_sums[2].item())
    force_count = float(error_sums[5].item())
    if energy_count <= 0 or force_count <= 0:
        raise ValueError("prediction metrics require non-empty energy and force targets")
    return (
        float(error_sums[0].item() / energy_count),
        math.sqrt(float(error_sums[1].item() / energy_count)),
        float(error_sums[3].item() / force_count),
        math.sqrt(float(error_sums[4].item() / force_count)),
    )


def save_checkpoint(
    path: Path,
    accelerator,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_validation_loss: float,
    stale_epochs: int,
    config: BenchmarkConfig,
    kind: str,
    seed: int,
    lambda_max: float,
    avg_num_neighbors: float,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "kind": kind,
            "result_namespace": (
                "epic_hierarchical"
                if kind == "epic_hierarchical"
                else kind
            ),
            "epic_variant": (
                "hierarchical_atomwise_shared_scale"
                if kind == "epic_hierarchical"
                else None
            ),
            "seed": seed,
            "lambda_max": lambda_max,
            "avg_num_neighbors": avg_num_neighbors,
            "epoch": epoch,
            "best_validation_loss": best_validation_loss,
            "stale_epochs": stale_epochs,
            "config": config.to_dict(),
            "model_state_dict": accelerator.unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        temporary,
    )
    os.replace(temporary, path)


def run_training(
    config: BenchmarkConfig,
    split_path: str | Path,
    output_dir: str | Path,
    kind: Literal[
        "deterministic",
        "epic",
        "epic_atomwise",
        "epic_hierarchical",
    ],
    seed: int,
    lambda_max: float = 0.0,
    resume: str | Path | None = None,
    reset_training_state: bool = False,
) -> Path:
    if kind == "deterministic" and lambda_max:
        raise ValueError("lambda_max is only valid for EPIC")
    if kind in ("epic", "epic_atomwise", "epic_hierarchical") and lambda_max not in config.training.epic_lambda_max:
        raise ValueError(f"EPIC lambda_max must be one of {config.training.epic_lambda_max}")
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if accelerator.local_process_index != local_rank:
        raise RuntimeError(
            f"Accelerator local rank {accelerator.local_process_index} does not match "
            f"LOCAL_RANK={local_rank}"
        )
    _prepare_cuda_device(accelerator.device)
    seed_everything(seed, accelerator.device)
    if accelerator.num_processes != config.training.num_gpus:
        raise RuntimeError(
            f"config requests {config.training.num_gpus} GPUs, "
            f"but accelerate started {accelerator.num_processes} processes"
        )
    output_dir = Path(output_dir)
    if kind == "epic_hierarchical":
        # All ranks enforce the namespace checks. Only rank 0 checks whether
        # the directory is occupied: it may create run_config.json before the
        # other ranks get here, which must not look like a reused run.
        _validate_hierarchical_output_dir(
            output_dir,
            resume,
            check_occupancy=accelerator.is_main_process,
        )
        accelerator.wait_for_everyone()
    output_dir.mkdir(parents=True, exist_ok=True)
    energy_loss_label = {
        "epic": "full_graph_nig_nll",
        "epic_atomwise": "independent_atomic_gaussian_moment_nll",
        "epic_hierarchical": "full_graph_nig_nll_plus_force_error_allocation",
    }.get(kind, config.training.energy_loss)
    if accelerator.is_main_process:
        (output_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "kind": kind,
                    "seed": seed,
                    "lambda_max": lambda_max,
                    "world_size": accelerator.num_processes,
                    "resume_checkpoint": None if resume is None else str(resume),
                    "reset_training_state": reset_training_state,
                    "energy_loss_kind": energy_loss_label,
                    "epic_variant": (
                        "hierarchical_atomwise_shared_scale"
                        if kind == "epic_hierarchical"
                        else None
                    ),
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "config": config.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    state = None
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        checkpoint_matches = (
            state["kind"] == kind
            and int(state["seed"]) == seed
            and float(state["lambda_max"]) == lambda_max
        )
        if not checkpoint_matches:
            raise ValueError("resume checkpoint does not match requested run")
        if "avg_num_neighbors" not in state:
            raise ValueError("resume checkpoint does not contain avg_num_neighbors")
        avg_num_neighbors = float(state["avg_num_neighbors"])
        if not np.isfinite(avg_num_neighbors) or avg_num_neighbors <= 0:
            raise ValueError(
                f"resume checkpoint contains invalid avg_num_neighbors={avg_num_neighbors}"
            )

    train_loader, validation_loader = _make_loaders(config, split_path)
    if state is None:
        avg_num_neighbors = _average_neighbors(train_loader)
    model = create_model(kind, config.model, avg_num_neighbors)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.training.scheduler_patience
    )
    start_epoch = 0
    best_validation_loss = float("inf")
    stale_epochs = 0
    if state is not None:
        model.load_state_dict(state["model_state_dict"], strict=True)
        start_epoch = int(state["epoch"]) + 1
        if not reset_training_state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            best_validation_loss = float(state["best_validation_loss"])
            stale_epochs = int(state["stale_epochs"])

    model, optimizer, train_loader, validation_loader = accelerator.prepare(
        model, optimizer, train_loader, validation_loader
    )
    termination = TerminationFlag()
    signal.signal(signal.SIGTERM, termination.handler)
    signal.signal(signal.SIGINT, termination.handler)
    latest = output_dir / "checkpoint_latest.pt"
    best = output_dir / "checkpoint_best.pt"
    progress_path = output_dir / "training_progress.json"
    iteration_log = (
        (output_dir / "iterations.jsonl").open("a", buffering=1)
        if accelerator.is_main_process
        else None
    )

    for epoch in range(start_epoch, config.training.max_epochs):
        epoch_started = time.perf_counter()
        _reset_peak_memory_stats(accelerator.device)
        regularization = (
            annealed_regularization(lambda_max, epoch, config.training.lambda_anneal_epochs)
            if kind in ("epic", "epic_atomwise", "epic_hierarchical")
            else 0.0
        )
        model.train()
        train_stage_started = time.perf_counter()
        n_train_batches = len(train_loader)
        train_progress_interval = max(1, n_train_batches // 100)
        train_objective_sum = torch.zeros(4, device=accelerator.device, dtype=torch.float64)
        train_error_sum = torch.zeros(6, device=accelerator.device, dtype=torch.float64)
        for batch_index, batch in enumerate(train_loader, start=1):
            with accelerator.accumulate(model):
                optimizer.zero_grad(set_to_none=True)
                output = model(batch, training=True)
                loss, energy_loss, force_loss = _loss(
                    kind,
                    output,
                    batch,
                    config.training.force_weight,
                    regularization,
                    config.training.force_loss,
                    config.training.force_huber_delta,
                    config.training.energy_loss,
                    config.epic.hierarchical_allocation_weight,
                    config.epic.hierarchical_force_error_floor,
                )
                accelerator.backward(loss)
                optimizer.step()
            n_graphs = int(batch.ptr.numel() - 1)
            train_objective_sum += torch.stack(
                (
                    loss.detach().double() * n_graphs,
                    energy_loss.detach().double() * n_graphs,
                    force_loss.detach().double() * n_graphs,
                    loss.detach().new_tensor(float(n_graphs), dtype=torch.float64),
                )
            )
            train_error_sum += _prediction_error_sums(output, batch)
            reduced_losses = accelerator.reduce(
                torch.stack(
                    (loss.detach(), energy_loss.detach(), force_loss.detach())
                ).to(torch.float64),
                reduction="mean",
            )
            if iteration_log is not None:
                iteration_log.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "iteration_in_epoch": batch_index,
                            "global_iteration": epoch * n_train_batches + batch_index,
                            "total_loss": float(reduced_losses[0].item()),
                            "energy_loss": float(reduced_losses[1].item()),
                            "energy_loss_kind": energy_loss_label,
                            "force_loss": float(reduced_losses[2].item()),
                            "force_weight": float(config.training.force_weight),
                            "force_loss_kind": config.training.force_loss,
                            "force_huber_delta": float(config.training.force_huber_delta),
                            "regularization": regularization,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "elapsed_seconds": time.perf_counter() - train_stage_started,
                            "world_size": accelerator.num_processes,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            if accelerator.is_main_process and (
                batch_index == 1
                or batch_index == n_train_batches
                or batch_index % train_progress_interval == 0
            ):
                elapsed = time.perf_counter() - train_stage_started
                _write_progress(
                    progress_path,
                    {
                        "stage": "train",
                        "epoch": epoch,
                        "max_epochs": config.training.max_epochs,
                        "completed_batches": batch_index,
                        "total_batches": n_train_batches,
                        "percent": round(100.0 * batch_index / n_train_batches, 2),
                        "stage_elapsed_seconds": round(elapsed, 1),
                        "stage_eta_seconds": round(elapsed * (n_train_batches - batch_index) / batch_index, 1),
                        "world_size": accelerator.num_processes,
                    },
                )
            if termination.requested:
                break

        train_objective_sum = accelerator.reduce(train_objective_sum, reduction="sum")
        train_error_sum = accelerator.reduce(train_error_sum, reduction="sum")
        train_total_loss_value = float((train_objective_sum[0] / train_objective_sum[3]).item())
        train_energy_loss_value = float((train_objective_sum[1] / train_objective_sum[3]).item())
        train_force_loss_value = float((train_objective_sum[2] / train_objective_sum[3]).item())
        (
            train_energy_mae,
            train_energy_rmse,
            train_force_mae,
            train_force_rmse,
        ) = _prediction_metrics(train_error_sum)

        model.eval()
        validation_objective_sum = torch.zeros(4, device=accelerator.device, dtype=torch.float64)
        validation_error_sum = torch.zeros(6, device=accelerator.device, dtype=torch.float64)
        validation_stage_started = time.perf_counter()
        n_validation_batches = len(validation_loader)
        validation_progress_interval = max(1, n_validation_batches // 100)
        for batch_index, batch in enumerate(validation_loader, start=1):
            # Forces require autograd even in evaluation.
            with torch.enable_grad():
                output = model(batch, training=False)
                validation_loss, validation_energy_loss, validation_force_loss = _loss(
                    kind,
                    output,
                    batch,
                    config.training.force_weight,
                    regularization,
                    config.training.force_loss,
                    config.training.force_huber_delta,
                    config.training.energy_loss,
                    config.epic.hierarchical_allocation_weight,
                    config.epic.hierarchical_force_error_floor,
                )
            n_graphs = int(batch.ptr.numel() - 1)
            validation_objective_sum += torch.stack(
                (
                    validation_loss.detach().double() * n_graphs,
                    validation_energy_loss.detach().double() * n_graphs,
                    validation_force_loss.detach().double() * n_graphs,
                    validation_loss.detach().new_tensor(float(n_graphs), dtype=torch.float64),
                )
            )
            validation_error_sum += _prediction_error_sums(output, batch)
            if accelerator.is_main_process and (
                batch_index == 1
                or batch_index == n_validation_batches
                or batch_index % validation_progress_interval == 0
            ):
                elapsed = time.perf_counter() - validation_stage_started
                _write_progress(
                    progress_path,
                    {
                        "stage": "validation",
                        "epoch": epoch,
                        "max_epochs": config.training.max_epochs,
                        "completed_batches": batch_index,
                        "total_batches": n_validation_batches,
                        "percent": round(100.0 * batch_index / n_validation_batches, 2),
                        "stage_elapsed_seconds": round(elapsed, 1),
                        "stage_eta_seconds": round(
                            elapsed * (n_validation_batches - batch_index) / batch_index, 1
                        ),
                        "world_size": accelerator.num_processes,
                    },
                )
        validation_objective_sum = accelerator.reduce(validation_objective_sum, reduction="sum")
        validation_error_sum = accelerator.reduce(validation_error_sum, reduction="sum")
        validation_loss_value = float(
            (validation_objective_sum[0] / validation_objective_sum[3]).item()
        )
        validation_energy_loss_value = float(
            (validation_objective_sum[1] / validation_objective_sum[3]).item()
        )
        validation_force_loss_value = float(
            (validation_objective_sum[2] / validation_objective_sum[3]).item()
        )
        (
            validation_energy_mae,
            validation_energy_rmse,
            validation_force_mae,
            validation_force_rmse,
        ) = _prediction_metrics(validation_error_sum)
        scheduler.step(validation_loss_value)
        improved = validation_loss_value < best_validation_loss
        if improved:
            best_validation_loss = validation_loss_value
            stale_epochs = 0
        else:
            stale_epochs += 1
        save_checkpoint(
            latest,
            accelerator,
            model,
            optimizer,
            scheduler,
            epoch,
            best_validation_loss,
            stale_epochs,
            config,
            kind,
            seed,
            lambda_max,
            avg_num_neighbors,
        )
        if improved:
            save_checkpoint(
                best,
                accelerator,
                model,
                optimizer,
                scheduler,
                epoch,
                best_validation_loss,
                stale_epochs,
                config,
                kind,
                seed,
                lambda_max,
                avg_num_neighbors,
            )
        if accelerator.is_main_process:
            with (output_dir / "epochs.jsonl").open("a") as log:
                log.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "train_total_loss": train_total_loss_value,
                            "train_energy_loss": train_energy_loss_value,
                            "energy_loss_kind": energy_loss_label,
                            "train_force_loss": train_force_loss_value,
                            "force_loss_kind": config.training.force_loss,
                            "force_weight": float(config.training.force_weight),
                            "force_huber_delta": float(config.training.force_huber_delta),
                            "train_energy_mae_ev_per_atom": train_energy_mae,
                            "train_energy_rmse_ev_per_atom": train_energy_rmse,
                            "train_force_mae_ev_per_angstrom": train_force_mae,
                            "train_force_rmse_ev_per_angstrom": train_force_rmse,
                            "validation_loss": validation_loss_value,
                            "validation_energy_loss": validation_energy_loss_value,
                            "validation_force_loss": validation_force_loss_value,
                            "validation_energy_mae_ev_per_atom": validation_energy_mae,
                            "validation_energy_rmse_ev_per_atom": validation_energy_rmse,
                            "validation_force_mae_ev_per_angstrom": validation_force_mae,
                            "validation_force_rmse_ev_per_angstrom": validation_force_rmse,
                            "best_validation_loss": best_validation_loss,
                            "stale_epochs": stale_epochs,
                            "regularization": regularization,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "epoch_seconds": time.perf_counter() - epoch_started,
                            "peak_memory_bytes": _peak_memory_bytes(accelerator.device),
                            "world_size": accelerator.num_processes,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            _write_progress(
                progress_path,
                {
                    "stage": "epoch_complete",
                    "epoch": epoch,
                    "max_epochs": config.training.max_epochs,
                    "validation_loss": validation_loss_value,
                    "best_validation_loss": best_validation_loss,
                    "stale_epochs": stale_epochs,
                    "epoch_seconds": round(time.perf_counter() - epoch_started, 1),
                    "peak_memory_bytes": _peak_memory_bytes(accelerator.device),
                    "world_size": accelerator.num_processes,
                },
            )
        if termination.requested or stale_epochs >= config.training.early_stopping_patience:
            break
    if iteration_log is not None:
        iteration_log.close()
    return best
