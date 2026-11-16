from __future__ import annotations

import torch

from uq_benchmark import training


def test_cpu_device_never_calls_cuda_api_when_cuda_is_visible(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append(("set", device)))
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: calls.append(("reset", device)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda device: calls.append(("max", device)) or 123,
    )
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed",
        lambda seed: calls.append(("seed", seed)),
    )

    device = torch.device("cpu")
    training._prepare_cuda_device(device)
    training._reset_peak_memory_stats(device)
    training.seed_everything(7, device)

    assert training._peak_memory_bytes(device) == 0
    assert calls == []
