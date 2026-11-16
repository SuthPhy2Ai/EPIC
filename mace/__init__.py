from .__version__ import __version__


def _preload_e3nn_constants_safely() -> None:
    """Load e3nn 0.4 constants without changing torch.load defaults globally."""
    import torch

    if hasattr(torch.serialization, "safe_globals"):
        with torch.serialization.safe_globals([slice]):
            from e3nn import o3  # noqa: F401
    else:  # PyTorch versions before safe_globals still default to full loading.
        from e3nn import o3  # noqa: F401


_preload_e3nn_constants_safely()
del _preload_e3nn_constants_safely
