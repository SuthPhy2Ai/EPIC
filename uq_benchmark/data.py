from __future__ import annotations

from pathlib import Path
import re

from torch.utils.data import ConcatDataset

from .config import BenchmarkConfig
from .hdf5_dataset import MANIFEST_NAME, load_sharded_hdf5
from .modeling import configure_mace_source


def _natural_path_key(path: Path) -> tuple:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", path.name)
    )


def load_dataset(config: BenchmarkConfig):
    """Load packed HDF5, standard MACE HDF5, or ASE DB in source row order."""
    configure_mace_source()
    from mace.tools import AtomicNumberTable

    source = Path(config.data.hdf5_path or config.data.db_path)
    z_table = AtomicNumberTable(list(range(1, config.model.num_elements + 1)))
    hdf5_kwargs = {
        "r_max": config.model.r_max,
        "z_table": z_table,
        "heads": ["Default"],
        "head": "Default",
    }

    if source.is_dir():
        if (source / MANIFEST_NAME).is_file():
            return load_sharded_hdf5(
                source,
                r_max=config.model.r_max,
                z_table=z_table,
                heads=["Default"],
            )
        from mace.data import HDF5Dataset

        files = sorted(
            (*source.glob("*.h5"), *source.glob("*.hdf5")),
            key=_natural_path_key,
        )
        if not files:
            raise ValueError(f"HDF5 directory contains no .h5/.hdf5 shards: {source}")
        return ConcatDataset([HDF5Dataset(str(path), **hdf5_kwargs) for path in files])
    if source.suffix.lower() in (".h5", ".hdf5"):
        from mace.data import HDF5Dataset

        if not source.is_file():
            raise FileNotFoundError(f"HDF5 dataset not found: {source}")
        return HDF5Dataset(str(source), **hdf5_kwargs)
    if not source.is_file():
        raise FileNotFoundError(f"ASE database not found: {source}")
    from mace.data.ase_db_dataset import load_from_ase_db
    from mace.data.utils import KeySpecification

    return load_from_ase_db(
        db_path=str(source),
        keyspec=KeySpecification.from_defaults(),
        atomic_numbers_table=z_table,
        r_max=config.model.r_max,
    )


def make_data_loader(
    dataset,
    config: BenchmarkConfig,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
):
    configure_mace_source()
    from mace.tools import torch_geometric

    num_workers = config.training.num_workers
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "pin_memory": config.training.pin_memory,
    }
    if num_workers:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=config.training.prefetch_factor,
        )
    return torch_geometric.dataloader.DataLoader(**kwargs)
