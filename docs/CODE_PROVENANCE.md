# Code provenance

This repository was assembled on 2026-08-06 from the isolated source snapshot
used for the selected Hierarchical atom-wise EPIC training run, and its frozen
two-model benchmark layer was completed on 2026-08-08. Training data,
checkpoints, logs, cluster launch files, private paths, and large pretrained
assets are intentionally excluded.

## Selected training-snapshot identity

The selected private run used source files with these SHA-256 identities:

| File | SHA-256 |
|---|---|
| `uq_benchmark/hierarchical_epic.py` | `a0532a7fed5acaa351e11d66d0e2cd2adea31ad7ba1f89d6c37df687d67448fd` |
| `uq_benchmark/distributions.py` | `4eb07203406a3dbd2f3b69df7a3bba5b16377634c845f6b0ecd15e1b2631d3aa` |
| `uq_benchmark/training.py` | `9b81557d382f484f2cba4147af0ce5efe526fede63107fed3d0c61daf1b3f5b3` |
| `mace/modules/models.py` | `efd57054399f26c4ab8fb09cf36e2f0f974c7208037f2a660e576d4822a89553` |
| `mace/modules/loss.py` | `342a403f74ce67cbda6064cbffdec81d9958e510d4b4b276f0c8c8ee4616ddc9` |

The selected private checkpoint had SHA-256
`df9e6d1629f166ce8b6c77988288faba8bf3f6a2d049627f3c4b2fb4f347c156`.
The checkpoint is not redistributed. The hash is retained so users who have
authorized access can verify its identity.

The release copies of `uq_benchmark/distributions.py`,
`mace/modules/models.py`, and `mace/modules/loss.py` remain byte-identical to
that snapshot. The release copies of `uq_benchmark/hierarchical_epic.py` and
`uq_benchmark/training.py` contain the operational safeguards listed below;
they do not change learned parameters, loss formulas, or state-dictionary keys.

## Release-layer changes

The open-source layer deliberately changes files that do not define learned
parameters:

- `uq_benchmark/modeling.py` removes private fallback paths and finds the
  bundled `mace/` package from the repository root.
- `uq_benchmark/training.py` gates CUDA-only setup and memory accounting on the
  device actually selected by Accelerate. This fixes CPU execution on a host
  where CUDA remains visible.
- `uq_benchmark/hierarchical_epic.py` fails immediately if heavy-tailed
  shared-scale sampling produces non-finite variance or energy values.
- `uq_benchmark/config.py` and the example YAML use relative placeholder data
  and run paths.
- `uq_benchmark/inference.py` directly stores all three derived atom-wise
  variance fields and rejects incomplete, duplicate, or structurally
  inconsistent shard sets during aggregation.
- `uq_benchmark/epic_pipeline.py` and `uq_benchmark/pipeline.py` record and
  validate conservative scientific limitations of force sampling.
- `uq_benchmark/diagnostics.py` adds NIG-domain and heavy-tail diagnostics.
- `uq_benchmark/checkpoint_audit.py` adds calibration-only finite-difference,
  closed-loop, and atom-permutation checks without changing model execution.
- `uq_benchmark/two_model_analysis.py` separates strict per-structure force
  scale convergence from fixed-seed aggregate-metric stability and stratifies
  the result by NIG tail parameter.
- `uq_benchmark/report.py`, `uq_benchmark/figures.py`, and
  `uq_benchmark/two_model_summary.py` generate the isolated MACE-versus-
  Hierarchical-EPIC store, tables, figures, and machine-readable summary.
- `uq_benchmark/downstream.py` discovers risk-bearing methods from the store,
  so the two-model release does not read deferred ensemble5 or LLPR groups.
- `maceedl_analysis` is reduced to the sampling helper actually imported by
  this release.
- `mace/__init__.py` safely preloads the e3nn 0.4 constants with a narrow
  `safe_globals([slice])` allowlist instead of changing PyTorch loading policy
  process-wide. Full-object MACE checkpoint loads explicitly use
  `weights_only=False` and therefore require trusted files.

None of these changes adds, removes, or renames a model state-dictionary key.
Loading remains strict (`strict=True`) so incompatible state-dictionary keys
or tensor shapes fail immediately. It cannot detect a semantic forward change
that preserves the same key and shape schema.

## Reproduction boundary

The public example configuration documents the architecture and commands but
cannot reproduce the reported numerical checkpoint without the original
dataset and exact split. Published numerical results should therefore state
the dataset version, split manifest/hash, training hardware, software lock,
checkpoint hash, and evaluation artifacts separately.
