# Two-model benchmark execution

## Active scope

The active paper benchmark compares exactly two selected checkpoints:

1. deterministic MACE (`mace` in the benchmark store);
2. Hierarchical atom-wise EPIC (`hierarchical_epic` in the benchmark store).

The older ensemble-of-five, LLPR, and graph-NIG EPIC comparison is deferred to
the final project stage. Its existing commands remain available but must not be
mixed into the two-model store or paper tables.

## Frozen evaluation protocol

- Use the existing material-grouped seed-42 split without regeneration.
- Fit the Hierarchical EPIC energy scale on `calibration` only.
- Evaluate point energy and force errors on the complete `test` split.
- Evaluate energy NLL, PIT, coverage, interval width, ranking, and risk-coverage
  on the complete `test` split.
- Fit force-UQ calibration on the fixed 10,000-structure calibration subset and
  report force-UQ metrics on the fixed 20,000-structure test subset.
- Treat force calibration and headline coverage as Cartesian-component
  weighted, and report atom-count strata so that large structures do not hide
  size-dependent miscalibration.
- Keep deterministic MACE uncertainty fields explicitly not applicable.
- Record dataset, split, checkpoint, source, configuration, prediction, and
  report hashes in the release manifest.

The formal run uses 1,264,314 / 158,036 / 79,025 / 79,020 structures for
train / validation / calibration / test. Its frozen identities are:

| Artifact | SHA-256 |
|---|---|
| MPTrj ASE database | `d2f1e88bdc16eefdaea52a695f2ec1b0d9a275ca6a238df3b47ab16b341b62b9` |
| Material-grouped split | `7810c9ef36aad9be2ee62f15c7ba56bef6ad4bfde58d0bad1fa9ec504dbebd99` |
| Deterministic MACE checkpoint | `16391b9a0b08d2ac570151fe52b5e08a2224d4d72be752da894de96fc4372d6e` |
| Hierarchical EPIC epoch-30 checkpoint | `df9e6d1629f166ce8b6c77988288faba8bf3f6a2d049627f3c4b2fb4f347c156` |

## Required Hierarchical EPIC audits

- NIG-domain and heavy-tail diagnostics, including the fraction `alpha <= 2`.
- Energy calibration stratified by atom count.
- Force Monte Carlo convergence over multiple sample counts and seeds.
- Separate the strict per-structure scale-convergence verdict from the sample
  count used for fixed-seed aggregate calibration, coverage, and ranking. A
  stable aggregate recommendation must meet median seed-CV, rank-correlation,
  coverage-range, and higher-sample-reference criteria; it must not be
  relabeled as convergence of every heavy-tail per-structure scale.
- Point-force and sampled-force energy-gradient consistency checks.
- Atom-permutation checks with explicit limits on cross-configuration random
  variable alignment claims.
- Long-tail error analysis by atom count and force-error quantile.

No independent atom-posterior, global random-PES, universally superior
point-accuracy, or universal per-structure force-scale convergence claim is
allowed. Calibrated-UQ statements must name the evaluated target, split,
calibration family, and whether the result is aggregate or per-structure.

## Frozen point-prediction results

The complete 79,020-structure test split gives:

| Metric | Deterministic MACE | Hierarchical EPIC |
|---|---:|---:|
| Energy MAE (eV/atom) | 0.033467 | 0.029190 |
| Energy RMSE (eV/atom) | 0.087915 | 0.070704 |
| Force-component MAE (eV/Angstrom) | 0.059076 | 0.076307 |
| Force-component RMSE (eV/Angstrom) | 0.504869 | 0.279769 |
| Force-vector MAE (eV/Angstrom) | 0.123700 | 0.159513 |
| Force-vector RMSE (eV/Angstrom) | 0.874458 | 0.484574 |

The paired material-cluster bootstrap difference in energy MAE
(Hierarchical minus MACE) is -0.004276 eV/atom, with a 95% interval from
-0.005268 to -0.003355 eV/atom. Point accuracy is nevertheless mixed:
Hierarchical EPIC improves the reported energy errors and force RMSE, whereas
deterministic MACE has the lower force MAE. These results do not support a
universal-superiority claim.

## Frozen energy-UQ results

The energy scale was fitted on all 79,025 calibration structures. The fitted
Student-t scale multiplier is `tau=1.3413318700646646`; the additive scale is
zero. On the complete test split:

| Metric | Calibrated value |
|---|---:|
| NLL (nats) | -2.253177 |
| PIT ECE-20 | 0.010425 |
| 50% coverage | 0.539509 |
| 80% coverage | 0.803012 |
| 90% coverage | 0.885130 |
| 95% coverage | 0.926348 |
| Mean absolute coverage error | 0.020261 |

Energy-risk Spearman is 0.443072. Detection of the top-5% energy errors gives
AUROC 0.879721 and AUPRC 0.401981. The paired point-result interval and all UQ
values use the frozen test split only after calibration selection was complete.

Calibration is size dependent. The calibrated 95% energy coverage across
atom-count bins `1-20`, `21-40`, `41-80`, `81-160`, and `>160` is respectively
0.9373, 0.9317, 0.9104, 0.8753, and 0.7679. These bins are a size stress test,
not an external-OOD benchmark. The frozen material-grouped test split remains
in-distribution.

## Force-MC protocol selection

The calibration-only convergence study used a fixed hashed subset of 256
structures, sample counts 32/64/128/256/512/1024, and seeds 42/43/44. With
1024 as the highest-sample reference, 256 samples is the first count satisfying
the fixed-seed aggregate calibration/coverage/ranking thresholds:

| 256-sample diagnostic | Value |
|---|---:|
| Median structure seed CV | 0.08044 |
| 95th-percentile structure seed CV | 0.28212 |
| Mean pairwise scale Spearman | 0.99198 |
| Raw 90% coverage seed range | 0.02544 |
| Maximum median relative difference vs. 1024 | 0.08345 |
| Maximum q95 relative difference vs. 1024 | 0.39343 |

The strict per-structure recommendation is null because the q95 seed CV exceeds
0.25. Even at 1024 samples, q95 seed CV is 0.31926. Its values stratified by
`alpha <= 1.5`, `1.5 < alpha <= 2`, and `alpha > 2` are 0.53672, 0.10011, and
0.06967. The formal force benchmark therefore uses `samples_256_seed_42` only
as a calibration-selected aggregate protocol; it is not evidence that every
per-structure force scale converged.

## Frozen force-UQ results

The force variance map was fitted on 10,000 calibration structures (947,448
Cartesian components). It has `tau=0.8401370890` and an additive scale floor
`sigma0=0.0843299711 eV/Angstrom`. The independent 20,000-structure test subset
contains 1,921,581 Cartesian components and gives:

| Metric | Calibrated value |
|---|---:|
| NLL (nats) | -0.423520 |
| PIT ECE-20 | 0.029184 |
| 50% coverage | 0.752268 |
| 80% coverage | 0.897999 |
| 90% coverage | 0.934451 |
| 95% coverage | 0.953643 |
| Mean absolute coverage error | 0.097090 |

The Gaussian moment-matched MC intervals are over-covering at the 50% and 80%
levels; only quoting the near-nominal 90% and 95% values would be misleading.
All headline force calibration values are component weighted, not
equal-structure weighted.

Force-component risk has Spearman 0.467201, top-5% AUROC 0.786721, and top-5%
AUPRC 0.166501. After reducing errors and risks to one value per structure,
the corresponding values are 0.570184, 0.812078, and 0.206580. Calibrated 95%
coverage across atom-count bins `1-20`, `21-40`, `41-80`, `81-160`, and `>160`
is 0.9563, 0.9489, 0.9544, 0.9535, and 0.9727. The final bin contains only 109
structures and remains a size stress test rather than external-OOD evidence.

## Checkpoint audit and cost

The formal 16-structure calibration audit uses eight sampled-force checks and
the numerical finite-difference step 0.01 Angstrom. Maximum point-autograd,
point finite-difference, sampled finite-difference, and sampled closed-loop
absolute-work discrepancies are respectively 1.13e-6 eV/Angstrom,
0.006694 eV/Angstrom, 0.025177 eV/Angstrom, and 1.03e-6 eV. Point prediction
and atomic-weight permutation checks pass. Same-sample-ID permutation
equivariance and a global random PES are not verified.

Recorded training and point-inference cost is:

| Method | Epochs | Training wall-hours | GPU-hours | Point ms/structure |
|---|---:|---:|---:|---:|
| Deterministic MACE | 107 | 45.967 | 367.738 | 3.574 |
| Hierarchical EPIC | 51 | 24.261 | 194.090 | 3.838 |

Training cost merges resumed segments and sums recorded epoch timers. Point
inference time sums shard-local device timers and is not allocation wall time.
Force-UQ calibration and test similarly sum to 22,204.10 and 44,240.02
shard-device seconds; peak per-device allocated memory is 962,993,664 and
974,788,096 bytes, respectively.
