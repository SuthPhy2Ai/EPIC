# Scientific limitations

1. The graph-level NIG is trained on total energies. No atom-resolved energy
   likelihood or force-uncertainty likelihood is used.
2. Atom-wise NIG-compatible parameters reproduce allocated graph moments but do
   not define independent atom-wise posteriors.
3. The learned `atomic_epistemic_weight` is an atom-wise energy-uncertainty
   attribution trained against a detached relative force-error proxy. It is not
   an atom-wise force variance.
4. Mean pooling omits explicit structure size from the graph NIG heads. Check
   NLL and coverage as functions of atom count before cross-size use.
5. The NIG aleatoric/epistemic split is model-defined and is not uniquely
   identifiable from one energy label per structure.
6. Sampled forces are energy-gradient-derived for each executed computation.
   The implementation does not prove a single global random potential across
   independently evaluated configurations, atom reorderings, or neighbor-list
   topology changes.
7. Force MC sampling excludes aleatoric observation noise. The same integer RNG
   seed is replayed for each structure, but cross-structure base-random-variable
   alignment is not verified. It is not a complete predictive force posterior.
8. With `alpha <= 2`, fourth moments do not exist and a 32-sample variance
   estimate can be unstable. Report alpha diagnostics and MC convergence.
9. Atomic energy contributions have a gauge/non-identifiability: only their sum
   and its coordinate gradient are supervised.
10. Calibration and test evaluation are required before any calibrated-UQ claim.
11. Very heavy NIG tails can make finite Monte Carlo samples unstable. The
    release fails on non-finite sampled energies or forces, but finite extreme
    values still require alpha diagnostics and a multi-seed convergence study.
12. An aggregate force-UQ sample-count recommendation applies only to the
    frozen seed's calibration, coverage, and ranking protocol. It does not
    imply that the force-scale estimate of every structure has converged,
    especially in the `alpha <= 2` strata where the fourth moment is absent.
13. The Gaussian force variance map and headline force coverage are fitted and
    evaluated over Cartesian force components. Larger structures therefore
    contribute more components. The separate atom-count strata are required to
    expose size-dependent behavior; the component-weighted headline must not
    be described as an equal-structure average.
