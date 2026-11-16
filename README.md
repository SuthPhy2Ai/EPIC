# EPIC: Energy-Propagated Intrinsic Confidence for Machine-Learned Interatomic Potentials

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**EPIC** (**E**nergy-**P**ropagated **I**ntrinsic **C**onfidence) is a differentiable, energy-centered uncertainty quantification (UQ) and physical observable propagation framework for equivariant machine-learned interatomic potentials (MLIPs).

---

## Core Principle

Conventional force-level evidential models place independent distributions on Cartesian force components without defining an integrable potential energy surface (PES), breaking the physical derivative chain connecting potential energy, atomic forces, and the force-constant Hessian:

$$
E(\mathbf{R}) \longrightarrow \mathbf{F}_i(\mathbf{R}) = -\frac{\partial E(\mathbf{R})}{\partial \mathbf{R}_i} \longrightarrow \Phi_{ij}^{\alpha\beta}(\mathbf{R}) = \frac{\partial^2 E(\mathbf{R})}{\partial R_i^\alpha \partial R_j^\beta}
$$

EPIC maintains differential consistency across this chain through four components:

### 1. Conservative Energy Readout and Graph-Level NIG Prior

EPIC places a Normal-Inverse-Gamma (NIG) conjugate prior $(\gamma_G, \nu_G, \alpha_G, \beta_G)$ over the total potential energy:

$$
\hat{E}(\mathbf{R}) \equiv \gamma_G(\mathbf{R}) = \sum_{i=1}^N \gamma_i(\mathbf{R}) + E_0, \qquad \mathbf{F}_i(\mathbf{R}) = -\nabla_{\mathbf{R}_i}\gamma_G(\mathbf{R})
$$

This guarantees a strictly conservative mean force field and yields closed-form structure-level aleatoric and epistemic energy variances in a single forward pass:

$$
\sigma_{\mathrm{al},G}^2 = \frac{\beta_G}{\alpha_G - 1}, \qquad \sigma_{\mathrm{ep},G}^2 = \frac{\beta_G}{\nu_G(\alpha_G - 1)}
$$

### 2. Hierarchical Atom-Wise Variance Allocation

An atomic allocation head maps invariant node features to normalized structure-wise softmax weights $w_i$ satisfying $\sum_{i=1}^N w_i = 1$, alongside uniform aleatoric weights $u_i = 1/N$. This decomposes the structure-level variances into atom-wise contributions:

$$
V_{i,\mathrm{alea}} = u_i\,\sigma_{\mathrm{al},G}^2, \qquad V_{i,\mathrm{epi}} = w_i\,\sigma_{\mathrm{ep},G}^2
$$

During training, $w_i$ is supervised by stop-gradient squared force residuals so that local error attribution does not perturb the conservative force branch.

### 3. Feature-Jacobian Force Covariance Coupled with Energy Evidence

To propagate scalar energy confidence into directional $3 \times 3$ atomic force uncertainties, EPIC couples a damped structure-level energy evidence scale with the regularized inverse Gram matrix of the last-layer invariant feature Jacobians:

$$
J_{h,i}(\mathbf{R}) = \frac{\partial \mathbf{H}(\mathbf{R})}{\partial \mathbf{R}_i}, \qquad \Sigma_F = \left(G_F + \lambda_{\mathrm{cov}}I\right)^{-1} = LL^\top
$$

$$
c_0(\mathbf{R}_{\mathrm{ref}}) = \left(\frac{\sigma_{\mathrm{ep},G}^2(\mathbf{R}_{\mathrm{ref}})}{N_{\mathrm{ref}}^2}\right)^{\eta}, \qquad \Sigma_{\mathbf{F}_i}(\mathbf{R}) = \tau_F^2\,c_0(\mathbf{R})^2\,J_{h,i}(\mathbf{R})\,\Sigma_F\,J_{h,i}(\mathbf{R})^\top
$$

### 4. Downstream Physical Observable Propagation

By sampling correlated weight perturbations from the Cholesky factor $L$ with $\mathbf{z}^{(k)} \sim \mathcal{N}(\mathbf{0}, I)$, EPIC defines conservatively perturbed potential energy surfaces and force realizations:

$$
\delta\mathbf{w}^{(k)} = \tau_F\,c_0(\mathbf{R}_{\mathrm{ref}})\,L\mathbf{z}^{(k)}
$$

$$
E_G^{(k)}(\mathbf{R}) = \gamma_G(\mathbf{R}) + \left[\mathbf{H}(\mathbf{R}) - \mathbf{H}_0\right]^\top \delta\mathbf{w}^{(k)}, \qquad \mathbf{F}_i^{(k)}(\mathbf{R}) = \mathbf{F}_i(\mathbf{R}) - J_{h,i}(\mathbf{R})\,\delta\mathbf{w}^{(k)}
$$

This enables self-consistent uncertainty propagation to:
- Real-space 3D atomic force uncertainty ellipsoids (exportable to VESTA CIF format);
- Volumetric equation-of-state (energy–volume) curves;
- Supercell harmonic phonon dispersions and vibrational density of states via `Phonopy`.

---

## Repository Structure

```text
EPIC/
├── mace/                              # Bundled MACE equivariant backbone with MACEEDL readout
├── uq_benchmark/                      # Core EPIC library
│   ├── hierarchical_epic.py           # Graph NIG readout, atomic softmax allocation & losses
│   ├── distributions.py               # Student-t NIG NLL, evidential regularizer & annealing
│   ├── training.py                    # Multi-GPU Accelerate training loop & checkpointing
│   ├── inference.py                   # Sharded inference & NPZ aggregation
│   ├── calibration.py                 # Post-hoc variance calibration routines
│   ├── s2c_covariance.py              # Feature-Jacobian covariance, 3x3 force UQ & perturbed PES
│   ├── s2c_ellipsoids.py              # 3x3 force covariance eigendecomposition & VESTA CIF export
│   ├── s2c_phonon.py                  # Supercell Monte Carlo harmonic phonon UQ with Phonopy
│   ├── s2c_eos.py                     # Volumetric Equation-of-State (E-V) uncertainty evaluation
│   └── tests/                         # Unit and integration smoke tests
├── scripts/                           # Standalone CLI workflow scripts
│   ├── prepare_s2c_covariance.py      # Accumulate calibration feature-Jacobian covariance
│   ├── run_s2c_force_uq.py            # Evaluate closed-form 3x3 force covariance across shards
│   ├── run_s2c_calibration.py         # Fit post-hoc affine variance calibration parameters
│   ├── run_s2c_ellipsoids.py          # Export 3D force uncertainty ellipsoids (VESTA CIF)
│   └── run_s2c_phonon.py              # Run correlated Monte Carlo harmonic phonon UQ
├── configs/                           # Configuration templates
└── docs/                              # Method-to-code mapping and technical notes
```

---

## Installation

Python **3.10+** and PyTorch **2.2+** are required. The modified MACE backbone is bundled under `mace/` (do not install an external `mace-torch` package in the same environment).

### Conda

```bash
git clone https://github.com/SuthPhy2Ai/EPIC.git
cd EPIC
conda env create -f environment.yml
conda activate epic-uq
```

### Pip / Virtualenv

```bash
git clone https://github.com/SuthPhy2Ai/EPIC.git
cd EPIC
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[all]"
```

---

## Quickstart

### 1. Run Smoke Tests

```bash
pytest -q
python uq_benchmark/tests/integration_mace_smoke.py
```

### 2. Prepare Dataset Split & Train EPIC

```bash
# Generate material-cluster-disjoint train/val/calibration/test split indices
epic-uq --config configs/epic_single_gpu.example.yaml make-split

# Single-GPU training
epic-uq --config configs/epic_single_gpu.example.yaml \
  train \
  --kind epic_hierarchical \
  --seed 42 \
  --lambda-max 0.001 \
  --output-dir runs/epic_hierarchical/seed42

# Multi-GPU training (via HuggingFace Accelerate)
accelerate launch --num_processes 8 -m uq_benchmark \
  --config configs/epic_mptrj_8gpu.yaml \
  train \
  --kind epic_hierarchical \
  --seed 42 \
  --lambda-max 0.001 \
  --output-dir runs/epic_hierarchical/seed42
```

### 3. Single-Pass Energy & NIG Inference

```bash
epic-uq --config configs/epic_mptrj_8gpu.yaml \
  infer-shard \
  --split calibration \
  --checkpoint runs/epic_hierarchical/seed42/checkpoint_best.pt \
  --output runs/predictions/calibration_point.npz \
  --device cuda
```

### 4. Feature-Jacobian Force Covariance & Calibration

```bash
# 1. Estimate regularized feature-Jacobian covariance on the calibration split
python scripts/prepare_s2c_covariance.py \
  --mode from-checkpoint \
  --config configs/epic_mptrj_8gpu.yaml \
  --checkpoint runs/epic_hierarchical/seed42/checkpoint_best.pt \
  --output-dir runs/s2c_covariance \
  --device cuda

# 2. Evaluate 3x3 force covariance on shards and aggregate
python scripts/run_s2c_force_uq.py \
  --mode worker \
  --config configs/epic_mptrj_8gpu.yaml \
  --checkpoint runs/epic_hierarchical/seed42/checkpoint_best.pt \
  --covariance-npz runs/s2c_covariance/s2c_covariance.npz \
  --test-force-npz runs/predictions/test_force.npz \
  --test-shard-dir runs/s2c_force_test/shards \
  --output-dir runs/s2c_force_test \
  --local-gpus 8 --world-size 8

python scripts/run_s2c_force_uq.py \
  --mode eval \
  --config configs/epic_mptrj_8gpu.yaml \
  --test-force-npz runs/predictions/test_force.npz \
  --test-shard-dir runs/s2c_force_test/shards \
  --output-dir runs/s2c_force_test \
  --world-size 8

# 3. Fit post-hoc affine variance calibration parameters
python scripts/run_s2c_calibration.py \
  --force-calibration runs/s2c_force_cal/force_calibration/hierarchical_epic_s2c/aggregated.npz \
  --force-test runs/s2c_force_test/force_test/hierarchical_epic_s2c/aggregated.npz \
  --energy-calibration runs/predictions/calibration_point.npz \
  --energy-test runs/predictions/test_point.npz \
  --output runs/s2c_calibration/calibration_metrics.json
```

### 5. 3D Force Uncertainty Ellipsoids & Phonon UQ Propagation

```bash
# Export 3D force uncertainty ellipsoids in VESTA CIF format
python scripts/run_s2c_ellipsoids.py \
  --checkpoint runs/epic_hierarchical/seed42/checkpoint_best.pt \
  --covariance-npz runs/s2c_covariance/s2c_covariance.npz \
  --structure-files path/to/crystal.cif \
  --output-dir runs/s2c_ellipsoids \
  --device cuda

# Propagate uncertainty to 2x2x2 supercell harmonic phonons via Phonopy
python scripts/run_s2c_phonon.py \
  --checkpoint runs/epic_hierarchical/seed42/checkpoint_best.pt \
  --covariance-npz runs/s2c_covariance/s2c_covariance.npz \
  --structure-file path/to/crystal.cif \
  --supercell 2 2 2 \
  --displacement 0.01 \
  --n-mc-samples 32 \
  --output-dir runs/s2c_phonon \
  --device cuda
```

---

## License

This project is released under the [MIT License](LICENSE).
