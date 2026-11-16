# Equation-to-Code Mapping and Mathematical Implementation Audit (`EPIC`)

> **Document Scope:** Complete equation-by-equation mapping from the EPIC manuscript (*EPIC: Differentiable Energy-Force Consistent Uncertainty Quantification and Propagation for Machine-Learned Interatomic Potentials*, Section 4 Methods **Eqs. 1–20** and Supplementary Information **Notes 1–11 / Eqs. S1–S7**) to the exact Python modules, classes, functions, and line ranges in this repository.

---

## 1. Master Index: Manuscript Equations to Code Implementation

| Manuscript Equation / Section | Mathematical Expression / Concept | Primary File & Line Range | Class / Function Name |
| :--- | :--- | :--- | :--- |
| **Eq. (1)** (`eq:mace-energy`) | $\hat{E}(\mathbf{R}) = \sum_{i=1}^{N} \hat{E}_i(\mathbf{R}) + E_0$ | `mace/modules/models.py:545-571, 614-620` | `MACEEDL.forward` |
| **Eq. (2)** (`eq:conservative-force`) | $\mathbf{F}_i = -\nabla_{\mathbf{R}_i}\hat{E}(\mathbf{R})$ | `mace/modules/models.py:622-636` | `get_outputs` (via `torch.autograd.grad`) |
| **Eq. (3)** (`eq:nig-prior`) | $p(\mu, \sigma^2) = \mathcal{N}(\mu \mid \gamma_G, \sigma^2/\nu_G)\operatorname{Inv-Gamma}(\sigma^2 \mid \alpha_G, \beta_G)$ | `uq_benchmark/distributions.py:9-16` | `nig_predictive_parameters` |
| **Eq. (4)** (`eq:uncertainty-decomposition`) | $\sigma_{\mathrm{al},G}^2 = \frac{\beta_G}{\alpha_G-1},\;\sigma_{\mathrm{ep},G}^2 = \frac{\beta_G}{\nu_G(\alpha_G-1)},\;\sigma_{\mathrm{tot},G}^2 = \sigma_{\mathrm{al},G}^2 + \sigma_{\mathrm{ep},G}^2$ | `uq_benchmark/hierarchical_epic.py:123-135`<br>`uq_benchmark/distributions.py:19-27` | `HierarchicalEPICReadout.hierarchical_parameters`<br>`nig_variances` |
| **Eq. (5)** (`eq:shared-hidden`) | $\tilde{\mathbf{h}}_i = \phi(W_{\mathrm{shared}}\mathbf{h}_i),\;\bar{\mathbf{h}} = \frac{1}{N}\sum_{i=1}^N \tilde{\mathbf{h}}_i$ | `uq_benchmark/hierarchical_epic.py:73-75, 90-96` | `HierarchicalEPICReadout._hidden`<br>`HierarchicalEPICReadout.hierarchical_parameters` |
| **Eq. (6)** (`eq:epic-heads`) | $\gamma_i = W_\gamma\tilde{\mathbf{h}}_i,\;\nu_G = \operatorname{softplus}(W_\nu\bar{\mathbf{h}})+\epsilon,\;\alpha_G = \operatorname{softplus}(W_\alpha\bar{\mathbf{h}})+1+\epsilon,\;\beta_G = \operatorname{softplus}(W_\beta\bar{\mathbf{h}})+\epsilon$ ($\epsilon = 10^{-6}$) | `uq_benchmark/hierarchical_epic.py:76-83, 98-110` | `HierarchicalEPICReadout.forward`<br>`HierarchicalEPICReadout.hierarchical_parameters` |
| **Eq. (7)** (`eq:nig-aggregation`) | $\hat{E}(\mathbf{R}) \equiv \gamma_G(\mathbf{R}) = \sum_{i=1}^N \gamma_i(\mathbf{R}) + E_0$ | `mace/modules/models.py:614-620`<br>`uq_benchmark/hierarchical_epic.py:155-167` | `MACEEDL.forward`<br>`HierarchicalAtomwiseEPIC.forward` |
| **Eq. (8)** (`eq:atomic-nig-allocation`) | $\alpha_i = \alpha_G,\;\beta_i = u_i\beta_G,\;\nu_i = \nu_G \frac{u_i}{w_i}$ ($u_i = 1/N,\;w_i = \operatorname{softmax}_G(\ell_i)$) | `uq_benchmark/hierarchical_epic.py:23-57, 111-121` | `segment_normalize`, `segment_softmax`<br>`HierarchicalEPICReadout.hierarchical_parameters` |
| **Eq. (9)** (`eq:atomic-variance-allocation`) | $V_{i,\mathrm{alea}} = u_i\sigma_{\mathrm{al},G}^2,\;V_{i,\mathrm{epi}} = w_i\sigma_{\mathrm{ep},G}^2$ ($\sum_i V_{i,\cdot} = \sigma_{\cdot,G}^2$) | `uq_benchmark/hierarchical_epic.py:127-144`<br>`uq_benchmark/tests/test_hierarchical_epic.py:39-64` | `HierarchicalEPICReadout.hierarchical_parameters` |
| **Eq. (10)** (`eq:nll-loss`) & **SI Note 1 (Eqs. S1–S5)** | $\mathcal{L}_{\mathrm{NLL}} = \frac{1}{2}\log(\frac{\pi}{\nu_G}) - \alpha_G\log\Omega_G + (\alpha_G+\frac{1}{2})\log[\nu_G(y-\gamma_G)^2+\Omega_G] + \log\Gamma(\alpha_G) - \log\Gamma(\alpha_G+\frac{1}{2})$ | `uq_benchmark/distributions.py:75-99` | `full_nig_nll_torch` |
| **Eq. (11)** (`eq:evidential-regularization`) & **SI Note 2 (Eq. S6)** | $\mathcal{L}_{\mathrm{reg}} = |y-\gamma_G|(2\nu_G+\alpha_G)$, $\lambda(t) = \lambda_{\max}\min((t+1)/5, 1)$ | `uq_benchmark/distributions.py:102-114, 163-166` | `full_nig_loss_torch`<br>`annealed_regularization` |
| **Eq. (12)** (`eq:alloc-loss`) | $\mathcal{L}_{\mathrm{alloc}} = -\sum_{i=1}^N \tilde{e}_i \log w_i,\;\mathbf{F}_i^{\mathrm{detach}} = \operatorname{stop\_grad}(-\nabla_{\mathbf{R}_i}\gamma_G(\mathbf{R}))$ | `uq_benchmark/hierarchical_epic.py:170-192` | `force_error_allocation_loss` |
| **Eq. (13)** (`eq:total-loss`) | $\mathcal{L} = \langle \mathcal{L}_{\mathrm{NLL}} + \lambda(t)\mathcal{L}_{\mathrm{reg}} + \lambda_{\mathrm{alloc}}\mathcal{L}_{\mathrm{alloc}}\rangle_{\mathcal{B}} + w_f\mathcal{L}_{\mathrm{F}}^{\mathrm{cond.\,Huber}}$ | `uq_benchmark/hierarchical_epic.py:195-225`<br>`uq_benchmark/training.py:158-184`<br>`mace/modules/loss.py:401-428` | `hierarchical_epic_energy_loss`<br>`compute_loss`<br>`ConditionalHuberForcesLoss` |
| **Eq. (14)** (`eq:sigma-f-gram`) | $G_F = \sum_{c\in\mathcal{D}_{\mathrm{cal}}}\sum_{i=1}^{N_c} J_{h,i}(\mathbf{R}_c)^\top J_{h,i}(\mathbf{R}_c),\;\Sigma_F = (G_F + \lambda_{\mathrm{cov}}I_D)^{-1} = LL^\top$ | `uq_benchmark/s2c_covariance.py:136-307`<br>`scripts/prepare_s2c_covariance.py` | `compute_feature_jacobian_batch`<br>`accumulate_force_jacobian_gram`<br>`build_s2c_covariance` |
| **Eq. (15)** (`eq:c0-scale`) | $c_0(\mathbf{R}_{\mathrm{ref}}) = (\sigma_{\mathrm{ep},G}^2(\mathbf{R}_{\mathrm{ref}})/N_{\mathrm{ref}}^2)^\eta$ ($\eta = 0.075,\;2\eta = 0.15$) | `uq_benchmark/s2c_covariance.py:56-82` | `compute_damped_evidence_scale` |
| **Eq. (16)** (`eq:mc-energy-sampling`) | $\delta\mathbf{w}^{(k)} = \tau_F c_0 L\mathbf{z}^{(k)},\;E_G^{(k)}(\mathbf{R}) = \gamma_G(\mathbf{R}) + [\mathbf{H}(\mathbf{R})-\mathbf{H}_0]^\top\delta\mathbf{w}^{(k)}$ | `uq_benchmark/s2c_covariance.py:398-487` | `draw_s2c_weight_perturbations`<br>`sample_s2c_perturbed_pes_and_forces` |
| **Eq. (17)** (`eq:mc-force`) | $\mathbf{F}_i^{(k)}(\mathbf{R}) = -\nabla_{\mathbf{R}_i} E_G^{(k)}(\mathbf{R}) = \mathbf{F}_i(\mathbf{R}) - J_{h,i}(\mathbf{R})\delta\mathbf{w}^{(k)}$ | `uq_benchmark/s2c_covariance.py:488-514` | `sample_s2c_perturbed_pes_and_forces` |
| **Eq. (18)** (`eq:closed-form-force-cov`) | $\Sigma_{\mathbf{F}_i}(\mathbf{R}) = \tau_F^2 c_0^2 J_{h,i}(\mathbf{R})\Sigma_F J_{h,i}(\mathbf{R})^\top,\;\mathrm{RMS}(\sigma_F) = \sqrt{\frac{1}{3N}\sum_i \operatorname{tr}(\Sigma_{\mathbf{F}_i})}$ | `uq_benchmark/s2c_covariance.py:310-395`<br>`scripts/run_s2c_force_uq.py` | `compute_s2c_atomic_force_covariances` |
| **Eq. (19)** (`eq:force-affine-cal`) & **SI Note 3** | $(\sigma_{F,i\alpha}^{\mathrm{cal}}(\mathbf{R}))^2 = (\Sigma_{\mathbf{F}_i}(\mathbf{R}))_{\alpha\alpha} + \sigma_{0,F}^2$ and $(\sigma_E^{\mathrm{cal}})^2 = \tau_E^2(\sigma_{\mathrm{tot},G}/N)^2 + \sigma_{0,E}^2$ | `uq_benchmark/calibration.py:26-57`<br>`uq_benchmark/s2c_covariance.py:43-54, 517-590`<br>`scripts/run_s2c_calibration.py` | `fit_gaussian_variance_map`<br>`fit_s2c_affine_calibration` |
| **Eq. (20)** (`eq:prescreen-score`) & **Sec. 2.1** | $S_{\mathrm{pre}} = \sum_j w_j x_j'$ and active-learning DFT review trigger | `uq_benchmark/s2c_covariance.py:593-668` | `compute_prescreen_score`<br>`compute_dft_review_trigger_score` |
| **Sec. 2.2, 4.6 & SI Note 9** | Volumetric Equation-of-State (EOS) $E(V/V_0)$, $\sigma_{\mathrm{ep}}(V/V_0)$, and offset $\Delta V/V_0$ | `uq_benchmark/s2c_eos.py:22-157` | `analyze_eos_uncertainty_profile`<br>`evaluate_eos_uncertainty_curve` |
| **Sec. 2.3, 4.6 & SI Note 8** | $3\times 3$ Force Ellipsoids $\Sigma_{\mathbf{F}_i} = U_i\operatorname{diag}(\lambda_{i,1},\lambda_{i,2},\lambda_{i,3})U_i^\top$, Wyckoff triaxiality, VESTA CIF | `uq_benchmark/s2c_ellipsoids.py:27-266`<br>`scripts/run_s2c_ellipsoids.py` | `eigendecompose_atomic_force_covariances`<br>`export_anisotropic_ellipsoid_cif`<br>`evaluate_structure_ellipsoids` |
| **Sec. 2.4, 4.6 & SI Note 10 (Eq. S7)** | $2\times 2\times 2$ Supercell Phonopy MC UQ, Reference-Force Subtraction, Stability Classes, Spectral Jacobian $(2\sqrt{\omega_{\mathrm{pt}}^2+\epsilon^2})^{-1}$ | `uq_benchmark/s2c_phonon.py:28-370`<br>`scripts/run_s2c_phonon.py` | `run_s2c_phonon_dispersion`<br>`classify_phonon_stability`<br>`compute_spectral_jacobian_perturbation` |

---

## 2. Detailed Equation-by-Equation Implementation Audit

### 2.1 Equivariant Message Passing Backbone (Section 4.1, Eqs. 1–2)

#### Equivariant Node Features $\mathbf{h}_i$
In `mace/modules/models.py:580-607`, the `MACEEDL` backbone executes $T = 2$ equivariant interaction and body-ordered tensor product layers ($\nu = 3$, hidden irreps $64\times 0e + 64\times 1o$, spherical harmonics up to $L_{\max} = 3$, radial cutoff $r_{\max} = 5.5\text{ \AA}$) and extracts the final-layer scalar invariant ($L = 0$) features $\mathbf{h}_i \in \mathbb{R}^{64}$ (`node_feats_final = node_feats_list[-1]`).

#### Equation (1) — Structure-Level Mean Potential Energy:
$$
\hat{E}(\mathbf{R}) = \sum_{i=1}^{N} \hat{E}_i(\mathbf{R}) + E_0
$$
- **Code Location:** `mace/modules/models.py:545-571, 614-620`
- **Implementation Details:**
  ```python
  gamma_total = scatter_sum(
      src=gamma_node, index=data["batch"], dim=0, dim_size=num_graphs
  )
  total_energy = e0 + pair_energy + gamma_total
  ```
  In the formal MPtrj training configuration (`uq_benchmark/modeling.py:225`), `atomic_energies` ($E_0$) are initialized to zero and `pair_repulsion` is disabled, so $\hat{E}(\mathbf{R}) \equiv \gamma_G(\mathbf{R}) = \sum_{i=1}^N \gamma_i(\mathbf{R})$ holds identically.

#### Equation (2) — Conservative Point Forces via Automatic Differentiation:
$$
\mathbf{F}_i = -\nabla_{\mathbf{R}_i}\hat{E}(\mathbf{R})
$$
- **Code Location:** `mace/modules/models.py:622-636`
- **Implementation Details:** `get_outputs(energy=total_energy, positions=positions, ...)` differentiates the scalar graph energy with respect to atomic Cartesian coordinates `positions`, guaranteeing zero curl ($\nabla \times \mathbf{F} = \mathbf{0}$) and exact energy-force consistency at the mean level.

---

### 2.2 Evidential Deep Learning Preliminaries (Section 4.2, Eqs. 3–4)

#### Equation (3) — Conjugate Normal-Inverse-Gamma (NIG) Prior:
$$
p(\mu, \sigma^2) = \mathcal{N}\!\left(\mu \;\middle|\; \gamma_G, \frac{\sigma^2}{\nu_G}\right)\operatorname{Inv-Gamma}(\sigma^2 \mid \alpha_G, \beta_G)
$$
- **Code Location:** `uq_benchmark/distributions.py:9-16` (`nig_predictive_parameters`)
- **Implementation Details:** Integrating out $(\mu, \sigma^2)$ (derived step-by-step in **Supplementary Note 1, Eqs. S1–S5**) yields a Student-$t$ marginal distribution with $2\alpha_G$ degrees of freedom, location $\gamma_G$, and scale $\sqrt{\beta_G(1+\nu_G)/(\alpha_G\nu_G)}$.

#### Equation (4) — Closed-Form Aleatoric, Epistemic, and Total Energy Variances:
$$
\sigma_{\mathrm{al},G}^2 = \mathbb{E}[\sigma^2] = \frac{\beta_G}{\alpha_G - 1}, \qquad
\sigma_{\mathrm{ep},G}^2 = \operatorname{Var}[\mu] = \frac{\beta_G}{\nu_G(\alpha_G - 1)}, \qquad
\sigma_{\mathrm{tot},G}^2 = \sigma_{\mathrm{al},G}^2 + \sigma_{\mathrm{ep},G}^2
$$
- **Code Location:** `uq_benchmark/hierarchical_epic.py:123-135` and `uq_benchmark/distributions.py:19-27` (`nig_variances`)
- **Implementation Details:**
  ```python
  graph_aleatoric = graph_beta / (graph_alpha - 1.0)
  graph_epistemic = graph_beta / (graph_nu * (graph_alpha - 1.0))
  "energy_aleatoric_variance": graph_aleatoric,
  "energy_epistemic_variance": graph_epistemic,
  "energy_total_variance": graph_aleatoric + graph_epistemic,
  ```
  Consequently, the epistemic-to-aleatoric standard-deviation ratio satisfies $\sigma_{\mathrm{ep},G}/\sigma_{\mathrm{al},G} = \nu_G^{-1/2}$.

---

### 2.3 Energy-Level Evidential Formulation & Hierarchical Allocation (Section 4.4, Eqs. 5–9)

#### Equation (5) — Shared Hidden Representation and Structure-Wise Mean Pooling:
$$
\tilde{\mathbf{h}}_i = \phi\!\left(W_{\mathrm{shared}}\mathbf{h}_i\right), \qquad
\bar{\mathbf{h}} = \frac{1}{N}\sum_{i=1}^{N}\tilde{\mathbf{h}}_i
$$
- **Code Location:** `uq_benchmark/hierarchical_epic.py:73-75, 90-96` (`HierarchicalEPICReadout._hidden` and `hierarchical_parameters`)
- **Implementation Details:**
  ```python
  def _hidden(self, node_features: torch.Tensor) -> torch.Tensor:
      return self.base.non_linearity(self.base.linear_1(node_features))

  hidden = self._hidden(node_features)
  pooled = hidden.new_zeros((num_graphs, hidden.shape[-1]))
  pooled.index_add_(0, batch, hidden)
  counts = torch.bincount(batch, minlength=num_graphs).to(hidden.dtype)
  pooled = pooled / counts[:, None]
  ```
  Here $\phi$ (`self.base.non_linearity`) is the second-moment-normalized SiLU activation on the $D = 64$ invariant channel representation.

#### Equation (6) — EPIC Readout Heads:
$$
\gamma_i = W_\gamma\tilde{\mathbf{h}}_i, \quad
\nu_G = \operatorname{softplus}(W_\nu\bar{\mathbf{h}}) + \epsilon, \quad
\alpha_G = \operatorname{softplus}(W_\alpha\bar{\mathbf{h}}) + 1 + \epsilon, \quad
\beta_G = \operatorname{softplus}(W_\beta\bar{\mathbf{h}}) + \epsilon
$$
- **Code Location:** `uq_benchmark/hierarchical_epic.py:76-83, 98-110`
- **Implementation Details:** With $\epsilon = 10^{-6}$ (`self.epsilon = 1e-6`), the parameterization strictly enforces the NIG positivity domain $\nu_G > 0$, $\alpha_G > 1$, and $\beta_G > 0$.

#### Equation (7) — Structure-Level Mean Energy Aggregation:
$$
\hat{E}(\mathbf{R}) \equiv \gamma_G(\mathbf{R}) = \sum_{i=1}^{N}\gamma_i(\mathbf{R}) + E_0
$$
- **Code Location:** `mace/modules/models.py:614-620` and `uq_benchmark/hierarchical_epic.py:155-167` (`HierarchicalAtomwiseEPIC.forward`)

#### Equation (8) & Equation (9) — Shared-Scale Atom-Wise NIG Marginals & Variance Allocation:
$$
\alpha_i = \alpha_G, \qquad \beta_i = u_i\beta_G, \qquad \nu_i = \nu_G\frac{u_i}{w_i}
$$
$$
V_{i,\mathrm{alea}} = u_i\frac{\beta_G}{\alpha_G - 1} = u_i\,\sigma_{\mathrm{al},G}^2, \qquad
V_{i,\mathrm{epi}} = w_i\frac{\beta_G}{\nu_G(\alpha_G - 1)} = w_i\,\sigma_{\mathrm{ep},G}^2
$$
- **Code Location:** `uq_benchmark/hierarchical_epic.py:23-57, 111-144`
- **Implementation Details:**
  ```python
  epistemic_logits = self.epistemic_allocation_head(hidden).squeeze(-1)
  epistemic_weight = segment_softmax(epistemic_logits, batch, num_graphs)
  aleatoric_weight = 1.0 / counts[batch]

  atomic_alpha = graph_alpha[batch]
  atomic_beta = aleatoric_weight * graph_beta[batch]
  atomic_nu = graph_nu[batch] * aleatoric_weight / epistemic_weight

  atomic_aleatoric = aleatoric_weight * graph_aleatoric[batch]
  atomic_epistemic = epistemic_weight * graph_epistemic[batch]
  ```
  `segment_softmax` (`uq_benchmark/hierarchical_epic.py:38-57`) applies a per-structure softmax with a floor of `minimum = 1e-8` followed by `segment_normalize` so $\sum_{i=1}^N w_i = 1$ and $\sum_{i=1}^N u_i = 1$ hold identically. As verified in `uq_benchmark/tests/test_hierarchical_epic.py:39-64`, $\sum_{i=1}^N V_{i,\mathrm{alea}} = \sigma_{\mathrm{al},G}^2$ and $\sum_{i=1}^N V_{i,\mathrm{epi}} = \sigma_{\mathrm{ep},G}^2$.

---

### 2.4 Training Objective and Regularization (Section 4.5, Eqs. 10–13 & SI Notes 1–2)

#### Equation (10) & Supplementary Note 1 (Eqs. S1–S5) — Exact NIG Marginal Negative Log-Likelihood:
$$
\mathcal{L}_{\mathrm{NLL}} = \frac{1}{2}\log\!\left(\frac{\pi}{\nu_G}\right) - \alpha_G\log\Omega_G + \left(\alpha_G + \frac{1}{2}\right)\log\!\left[\nu_G(y - \gamma_G)^2 + \Omega_G\right] + \log\Gamma(\alpha_G) - \log\Gamma\!\left(\alpha_G + \frac{1}{2}\right)
$$
where $\Omega_G = 2\beta_G(1 + \nu_G)$.
- **Code Location:** `uq_benchmark/distributions.py:75-99` (`full_nig_nll_torch`)
- **Implementation Details:**
  ```python
  two_blambda = 2.0 * beta * (1.0 + nu)
  nll = (
      0.5 * torch.log(torch.pi / nu)
      - alpha * torch.log(two_blambda)
      + (alpha + 0.5) * torch.log(nu * (target - gamma).square() + two_blambda)
      + torch.lgamma(alpha)
      - torch.lgamma(alpha + 0.5)
  )
  ```

#### Equation (11) & Supplementary Note 2 (Eq. S6) — Evidential Regularization & Annealing:
$$
\mathcal{L}_{\mathrm{reg}} = |y - \gamma_G|(2\nu_G + \alpha_G), \qquad
\lambda(t) = \lambda_{\max}\min\!\left(\frac{t + 1}{T_{\mathrm{anneal}}}, 1\right)
$$
- **Code Location:** `uq_benchmark/distributions.py:102-114, 163-166` (`full_nig_loss_torch` and `annealed_regularization`)
- **Implementation Details:** With $\lambda_{\max} = 10^{-3}$ and $T_{\mathrm{anneal}} = 5$ epochs (`configs/epic_mptrj_8gpu.yaml:38-39`), $\lambda(t)$ ramps linearly through $\{0.2, 0.4, 0.6, 0.8, 1.0\}\times 10^{-3}$ over epochs $0\text{--}4$ (`uq_benchmark/training.py:443-446`).

#### Equation (12) — Detached Force-Residual Atomic Allocation Loss:
$$
\mathcal{L}_{\mathrm{alloc}} = -\sum_{i=1}^{N} \tilde{e}_i \log w_i, \qquad
\mathbf{F}_i^{\mathrm{detach}} = \operatorname{stop\_grad}\!\left(-\nabla_{\mathbf{R}_i}\gamma_G(\mathbf{R})\right), \qquad
\tilde{e}_i = \frac{\|\mathbf{F}_i^{\mathrm{detach}} - \mathbf{F}_i^{\mathrm{DFT}}\|_2^2 + \delta}{\sum_{j=1}^N \left(\|\mathbf{F}_j^{\mathrm{detach}} - \mathbf{F}_j^{\mathrm{DFT}}\|_2^2 + \delta\right)}
$$
- **Code Location:** `uq_benchmark/hierarchical_epic.py:170-192` (`force_error_allocation_loss`)
- **Implementation Details:**
  ```python
  error_score = (
      (force_prediction.detach() - force_target.detach()).square().sum(dim=1)
      + float(floor)
  )
  target_weight = segment_normalize(error_score, batch, num_graphs)
  tiny = torch.finfo(epistemic_weight.dtype).tiny
  per_atom = -target_weight * torch.log(epistemic_weight.clamp_min(tiny))
  per_graph = per_atom.new_zeros(num_graphs)
  per_graph.index_add_(0, batch, per_atom)
  return per_graph.mean()
  ```
  Here `floor` is $\delta = 10^{-6}$ (`hierarchical_force_error_floor: 1.0e-6` in `configs/epic_mptrj_8gpu.yaml:46`). Calling `.detach()` on `force_prediction` ensures that allocation supervision never perturbs the conservative force gradients.

#### Equation (13) — Complete Training Objective:
$$
\mathcal{L} = \left\langle \mathcal{L}_{\mathrm{NLL}} + \lambda(t)\mathcal{L}_{\mathrm{reg}} + \lambda_{\mathrm{alloc}}\mathcal{L}_{\mathrm{alloc}} \right\rangle_{\mathcal{B}} + w_f\mathcal{L}_{\mathrm{F}}^{\mathrm{cond.\,Huber}}
$$
- **Code Location:** `uq_benchmark/hierarchical_epic.py:195-225` (`hierarchical_epic_energy_loss`), `uq_benchmark/training.py:158-184` (`compute_loss`), and `mace/modules/loss.py:401-428` (`ConditionalHuberForcesLoss`)
- **Implementation Details:** Uses $\lambda_{\mathrm{alloc}} = 0.1$, $w_f = 2000.0$, and a 4-tier conditional Huber force loss with base threshold $\delta_F = 0.01\text{ eV/\AA}$ scaled by $\{1.0, 0.7, 0.4, 0.1\}$ for reference atomic force magnitudes in $[0, 100), [100, 200), [200, 300), [300, \infty)\text{ eV/\AA}$.

---

### 2.5 Energy-Propagated Force Covariance & Downstream Propagation — Scheme 2c (`hepic_s2c_nig015`, Section 4.6, Eqs. 14–20)

#### Equation (14) — Feature-Jacobian Gram Matrix $G_F$ and Regularized Inverse Covariance $\Sigma_F$:
$$
G_F = \sum_{c\in\mathcal{D}_{\mathrm{cal}}}\sum_{i=1}^{N_c} J_{h,i}(\mathbf{R}_c)^\top J_{h,i}(\mathbf{R}_c), \qquad
\Sigma_F = \left(G_F + \lambda_{\mathrm{cov}}I_D\right)^{-1} = LL^\top \in \mathbb{R}^{D\times D}
$$
with $D = 64$, $J_{h,i}(\mathbf{R}) = \partial\mathbf{H}(\mathbf{R})/\partial\mathbf{R}_i \in \mathbb{R}^{3\times 64}$, $\mathbf{H}(\mathbf{R}) = \sum_{i=1}^N \tilde{\mathbf{h}}_i(\mathbf{R})$, and $\lambda_{\mathrm{cov}} = 10^{-3}\operatorname{tr}(G_F)/D$.
- **Code Location:** `uq_benchmark/s2c_covariance.py:104-307` (`forward_pooled_hidden`, `compute_feature_jacobian_batch`, `accumulate_force_jacobian_gram`, `build_s2c_covariance`) and `scripts/prepare_s2c_covariance.py`
- **Implementation Note on Normalization:** Whether $G_F$ is accumulated as the unnormalized sum $\sum_{c\in\mathcal{D}_{\mathrm{cal}}}\sum_{i=1}^{N_c} J_{h,i}^\top J_{h,i}$ (`normalize_by_structures=False`, matching the frozen calibration archive where $\operatorname{tr}(G_F) = 8.4024\times 10^7$ and $\lambda_{\mathrm{cov}} = 1312.87$) or divided by $N_{\mathrm{cal}}$ (`normalize_by_structures=True`), $\Sigma_F$ differs only by the constant scalar factor $N_{\mathrm{cal}}$, which is absorbed identically into the fitted multiplicative calibration parameter $a = \tau_F^2$.
- **Spectral Anatomy (Supplementary Note 8, Fig. S8a):** `build_s2c_covariance` (`uq_benchmark/s2c_covariance.py:288-306`) also computes the effective participation ratios $D_{\mathrm{eff}}(G_F) = (\sum_m \mu_m)^2/\sum_m \mu_m^2 = 2.63$ and $D_{\mathrm{eff}}(\Sigma_F) = 25.56$ ($9.7\times$ spectral subspace expansion).

#### Equation (15) — Damped Structure-Level Energy-Evidence Scale $c_0(\mathbf{R}_{\mathrm{ref}})$:
$$
c_0(\mathbf{R}_{\mathrm{ref}}) = \left(\frac{\sigma_{\mathrm{ep},G}^2(\mathbf{R}_{\mathrm{ref}})}{N_{\mathrm{ref}}^2}\right)^{\eta}, \qquad \eta = 0.075 \quad (2\eta = 0.15)
$$
- **Code Location:** `uq_benchmark/s2c_covariance.py:56-82` (`compute_damped_evidence_scale`)
- **Implementation Details:** Computes the per-atom epistemic variance $\sigma_{\mathrm{ep},G}^2(\mathbf{R}_{\mathrm{ref}})/N_{\mathrm{ref}}^2$ in `float64` without $\alpha_G$ clipping and raises it to $\eta = 0.075$ (so $c_0^2 = (\sigma_{\mathrm{ep},G}^2/N_{\mathrm{ref}}^2)^{0.15}$ in the force covariance).

#### Equation (16) & Equation (17) — Perturbed Scalar PES $E_G^{(k)}(\mathbf{R})$ and Conservative Force Realizations $\mathbf{F}_i^{(k)}(\mathbf{R})$:
$$
\delta\mathbf{w}^{(k)} = \tau_F\,c_0 L\mathbf{z}^{(k)}, \qquad
E_G^{(k)}(\mathbf{R}) = \gamma_G(\mathbf{R}) + \left[\mathbf{H}(\mathbf{R}) - \mathbf{H}_0\right]^\top \delta\mathbf{w}^{(k)}
$$
$$
\mathbf{F}_i^{(k)}(\mathbf{R}) = -\nabla_{\mathbf{R}_i} E_G^{(k)}(\mathbf{R}) = \mathbf{F}_i(\mathbf{R}) - J_{h,i}(\mathbf{R})\,\delta\mathbf{w}^{(k)}
$$
- **Code Location:** `uq_benchmark/s2c_covariance.py:398-514` (`draw_s2c_weight_perturbations` and `sample_s2c_perturbed_pes_and_forces`)
- **Implementation Details:**
  ```python
  mean_energy = output["energy"].sum()
  perturbations = (hidden_sum - anchor) @ weights_t.T
  sampled_energies = (mean_energy.detach() + perturbations.detach()).cpu().numpy()

  mean_force = -torch.autograd.grad(mean_energy, batch.positions, retain_graph=True)[0]
  mean_force_detached = mean_force.detach()

  for idx, pert in enumerate(perturbations.unbind()):
      grad_pert = torch.autograd.grad(
          pert, batch.positions, retain_graph=(idx + 1 < len(weights_t)),
          create_graph=False, allow_unused=False,
      )[0]
      force_realizations.append((mean_force_detached - grad_pert.detach()).cpu().numpy())
  ```
  Because $\delta\mathbf{w}^{(k)}$, $c_0(\mathbf{R}_{\mathrm{ref}})$, and $\mathbf{H}_0 = \mathbf{H}(\mathbf{R}_{\mathrm{ref}})$ are held frozen across displaced supercell configurations, each realization $E_G^{(k)}(\mathbf{R})$ defines a smooth, conservative perturbed potential energy surface whose exact spatial derivatives generate $\mathbf{F}_i^{(k)}(\mathbf{R})$ and perturbed harmonic force constants $\Phi_{ij}^{\alpha\beta,(k)}$.

#### Equation (18) — Closed-Form $3\times 3$ Atomic Force Covariance Tensor $\Sigma_{\mathbf{F}_i}(\mathbf{R})$:
$$
\Sigma_{\mathbf{F}_i}(\mathbf{R}) = \operatorname{Cov}\!\left[\mathbf{F}_i^{(k)}(\mathbf{R})\right] = \tau_F^2\,c_0^2\,J_{h,i}(\mathbf{R})\,\Sigma_F\,J_{h,i}(\mathbf{R})^\top, \qquad
\mathrm{RMS}(\sigma_F) = \sqrt{\frac{1}{3N}\sum_{i=1}^{N}\operatorname{tr}\!\left(\Sigma_{\mathbf{F}_i}(\mathbf{R})\right)}
$$
- **Code Location:** `uq_benchmark/s2c_covariance.py:310-395` (`compute_s2c_atomic_force_covariances`) and `scripts/run_s2c_force_uq.py`
- **Implementation Details:** Evaluates `cov_geom = np.einsum("nai,ij,nbj->nab", j_blocks, sigma_f_arr, j_blocks)`, symmetrizes `0.5 * (cov_geom + cov_geom.transpose(0, 2, 1))`, and scales by $\tau_F^2 c_0^2$.

#### Equation (19) & Supplementary Note 3 — Post-Hoc Affine Variance Recalibration:
$$
\left(\sigma_{F,i\alpha}^{\mathrm{cal}}(\mathbf{R})\right)^2 = \left(\Sigma_{\mathbf{F}_i}(\mathbf{R})\right)_{\alpha\alpha} + \sigma_{0,F}^2 = \tau_F^2\,c_0^2\left(J_{h,i}(\mathbf{R})\,\Sigma_F\,J_{h,i}(\mathbf{R})^\top\right)_{\alpha\alpha} + \sigma_{0,F}^2
$$
$$
\left(\sigma_E^{\mathrm{cal}}\right)^2 = \tau_E^2\left(\frac{\sigma_{\mathrm{tot},G}}{N}\right)^2 + \sigma_{0,E}^2
$$
- **Code Location:** `uq_benchmark/calibration.py:26-57` (`fit_gaussian_variance_map`, `apply_variance_map`), `uq_benchmark/s2c_covariance.py:43-54, 517-590` (`fit_s2c_affine_calibration`), `scripts/run_s2c_calibration.py`, and `configs/s2c_calibration_reference.json`
- **Fitted Calibration Constants on MPtrj ($79{,}025$ Calibration Structures):**
  - Force components ($7{,}501{,}668$ calibration components): $\tau_F = 47.423172$ ($a = \tau_F^2 = 2248.9572$), $\sigma_{0,F} = 0.0102403\text{ eV/\AA}$ ($b = \sigma_{0,F}^2 = 1.048645\times 10^{-4}\text{ (eV/\AA)}^2$).
  - Per-atom total energy: $\tau_E = 1.007753$, $\sigma_{0,E} = 0.02228\text{ eV/atom}$ (epistemic-only energy fit: $\tau_{E,\mathrm{ep}} = 0.5698$, $\sigma_{0,E,\mathrm{ep}} = 0.008023\text{ eV/atom}$).

#### Equation (20) & Section 2.1 — Candidate Prescreen Score & Active-Learning DFT Trigger:
$$
S_{\mathrm{pre}} = \sum_j w_j x_j', \qquad x_j' = \frac{x_j - \min(x_j)}{\max(x_j) - \min(x_j)}
$$
- **Code Location:** `uq_benchmark/s2c_covariance.py:593-668` (`compute_prescreen_score` and `compute_dft_review_trigger_score`)
- **Implementation Details:** Supports both min-max normalization (`normalization="minmax"`, Eq. 20) and percentile-rank normalization (`normalization="percentile"`, used in the Section 2.1 active-learning DFT review prioritization trigger with weights $(0.5, 0.2, 0.3)$ on $\mathrm{RMS}(\sigma_F)$, $\sigma_{\mathrm{ep},G}/N$, and $\max_i\|\mathbf{F}_i\|_2$).

---

## 3. Downstream Physical Observables & Supplementary Notes Mapping

### 3.1 Real-Space 3D Force Uncertainty Ellipsoids & Wyckoff Symmetry Rules (Section 2.3, Section 4.6, SI Note 8)
- **Code Location:** `uq_benchmark/s2c_ellipsoids.py:27-266` and `scripts/run_s2c_ellipsoids.py`
- **Implemented Quantities:**
  - `eigendecompose_atomic_force_covariances` (`s2c_ellipsoids.py:27-90`): Computes eigenvalues $0 \le \lambda_{i,1} \le \lambda_{i,2} \le \lambda_{i,3}$, principal standard-deviation semi-axes $\sigma_{\min} \le \sigma_{\mathrm{mid}} \le \sigma_{\max}$ ($\sqrt{\lambda_{i,m}}$), eigenvectors $U_i$, Zingg triaxiality ratios ($\sigma_{\min}/\sigma_{\mathrm{mid}}$ and $\sigma_{\mathrm{mid}}/\sigma_{\max}$), 3D effective participation rank $d_{\mathrm{eff},i} = (\sum_{m=1}^3 \lambda_{i,m})^2 / \sum_{m=1}^3 \lambda_{i,m}^2$, and fractional anisotropy $\mathrm{FA}$.
  - `export_anisotropic_ellipsoid_cif` (`s2c_ellipsoids.py:93-187`): Exports crystallographic CIF files containing `_atom_site_aniso_U_11..U_12` tensors for VESTA rendering, annotating exact physical magnitudes $\mathrm{RMS}(\sigma_F)$ and $\max_i\sqrt{\lambda_{i,3}}$ in $\text{eV/\AA}$ in the CIF header.
  - `evaluate_structure_ellipsoids` (`s2c_ellipsoids.py:190-266`): Computes both closed-form analytical and $K$-sample Monte Carlo force covariance tensors and verifies Wishart $K^{-1/2}$ relative Frobenius convergence.

### 3.2 Volumetric Equation-of-State (EOS) Topography (Section 2.2, Section 4.6, SI Note 9)
- **Code Location:** `uq_benchmark/s2c_eos.py:22-157`
- **Implemented Quantities:**
  - `evaluate_eos_uncertainty_curve` (`s2c_eos.py:107-157`): Scales crystal lattices isotropically across $V/V_0 \in [0.85, 1.15]$ ($s = (V/V_0)^{1/3}$) and evaluates $\gamma_G(V)/N$, $\sigma_{\mathrm{ep},G}(V)/N$, $\sigma_{\mathrm{al},G}(V)/N$, and $\sigma_{\mathrm{tot},G}(V)/N$.
  - `analyze_eos_uncertainty_profile` (`s2c_eos.py:22-104`): Extracts physical equilibrium volume $V_0 = \arg\min_V \gamma_G(V)$, minimum-epistemic-uncertainty volume $V_{\min}^{\mathrm{epist}} = \arg\min_V \sigma_{\mathrm{ep},G}(V)$, relative volumetric offset $\Delta V/V_0 = (V_{\min}^{\mathrm{epist}} - V_0)/V_0$, volumetric confidence regime (`compression_shifted` $<-3\%$, `equilibrium_aligned` $[-3\%, +3\%]$, `expansion_shifted` $>+3\%$), equilibrium stiffness curvature $\partial^2(E/N)/\partial(V/V_0)^2|_{V_0}$, and tensile vs. compressive epistemic steepness $\langle|\partial\sigma_{\mathrm{ep}}/\partial(V/V_0)|\rangle$.

### 3.3 Harmonic Phonon Dispersions, Stability Screening & Spectral Perturbation Theory (Section 2.4, Section 4.6, SI Note 10)
- **Code Location:** `uq_benchmark/s2c_phonon.py:28-370` and `scripts/run_s2c_phonon.py`
- **Implemented Quantities:**
  - `run_s2c_phonon_dispersion` (`s2c_phonon.py:201-370`):
    1. Evaluates reference-cell NIG parameters and damped scale $c_0(\mathbf{R}_{\mathrm{ref}})$ once on the primitive/unit cell (`evaluate_reference_structure_properties`, lines 155–187).
    2. Generates $2\times 2\times 2$ supercells with $0.01\text{ \AA}$ finite atomic displacements in `Phonopy` and high-symmetry Brillouin-zone paths via `Seekpath` (lines 255–270).
    3. Evaluates undisplaced equilibrium supercell forces $\mathbf{F}_{\mathrm{ref}}^{(k)}$ and displaced supercell forces $\mathbf{F}_{\mathrm{disp}}^{(k)}$ under frozen $\delta\mathbf{w}^{(k)}$ and anchor $\mathbf{H}_0$, then subtracts the undisplaced reference forces:
       $$
       \Delta\mathbf{F}^{(k)} = \mathbf{F}_{\mathrm{disp}}^{(k)} - \mathbf{F}_{\mathrm{ref}}^{(k)}
       $$
       (`s2c_phonon.py:296-303`), canceling static background force fluctuations by $>10^5\times$ and preserving the supercell acoustic sum rule (Supplementary Note 10, Fig. S10c).
  - `classify_phonon_stability` (`s2c_phonon.py:28-109`): Evaluates $\omega_{\min}^{\mathrm{pt}}$, 5th-percentile MC minimum $\omega_{\min,q05}$, and stable fraction $P_{\mathrm{stable}} = \frac{1}{K}\sum_{k=1}^K \mathbb{I}(\omega_{\min}^{(k)} \ge \omega_{\mathrm{tol}})$ with $\omega_{\mathrm{tol}} = -1.0\text{ THz}$, assigning each crystal to `robustly_stable`, `marginal` (high-suspicion soft-mode candidate), or `unstable`.
  - `compute_spectral_jacobian_perturbation` (`s2c_phonon.py:112-152`): Implements **Supplementary Note 10, Eq. (S7)**:
    $$
    \frac{\sigma_\omega(\mathbf{q},\nu)}{\sigma_D(\mathbf{q},\nu)} \approx \frac{1}{2\sqrt{\omega_{\mathrm{pt}}(\mathbf{q},\nu)^2 + \epsilon^2}} \qquad (\epsilon = 0.038\text{ THz})
    $$
    quantifying the $>500\times$ inverse-frequency amplification that converts localized force uncertainty into large imaginary-frequency excursions along soft lattice modes.
