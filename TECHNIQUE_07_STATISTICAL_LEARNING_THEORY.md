# TECHNIQUE_07_STATISTICAL_LEARNING_THEORY.md — Statistical Learning Theory

> ## CRITICAL: NEVER OMIT, SIMPLIFY, OR WEAKEN
>
> Claude Code must NEVER:
> - Omit features, techniques, or subsystems to fit within a single session
> - Simplify implementations to make them "easier" or "more practical"
> - Replace a specified technique with a simpler alternative without explicit approval
> - Skip sections because they seem redundant or low-priority
> - Truncate implementations to keep diffs small — instead, split across commits
>
> If a task is too large for one session: split it into sub-tasks across multiple
> sessions. Document what remains in PROGRESS.md and pick it up next session.
> The project must contain the FULL depth of every technique specified.
> "Good enough" is not good enough. Every feature exists at maximum fidelity.

> **Purpose**: Feed this to Claude Code after the main phases and earlier techniques
> are complete. Implements rigorous uncertainty quantification, prediction guarantees,
> and adaptive learning: conformal prediction, PAC-Bayes bounds, online learning,
> GPs, ensemble methods, hierarchical Bayes, drift detection, causal inference,
> information theory, calibration, fairness, and production monitoring.
>
> **How to use**: Tell Claude Code: "Read TECHNIQUE_07_STATISTICAL_LEARNING_THEORY.md
> and implement incrementally, starting from SLT-1."
>
> **This is 12 sub-domains. Implement ALL of them. Do not skip any.**

---

## Key Papers (Referenced Throughout)

- Vovk, Gammerman & Shafer (2005). *Algorithmic Learning in a Random World* — Conformal prediction foundations
- Barber, Candès, Ramdas & Tibshirani (2019). "Jackknife+." arXiv:1905.02928
- Romano, Patterson & Candès (2019). "CQR." arXiv:1905.03222
- Gibbs & Candès (2021). "Adaptive Conformal Inference." arXiv:2106.00170
- Xu & Xie (2021). "EnbPI." arXiv:2010.09107
- Tibshirani et al. (2019). "Weighted Conformal." arXiv:1904.06019
- Dziugaite & Roy (2017). "Non-vacuous PAC-Bayes." arXiv:1703.11008
- Pérez-Ortiz et al. (2021). "Tighter PAC-Bayes." arXiv:2007.12911
- Catoni (2007). "PAC-Bayes bounds." arXiv:0712.0248
- Rodríguez-Gálvez et al. (2024). "Strengthened Catoni." arXiv:2306.12214
- Alquier (2024). "User-friendly PAC-Bayes." arXiv:2110.11216
- Grinsztajn, Oyallon & Varoquaux (2022). "Trees beat NNs on tabular." arXiv:2207.08815
- Wilson & Adams (2013). "Spectral Mixture Kernels." arXiv:1302.4245
- Chernozhukov et al. (2018). "Double ML." arXiv:1608.00060
- Gal & Ghahramani (2016). "MC Dropout." arXiv:1506.02142
- Hébert-Johnson et al. (2018). "Multi-calibration." arXiv:1711.08513
- Hellström et al. (2023). "Info-theoretic PAC-Bayes." arXiv:2309.04381
- Xu & Raginsky (2017). "MI generalization." arXiv:1705.07809
- Esfahani & Kuhn (2018). "Wasserstein DRO." DOI:10.1007/s10107-017-1172-1

---

## Architecture Overview

```
apps/
  ml-api/
    src/
      learning/
        conformal/
          split_conformal.py       — Split conformal prediction
          cqr.py                   — Conformalized Quantile Regression
          aci.py                   — Adaptive Conformal Inference (time series)
          enbpi.py                 — Ensemble batch PI (time series)
          weighted.py              — Weighted conformal (distribution shift)
        pac_bayes/
          bounds.py                — PAC-Bayes-kl, Catoni, McAllester bounds
          model_selection.py       — SRM via PAC-Bayes
        bandits/
          hedge.py                 — Multiplicative weights (full information)
          ucb.py                   — UCB1 (stochastic bandits)
          exp3.py                  — EXP3 (adversarial bandits)
          lin_ucb.py               — LinUCB (contextual bandits)
          thompson.py              — Thompson Sampling
        gp/
          venue_demand_gp.py       — GP regression with composite kernels
          bayesian_opt.py          — BoTorch-based price optimization
          scalable_gp.py           — SVGP, KISS-GP+LOVE for large data
        ensembles/
          quantile_forest.py       — QRF with prediction intervals
          boosting.py              — CatBoost/LightGBM/XGBoost wrappers
          shap_explanations.py     — TreeSHAP feature importance
        bayesian/
          hierarchical_venue.py    — PyMC hierarchical model (partial pooling)
          mc_dropout.py            — MC Dropout BNN approximation
        drift/
          detectors.py             — ADWIN, DDM, Page-Hinkley (River)
          covariate_shift.py       — KMM, DANN domain adaptation
          dro.py                   — Distributionally Robust Optimization
          ewc.py                   — Elastic Weight Consolidation
        causal/
          dml.py                   — Double/Debiased ML (EconML)
          iv.py                    — Instrumental Variables (2SLS)
          uplift.py                — CausalML uplift modeling
          sequential_testing.py    — Always-valid sequential A/B tests
          synthetic_control.py     — Synthetic control methods
        information/
          feature_selection.py     — MI-based mRMR feature selection
          divergence.py            — KL, JSD for drift monitoring
          mdl.py                   — MDL model selection
          fisher_oed.py            — D-optimal experimental design
        calibration/
          platt_scaling.py         — Platt calibration
          isotonic.py              — Isotonic regression
          temperature_scaling.py   — Temperature scaling
          multi_calibration.py     — Multi-calibration (subgroup fairness)
        fairness/
          monitoring.py            — Demographic parity, individual fairness
          debiasing.py             — fairlearn ExponentiatedGradient
        monitoring/
          coverage_tracker.py      — Rolling conformal coverage monitoring
          calibration_monitor.py   — PIT histogram calibration checks
          retraining_trigger.py    — Drift → retraining pipeline
      routes/
        learning.py                — FastAPI endpoints

packages/
  learning-core/                   — TypeScript types + client-side compute
    src/
      types.ts                     — PricePredictionResponse, UncertaintyLevel
      pac_bayes.ts                 — PAC-Bayes bound computation (browser)
      bandits/
        hedge.ts                   — Hedge/MW for client-side pricing
        ucb.ts                     — UCB1
        thompson.ts                — Thompson Sampling
      visualization/
        PredictionIntervals.tsx    — Fan chart (recharts Area + Line)
        UncertaintyBadge.tsx       — Traffic light uncertainty indicator
        SHAPWaterfall.tsx          — SHAP feature importance waterfall
        CalibrationPlot.tsx        — Reliability diagram
        CoverageDashboard.tsx      — Rolling coverage monitoring
        BanditExplorer.tsx         — Live bandit arm selection visualization
        ParetoTradeoff.tsx         — Fairness-accuracy Pareto front
```

### Python Dependencies

```
mapie>=0.9                    # Conformal prediction (split, jackknife+, CQR, EnbPI)
pymc>=5.0                     # Hierarchical Bayesian models
numpyro>=0.15                 # JAX-accelerated MCMC (11× faster than PyMC CPU)
gpytorch>=1.12                # Gaussian processes (GPU-accelerated)
botorch>=0.12                 # Bayesian optimization with GPs
xgboost>=2.0                  # Gradient boosting
catboost>=1.2                 # Categorical-native gradient boosting
lightgbm>=4.0                 # Large-scale gradient boosting
quantile-forest>=1.3          # Quantile Regression Forests (Zillow, JOSS 2024)
shap>=0.45                    # TreeSHAP explanations
econml>=0.16                  # Causal inference (DML, CausalForest, DeepIV)
causalml>=0.16                # Uplift modeling (Uber)
dowhy>=0.11                   # Causal pipeline (PyWhy)
river>=0.21                   # Online learning + drift detection (ADWIN, DDM)
alibi-detect>=0.12            # Statistical drift detection (KSD, MMD)
nannyml>=0.12                 # Performance estimation + concept drift
fairlearn>=0.11               # Fairness constraints
confseq>=0.0.10               # Always-valid confidence sequences
vowpalwabbit>=9.0             # Contextual bandits at scale
dro>=0.1                      # Distributionally Robust Optimization
```

---

## SLT-1: Conformal Prediction Suite

### What to Build

Distribution-free prediction intervals with guaranteed coverage for venue
demand forecasting. The centrepiece of the uncertainty system.

### Split Conformal + Jackknife+ + CQR

```python
# apps/ml-api/src/learning/conformal/split_conformal.py

from mapie.regression import MapieRegressor, MapieQuantileRegressor
from mapie.time_series import MapieTimeSeriesRegressor
from sklearn.ensemble import GradientBoostingRegressor

class VenueConformalPredictor:
    """
    Conformal prediction for venue demand/pricing.

    Coverage guarantee: P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 - α
    under exchangeability (Vovk et al. 2005).

    Methods:
    - Split conformal: simplest, constant-width intervals
    - Jackknife+ (arXiv:1905.02928): LOO-based, tighter, P ≥ 1-2α
    - CV+ (MAPIE cv=10): computationally efficient Jackknife+
    - CQR (arXiv:1905.03222): RECOMMENDED — heteroscedastic intervals,
      wider when demand volatile (weekends/holidays), tighter for predictable periods
    """

    def __init__(self, method: str = "cqr", alpha: float = 0.1):
        self.method = method
        self.alpha = alpha
        self.base_model = GradientBoostingRegressor(n_estimators=200)

    def fit(self, X_train, y_train, X_cal=None, y_cal=None):
        if self.method == "split":
            self.base_model.fit(X_train, y_train)
            self.mapie = MapieRegressor(
                estimator=self.base_model, method="base", cv="prefit")
            self.mapie.fit(X_cal, y_cal)

        elif self.method == "jackknife_plus":
            # Full LOO: cv=-1, or efficient CV+: cv=10
            self.mapie = MapieRegressor(
                estimator=self.base_model, method="plus", cv=10)
            self.mapie.fit(X_train, y_train)

        elif self.method == "cqr":
            # Conformalized Quantile Regression (Romano et al. 2019)
            # Heteroscedastic intervals — RECOMMENDED for venue pricing
            # Score: E_i = max{q̂_{α/2}(X_i) - Y_i, Y_i - q̂_{1-α/2}(X_i)}
            self.mapie = MapieQuantileRegressor(
                estimator=GradientBoostingRegressor(
                    n_estimators=200, loss="quantile"),
                alpha=self.alpha)
            self.mapie.fit(X_train, y_train)

    def predict(self, X_new):
        y_pred, y_intervals = self.mapie.predict(X_new, alpha=self.alpha)
        return {
            "point_estimate": y_pred,
            "lower": y_intervals[:, 0, 0],
            "upper": y_intervals[:, 1, 0],
            "confidence_level": 1 - self.alpha,
        }
```

### Time Series Conformal Methods

```python
# apps/ml-api/src/learning/conformal/aci.py

class AdaptiveConformalInference:
    """
    ACI (Gibbs & Candès 2021, arXiv:2106.00170):
    Online-updates significance level for non-exchangeable time series.

    α_{t+1} = α_t + γ(α - err_t) where err_t = 1{Y_t ∉ C_t}

    Miscoverage → α increases (wider intervals)
    Coverage → α decreases (tighter intervals)

    Guarantee: |Σ err_t/T - α| ≤ (max{α₁, 1-α₁} + 1)/(γT)

    DtACI (arXiv:2208.08401) removes need to tune γ.
    """
    def __init__(self, base_predictor, alpha: float = 0.1, gamma: float = 0.005):
        self.predictor = base_predictor
        self.alpha_target = alpha
        self.alpha_t = alpha
        self.gamma = gamma

    def update(self, y_true, y_pred, interval):
        err_t = 1.0 if y_true < interval[0] or y_true > interval[1] else 0.0
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha_target - err_t)
        self.alpha_t = max(0.001, min(0.999, self.alpha_t))
        return self.alpha_t

class EnbPIPredictor:
    """
    EnbPI (Xu & Xie 2021, arXiv:2010.09107):
    Bootstrap ensemble PI for time series without data splitting or retraining.

    MAPIE: MapieTimeSeriesRegressor(method="enbpi") with partial_fit()
    for online residual updates.
    """
    def __init__(self, alpha: float = 0.1):
        self.mapie_ts = MapieTimeSeriesRegressor(
            method="enbpi",
            estimator=GradientBoostingRegressor(n_estimators=100))

    def fit(self, X_train, y_train):
        self.mapie_ts.fit(X_train, y_train)

    def predict_and_update(self, X_new, y_true=None):
        y_pred, y_intervals = self.mapie_ts.predict(X_new, alpha=0.1)
        if y_true is not None:
            self.mapie_ts.partial_fit(X_new, y_true)  # Online update
        return y_pred, y_intervals
```

### Weighted Conformal for Distribution Shift

```python
# apps/ml-api/src/learning/conformal/weighted.py

"""
Weighted conformal (Tibshirani et al. 2019, arXiv:1904.06019):
Handles covariate shift by computing weighted quantiles:
  w_i = p_test(X_i) / p_train(X_i)

For venue models: corrects for seasonal distributional changes
or geographic domain shifts.
"""
```

---

## SLT-2: PAC-Bayes Bounds and Model Selection

### What to Build

Non-vacuous generalization guarantees for venue demand models. PAC-Bayes
provides the tightest known bounds for neural networks.

### TypeScript (Browser-Side Bound Computation)

```typescript
// packages/learning-core/src/pac_bayes.ts

/**
 * PAC-Bayes-kl bound (Seeger 2002, Langford 2005, Maurer 2004):
 *
 * kl(R̂(ρ) ∥ R(ρ)) ≤ [KL(ρ∥π) + ln(n/δ)] / n
 *
 * where kl(q∥p) = q·ln(q/p) + (1-q)·ln((1-q)/(1-p))
 * R̂(ρ) = expected empirical risk, R(ρ) = expected true risk
 *
 * Must be inverted numerically via binary search.
 *
 * For venue models with n=10,000 records, emp risk 0.08, KL=50 nats:
 *   PAC-Bayes-kl → R ≤ ~0.133 (non-vacuous, informative!)
 *   McAllester   → R ≤ ~0.138
 * At n=1,000: R ≤ ~0.30 (suggests simple models)
 * At n=50,000+: R < 0.10 (deep architectures viable)
 */
export function pacBayesKLBound(
  empRisk: number,
  klDivergence: number,
  n: number,
  delta: number = 0.05,
): number {
  const c = (klDivergence + Math.log(n / delta)) / n;

  // Invert binary KL via binary search
  let lo = empRisk;
  let hi = 1 - 1e-9;
  while (hi - lo > 1e-9) {
    const mid = (lo + hi) / 2;
    const kl =
      empRisk * Math.log(empRisk / mid) +
      (1 - empRisk) * Math.log((1 - empRisk) / (1 - mid));
    if (kl <= c) lo = mid;
    else hi = mid;
  }
  return lo;
}

/**
 * McAllester bound (Pinsker relaxation):
 * R(ρ) ≤ R̂(ρ) + √{[KL(ρ∥π) + ln(n/δ)] / (2n)}
 */
export function mcAllesterBound(
  empRisk: number,
  klDivergence: number,
  n: number,
  delta: number = 0.05,
): number {
  return empRisk + Math.sqrt((klDivergence + Math.log(n / delta)) / (2 * n));
}

/**
 * Catoni bound with optimized λ* (arXiv:0712.0248):
 * R ≤ R̂ + B·√{[KL + ln(1/δ)] / (2n)}
 * for losses bounded in [0, B].
 */
export function catoniBound(
  empRisk: number,
  klDivergence: number,
  n: number,
  lossBound: number = 1.0,
  delta: number = 0.05,
): number {
  return empRisk + lossBound * Math.sqrt(
    (klDivergence + Math.log(1 / delta)) / (2 * n)
  );
}

/**
 * Structural Risk Minimization via PAC-Bayes:
 * Select model minimizing R̂(ĥ_i) + ε_i(n, δ)
 *
 * Practical guidelines for venue data:
 *   n=1K   → linear models
 *   n=5-10K → small NNs (validated by PAC-Bayes)
 *   n=50K+ → deep architectures with PAC-Bayes regularization
 */
export function selectModelComplexity(
  candidates: Array<{ name: string; empRisk: number; klDiv: number }>,
  n: number,
  delta: number = 0.05,
): string {
  let bestName = '';
  let bestBound = Infinity;
  for (const c of candidates) {
    const bound = pacBayesKLBound(c.empRisk, c.klDiv, n, delta);
    if (bound < bestBound) {
      bestBound = bound;
      bestName = c.name;
    }
  }
  return bestName;
}
```

### Python (Server-Side Bound Evaluation)

```python
# apps/ml-api/src/learning/pac_bayes/bounds.py

"""
PAC-Bayes connection to regularization:
When π = N(0, σ²_prior·I) and ρ = N(w, σ²_post·I):
  KL(ρ∥π) reduces to weighted L2 penalty → weight decay!

Training with Bayes by Backprop directly optimizes a PAC-Bayes bound.

Data-dependent priors (essential for small venue datasets):
Use 30% of early booking records to set the prior,
evaluate the bound on the remaining 70%.
This mirrors the temporal structure of venue data.
"""
```

---

## SLT-3: VC Dimension and Sample Complexity

### What to Build

Answer the question: "How many booking records do we need before our demand
model is reliable?"

```python
# apps/ml-api/src/learning/pac_bayes/model_selection.py

"""
Sample complexity bounds for venue model selection:

  m ≥ (1/ε)(d·ln(1/ε) + ln(1/δ))

VC dimensions for common models:
  - Logistic regression (20 features): d = 21 → m ≥ ~1,320 for ε=0.05
  - Decision tree (depth 5): d = 32 → m ≥ ~640
  - Neural net: d = O(W·L·log W) → VACUOUS bounds
    (PAC-Bayes gives better bounds for NNs)

Rademacher complexity (data-dependent, tighter):
  R(h) ≤ R̂(h) + 2·Rad_S(A) + 3·√{ln(2/δ)/(2m)}
  For linear with bounded norms: Rad_S = BR/√m (norm-based, not dimension-based)

SRM decision table for venue platforms:
  1K records  → linear/logistic regression
  5-10K       → small NN or tree ensemble
  50K+        → deep architectures with PAC-Bayes regularization
"""

def compute_sample_complexity(
    vc_dimension: int,
    epsilon: float = 0.05,
    delta: float = 0.05,
) -> int:
    """Minimum samples for (ε, δ)-PAC learning."""
    import math
    return int(math.ceil(
        (1 / epsilon) * (vc_dimension * math.log(1 / epsilon) + math.log(1 / delta))
    ))
```

---

## SLT-4: Online Learning and Bandit Systems

### What to Build

Provable-regret pricing optimization via bandits. Each price point is an "arm."
The platform learns the revenue-optimal price with regret O(√T).

### Python Implementation

```python
# apps/ml-api/src/learning/bandits/lin_ucb.py

import numpy as np

class LinUCB:
    """
    LinUCB (Li et al. 2010) for personalized venue pricing.

    Context x_t: day of week, event type, group size, season, ...
    Arms: discrete price points
    Reward: booking revenue

    Select arm maximizing: x_t^T θ̂_a + α√(x_t^T A_a^{-1} x_t)

    Regret bound: Õ(d√T) where d = context dimension.
    """
    def __init__(self, n_arms: int, d: int, alpha: float = 1.0):
        self.K = n_arms
        self.d = d
        self.alpha = alpha
        self.A = [np.eye(d) for _ in range(n_arms)]
        self.b = [np.zeros(d) for _ in range(n_arms)]

    def select(self, context: np.ndarray) -> int:
        x = np.array(context)
        ucb_values = np.zeros(self.K)
        for a in range(self.K):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            ucb_values[a] = x @ theta + self.alpha * np.sqrt(x @ A_inv @ x)
        return int(np.argmax(ucb_values))

    def update(self, arm: int, context: np.ndarray, reward: float):
        x = np.array(context)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
```

```python
# apps/ml-api/src/learning/bandits/thompson.py

class ThompsonSampling:
    """
    Thompson Sampling for venue price A/B testing.

    Bayesian: sample θ_i ~ Beta(α_i, β_i) for each arm,
    select arm with highest sample.

    Matches UCB's O(√(KT ln T)) regret but converges faster in practice.
    Naturally quantifies uncertainty → ideal for quick A/B convergence.
    """
    def __init__(self, n_arms: int):
        self.alpha = np.ones(n_arms)  # successes + 1
        self.beta = np.ones(n_arms)   # failures + 1

    def select(self) -> int:
        samples = [np.random.beta(self.alpha[i], self.beta[i])
                   for i in range(len(self.alpha))]
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        # Bernoulli reward: booked (1) or not (0)
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
```

```python
# apps/ml-api/src/learning/bandits/hedge.py

class Hedge:
    """
    Hedge / Multiplicative Weights Update (full information).

    w_i^{t+1} = w_i^t · exp(-η · loss_i^t)
    Regret: O(√(T ln N)) with optimal η = √(ln N / T)

    Use when: all pricing strategies' outcomes are observable
    (e.g., simulated or historical data).

    For N=5 price tiers over T=1000 events:
    worst-case regret ≈ √(1000 · ln 5) ≈ 40 suboptimal decisions.
    """
    def __init__(self, n_experts: int, eta: float = 0.1):
        self.weights = np.ones(n_experts)
        self.eta = eta

    def get_distribution(self) -> np.ndarray:
        return self.weights / self.weights.sum()

    def update(self, losses: np.ndarray):
        self.weights *= np.exp(-self.eta * losses)
```

### TypeScript (Client-Side Bandits)

```typescript
// packages/learning-core/src/bandits/thompson.ts

/**
 * Client-side Thompson Sampling for real-time pricing decisions.
 * Beta-Bernoulli model: no server round-trip needed.
 */
export class ThompsonSampling {
  private alpha: number[];
  private beta: number[];

  constructor(nArms: number) {
    this.alpha = new Array(nArms).fill(1);
    this.beta = new Array(nArms).fill(1);
  }

  select(): number {
    const samples = this.alpha.map((a, i) => betaSample(a, this.beta[i]));
    return samples.indexOf(Math.max(...samples));
  }

  update(arm: number, reward: number): void {
    if (reward > 0) this.alpha[arm] += 1;
    else this.beta[arm] += 1;
  }
}
```

### Production: Vowpal Wabbit for Scale

```python
"""
Vowpal Wabbit (vowpalwabbit.org): gold standard for contextual bandits at scale.
Powers Azure Personalizer.

  --cb_explore_adf: action-dependent features
  Exploration: ε-greedy, bag, cover, RND
  Doubly robust off-policy evaluation

Also: River (pip install river) → river.bandit for streaming API.
"""
```

---

## SLT-5: Gaussian Processes for Demand Forecasting

### What to Build

GP regression with composite venue-specific kernels, automatic uncertainty
quantification, and Bayesian optimization of pricing.

```python
# apps/ml-api/src/learning/gp/venue_demand_gp.py

import gpytorch
import torch
from botorch.models import SingleTaskGP

class VenueDemandGP(gpytorch.models.ExactGP):
    """
    GP regression for venue demand with composite kernel:

    k = ScaleKernel(Periodic(p=7) × Matérn(ν=2.5))     — weekly cycles
      + ScaleKernel(Periodic(p=365) × RBF)              — annual seasonality
      + ScaleKernel(Matérn(ν=1.5))                       — price elasticity

    Posterior mean: μ* = K*^T (K + σ²I)^{-1} y
    Posterior var:  σ*² = K** - K*^T (K + σ²I)^{-1} K*

    Variance automatically widens in data-sparse regions.

    Kernel recommendations:
    - Matérn-3/2: price-demand (once-differentiable, less restrictive than RBF)
    - Periodic: weekly/yearly booking cycles
    - Spectral Mixture (Wilson & Adams 2013, arXiv:1302.4245):
      auto-discovers periodicities and long-range structure
    """
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        # Composite kernel for venue demand
        weekly = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel(period_length=7.0)
            * gpytorch.kernels.MaternKernel(nu=2.5))
        annual = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel(period_length=365.0)
            * gpytorch.kernels.RBFKernel())
        elasticity = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5))

        self.covar_module = weekly + annual + elasticity

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


class ScalableVenueGP:
    """
    Scalability at venue-relevant data volumes:

    | Method           | 1K pts  | 10K pts | 100K pts | 1M pts    |
    |------------------|---------|---------|----------|-----------|
    | Exact GP (GPU)   | ~0.1s   | ~5s     | OOM      | Infeasible|
    | SGPR (M=500)     | ~0.3s   | ~3s     | ~30s     | ~300s     |
    | SVGP (M=1024)    | 0.1s/it | 0.1s/it | 0.1s/it  | 0.1s/it   |
    | KISS-GP + LOVE   | ~0.05s  | ~0.5s   | ~5s      | ~50s      |

    Recommendation: SVGP (Hensman et al. 2013) for >10K points.
    KISS-GP (Wilson & Nickisch 2015, arXiv:1503.01057) for ≤5 input dims.
    LOVE (Pleiss et al. 2018, arXiv:1803.06058) for O(1) predictive variances.
    """
    ...


class VenuePriceOptimizer:
    """
    Bayesian Optimization with GP surrogates via BoTorch.
    GP models price→revenue mapping.
    Acquisition function suggests next price to try (explore/exploit).

    BoTorch (Balandat et al. 2020, arXiv:1910.06403):
    - qLogExpectedImprovement for single-objective
    - qNEHVI for multi-objective (revenue + occupancy + satisfaction)
    """
    ...
```

### Multi-Output GPs and Deep Kernel Learning

```python
"""
Multi-output GP: jointly model correlated venues.
  cov(f_i(x), f_j(x')) = k(x,x') · B[i,j]
  B captures cross-venue correlations.
  GPyTorch: MultitaskKernel. Sparse venues benefit from correlated venues' data.

Deep Kernel Learning (Wilson et al. 2016, arXiv:1511.02222):
  k_deep(x,x') = k_base(g(x;θ), g(x';θ))
  Neural feature extraction + GP inference, jointly trained via marginal likelihood.
"""
```

---

## SLT-6: Tree Ensembles — The Correct Default for Venue Tabular Data

### What to Build

Gradient-boosted trees and random forests as base models, with quantile
regression forests for prediction intervals and SHAP for explanations.

```python
# apps/ml-api/src/learning/ensembles/quantile_forest.py

from quantile_forest import RandomForestQuantileRegressor

class VenueQuantileForest:
    """
    Quantile Regression Forests (Meinshausen 2006, JMLR 7).
    Predicts full conditional distribution, not just mean.

    quantile-forest library by Zillow (JOSS 2024):
    Cython-optimized, scikit-learn compatible.

    Trees beat neural networks on tabular data:
    Grinsztajn et al. (2022, arXiv:2207.08815, NeurIPS):
    45 datasets, 20K GPU-hours → trees consistently win on medium tabular.

    CatBoost recommended when venue data has many categoricals
    (venue_type, neighborhood, event_category, amenities).
    """
    def __init__(self, n_estimators: int = 500):
        self.qrf = RandomForestQuantileRegressor(
            n_estimators=n_estimators, min_samples_leaf=5)

    def fit(self, X, y):
        self.qrf.fit(X, y)

    def predict(self, X, quantiles=(0.05, 0.50, 0.95)):
        return self.qrf.predict(X, quantiles=list(quantiles))
```

```python
# apps/ml-api/src/learning/ensembles/shap_explanations.py

import shap

def explain_price(model, X_instance, X_background):
    """
    TreeSHAP (Lundberg & Lee, NeurIPS 2017):
    O(TLD²) per-prediction explanations.

    Waterfall shows venue managers WHY a price was recommended:
    "Weekend premium (+$200), high capacity (+$150), nearby event (+$100)"

    Critical for manager trust and adoption.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_instance)
    return shap_values
```

---

## SLT-7: Hierarchical Bayesian Models (Cold-Start Solution)

### What to Build

Partial pooling across venues: new venues get population-mean predictions,
well-observed venues retain individual estimates.

```python
# apps/ml-api/src/learning/bayesian/hierarchical_venue.py

import pymc as pm
import numpy as np

def build_hierarchical_venue_model(
    venue_ids: np.ndarray,
    X_features: np.ndarray,
    y_price: np.ndarray,
    n_venues: int,
):
    """
    Hierarchical Bayesian model for multi-venue pricing.

    μ_venue[j] ~ Normal(μ_global, σ_venue)

    New venues (zero history) → predictions default to population prior.
    Well-observed venues → individual estimates dominate.
    This IS the cold-start solution — no ad hoc rules needed.

    Non-centered parameterization for better MCMC sampling.
    Use NumPyro backend for ~11× speedup on GPU:
      pm.sample(nuts_sampler="numpyro")

    MC Dropout (Gal & Ghahramani 2016, arXiv:1506.02142):
    Alternative BNN approximation — T forward passes yield:
      Ê[y] = (1/T) Σ_t f(x; W̃_t)
      V̂ar[y] ≈ epistemic + aleatoric uncertainty decomposition
    """
    with pm.Model() as model:
        # Population-level parameters
        mu_global = pm.Normal("mu_global", mu=500, sigma=200)
        sigma_venue = pm.HalfNormal("sigma_venue", sigma=100)

        # Non-centered parameterization
        venue_offset_raw = pm.Normal("venue_offset_raw", mu=0, sigma=1, shape=n_venues)
        venue_intercept = pm.Deterministic(
            "venue_intercept", mu_global + sigma_venue * venue_offset_raw)

        # Feature effects (shared across venues)
        beta_day = pm.Normal("beta_day", mu=0, sigma=100)
        beta_capacity = pm.Normal("beta_capacity", mu=0, sigma=50)

        # Observation noise
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=50)

        # Likelihood
        mu = venue_intercept[venue_ids] + beta_day * X_features[:, 0]
        price = pm.Normal("price", mu=mu, sigma=sigma_obs, observed=y_price)

        # Sample — use NumPyro for GPU acceleration
        trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95,
                          nuts_sampler="numpyro")

    return model, trace
```

---

## SLT-8: Distribution Shift Detection and Robustness

### What to Build

Detect when booking patterns shift and automatically trigger retraining.
DRO for worst-case guarantees.

```python
# apps/ml-api/src/learning/drift/detectors.py

from river import drift

class VenueDriftMonitor:
    """
    Three complementary drift detectors:

    ADWIN (Bifet & Gavalda 2007): variable-length window,
      detects when |μ₀ - μ₁| ≥ ε_cut between subwindows.
    DDM: monitors error rate, flags when p_i + s_i ≥ p_min + 3·s_min.
    Page-Hinkley: cumulative shift detection.

    All from River (pip install river).
    """
    def __init__(self):
        self.adwin = drift.ADWIN(delta=0.002)
        self.ddm = drift.DDM(min_num_instances=30)
        self.ph = drift.PageHinkley(delta=0.005)

    def update(self, residual: float) -> dict:
        self.adwin.update(residual)
        self.ddm.update(residual)
        self.ph.update(residual)
        return {
            "adwin_drift": self.adwin.drift_detected,
            "ddm_drift": self.ddm.drift_detected,
            "ph_drift": self.ph.drift_detected,
        }
```

```python
# apps/ml-api/src/learning/drift/dro.py

"""
Distributionally Robust Optimization (Esfahani & Kuhn 2018):
  min_θ max_{P ∈ U(P̂, ε)} E_P[loss(θ)]
  U = Wasserstein ball around empirical distribution.

The `dro` library (Namkoong Lab, 2025, arXiv:2505.23565):
79 method combinations (14 DRO formulations × 9 model backbones).
scikit-learn and PyTorch interfaces.

Elastic Weight Consolidation (Kirkpatrick et al. 2017, arXiv:1612.00796):
  L(θ) = L_new(θ) + (λ/2) Σ_i F_i (θ_i - θ*_old,i)²
  F_i = Fisher Information for parameter importance.
  Prevents catastrophic forgetting when updating venue models.
"""
```

---

## SLT-9: Causal Inference for Venue Optimization

### What to Build

Separate causation from correlation. Does lowering price CAUSE more bookings,
or do both correlate with seasonal demand?

```python
# apps/ml-api/src/learning/causal/dml.py

from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor

class VenueCausalAnalysis:
    """
    Double/Debiased ML (Chernozhukov et al. 2018, arXiv:1608.00060):

    Two-stage with cross-fitting:
    1. Partial out confounders: Ỹ = Y - ĝ(W), T̃ = T - m̂(W)
    2. Estimate: θ̂ = (Σ T̃_i Ỹ_i) / (Σ T̃_i²)

    Neyman orthogonality → √n-consistency even with slow ML convergence.

    CausalForestDML provides heterogeneous treatment effects:
    "This venue type benefits MORE from a price decrease than that type."
    """
    def __init__(self):
        self.model = CausalForestDML(
            model_y=GradientBoostingRegressor(),
            model_t=GradientBoostingRegressor(),
            n_estimators=200)

    def fit(self, Y, T, X, W):
        """Y=bookings, T=price, X=venue features, W=confounders."""
        self.model.fit(Y=Y, T=T, X=X, W=W)

    def estimate_effect(self, X, alpha=0.05):
        ate = self.model.ate(X)
        lb, ub = self.model.ate_interval(X, alpha=alpha)
        return {"ate": ate, "ci_lower": lb, "ci_upper": ub}
```

```python
# apps/ml-api/src/learning/causal/sequential_testing.py

"""
Always-valid sequential testing (Johari et al. 2015, arXiv:1512.04922):
Peek at A/B test results without inflating Type I error.

confseq library: gamma-exponential mixture boundaries for
time-uniform confidence sequences.

GrowthBook (open-source): A/B platform with sequential analysis,
CUPED variance reduction, and automatic safe rollbacks.
"""
```

---

## SLT-10: Information-Theoretic Methods

```python
# apps/ml-api/src/learning/information/feature_selection.py

"""
Mutual Information feature selection:
I(X;Y) = H(Y) - H(Y|X)

mRMR (Peng et al. 2005):
  Score = I(f;Y) - (1/|S|) Σ_{s∈S} I(f;s)
  Maximizes relevance, minimizes redundancy.
  Prevents selecting correlated features (e.g., capacity + square footage).

KL divergence for drift monitoring:
  JSD(P∥Q) = ½KL(P∥M) + ½KL(Q∥M) where M = ½(P+Q)
  √JSD is a true metric, bounded [0, √ln2].

MDL model selection:
  Minimize L(M) + L(D|M) — equivalent to BIC asymptotically.

Fisher information for optimal experimental design:
  D-optimal: maximize det(I(θ)) → minimize confidence ellipsoid.
  Tells you WHICH A/B tests to run for maximum information.

Deep connection (Hellström et al. 2023, arXiv:2309.04381):
  KL(ρ∥π) in PAC-Bayes IS an information-theoretic quantity.
  Xu & Raginsky (2017, arXiv:1705.07809):
  |gen(μ,A)| ≤ √{2σ²·I(W;S_n)/n}
  → Low mutual information between model and training data = better generalization.
"""
```

---

## SLT-11: Calibration and Fairness

```python
# apps/ml-api/src/learning/calibration/temperature_scaling.py

"""
Temperature scaling (Guo et al. ICML 2017, 5000+ citations):
Divides logits by learned scalar T > 0.
Most effective NN calibration method. Preserves accuracy.
scikit-learn ≥1.8: CalibratedClassifierCV(method='temperature')

Multi-calibration (Hébert-Johnson et al. 2018, arXiv:1711.08513):
Guarantees calibration SIMULTANEOUSLY across subgroups:
  |E[Y - f(X) | f(X)=v, X ∈ C]| ≤ α for every subgroup C
Ensures predictions calibrated for small venues, large venues,
urban venues, weekend bookings, and every subpopulation.

Fairness impossibility (Chouldechova 2016; Kleinberg 2016):
Demographic parity + calibration + equal FPR cannot all hold
when base rates differ → must deliberately prioritize.

Libraries: fairlearn (ExponentiatedGradient with DemographicParity/EqualizedOdds),
AIF360 (IBM, 9+ bias mitigation algorithms).
"""
```

---

## SLT-12: Production API and Monitoring Architecture

### What to Build

The uncertainty-aware prediction API and continuous monitoring dashboard.

### API Response Type

```typescript
// packages/learning-core/src/types.ts

export interface PricePredictionResponse {
  prediction: {
    point_estimate: number;       // e.g., 1200.00
    prediction_intervals: Array<{
      confidence_level: number;   // e.g., 0.90
      lower: number;              // e.g., 950.00
      upper: number;              // e.g., 1450.00
    }>;
  };
  uncertainty: {
    level: 'low' | 'medium' | 'high';
    interval_width: number;
    epistemic_std: number;        // Model uncertainty (data-sparse?)
    aleatoric_std: number;        // Inherent noise (volatile demand?)
    data_support: number;         // Similar historical bookings
  };
  explanation: {
    top_drivers: Array<{
      feature: string;
      shap_value: number;
      direction: 'increases' | 'decreases';
    }>;
  };
  metadata: {
    model_version: string;
    calibration_coverage: number; // Recent coverage metric
    is_cold_start: boolean;
    pac_bayes_bound: number;      // Generalization guarantee
  };
}
```

### Visualization Components

```typescript
// packages/learning-core/src/visualization/PredictionIntervals.tsx

/**
 * Fan chart: recharts Area + Line components.
 * ci90 (light blue, 15% opacity) + ci50 (medium, 30%) + point line.
 *
 * Recommended UX:
 *   "Demand forecast: **45-72 bookings** (90% confidence)"
 *   Traffic light badge: green (narrow CI) / yellow / red (wide CI)
 *   SHAP waterfall for WHY
 */

// packages/learning-core/src/visualization/CalibrationPlot.tsx
/**
 * Reliability diagram: predicted probability vs observed frequency.
 * PIT histogram: should be uniform if calibrated.
 * Rolling coverage plot: tracks whether 90% intervals cover 90% of outcomes.
 */

// packages/learning-core/src/visualization/BanditExplorer.tsx
/**
 * Live visualization of bandit arm selection:
 * - Bar chart of posterior distributions (Beta for Thompson)
 * - Arm pull counts over time
 * - Cumulative regret curve
 * - UCB confidence intervals per arm
 */
```

### Monitoring Pipeline

```python
# apps/ml-api/src/learning/monitoring/coverage_tracker.py

"""
Production monitoring pipeline:
  data → model training → conformal calibration → inference serving
    → coverage monitoring → drift detection → retraining trigger

1. Coverage tracking: rolling empirical coverage, alert when >5% below nominal
2. Drift detection: ADWIN on residuals (River), MMDDrift/KSDrift (Alibi Detect)
3. A/B management: sequential analysis (confseq), CUPED (GrowthBook)
4. Calibration: PIT histograms verify quantiles are uniform
5. Fairness: fairlearn audit on subgroups (venue type, location, size)

All metrics → dashboard with recharts time series + alert badges.
"""
```

---

## Integration with Other Techniques

- **Stochastic Pricing** (SP): Conformal intervals on stochastic price predictions;
  bandits select optimal dynamic pricing strategy; GPs provide surrogate model for
  Bayesian optimization of price parameters
- **Physics Solvers** (PS): PAC-Bayes model selection determines which physics model
  complexity is justified given available venue data
- **Optimal Transport** (OT): GP embeddings of venues used as source/target for OT
  matching; calibrated predictions ensure transport costs reflect true uncertainty
- **Category Theory** (CT): Conformal prediction as a natural transformation from
  point predictions to interval predictions (functorial)
- **TDA**: Persistent homology features fed to tree ensembles as inputs; conformal
  wraps the TDA-enhanced model for guaranteed coverage

---

## Session Management

1. **SLT-1** (Conformal: split, jackknife+, CQR, ACI, EnbPI, weighted) — 1-2 sessions
2. **SLT-2** (PAC-Bayes: kl, Catoni, McAllester + SRM model selection) — 1 session
3. **SLT-3** (VC dimension + sample complexity calculator) — 1 session
4. **SLT-4** (Bandits: Hedge, UCB, EXP3, LinUCB, Thompson, VW integration) — 1-2 sessions
5. **SLT-5** (GPs: composite kernels, SVGP, KISS-GP, BoTorch optimization) — 1-2 sessions
6. **SLT-6** (Tree ensembles: QRF, CatBoost/LightGBM/XGBoost, SHAP) — 1 session
7. **SLT-7** (Hierarchical Bayes: PyMC partial pooling, MC Dropout) — 1 session
8. **SLT-8** (Drift: ADWIN/DDM/PH, DRO, EWC, covariate shift) — 1 session
9. **SLT-9** (Causal: DML, IV, uplift, sequential testing) — 1-2 sessions
10. **SLT-10** (Information theory: MI, KL/JSD, MDL, Fisher OED) — 1 session
11. **SLT-11** (Calibration: Platt/isotonic/temperature, multi-cal, fairness) — 1 session
12. **SLT-12** (Production API, monitoring dashboard, retraining pipeline) — 1-2 sessions

Total: ~12-16 Claude Code sessions.

### Inference Serving (Rust for Latency)

```
Train in Python (PyMC/GPyTorch/XGBoost) → export via ONNX → serve from Rust.
linfa: Decision Trees, Linear, SVM, KNN, PCA (pure Rust, BLAS backends)
smartcore: Random Forest, Extra Trees
egobox-gp: kriging/GP regression in Rust
Production pattern: 10-100× latency reduction in scoring endpoints.
```
