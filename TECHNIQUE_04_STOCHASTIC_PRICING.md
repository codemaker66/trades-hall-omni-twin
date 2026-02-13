# TECHNIQUE_04_STOCHASTIC_PRICING.md — Quantitative Finance Pricing Engine

<!-- COMMAND_TRACK_LINKS_START -->
## Canonical Command Track
Canonical command track: `docs/commands/stochastic-pricing/`
Execution authority for agent command specs is `docs/commands/**`.
This document remains a research/reference source.

## ID Mapping Notes
SP-N maps to SP-N.
Integration checkpoints use IDs `SP-INT-1`, `SP-INT-2`, and `SP-INT-3`.
<!-- COMMAND_TRACK_LINKS_END -->


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
> are complete. Implements a full quantitative finance pricing engine — stochastic
> models, options pricing, optimal control, mean-field games, order book mechanics,
> self-exciting demand, Monte Carlo simulation, revenue management, portfolio theory,
> and risk metrics.
>
> **How to use**: Tell Claude Code: "Read TECHNIQUE_04_STOCHASTIC_PRICING.md and
> implement incrementally, starting from SP-1."
>
> **This is 12 sub-domains. Implement ALL of them. Do not skip any.**

---

## Key Papers (Referenced Throughout)

- Guizzardi et al. (2022). "Hotel dynamic pricing, stochastic demand and covid-19." *Annals of Tourism Research* 97:103495
- Holý & Tomanová (2018). "Estimation of OU Process Using Ultra-High-Frequency Data." arXiv:1811.09312
- arXiv:2505.03980 (2025). "Comparing SDE estimation techniques" — LSTM beats MLE for OU
- Cartea & Figueroa (2005). "Pricing in Electricity Markets: A Mean Reverting Jump Diffusion Model with Seasonality"
- arXiv:2506.04542 (2025). "Neural MJD" — neural parameterization of non-stationary jump-diffusion
- Anderson, Davison & Rasmussen (2004). "Real options in RM." *Naval Research Logistics* 51(5):686–703
- Gallego & van Ryzin (1994). "Optimal Dynamic Pricing of Inventories." *Management Science* 40(8):999–1020
- Han, Jentzen & E (2018). "Deep BSDE for HJB." *PNAS* 115(34):8505–8510
- Jiao et al. (2024). "PINNs for HJB." arXiv:2402.15592
- Gomes & Saude (2020). "MFG Approach to Price Formation." *Dynamic Games and Applications* 10:892–922
- Kyle (1985). "Continuous Auctions and Insider Trading." *Econometrica*
- Miscouridou et al. (2022). "Cox-Hawkes hybrid processes." arXiv:2210.11844
- Rockafellar & Uryasev (2000). "CVaR optimization." *J. Risk* 2(3):21–41
- Talluri & van Ryzin (2004). *The Theory and Practice of Revenue Management*. Springer
- Andersson et al. (2017). "Event portfolios: Markowitz for events." *Int. J. Event and Festival Management*
- Schur, Gönsch & Hassler (2019). "CVaR-optimal pricing is constant." *EJOR*
- Platanakis et al. (2023). "Black-Litterman for tourism." *Tourism Management*

---

## Architecture Overview

```
packages/
  pricing-engine/              — TypeScript core (types, API, orchestration)
    src/
      types.ts                 — All pricing domain types
      index.ts                 — Public API
  pricing-engine-wasm/         — Rust WASM (compute-heavy stochastic simulation)
    src/
      lib.rs                   — Entry point
      gbm.rs                   — GBM simulation
      ou.rs                    — Ornstein-Uhlenbeck simulation + calibration
      merton.rs                — Jump-diffusion simulation + calibration
      black_scholes.rs         — BS pricing + all Greeks + implied vol
      binomial.rs              — CRR binomial tree for American options
      monte_carlo.rs           — MC engine with variance reduction + QMC
      sobol.rs                 — Sobol sequence generation
      hawkes.rs                — Hawkes process simulation + estimation
      order_book.rs            — Limit order book engine
      hjb_solver.rs            — Finite difference HJB solver
      portfolio.rs             — Markowitz optimization + CVaR LP

apps/
  ml-api/
    src/
      pricing/
        demand_model.py        — Cox-Hawkes hybrid demand model (tick library)
        revenue_management.py  — EMSRb, bid-price control, choice-based RM
        mean_field.py          — Mean-field game solver (Picard iteration + deep)
        neural_mjd.py          — Neural parameterized jump-diffusion
        deep_bsde.py           — Deep BSDE solver for high-dimensional HJB
      routes/
        pricing.py             — FastAPI endpoints for all pricing operations

  web/
    src/
      components/
        pricing/
          FanChart.tsx         — Price path confidence cones
          OptionSurface.tsx    — 3D option value surface
          OrderBookDepth.tsx   — Real-time order book depth chart
          DemandHeatmap.tsx    — Time-of-day × day-of-week booking heatmap
          MonteCarloHist.tsx   — MC simulation distribution with VaR overlay
          RiskDashboard.tsx    — Portfolio efficient frontier + CVaR
          PricingControls.tsx  — Interactive parameter controls for venue manager
```

### Rust Crate Dependencies

```toml
# crates/pricing-engine-wasm/Cargo.toml
[dependencies]
wasm-bindgen = "0.2"
serde = { version = "1", features = ["derive"] }
serde-wasm-bindgen = "0.6"
RustQuant = "0.2"           # SDE simulation, BS pricing, auto-diff
stochastic-rs = "0.7"       # SIMD-accelerated GBM, OU, CIR, Poisson
statrs = "0.18"             # 30+ distributions
nalgebra = "0.33"           # Linear algebra
faer = "0.20"               # High-perf linear algebra for large matrices
argmin = "0.10"             # Optimization: L-BFGS, Nelder-Mead
sobol_burley = "0.5"        # Owen-scrambled Sobol sequences, SIMD
rand = "0.8"
rand_distr = "0.4"
```

### TypeScript Dependencies

```json
{
  "@uqee/black-scholes": "^1.0.7",
  "@stdlib/stdlib": "latest",
  "simple-statistics": "^7.8.8",
  "lobos": "^0.10.0",
  "lightweight-charts": "^5.1"
}
```

---

## SP-1: Stochastic Price Models (GBM + OU + Seasonal + Jump-Diffusion)

### What to Build

The composite venue pricing model combining mean-reversion, seasonality, and jumps:

```
dP(t) = θ(μ(t) - P(t))dt + σdW(t) + ξdJ(t)
```

where:
- P(t) is log-price
- μ(t) = μ₀ + Σₖ(aₖcos(2πkt/365) + bₖsin(2πkt/365)) is seasonal fair value
- θ is mean-reversion speed (half-life = ln(2)/θ)
- σ is diffusion volatility
- J is compound Poisson process (intensity λ, jump size ~ N(μ_J, σ_J²))

### Rust Implementation

```rust
// crates/pricing-engine-wasm/src/ou.rs

use wasm_bindgen::prelude::*;
use rand::prelude::*;
use rand_distr::{Normal, Poisson};

#[wasm_bindgen]
pub struct OUParams {
    pub theta: f64,     // Mean-reversion speed
    pub mu: f64,        // Long-term mean (base fair value)
    pub sigma: f64,     // Diffusion volatility
    // Seasonal Fourier components
    pub seasonal_a: Vec<f64>,  // cosine coefficients
    pub seasonal_b: Vec<f64>,  // sine coefficients
}

#[wasm_bindgen]
pub struct JumpParams {
    pub lambda: f64,    // Jump intensity (jumps per year)
    pub mu_j: f64,      // Mean jump size (log)
    pub sigma_j: f64,   // Jump volatility
}

/// Simulate composite OU + seasonal + jump-diffusion paths
/// Returns: n_paths × n_steps matrix (row-major Float64Array)
#[wasm_bindgen]
pub fn simulate_venue_price(
    s0: f64,
    ou: &OUParams,
    jumps: &JumpParams,
    t_years: f64,
    dt: f64,
    n_paths: u32,
) -> Vec<f64> {
    let n_steps = (t_years / dt) as usize;
    let mut paths = vec![0.0f64; (n_paths as usize) * (n_steps + 1)];
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let jump_normal = Normal::new(jumps.mu_j, jumps.sigma_j).unwrap();

    for p in 0..n_paths as usize {
        let offset = p * (n_steps + 1);
        paths[offset] = s0.ln(); // Work in log-price space

        for t in 1..=n_steps {
            let time = t as f64 * dt;

            // Seasonal fair value: μ(t) = μ₀ + Σ(aₖcos + bₖsin)
            let mut mu_t = ou.mu;
            for k in 0..ou.seasonal_a.len() {
                let freq = 2.0 * std::f64::consts::PI * (k + 1) as f64 / 365.0;
                mu_t += ou.seasonal_a[k] * (freq * time * 365.0).cos()
                      + ou.seasonal_b[k] * (freq * time * 365.0).sin();
            }

            // OU mean-reversion
            let drift = ou.theta * (mu_t - paths[offset + t - 1]) * dt;

            // Diffusion
            let diffusion = ou.sigma * (dt.sqrt()) * rng.sample::<f64, _>(normal);

            // Poisson jumps
            let n_jumps = rng.sample(Poisson::new(jumps.lambda * dt).unwrap()) as u32;
            let mut jump_sum = 0.0;
            for _ in 0..n_jumps {
                jump_sum += rng.sample::<f64, _>(jump_normal);
            }

            paths[offset + t] = paths[offset + t - 1] + drift + diffusion + jump_sum;
        }

        // Convert back from log-price to price
        for t in 0..=n_steps {
            paths[offset + t] = paths[offset + t].exp();
        }
    }

    paths
}
```

### Calibration Pipeline (Python)

```python
# apps/ml-api/src/pricing/calibration.py

def calibrate_venue_model(booking_history: list[dict]) -> dict:
    """
    5-step calibration pipeline:
    1. Remove seasonality via Fourier regression → seasonal coefficients
    2. Fit OU parameters to residuals via MLE (AR(1) representation)
    3. Detect jumps using threshold |rₜ| > 3σ√Δt
    4. Estimate jump parameters (λ, μ_J, σ_J) from detected events
    5. Re-estimate diffusion σ excluding jump periods

    Returns complete model parameters for simulation.
    """
    ...
```

### For Trades Hall Glasgow (Single-Venue)

Calibrate against Trades Hall's actual booking history:
- Seasonal pattern: wedding season (May-Oct), corporate season (Sep-Nov, Jan-Mar)
- Mean-reversion: Trades Hall has a "fair value" per room based on market position
- Jumps: Glasgow festivals (Celtic Connections, Edinburgh spillover), large conferences

### For Multi-Venue (Planetary Scale)

Each venue gets its own calibrated model. Cross-venue correlations captured by a
correlation matrix on the Brownian motions (like equity correlation in a multi-asset model).

---

## SP-2: Black-Scholes Booking Options (Hold Fees)

### What to Build

A client pays a non-refundable premium (hold fee) to reserve a venue slot with the
right but not obligation to confirm by a deadline. This IS an option contract.

### Parameter Mapping

| Black-Scholes | Venue Booking |
|---|---|
| Underlying S | Expected revenue from the slot at current demand |
| Strike K | Locked-in booking price |
| Expiry T | Confirmation deadline (days to decision) |
| Volatility σ | Demand uncertainty for that date/venue type |
| Risk-free rate r | Opportunity cost of holding slot unavailable |
| Premium C | Non-refundable hold fee |
| Delta Δ | How much hold fee changes per $1 demand shift |
| Theta Θ | Daily time decay of the hold — justifies declining refund schedules |
| Vega ν | Sensitivity to uncertainty — hold fees rise during volatile periods |
| Gamma Γ | Signals when to re-price hold fees more frequently |

### Rust Implementation

```rust
// crates/pricing-engine-wasm/src/black_scholes.rs

use wasm_bindgen::prelude::*;
use std::f64::consts::PI;

#[wasm_bindgen]
pub struct OptionResult {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
}

/// Standard normal CDF (Abramowitz & Stegun approximation, <1e-7 error)
fn norm_cdf(x: f64) -> f64 {
    let a1 = 0.254829592; let a2 = -0.284496736; let a3 = 1.421413741;
    let a4 = -1.453152027; let a5 = 1.061405429; let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / (2.0f64).sqrt();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * (-x*x).exp();
    0.5 * (1.0 + sign * y)
}

/// Standard normal PDF
fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// European call option price + all Greeks
#[wasm_bindgen]
pub fn black_scholes_call(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> OptionResult {
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();

    OptionResult {
        price: s * norm_cdf(d1) - k * (-r * t).exp() * norm_cdf(d2),
        delta: norm_cdf(d1),
        gamma: norm_pdf(d1) / (s * sigma * t.sqrt()),
        theta: -(s * norm_pdf(d1) * sigma) / (2.0 * t.sqrt())
               - r * k * (-r * t).exp() * norm_cdf(d2),
        vega: s * t.sqrt() * norm_pdf(d1),
        rho: k * t * (-r * t).exp() * norm_cdf(d2),
    }
}

/// European put option price + all Greeks
#[wasm_bindgen]
pub fn black_scholes_put(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> OptionResult {
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();

    OptionResult {
        price: k * (-r * t).exp() * norm_cdf(-d2) - s * norm_cdf(-d1),
        delta: norm_cdf(d1) - 1.0,
        gamma: norm_pdf(d1) / (s * sigma * t.sqrt()),
        theta: -(s * norm_pdf(d1) * sigma) / (2.0 * t.sqrt())
               + r * k * (-r * t).exp() * norm_cdf(-d2),
        vega: s * t.sqrt() * norm_pdf(d1),
        rho: -k * t * (-r * t).exp() * norm_cdf(-d2),
    }
}

/// Implied volatility via Newton-Raphson (bisection fallback)
#[wasm_bindgen]
pub fn implied_volatility(
    market_price: f64, s: f64, k: f64, t: f64, r: f64, is_call: bool
) -> f64 {
    let mut sigma = 0.3; // Initial guess
    for _ in 0..100 {
        let result = if is_call {
            black_scholes_call(s, k, t, r, sigma)
        } else {
            black_scholes_put(s, k, t, r, sigma)
        };
        let diff = result.price - market_price;
        if diff.abs() < 1e-8 { break; }
        let d_sigma = result.vega;
        if d_sigma.abs() < 1e-10 { break; }
        sigma -= diff / d_sigma;
        sigma = sigma.max(0.001).min(5.0);
    }
    sigma
}

/// CRR Binomial tree for American-style booking options
/// (exercisable anytime before deadline)
#[wasm_bindgen]
pub fn american_option_binomial(
    s: f64, k: f64, t: f64, r: f64, sigma: f64,
    n_steps: u32, is_call: bool,
) -> f64 {
    let dt = t / n_steps as f64;
    let u = (sigma * dt.sqrt()).exp();
    let d = 1.0 / u;
    let p = ((r * dt).exp() - d) / (u - d);
    let disc = (-r * dt).exp();

    // Build terminal payoffs
    let n = n_steps as usize;
    let mut values = vec![0.0; n + 1];
    for i in 0..=n {
        let spot = s * u.powi(i as i32) * d.powi((n - i) as i32);
        values[i] = if is_call {
            (spot - k).max(0.0)
        } else {
            (k - spot).max(0.0)
        };
    }

    // Backward induction with early exercise check
    for step in (0..n).rev() {
        for i in 0..=step {
            let hold = disc * (p * values[i + 1] + (1.0 - p) * values[i]);
            let spot = s * u.powi(i as i32) * d.powi((step - i) as i32);
            let exercise = if is_call {
                (spot - k).max(0.0)
            } else {
                (k - spot).max(0.0)
            };
            values[i] = hold.max(exercise); // American: can exercise early
        }
    }

    values[0]
}
```

### Critical Insight — Floor Price

Anderson et al. (2004) proved venues systematically discount too deeply. The option
framework provides a **floor price** below which the venue should NOT rent capacity,
even to fill empty slots — because the value of waiting for a higher-paying customer
exceeds the immediate revenue.

```rust
/// Compute the minimum price the venue should accept for a slot.
/// Below this, it's better to wait for a higher-paying booking.
#[wasm_bindgen]
pub fn compute_floor_price(
    current_demand_value: f64,  // S: what the slot is worth at current demand
    days_until_event: f64,      // T: time until the event date
    demand_volatility: f64,     // σ: how uncertain is future demand
    opportunity_rate: f64,      // r: rate of return on alternative use
) -> f64 {
    // The floor price is the present value of the option to wait
    let option_value = black_scholes_call(
        current_demand_value,
        current_demand_value, // ATM option
        days_until_event / 365.0,
        opportunity_rate,
        demand_volatility,
    );
    // Don't sell below: current demand - option value of waiting
    (current_demand_value - option_value.price).max(0.0)
}
```

---

## SP-3: Hamilton-Jacobi-Bellman Dynamic Pricing

### What to Build

The HJB equation finds the optimal price for each (remaining capacity, time until event)
pair. The venue owner maximizes expected revenue subject to capacity constraints and
stochastic demand.

### The HJB PDE

With remaining capacity n and time-to-event t, exponential demand D(p,t) = λ₀(t)e^{-αp}:

```
∂V/∂t + max_p { D(p,t) · [p + V(n-1,t) - V(n,t)] } = 0
V(n, 0) = 0  (unsold slots have zero salvage value)
```

The optimal price from the first-order condition:

```
p*(n,t) = 1/α + [V(n,t) - V(n-1,t)]
```

This is markup (1/α) plus the **shadow price of capacity**. As capacity becomes scarce,
the shadow price rises, pushing prices up automatically.

### Rust Implementation — Finite Difference Solver

```rust
// crates/pricing-engine-wasm/src/hjb_solver.rs

#[wasm_bindgen]
pub struct HJBResult {
    pub optimal_prices: Vec<f64>,   // n_capacity × n_time_steps
    pub value_function: Vec<f64>,   // n_capacity × n_time_steps
    pub shadow_prices: Vec<f64>,    // n_capacity × n_time_steps (marginal value of capacity)
}

/// Solve the HJB via policy iteration (converges in 5-10 iterations)
#[wasm_bindgen]
pub fn solve_hjb_pricing(
    max_capacity: u32,
    time_horizon_days: f64,
    dt: f64,
    base_demand_rate: f64,      // λ₀ base arrival rate
    price_sensitivity: f64,      // α in D(p) = λ₀·e^{-αp}
    seasonal_factors: Vec<f64>,  // multipliers for each time step
) -> HJBResult {
    let n_cap = max_capacity as usize;
    let n_steps = (time_horizon_days / dt) as usize;

    // V[n][t] = value function
    let mut v = vec![vec![0.0f64; n_steps + 1]; n_cap + 1];
    // p[n][t] = optimal price
    let mut p = vec![vec![0.0f64; n_steps + 1]; n_cap + 1];

    // Terminal condition: V(n, 0) = 0 for all n
    // Already initialized to 0

    // Policy iteration
    for _iter in 0..20 {
        let v_old = v.clone();

        // Backward in time
        for t in (0..n_steps).rev() {
            let lambda_t = base_demand_rate * seasonal_factors.get(t).unwrap_or(&1.0);

            for n in 1..=n_cap {
                // Optimal price from FOC: p* = 1/α + shadow_price
                let shadow = v[n][t + 1] - v[n - 1][t + 1];
                let p_star = (1.0 / price_sensitivity + shadow).max(0.0);

                // Demand at optimal price
                let demand = lambda_t * (-price_sensitivity * p_star).exp();

                // Value function update
                v[n][t] = demand * dt * (p_star + v[n - 1][t + 1] - v[n][t + 1])
                         + v[n][t + 1];

                p[n][t] = p_star;
            }
        }

        // Check convergence
        let max_diff: f64 = v.iter().zip(v_old.iter())
            .flat_map(|(a, b)| a.iter().zip(b.iter()))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        if max_diff < 1e-6 { break; }
    }

    // Flatten for WASM return
    let mut prices = Vec::with_capacity((n_cap + 1) * (n_steps + 1));
    let mut values = Vec::with_capacity((n_cap + 1) * (n_steps + 1));
    let mut shadows = Vec::with_capacity((n_cap + 1) * (n_steps + 1));

    for n in 0..=n_cap {
        for t in 0..=n_steps {
            prices.push(p[n][t]);
            values.push(v[n][t]);
            shadows.push(if n > 0 { v[n][t] - v[n-1][t] } else { 0.0 });
        }
    }

    HJBResult { optimal_prices: prices, value_function: values, shadow_prices: shadows }
}
```

### Python — Deep BSDE Solver for High-Dimensional HJB

When the venue has many rooms × many time slots × many planner types, the state
space is too large for finite differences. Use the Deep BSDE method (Han et al. 2018).

```python
# apps/ml-api/src/pricing/deep_bsde.py

"""
Deep BSDE solver for high-dimensional HJB equations.

Reformulates HJB as a backward SDE, neural networks approximate ∇V at each step.
Achieves <0.5% relative error on 100-dimensional HJB.

Architecture:
  - Input: (time, state) at each step
  - Network: 2-layer MLP predicting gradient ∇V(t, x)
  - Loss: terminal condition mismatch + HJB residual
  - Optimizer: Adam with exponential LR decay

Use when: venue has >5 rooms with independent pricing,
          or >10 time slot categories, or >3 planner type segments.
"""
```

---

## SP-4: Mean-Field Games — Competing Planners

### What to Build

When N planners compete for venue slots, model the Nash equilibrium via the
coupled HJB-Fokker-Planck system. The equilibrium venue price emerges as a
Lagrange multiplier for supply-demand balance.

### The System

**(a) HJB (backward):** Each planner optimizes their bidding strategy:
```
-∂V/∂t - ν∆V + H(x, ∇V, m) = 0
```

**(b) Fokker-Planck (forward):** Population distribution evolves:
```
∂m/∂t - ν∆m - div(m · ∇_pH) = 0
```

**(c) Fixed-point:** Distribution m in HJB = distribution generated by optimal V.

### Python Implementation — Picard Iteration

```python
# apps/ml-api/src/pricing/mean_field.py

def solve_mfg_picard(
    n_planners: int,
    venue_capacity: int,
    time_horizon: float,
    congestion_param: float,    # How much competition raises prices
    planner_budgets: np.ndarray,
    max_iterations: int = 100,
    damping: float = 0.5,       # Damped updates for stability
) -> dict:
    """
    Picard iteration for the coupled HJB-FPK system:

    1. Initialize distribution m⁰ (e.g., uniform over planner states)
    2. Solve HJB backward given m^k → get value function V^k and optimal α^k
    3. Solve FPK forward given α^k → get new distribution m^{k+1}
    4. Damp: m^{k+1} ← damping·m^{k+1} + (1-damping)·m^k
    5. Repeat until convergence

    Returns:
        equilibrium_prices: price per slot at equilibrium
        planner_strategies: optimal bidding strategy per planner type
        distribution: equilibrium distribution of planner states
    """
    ...
```

### Deep Galerkin Method for High-Dimensional MFG

```python
"""
For >3 planner types with continuous state spaces, use the Deep Galerkin Method
(Assouli & Missaoui, 2023, arXiv:2301.02877):

Two neural networks:
  V_θ(x, t) — approximates value function
  m_φ(x, t) — approximates population distribution

Loss = ||HJB residual||² + ||FPK residual||² + ||boundary conditions||²

Scales to 300 dimensions with a single hidden layer.
"""
```

---

## SP-5: Order Book for Venue Availability Market

### What to Build

A limit order book that organizes venue supply and demand transparently — like a
financial exchange but for venue slots.

### Rust Implementation

```rust
// crates/pricing-engine-wasm/src/order_book.rs

#[wasm_bindgen]
pub struct Order {
    pub id: u64,
    pub side: OrderSide,      // Bid (planner wants) or Ask (venue offers)
    pub price: f64,
    pub quantity: u32,         // Number of slots/hours
    pub timestamp: f64,
    pub venue_id: String,
    pub slot_date: String,
    pub order_type: OrderType, // Limit, Market, IOC, FOK, GTD
}

#[wasm_bindgen]
pub enum OrderSide { Bid, Ask }
#[wasm_bindgen]
pub enum OrderType { Limit, Market, IOC, FOK, GTD }

#[wasm_bindgen]
pub struct OrderBook {
    bids: BTreeMap<OrderedFloat<f64>, VecDeque<Order>>,  // price-time priority
    asks: BTreeMap<OrderedFloat<f64>, VecDeque<Order>>,
}

#[wasm_bindgen]
impl OrderBook {
    pub fn new() -> Self { ... }

    /// Add a limit order. Returns fills if crossing.
    pub fn add_limit_order(&mut self, order: Order) -> Vec<Fill> { ... }

    /// Execute a market order immediately at best available price.
    pub fn add_market_order(&mut self, order: Order) -> Vec<Fill> { ... }

    /// Current best bid and ask
    pub fn best_bid(&self) -> Option<f64> { ... }
    pub fn best_ask(&self) -> Option<f64> { ... }

    /// Bid-ask spread — signal of demand/supply mismatch
    pub fn spread(&self) -> Option<f64> { ... }

    /// Order book imbalance: (Q_bid - Q_ask) / (Q_bid + Q_ask)
    /// Positive = more demand than supply → price should rise
    /// THE single most actionable signal for dynamic pricing
    pub fn imbalance(&self) -> f64 { ... }

    /// Kyle's lambda: price impact per booking
    /// λ = ΔAvgPrice / ΔBookings for comparable venues
    /// High λ = tight market, few substitutes
    pub fn kyle_lambda(&self, recent_trades: &[Fill]) -> f64 { ... }

    /// Depth at each price level (for visualization)
    pub fn depth(&self, n_levels: u32) -> OrderBookDepth { ... }

    /// Almgren-Chriss optimal inventory release schedule
    /// Determines how a venue should release slots over time
    pub fn optimal_release_schedule(
        &self,
        total_slots: u32,
        time_horizon: f64,
        risk_aversion: f64,
        volatility: f64,
    ) -> Vec<(f64, u32)> {
        // x_j = sinh(κ(T - t_j)) / sinh(κT) * X₀
        // High κ̃ (volatile demand): front-load release
        // Low κ̃ (stable demand): distribute evenly
        ...
    }
}
```

### Glosten-Milgrom Spread for Platform Commission

The bid-ask spread compensates for adverse selection — planners who know about
unannounced events booking before prices rise. Platform commission should cover
at least the adverse selection cost:

```
spread = E[V|buy] - E[V|sell] ∝ π (fraction of informed traders)
```

---

## SP-6: Stochastic Demand Models (Poisson → Hawkes → Cox)

### What to Build

Three progressively sophisticated demand models:

**Non-homogeneous Poisson** (baseline):
```
λ(t) = λ₀ · [1 + α·sin(2πt/7)] · seasonal(t)
```
Independent arrivals, good for stable venues.

**Hawkes self-exciting process** (FOMO/social proof):
```
λ*(t) = μ + Σ_{tᵢ < t} α · e^{-β(t-tᵢ)}
```
Each booking increases probability of future bookings. Branching ratio n* = α/β
measures virality. A venue posting a booking confirmation on Instagram generates
a burst of inquiries captured by the exponential kernel.

**Cox-Hawkes hybrid** (state of the art):
Log-Gaussian Cox background rate + Hawkes self-excitation captures both endemic
demand fluctuations AND booking cascades simultaneously.

### Python Implementation

```python
# apps/ml-api/src/pricing/demand_model.py

from tick.hawkes import HawkesExpKern, HawkesSumExpKern
from tick.plot import plot_point_process

def fit_hawkes_demand(booking_timestamps: list[float]) -> dict:
    """
    Fit Hawkes process to booking arrival timestamps.
    Uses tick library (C++ backend, multi-core).

    Returns: μ (background rate), α (excitation), β (decay),
             branching_ratio (α/β — virality metric)
    """
    hawkes = HawkesExpKern(decays=[[1.0]], max_iter=1000)
    hawkes.fit([np.array(booking_timestamps)])
    mu = hawkes.baseline[0]
    alpha = hawkes.adjacency[0, 0]
    beta = hawkes.decays[0, 0]
    return {
        'mu': float(mu),
        'alpha': float(alpha),
        'beta': float(beta),
        'branching_ratio': float(alpha / beta),
        'half_life_hours': float(np.log(2) / beta),
    }

def fit_cox_hawkes_hybrid(
    booking_timestamps: list[float],
    covariates: np.ndarray,  # weather, day-of-week, seasonality
) -> dict:
    """
    Cox-Hawkes hybrid (Miscouridou et al. 2022, arXiv:2210.11844):
    Log-Gaussian Cox Process background + Hawkes self-excitation.

    Captures BOTH endemic demand fluctuations AND booking cascades.
    """
    ...
```

### Rust Implementation (for WASM)

```rust
// crates/pricing-engine-wasm/src/hawkes.rs

/// Simulate Hawkes process via Ogata's thinning algorithm
#[wasm_bindgen]
pub fn simulate_hawkes(
    mu: f64,        // Background rate
    alpha: f64,     // Excitation magnitude
    beta: f64,      // Decay rate
    t_max: f64,     // Simulation horizon
) -> Vec<f64> {
    let mut events = Vec::new();
    let mut t = 0.0;
    let mut lambda_star = mu; // Upper bound on intensity

    let mut rng = thread_rng();

    while t < t_max {
        // Generate next candidate event (thinning)
        let u: f64 = rng.gen();
        t -= (1.0 - u).ln() / lambda_star;
        if t >= t_max { break; }

        // Accept/reject
        let lambda_t = mu + alpha * events.iter()
            .map(|&ti: &f64| (-beta * (t - ti)).exp())
            .sum::<f64>();

        let d: f64 = rng.gen();
        if d * lambda_star <= lambda_t {
            events.push(t);
            lambda_star = lambda_t + alpha; // Update bound after new event
        }
    }

    events
}
```

---

## SP-7: Monte Carlo Engine with Variance Reduction + QMC

### What to Build

Efficient Monte Carlo for pricing path-dependent booking options and simulating
revenue scenarios.

### Rust Implementation

```rust
// crates/pricing-engine-wasm/src/monte_carlo.rs

use sobol_burley::sample;

#[wasm_bindgen]
pub struct MCConfig {
    pub n_paths: u32,
    pub use_antithetic: bool,
    pub use_control_variate: bool,
    pub use_sobol: bool,        // Quasi-MC via Sobol sequences
    pub confidence_level: f64,  // For VaR/CVaR computation
}

#[wasm_bindgen]
pub struct MCResult {
    pub mean: f64,
    pub std_error: f64,
    pub var: f64,           // Value at Risk
    pub cvar: f64,          // Conditional VaR
    pub percentiles: Vec<f64>,  // [5th, 25th, 50th, 75th, 95th]
    pub paths: Vec<f64>,    // All terminal values (for histogram)
}

/// Monte Carlo revenue simulation with all variance reduction techniques
#[wasm_bindgen]
pub fn simulate_revenue_mc(
    model_params: &OUParams,
    jump_params: &JumpParams,
    config: &MCConfig,
) -> MCResult {
    // 1. Generate random numbers (Sobol or pseudorandom)
    // 2. If antithetic: generate (Z, -Z) pairs
    // 3. Simulate price paths using composite model
    // 4. Compute terminal revenue for each path
    // 5. If control variate: adjust using E[N(T)] = λT
    //    θ̂_CV = θ̂ - c*(N̄(T) - λT)  — variance reduction >90%
    // 6. Compute statistics: mean, std error, VaR, CVaR, percentiles
    ...
}
```

### Sobol Quasi-Monte Carlo

Sobol sequences fill the unit hypercube more uniformly than pseudorandom.
Koksma-Hlawka bound: error O((log N)^s / N) vs O(1/√N) for standard MC.
For smooth venue revenue functions: 10-1000× speedup.

```rust
// Using sobol_burley crate (Owen-scrambled, SIMD-accelerated)
fn sobol_point(index: u32, dimension: u32, seed: u32) -> f64 {
    sobol_burley::sample(index, dimension, seed) as f64
}
```

---

## SP-8: Revenue Management (EMSRb → Bid-Price → Choice-Based → Deep RL)

### What to Build

The full revenue management stack, from classical to cutting-edge.

### Python Implementation

```python
# apps/ml-api/src/pricing/revenue_management.py

def emsrb(
    fare_classes: list[dict],  # [{name, revenue, mean_demand, std_demand}, ...]
) -> dict:
    """
    EMSRb (Belobaba 1989): Expected Marginal Seat Revenue for n fare classes.

    Venue mapping: fare classes = event types
      Wedding: $15K, Corporate: $8K, Party: $3K
    Seats = time slots. Nested control: weddings can access any slot,
    parties only see unreserved slots.

    Returns protection levels for each class.
    Consistently within 0.5% of optimal.
    """
    ...

def bid_price_control(
    resources: list[dict],     # [{name, capacity}, ...] — rooms, time slots
    requests: list[dict],      # [{revenue, resource_usage: {room: n, ...}}, ...]
) -> dict:
    """
    Bid-price control via Deterministic Linear Program.

    Shadow prices λᵢ for each resource. Accept a multi-resource request
    only if revenue ≥ Σ aᵢⱼ·λᵢ.

    Venue application: a wedding requiring Ballroom + Garden + Prep Room
    is accepted only if total fee exceeds sum of bid prices across all three.
    """
    ...

def choice_based_rm(
    offer_sets: list[list[str]],   # Possible product assortments to offer
    utility_params: dict,           # MNL parameters per product
    capacity: dict,                 # Resource capacities
) -> dict:
    """
    Choice-Based RM via CDLP (Liu & van Ryzin 2008).

    Customers choose based on Multinomial Logit:
    P(choose j | offer set S) = e^{vⱼ} / (v₀ + Σ_{k∈S} e^{vₖ})

    Captures customer substitution: if preferred slot is full,
    customer may choose an alternative rather than leaving.

    Column generation determines optimal offer sets.
    """
    ...

def deep_rl_pricing(
    env_config: dict,
) -> dict:
    """
    Deep RL for revenue management (SAC or PPO).

    State: (remaining capacity per room, days until each date, demand forecast)
    Action: price vector for each room-date combination
    Reward: booking revenue

    Achieves >93% of theoretical optimal (Shihab & Wei 2022).
    Outperforms EMSRb under model misspecification.
    """
    ...
```

---

## SP-9: Portfolio Theory — Booking Mix Optimization

### What to Build

Markowitz portfolio optimization for the venue's mix of event types, plus CVaR
risk management.

### Rust Implementation

```rust
// crates/pricing-engine-wasm/src/portfolio.rs

#[wasm_bindgen]
pub struct PortfolioResult {
    pub weights: Vec<f64>,           // Optimal allocation per event type
    pub expected_revenue: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub var_95: f64,
    pub cvar_95: f64,
    pub efficient_frontier: Vec<(f64, f64)>,  // (volatility, return) pairs
}

/// Markowitz efficient frontier for event type diversification
///
/// Treat each event type as an asset class:
///   Wedding: high revenue, seasonal (May-Oct), weekend-heavy
///   Corporate: medium revenue, fiscal-year-driven (Sep-Nov, Jan-Mar), weekday
///   Party: low revenue, year-round, Friday-Saturday
///   Conference: high revenue, sporadic, multi-day
///
/// Diversification insight: Wedding demand has LOW or NEGATIVE correlation
/// with corporate events. A venue hosting only weddings faces extreme
/// seasonality risk. Venues diversifying across types achieve 40-60%
/// higher annual revenue (industry data).
#[wasm_bindgen]
pub fn optimize_booking_mix(
    expected_returns: Vec<f64>,  // Expected revenue per event type
    covariance_matrix: Vec<f64>, // Flattened n×n covariance
    n_types: u32,
    n_frontier_points: u32,
) -> PortfolioResult {
    // Quadratic programming: min w'Σw s.t. w'μ ≥ R, w'1 = 1, w ≥ 0
    // Sweep R from min to max expected return
    // At each R, solve the QP for minimum variance
    ...
}

/// CVaR optimization via Rockafellar-Uryasev LP formulation
///
/// CVaR_α = E[R | R ≤ VaR_α] — expected revenue in worst (1-α)% of outcomes
/// Unlike VaR, CVaR is sub-additive (always rewards diversification)
///
/// LP formulation:
///   min  α + (1/(S·(1-β))) · Σₛ uₛ
///   s.t. uₛ ≥ -(w'rₛ) - α,  uₛ ≥ 0,  w'1 = 1,  w ≥ 0
///
/// Handles 100s of scenarios × dozens of event types in seconds.
#[wasm_bindgen]
pub fn optimize_cvar(
    scenarios: Vec<f64>,         // S scenarios × n_types revenue matrix
    n_scenarios: u32,
    n_types: u32,
    confidence: f64,             // β (e.g., 0.95 for 95% CVaR)
    min_expected_return: f64,
) -> PortfolioResult {
    ...
}

/// Black-Litterman for incorporating venue manager views
///
/// Combine market equilibrium with manager beliefs:
///   "Corporate events will increase 15% next quarter" (absolute view)
///   "Weddings will outperform concerts by $2K/event" (relative view)
///
/// E[r] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ · [(τΣ)⁻¹π + P'Ω⁻¹Q]
#[wasm_bindgen]
pub fn black_litterman(
    market_weights: Vec<f64>,    // Current booking mix (equilibrium)
    covariance: Vec<f64>,        // n×n covariance
    views_p: Vec<f64>,           // View pick matrix (k×n)
    views_q: Vec<f64>,           // View expected returns (k)
    view_confidence: Vec<f64>,   // Ω diagonal (uncertainty per view)
    risk_aversion: f64,          // δ
    tau: f64,                    // Uncertainty scaling (typically 0.025-0.05)
    n_types: u32,
    n_views: u32,
) -> Vec<f64> {
    // Returns posterior expected returns incorporating manager views
    ...
}
```

---

## SP-10: Visualization Suite

### Fan Chart — Price Path Confidence Cones

```typescript
// apps/web/src/components/pricing/FanChart.tsx

/**
 * Simulate N price paths via WASM, compute percentile bands at each timestep.
 * Render as layered filled areas: 5th-95th (lightest), 25th-75th, median line.
 *
 * Interactive:
 * - Slider: adjust volatility σ → watch cone widen/narrow in real-time
 * - Toggle: show/hide jump events as vertical markers
 * - Hover: exact price and confidence interval at any time point
 *
 * Use visx (@visx/shape Area) for custom D3-level control in React.
 */
```

### 3D Option Value Surface

```typescript
// apps/web/src/components/pricing/OptionSurface.tsx

/**
 * Interactive 3D surface: option value vs (spot price × time to expiry)
 * or (spot price × volatility).
 *
 * Use react-plotly.js with type: 'surface', colorscale: 'Viridis'.
 * Rotate, zoom, hover for exact values.
 * Second view: Greeks surface (delta, gamma as 3D surface).
 */
```

### Order Book Depth Chart

```typescript
// apps/web/src/components/pricing/OrderBookDepth.tsx

/**
 * Real-time order book depth chart using TradingView Lightweight Charts.
 * Bid side (green, left), Ask side (red, right), staircase curve.
 * Live updates via WebSocket as orders come in.
 *
 * Below the depth: imbalance indicator bar showing (Q_bid - Q_ask)/(Q_bid + Q_ask)
 * with color gradient from red (excess supply) to green (excess demand).
 */
```

### Demand Intensity Heatmap

```typescript
// apps/web/src/components/pricing/DemandHeatmap.tsx

/**
 * Time-of-day × day-of-week heatmap using Nivo @nivo/heatmap.
 * Color intensity = booking arrival rate (from Hawkes model).
 * Second view: calendar heatmap (Cal-HeatMap style) showing demand over months.
 * Third view: Hawkes intensity plot showing self-exciting bursts.
 */
```

### Monte Carlo Distribution with VaR Overlay

```typescript
// apps/web/src/components/pricing/MonteCarloHist.tsx

/**
 * Histogram of terminal revenue from MC simulation.
 * VaR line (dashed vertical), CVaR shaded tail region.
 * Interactive sliders: σ, μ, time horizon, confidence level.
 * Reactive re-computation via WASM when parameters change.
 */
```

### Risk Dashboard — Efficient Frontier

```typescript
// apps/web/src/components/pricing/RiskDashboard.tsx

/**
 * Markowitz efficient frontier curve (volatility vs expected return).
 * Current booking mix plotted as a point on the chart.
 * Optimal portfolio highlighted. Individual event types as scatter points.
 * Tangent line showing maximum Sharpe ratio portfolio.
 *
 * Second panel: CVaR comparison across booking mix strategies.
 * Third panel: Black-Litterman adjusted returns after manager views.
 */
```

---

## Integration with Other Techniques

- **Category Theory** (CT): Pricing morphisms compose — `calibrate ∘ simulate ∘ price ∘ hedge`
  is a pipeline in the pricing category. The option pricing functor maps between the
  demand category and the pricing category.
- **Optimal Transport** (OT): The cost matrix for venue-event matching incorporates
  stochastic prices — the OT cost of assigning an event to a venue includes the
  expected price trajectory, not just the spot price.
- **TDA**: Regime change detection (TDA-4) triggers price model recalibration.
  When persistence landscape norms spike, the Hawkes/OU parameters need re-estimation.
- **Physics-Inspired Solvers** (next technique): Simulated annealing optimizes the
  multi-room, multi-date pricing problem that HJB alone can't handle efficiently.

---

## Session Management

This is the largest technique. Split across sessions:

1. **SP-1** (OU + GBM + jump-diffusion + calibration pipeline) — 1-2 sessions
2. **SP-2** (Black-Scholes + Greeks + American binomial + floor price) — 1 session
3. **SP-3** (HJB finite difference solver + deep BSDE stub) — 1-2 sessions
4. **SP-4** (Mean-field game: Picard iteration + deep Galerkin stub) — 1 session
5. **SP-5** (Order book engine + Kyle's lambda + Almgren-Chriss) — 1-2 sessions
6. **SP-6** (Hawkes + Cox-Hawkes demand models) — 1 session
7. **SP-7** (Monte Carlo engine + variance reduction + Sobol QMC) — 1 session
8. **SP-8** (Revenue management: EMSRb + bid-price + choice-based + RL stub) — 1-2 sessions
9. **SP-9** (Portfolio optimization + CVaR LP + Black-Litterman) — 1 session
10. **SP-10** (Full visualization suite) — 1-2 sessions

Each session: implement in Rust (WASM) where compute-heavy, Python where ML-heavy,
TypeScript for orchestration and visualization. Write tests. Update PROGRESS.md.
Commit after each section.

Total: ~10-14 Claude Code sessions for the full pricing engine.
