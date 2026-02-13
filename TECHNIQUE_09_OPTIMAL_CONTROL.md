# TECHNIQUE_09_OPTIMAL_CONTROL.md — Optimal Control for Venue Operations

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
> are complete. Implements the full optimal control stack: LQR/LQG resource regulation,
> MPC dynamic pricing, Pontryagin's Maximum Principle, DP/HJB booking acceptance,
> RL approximate control, feedback linearization, multi-agent venue networks,
> crowd flow control, stochastic/robust MPC, optimal experiment design, and
> real-time control architecture.
>
> **How to use**: Tell Claude Code: "Read TECHNIQUE_09_OPTIMAL_CONTROL.md
> and implement incrementally, starting from OC-1."
>
> **This is 12 sub-domains. Implement ALL of them. Do not skip any.**

---

## Venue Operations as a Control System

The platform is a **hierarchical optimal control stack** coupling three timescales:
1. **Slow/strategic** (seasonal): pricing bands, staffing contracts
2. **Medium-horizon** (weekly): marketing schedules, staff rosters
3. **Fast/operational** (hourly/daily): prices, capacity gating, crowd ops

Canonical controlled stochastic dynamical system:

```
x_{t+1} = f(x_t, u_t, d_t, θ_t) + w_t
y_t = h(x_t) + v_t
```

Where:
- x_t: latent state (occupancy, staffing, inventory, revenue rate)
- u_t: controls (price adjustment, staff allocation, marketing spend)
- d_t: exogenous drivers (event calendar, day-of-week, competitors, weather)
- w_t, v_t: process and measurement noise
- θ_t: slowly-varying parameters (elasticity drift, cancellation rate drift)

### Architecture Selection by Problem Type

- **LQR/LQG**: resource regulation (staff/inventory stabilization), soft capacity smoothing
- **MPC/NMPC**: pricing under constraints, economic objectives — THE central workhorse
- **DP/HJB/Optimal stopping**: booking acceptance, capacity protection
- **RL**: complex environments with high model uncertainty, under a safety layer

---

## Key Papers & References

- Kalman (1960). Linear Filtering and Prediction
- Pontryagin et al. (1962). The Mathematical Theory of Optimal Processes
- Bellman (1957). Dynamic Programming
- Crandall & Lions (1983). Viscosity Solutions of HJB PDEs
- Helbing & Molnár (1995). Social Force Model. Physical Review E
- Hughes (2002). Continuum Theory for Crowd Flow
- Kumar & Varaiya (1986). Stochastic Systems (dual control)
- Rawlings & Mayne (2017). Model Predictive Control: Theory and Practice
- Gasse et al. (2019). GNN for branch-and-bound. NeurIPS
- Russo & Van Roy (2018). Information-Directed Sampling

---

## Architecture Overview

```
apps/
  ml-api/
    src/
      control/
        lqr/
          discrete_lqr.py             — DARE-based LQR with venue state
          continuous_lqr.py            — CARE-based continuous LQR
          time_varying_lqr.py          — Finite-horizon backward Riccati
          tracking_lqr.py              — Integral action + reference feedforward
          lqg.py                       — LQR + Kalman filter (separation principle)
          q_r_tuning.py                — Inverse-tolerance Q/R construction
        mpc/
          linear_mpc.py                — CVXPY + OSQP QP-based pricing MPC
          economic_mpc.py              — Maximize profit (not track reference)
          nonlinear_mpc.py             — CasADi / do-mpc NMPC with demand model
          robust_mpc.py                — Tube MPC for hard constraints
          stochastic_mpc.py            — Chance-constrained SMPC
          explicit_mpc.py              — Piecewise-affine lookup for edge/browser
          mixed_integer_mpc.py         — MI-MPC for discrete decisions
        pmp/
          hamiltonian.py               — Pontryagin's Maximum Principle formulation
          shooting.py                  — Single/multiple shooting BVP solvers
          collocation.py               — Direct collocation → large sparse NLP
          bang_bang.py                  — Switching function analysis
        dp/
          bellman.py                   — Discrete-time Bellman recursion
          hjb.py                       — HJB PDE viscosity solution
          approximate_dp.py            — Fitted value iteration, NN value function
          bid_price.py                 — LP-based bid-price control
          optimal_stopping.py          — Threshold policies for booking acceptance
        rl/
          sac_pricing.py               — SAC for continuous pricing
          ppo_operations.py            — PPO for mixed continuous/discrete
          offline_rl.py                — CQL, IQL, Decision Transformer
          safe_rl.py                   — Constrained MDP, safety layer
          reward_design.py             — Shaped reward with economic + operational terms
          model_based_rl.py            — Learn dynamics → MPC planning
          rl_mpc_hybrid.py             — RL proposes, MPC verifies feasibility
        nonlinear/
          feedback_linearization.py    — Lie derivative, relative degree
          backstepping.py              — Recursive strict-feedback stabilization
          sliding_mode.py              — Robust, matched uncertainty
          adaptive_control.py          — MRAC / L1 adaptive for parameter drift
        multi_agent/
          decentralized_mpc.py         — Per-venue local MPC with coordination
          nash_equilibrium.py          — Game-theoretic pricing
          stackelberg.py               — Platform-leader, venues-followers
          mean_field.py                — Mean-field control for many venues
          mappo.py                     — Multi-Agent PPO (centralized critic)
          qmix.py                      — Value decomposition (monotonic mixing)
        crowd/
          hughes_model.py              — Continuum PDE + eikonal potential
          social_force.py              — Helbing microscopic model
          evacuation_control.py        — Dynamic signage, exit assignment MPC
          density_constraints.py       — ρ(x,t) ≤ ρ_max enforcement
        stochastic/
          stochastic_dp.py             — E[Σ r(x_t, u_t)] with random transitions
          risk_sensitive.py            — Exponential utility, CVaR
          distributionally_robust.py   — Wasserstein DRO
          chance_constrained.py        — P(violation) ≤ ε
          scenario_mpc.py              — Scenario tree / scenario fan
        experiment/
          dual_control.py              — Explore-exploit as control
          information_directed.py      — IDS (regret vs information gain)
          active_learning_mpc.py       — Economic objective + information value
          thompson_pricing.py          — Constrained Thompson sampling
          bayesian_optimization.py     — Safe BO for promotional experiments
        architecture/
          control_loop.py              — Sense→Estimate→Decide→Act→Observe
          sample_rates.py              — Pricing hourly, staffing daily, marketing weekly
          state_estimation.py          — Multi-sensor fusion (Kalman/MHE)
          fault_tolerance.py           — Graceful degradation
          visualization.py             — Occupancy bands, shadow prices, what-if curves
      routes/
        control.py                     — FastAPI endpoints

packages/
  control-core/                        — TypeScript types + browser-side controllers
    src/
      types.ts                         — ControlState, ControlAction, MPC types
      controllers/
        linear_feedback.ts             — LQR gain exported for browser preview
        explicit_mpc.ts                — Piecewise-affine MPC lookup
        policy_inference.ts            — ONNX neural policy for browser
      visualization/
        OccupancyTrajectory.tsx        — Predicted occupancy bands + constraints
        PricingDashboard.tsx           — MPC predictions, shadow prices
        ControlExplainer.tsx           — Why system chose these actions
        WhatIfCurves.tsx               — Price vs occupancy/revenue interactive
        CrowdDensityMap.tsx            — Real-time density visualization
```

### Python Dependencies

```
scipy>=1.17.0                  # DARE/CARE Riccati solvers
control==0.10.2                # python-control: LQR, LQG, LQE, state-space
cvxpy==1.8.1                   # Convex optimization modeling
osqp==1.1.1                    # ADMM QP solver (warm-start, cached factorization)
clarabel==0.11.1               # Interior-point conic/QP solver (also in Rust)
casadi==3.7.2                  # Symbolic modeling, AD, NLP integration
do-mpc==5.1.1                  # Robust NMPC + MHE toolbox
gymnasium==1.2.3               # RL environment API
stable-baselines3==2.7.1       # PPO, SAC, TD3, DQN
d3rlpy==2.8.1                  # Offline RL (CQL, IQL, Decision Transformer)
ray[rllib]==2.53.0             # Scalable RL training
ortools==9.15.6755             # MIP/LP for network RM formulations
hj-reachability==0.7.0         # HJB PDE viscosity solutions
jupedsim==1.3.1                # Pedestrian dynamics simulation
numpy>=2.0
```

### Rust Dependencies

```toml
[dependencies]
clarabel = "0.11"              # QP/conic solver (native Rust)
control-sys = "0.1"            # State-space modeling
lqr = "0.1"                    # LQR controller
kalman_filters = "1.0"         # Kalman filter variants
```

---

## OC-1: LQR for Venue Resource Management

### What to Build

Linear quadratic regulation for staff/inventory stabilization around
operating targets. Foundation for the entire control hierarchy.

```python
# apps/ml-api/src/control/lqr/discrete_lqr.py

import numpy as np
from scipy.linalg import solve_discrete_are

class VenueLQR:
    """
    State: x = [occupancy, staff_level, inventory, revenue_rate]
    Control: u = [pricing_adjustment, staff_allocation, marketing_spend]

    Infinite-horizon discrete LQR:
      J = Σ_{t=0}^∞ (x_t^T Q x_t + u_t^T R u_t + 2 x_t^T N u_t)
      u_t = -K x_t
      K = (R + B^T P B)^{-1} (B^T P A + N^T)
      P solves DARE: P = Q + A^T P A - (A^T P B + N)(R + B^T P B)^{-1}(B^T P A + N^T)

    Q/R tuning via inverse-tolerance (physically meaningful):
      Q = diag(w_o/σ_o², w_s/σ_s², w_i/σ_i², w_r/σ_r²)
      R = diag(w_Δp/σ_Δp², w_Δs/σ_Δs², w_m/σ_m²)

    Where σ values are acceptable deviations:
      σ_o = 5% occupancy points
      σ_s = ±2 staff
      σ_i = ±10 inventory units
      σ_r = ±£200/hour revenue rate

    w values are dimensionless priority weights.
    This keeps Q,R interpretable and stable across venues.

    LQR does NOT handle constraints natively.
    Clipping controls breaks optimality → graduate to MPC.

    Libraries:
      scipy 1.17.0: solve_discrete_are (DARE), solve_continuous_are (CARE)
      control 0.10.2: dlqr(A,B,Q,R,N) — supports cross-term N
    """

    def __init__(self, A, B, Q, R, N=None):
        self.A, self.B = A, B
        self.Q, self.R = Q, R
        self.N = N if N is not None else np.zeros((A.shape[0], B.shape[1]))

        # Solve DARE
        self.P = solve_discrete_are(A, B, Q, R, e=None, s=self.N)
        # Compute gain
        self.K = np.linalg.inv(
            R + B.T @ self.P @ B) @ (B.T @ self.P @ A + self.N.T)

    def policy(self, x):
        return -self.K @ x

    def simulate(self, x0, T, clip_bounds=None):
        """
        Simulate closed-loop. Optional clipping (breaks LQR optimality
        but enforces physical constraints — motivates MPC).
        """
        traj = [x0.copy()]
        controls = []
        x = x0.copy()
        for t in range(T):
            u = self.policy(x)
            if clip_bounds:
                for i, (lo, hi) in enumerate(clip_bounds):
                    u[i] = np.clip(u[i], lo, hi)
            x = self.A @ x + self.B @ u
            traj.append(x.copy())
            controls.append(u.copy())
        return np.vstack(traj), np.vstack(controls)
```

### Time-Varying and Tracking LQR

```python
# apps/ml-api/src/control/lqr/time_varying_lqr.py

"""
Finite-horizon LQR via backward Riccati recursion:
  P_N = Q_f (terminal cost)
  P_t = Q_t + A_t^T P_{t+1} A_t
      - (A_t^T P_{t+1} B_t + N_t)(R_t + B_t^T P_{t+1} B_t)^{-1}(B_t^T P_{t+1} A_t + N_t^T)
  K_t = (R_t + B_t^T P_{t+1} B_t)^{-1}(B_t^T P_{t+1} A_t + N_t^T)

Time variation from:
  - Day-of-week elasticity changes
  - Event proximity effects
  - Staff scheduling constraints changing by policy
  - Seasonal demand multipliers
"""

# apps/ml-api/src/control/lqr/tracking_lqr.py

"""
Tracking LQR — follow an occupancy/revenue trajectory:

Method 1: State augmentation with integral action (eliminates steady-state error):
  z_{t+1} = z_t + (y_t - y_ref_t)
  x_aug = [x_t; z_t]

Method 2: Reference feedforward via steady-state target:
  x_ss = A x_ss + B u_ss
  y_ss = C x_ss = y_ref
  Regulate x̃ = x - x_ss, ũ = u - u_ss
"""
```

### LQG (Separation Principle)

```python
# apps/ml-api/src/control/lqr/lqg.py

"""
LQG = LQR + Kalman Filter (separation principle):
  u_t = -K x̂_t

Noisy venue observations:
  - Occupancy: booking system lag (delayed)
  - Staff: effective ≠ assigned (absences)
  - Inventory: sensor noise
  - Revenue: incomplete transaction streams

Kalman filter estimates true state from noisy observations:
  x_{t+1} = A x_t + B u_t + G w_t
  y_t = C x_t + D u_t + v_t
  E[ww^T] = Q_N, E[vv^T] = R_N

python-control 0.10.2: lqe() for continuous, dlqe() for discrete.
Separation principle: design LQR and Kalman independently,
compose for LQG. Optimal under linear-Gaussian assumptions.
"""
```

---

## OC-2: MPC Dynamic Pricing — The Central Workhorse

### What to Build

Constrained receding-horizon optimization for pricing, staffing, marketing.

```python
# apps/ml-api/src/control/mpc/linear_mpc.py

import numpy as np
import cvxpy as cp

class VenuePricingMPC:
    """
    At each time t, solve horizon-H problem:
      min_{u_{t:t+H-1}} Σ_{k=0}^{H-1} ℓ(x_{t+k}, u_{t+k}) + ℓ_f(x_{t+H})
      s.t. x_{t+k+1} = A x_{t+k} + B u_{t+k} + E d_{t+k}
           u_min ≤ u_{t+k} ≤ u_max
           x_min ≤ x_{t+k} ≤ x_max
    Apply only u_t*, repeat at t+1.

    Linear MPC = QP when dynamics linear and cost quadratic.

    Solver stack:
      CVXPY 1.8.1: modeling layer (not a solver itself)
      OSQP 1.1.1: ADMM QP solver
        - Warm-start: reuse cached factorization/previous solution
        - DPP (Disciplined Parametrized Programming): efficient repeated solves
      Clarabel 0.11.1: interior-point conic/QP (Rust-native)

    Performance reality checks:
      acados: ~1.05ms median per NMPC iteration (chain-of-masses benchmark)
      OSQP: scalable across dimensions, warm-start critical for MPC
      Clarabel: 10⁻⁵ to 10⁻² seconds for small-to-mid QPs

    Constraints enforced:
      - Price bounds and rate-of-change limits
      - Capacity constraints (fire code)
      - Minimum staffing (safety)
      - Inventory non-negativity
      - Marketing budget limits
    """

    def __init__(self, A, B, Q, R, H=24, x_ref=None):
        self.nx, self.nu = A.shape[0], B.shape[1]
        self.H = H

        # CVXPY parameter for current state
        self.x0 = cp.Parameter(self.nx)
        self.x = cp.Variable((self.nx, H + 1))
        self.u = cp.Variable((self.nu, H))

        x_ref = x_ref if x_ref is not None else np.zeros(self.nx)

        constraints = [self.x[:, 0] == self.x0]
        cost = 0

        for k in range(H):
            # Dynamics
            constraints += [self.x[:, k+1] == A @ self.x[:, k] + B @ self.u[:, k]]

            # State constraints
            constraints += [
                0.0 <= self.x[0, k+1], self.x[0, k+1] <= 1.0,   # occupancy
                10.0 <= self.x[1, k+1], self.x[1, k+1] <= 40.0,  # staff
                0.0 <= self.x[2, k+1],                             # inventory
            ]

            # Control constraints
            constraints += [
                -0.15 <= self.u[0, k], self.u[0, k] <= 0.15,     # price adj ±15%
                -5.0 <= self.u[1, k], self.u[1, k] <= 5.0,       # staff change
                0.0 <= self.u[2, k], self.u[2, k] <= 1.0,        # marketing
            ]

            # Rate-of-change constraint on pricing
            if k > 0:
                constraints += [cp.abs(self.u[0, k] - self.u[0, k-1]) <= 0.05]

            # Quadratic tracking cost
            dx = self.x[:, k] - x_ref
            cost += cp.quad_form(dx, Q) + cp.quad_form(self.u[:, k], R)

        # Terminal cost
        cost += cp.quad_form(self.x[:, H] - x_ref, Q)
        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def step(self, current_state):
        self.x0.value = current_state
        self.prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if self.prob.status == "optimal":
            return self.u.value[:, 0], self.prob.status
        return None, self.prob.status
```

### Economic MPC (Maximize Profit)

```python
"""
Economic MPC: ℓ = -profit rather than tracking reference.
  profit_t = price_t × demand(price_t) - cost_staff(s_t) - cost_marketing(m_t)

Stability requires dissipativity theory — active research area.
More natural for venue pricing than tracking-MPC since the
objective IS economic, not regulation around a setpoint.
"""
```

### Nonlinear MPC

```python
# apps/ml-api/src/control/mpc/nonlinear_mpc.py

"""
When demand is nonlinear (logit, constant elasticity, saturation):
  o_{t+1} = o_t + Δt·clip(D(p_t, m_t, t; θ) - churn(o_t), 0, cap - o_t)
  r_t ≈ p_t·D(p_t, m_t, t; θ) - c_s(s_t) - c_m(m_t) - c_penalty(overcrowding)

NMPC tooling:
  do-mpc 5.1.1: robust NMPC + MHE, uncertainty handling, discretization
  CasADi 3.7.2: symbolic modeling, automatic differentiation, NLP integration
  acados 0.5.3: C-based embedded OC, Python acados_template generates C solver,
    ctypes wrapper (Cython optional), ~1ms/iteration

Demand model options:
  - ML forecaster (gradient-boosting/NN) as differentiable surrogate
  - Parametric logit model updated online
  - Hybrid: parametric core + residual NN

Architecture: demand model feeds NMPC rollout → constrained optimization.
"""
```

### Robust and Stochastic MPC

```python
# apps/ml-api/src/control/mpc/robust_mpc.py

"""
Demand uncertainty is structured:
  - Forecast error (continuous)
  - Event shocks (rare but large)
  - Cancellations/no-shows (state-dependent)
  - Competitor actions (game-theoretic)

Robust MPC: bounded disturbance w_t ∈ W, enforce constraints for ALL w.
  Use for HARD constraints: fire code occupancy, safety staffing.

Tube MPC: plan nominal trajectory, keep true trajectory inside
  robust invariant "tube" via ancillary feedback law.
  Guaranteed constraint satisfaction under bounded disturbances.

Stochastic MPC (SMPC): probabilistic disturbances,
  chance constraints P(g(x,u) ≤ 0) ≥ 1 - ε.
  Use for SOFT but risk-limited constraints:
    P(inventory stockout) ≤ 5%
    P(queue time > X) ≤ 10%

Scenario-based MPC: sample disturbance scenarios, optimize over tree.
"""
```

---

## OC-3: Pontryagin's Maximum Principle for Price Trajectories

```python
# apps/ml-api/src/control/pmp/hamiltonian.py

"""
Continuous-time pricing with remaining capacity:
  State: c(t) = remaining capacity
  Control: p(t) = price
  Dynamics: ċ(t) = -D(p(t), t) with c(t) ≥ 0, p ∈ [p_min, p_max]
  Objective: max ∫₀ᵀ p(t)·D(p(t),t) dt

Hamiltonian:
  H(c, p, λ, t) = p·D(p,t) + λ·(-D(p,t))

Costate:
  λ̇(t) = -∂H/∂c = 0  →  λ(t) = λ* constant (simplified model)

Optimality (interior):
  ∂H/∂p = D(p,t) + (p - λ*)·D_p(p,t) = 0

Interpretation: λ* is the SHADOW PRICE OF CAPACITY (opportunity cost).
This is exactly bid-price control in revenue management.

With price bounds p ∈ [p_min, p_max]:
  Solution can become BANG-BANG.
  Switching function based on sign of ∂H/∂p.

When does PMP produce closed forms?
  - D(p,t) linear in p, separable in t: analytic p*(t)
  - Logistic/logit demand: implicit equation, numerical root-finding
  - Additional state coupling: λ̇ ≠ 0, solve BVP
"""

# apps/ml-api/src/control/pmp/shooting.py

"""
Numerical methods for PMP boundary value problem:
  Given c(0) = c_0, terminal constraint c(T) or transversality.

1. Single shooting: parameterize controls, integrate forward, optimize.
   Can be ill-conditioned.
2. Multiple shooting: break horizon into segments, enforce continuity.
   Standard in NMPC solvers. Improved stability.
3. Direct collocation: enforce dynamics at collocation points → large sparse NLP.

In practice: if using CasADi + IPOPT or acados, implement PMP as
direct transcription (collocation/multiple shooting) rather than
solving costate equations explicitly — handles constraints cleanly
and fits same infrastructure as NMPC.
"""
```

---

## OC-4: Dynamic Programming / HJB for Booking Acceptance

### What to Build

Optimal booking acceptance policies via Bellman recursion and bid-price control.

```python
# apps/ml-api/src/control/dp/optimal_stopping.py

"""
Booking acceptance as optimal stopping:
  "10 Saturday evenings per quarter — accept £5K corporate vs hold for £15K wedding"

State: c = remaining prime slots (integer), t = time remaining.
Arrivals: stochastic offers with value distribution conditioned on time/seasonality.

Threshold/bid-price policy:
  Accept if offer_value ≥ ΔV(c,t) ≡ V(c,t) - V(c-1,t)

ΔV is the opportunity cost of consuming one unit of capacity.

Bellman equation (discounted):
  V(x,t) = max_{u∈U(x)} { r(x,u,t) + β·E[V(x',t+1) | x,u] }

HJB PDE (continuous-time analog):
  ∂V/∂t + max_u { f(x,u,t)·∇V + L(x,u,t) } = 0

Key: value functions often non-smooth (kinks from max/constraints).
Theory: viscosity solutions (Crandall & Lions 1983).
Implementation: hj-reachability 0.7.0 for grid-based HJB solves.

Curse of dimensionality:
  Multiple time slots × room types × staffing × choice models × cancellations.

Practical approximations:
  - Approximate DP / fitted value iteration (NN value function)
  - LP-based bid-price control (network RM)
  - Choice-based RM approximations
  - POMDP for partially observed demand regime

Revenue management tools:
  RevPy: EMSRb + LP solver for static bid prices/allocations
  RMOL: C++ RM optimization (airlines/hotels/theaters)
  OR-Tools 9.15.6755: MIP/LP for network RM formulations
"""
```

---

## OC-5: RL as Approximate Optimal Control

### What to Build

SAC/PPO for continuous pricing, offline RL from historical logs,
safe RL with constraint layers, RL-MPC hybrid.

```python
# apps/ml-api/src/control/rl/reward_design.py

"""
Reward design for venue pricing:

R_t = (revenue_t - variable_costs_t)           # economic
    - α·max(0, o_t - o_safe)²                  # overcrowding penalty
    - β·Δp_t²                                  # price stability
    - γ·max(0, queue_t - q_max)                # queue penalty
    - η·cancellations_t                         # churn penalty

Then constrained MDP (CMDP) layer for hard safety constraints.

Libraries:
  stable-baselines3 2.7.1: PPO, SAC, TD3, DQN (PyTorch)
  gymnasium 1.2.3: environment API
  d3rlpy 2.8.1: offline RL (CQL, IQL, Decision Transformer)
  ray[rllib] 2.53.0: scalable distributed training
"""

# apps/ml-api/src/control/rl/offline_rl.py

"""
Offline RL is EXTREMELY relevant to venue pricing:
  Cannot explore arbitrarily (legal/fairness constraints, revenue risk).
  Learn from historical booking/pricing logs.

CQL (Conservative Q-Learning):
  Learn conservative Q-values under dataset shift.
  Penalizes Q-values for out-of-distribution actions.

IQL (Implicit Q-Learning):
  Avoids querying OOD actions during training.
  Strong results on offline RL benchmarks.

Decision Transformer:
  Reframe RL as sequence modeling.
  Attractive with rich logs and long horizons.

Safe RL survey: constrained MDPs + multi-agent extensions.
"""

# apps/ml-api/src/control/rl/rl_mpc_hybrid.py

"""
Hybrid patterns (common in modern control):

1. Model-based RL → MPC:
   Learn dynamics f_θ, then plan with MPC (receding horizon).

2. RL policy with MPC safety shield:
   RL proposes actions → MPC checks feasibility →
   minimally modifies to satisfy constraints.

3. Offline RL init + online fine-tuning with strict safety.

MPC dominates when: predictive model available, constraints crucial.
RL dominates when: too complex to model, high-fidelity simulator available.
"""
```

---

## OC-6: Feedback Linearization and Nonlinear Control

```python
# apps/ml-api/src/control/nonlinear/feedback_linearization.py

"""
For smooth control-affine model ẋ = f(x) + g(x)u:
  Compute Lie derivatives, determine relative degree.
  Zero dynamics stability required for feasibility.

Linear models fail at:
  - Threshold demand effects (luxury venues)
  - Capacity cliff effects (hard sell-out)
  - Multimodal demand (corporate vs weddings)
  - Nonlinear cancellations

Nonlinear control tools:
  Feedback linearization: Lie derivatives, relative degree, zero dynamics
  Backstepping: recursive strict-feedback stabilization
  Sliding mode: robust to matched uncertainties (may chatter)
  Adaptive control (MRAC / L1): demand parameter drift, fast adaptation

For venues, these are usually LESS direct than:
  - NMPC with learned/parametric demand models
  - Robust/tube MPC under uncertainty
  - Online parameter estimation + MPC

Use nonlinear tools when you truly have a dynamical nonlinear plant
with continuous actuation — pricing is often better served by MPC.
"""
```

---

## OC-7: Multi-Agent Control for Venue Networks

```python
# apps/ml-api/src/control/multi_agent/decentralized_mpc.py

"""
Platform manages MANY venues — pricing decisions interact:
  - Cannibalization (venues compete for same demand)
  - Shared marketing budget
  - Local event conflicts

Approaches:
1. Decentralized MPC: each venue solves local MPC with coordination constraints
2. Game-theoretic: Nash (simultaneous), Stackelberg (platform-leader, venues-follow)
3. Mean-field: many similar venues → continuum approximation
4. MARL: MAPPO (centralized critic, strong empirical results),
   QMIX (value decomposition, monotonic mixing, decentralized execution)

MAPPO: centralized training, decentralized execution.
  Strong in cooperative multi-agent environments.
  Feasible for distributed venue policy learning.

QMIX: Q-value decomposition enforcing monotonicity.
  Each venue has local Q-function.
  Central mixer ensures consistency.
"""
```

---

## OC-8: Crowd Flow and Evacuation Control

```python
# apps/ml-api/src/control/crowd/evacuation_control.py

"""
Safety-critical crowd dynamics:
  - Density constraints ρ(x,t) ≤ ρ_max
  - Egress time minimization
  - Dynamic exit assignment/signage
  - Layout optimization for evacuation compliance

Hughes model (continuum PDE + eikonal potential):
  Standard continuum theory for crowd flow.
  PDE-constrained optimization for offline layout design.

Social Force Model (Helbing & Molnár, PRE 1995):
  m_i·(dv_i/dt) = F_desire + Σ F_social + Σ F_obstacle
  Microscopic agent-based simulation.

JuPedSim 1.3.1: pip install jupedsim
  Open-source pedestrian dynamics library.
  Social force model based.

PedSim: ROS-based simulators using social force models.

Control formulation:
  Objective: minimize integrated evacuation time OR maximize minimum clearance rate
  Constraints: density ≤ ρ_max, exit capacity limits
  Controls: dynamic signage fields, exit assignment, barrier actuation

Solved as:
  - PDE-constrained optimization (offline layout design)
  - MPC-like receding horizon on reduced-order crowd model (real-time ops)
"""
```

---

## OC-9: Stochastic Optimal Control

```python
# apps/ml-api/src/control/stochastic/risk_sensitive.py

"""
Stochastic DP: E[Σ r(x_t, u_t)] with x_{t+1} = f(x_t, u_t, w_t)

Risk-sensitive control:
  Exponential utility: J = -1/θ · log E[exp(-θ·Σ r_t)]
  θ > 0: risk-averse, θ < 0: risk-seeking, θ → 0: risk-neutral

CVaR (Conditional Value at Risk):
  min CVaR_ε = E[cost | cost ≥ VaR_ε]
  Protects against worst-case 5% scenarios.
  Linear programming formulation with auxiliary variables.

Distributionally Robust (DRO):
  Optimize over worst-case distribution in Wasserstein ball around empirical.
  Connection to OT technique (TECHNIQUE_02).

Chance-constrained:
  P(violation) ≤ ε
  Converted to deterministic constraints via:
  - Scenario approximation
  - Boole's inequality for joint chance constraints
  - Distributionally robust chance constraints
"""
```

---

## OC-10: Optimal Experiment Design (Dual Control)

```python
# apps/ml-api/src/control/experiment/dual_control.py

"""
New venue/market must LEARN its demand curve.
This is dual control: maximize revenue WHILE learning.

Formalization:
  Belief b_t(θ) over demand parameters θ
  Action u_t (price)
  Observation y_t (requests/bookings)
  Objective: combine profit + information gain

Information-Directed Sampling (Russo & Van Roy 2018):
  Balance expected regret vs information gain (mutual information).
  Principled explore-exploit.

Production-friendly pricing learning loop:
  1. Constrained Thompson sampling (price bounds + rate limits)
  2. Safe Bayesian optimization for promotional experiments
  3. Active learning MPC: economic objective + explicit information value term

Connection to SLT-4 (bandits) and SLT-5 (Bayesian optimization):
  Thompson sampling for exploration, GP/BO for experiment optimization.
"""
```

---

## OC-11: Real-Time Control Architecture

```python
# apps/ml-api/src/control/architecture/control_loop.py

"""
Production closed-loop: Sense → Estimate → Decide → Act → Observe

Sense:
  Bookings, website analytics, POS transactions,
  competitor scraping, social media/event signals.

Estimate:
  Kalman/MHE/state-estimation, demand parameter updates.
  (Connection to SP-3 Kalman filtering)

Decide:
  MPC/RL/DP policy selection based on problem type.

Act:
  Price publishing, staffing schedule updates,
  marketing budget allocations, on-site operational decisions.

Observe:
  Realized bookings, conversions, no-shows, feedback.

Sample rates:
  Pricing: hourly/daily
  Staffing: daily/weekly
  Marketing: weekly/monthly
  Crowd ops: real-time (~seconds)

Computational deadlines:
  MPC QP must solve within control period.
  acados: ~1ms/iteration achievable with codegen.
  OSQP: cached factorization + warm-start for repeated solves.
  CVXPY DPP: efficient parametric repeated solves.
"""
```

### Visualization for Venue Managers

```python
"""
Operational trust requires explainability:
  - Predicted occupancy bands (nominal + uncertainty tube)
  - Constraint margins (capacity, staffing)
  - Shadow price of capacity (bid price from DP)
  - What-if curves (price vs expected occupancy/revenue)
  - Action-rate limits (why system "refuses" aggressive changes)

This is NOT cosmetic — it operationalizes constrained control
in a human-managed environment.
"""
```

---

## OC-12: Edge / Browser Deployment

```python
"""
Browser-side preview of control actions:
  1. Explicit MPC: piecewise-affine controller → lookup tables
  2. Linear state-feedback: exported LQR gain K → u = -Kx in TypeScript
  3. Distilled neural policy: RL → MLP → ONNX → ONNX Runtime Web

Rust → WASM deployment for solver-side:
  Clarabel (Rust-native): QP/conic solving in Rust
    cargo add clarabel
  OSQP: wasm32 wheels available (noted in release tooling)
  Clarabel in Pyodide: BLAS complications for full Python+solver in browser

Rust control libraries:
  control-sys: state-space modeling and simulation
  lqr crate: LQR controller abstraction
  kalman_filters 1.0.1: Kalman filter with builder patterns
  solvr: discrete Riccati solver (Schur-based symplectic method)
"""
```

```typescript
// packages/control-core/src/controllers/linear_feedback.ts

/**
 * Browser-side LQR controller.
 * K matrix exported from Python solve, applied in TypeScript.
 * Runs at microsecond latency — no server round-trip.
 */
export class LinearFeedbackController {
    constructor(private K: number[][]) {}

    policy(x: number[]): number[] {
        // u = -Kx
        return this.K.map(row =>
            -row.reduce((sum, k, i) => sum + k * x[i], 0));
    }
}

// packages/control-core/src/controllers/explicit_mpc.ts

/**
 * Piecewise-affine MPC lookup.
 * Pre-compute critical regions offline → browser lookup.
 * Exact MPC solution without solving QP at runtime.
 */
```

---

## Integration with Other Techniques

- **Signal Processing** (SP-3): Kalman-filtered demand state feeds MPC/LQR state estimate;
  spectral decomposition of demand feeds seasonal A_t, B_t matrices for time-varying LQR
- **Stochastic Pricing** (SP-tech): PMP shadow price λ* connects to stochastic pricing
  models; HJB value function connects to SP option pricing; DRO connects to OT Wasserstein
- **Statistical Learning Theory** (SLT-4/SLT-5): Thompson sampling and Bayesian optimization
  from SLT feed dual control exploration; conformal prediction wraps MPC predictions
- **Physics Solvers** (PS-5): Crowd flow egress energy integrates with evacuation control;
  social force model used in both PS crowd simulation and OC crowd control
- **Graph Neural Networks** (GNN-8): GNN-guided combinatorial optimization warm-starts
  MIP solver for event-room assignment; GNN demand prediction feeds MPC
- **HPC** (HPC): WASM Clarabel for browser-side QP; Rust control-sys for platform services;
  WebGPU for scenario tree MPC parallelization

---

## Session Management

1. **OC-1** (LQR: DARE, Q/R tuning, time-varying, tracking, LQG) — 1-2 sessions
2. **OC-2** (MPC: linear QP, economic, NMPC, constraints) — 2-3 sessions
3. **OC-3** (PMP: Hamiltonian, shooting, collocation, bang-bang) — 1 session
4. **OC-4** (DP/HJB: Bellman, bid-price, optimal stopping, approximate DP) — 1-2 sessions
5. **OC-5** (RL: SAC/PPO, offline CQL/IQL/DT, safe RL, hybrid) — 2-3 sessions
6. **OC-6** (Nonlinear: feedback linearization, backstepping, sliding mode, adaptive) — 1 session
7. **OC-7** (Multi-agent: decentralized MPC, Nash, Stackelberg, MAPPO, QMIX) — 1-2 sessions
8. **OC-8** (Crowd: Hughes, social force, evacuation MPC, JuPedSim) — 1 session
9. **OC-9** (Stochastic: risk-sensitive, CVaR, DRO, chance-constrained, scenario) — 1-2 sessions
10. **OC-10** (Experiment: dual control, IDS, Thompson, safe BO, active learning MPC) — 1 session
11. **OC-11** (Architecture: control loop, sample rates, state estimation, visualization) — 1-2 sessions
12. **OC-12** (Edge/browser: explicit MPC, LQR TypeScript, ONNX policy, Rust WASM) — 1 session

Total: ~14-18 Claude Code sessions.
