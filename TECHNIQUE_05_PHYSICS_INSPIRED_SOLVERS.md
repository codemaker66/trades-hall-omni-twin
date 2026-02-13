# TECHNIQUE_05_PHYSICS_INSPIRED_SOLVERS.md — Physics-Inspired Optimization

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
> are complete. Implements physics-inspired computation — simulated annealing,
> parallel tempering, spin glass scheduling, simulated bifurcation, Boltzmann machines,
> MCMC layout sampling, evolutionary multi-objective optimization, MIP scheduling,
> diffusion-based layout generation, and the complete venue layout energy function.
>
> **How to use**: Tell Claude Code: "Read TECHNIQUE_05_PHYSICS_INSPIRED_SOLVERS.md and
> implement incrementally, starting from PS-1."
>
> **This is 12 sub-domains. Implement ALL of them. Do not skip any.**

---

## Key Papers (Referenced Throughout)

- Yu et al. (2011). "Make it Home." *ACM TOG* (SIGGRAPH). DOI:10.1145/1964921.1964981 — Foundational layout energy function
- Merrell et al. (2011). "Interactive Furniture Layout." *ACM TOG* (SIGGRAPH) — Circulation analysis + MCMC layout
- Lin et al. (2024). "RL-controlled SA for unequal-area facility layout." *Soft Computing* 28:5667–5682
- Di Gaspero et al. (2022). "SA for ITC2021 timetabling." *J. Scheduling*. DOI:10.1007/s10951-022-00740-y
- Goto et al. (2019). "Simulated Bifurcation." *Science Advances* 5:eaav2372
- Zeng et al. (2024). "dSB benchmarks." *Communications Physics* 7:249
- Lucas (2014). "Ising formulations of NP problems." *Frontiers in Physics* 2:5. arXiv:1302.5843
- Shabani et al. (2023). "HouseDiffusion." CVPR. arXiv:2211.13287
- Tang et al. (2024). "DiffuScene." CVPR
- Feng et al. (2023). "LayoutGPT." NeurIPS. arXiv:2305.15393
- Hoffman & Gelman (2014). "NUTS." *JMLR* 15:1351–1381. arXiv:1111.4246
- Vousden et al. (2016). "Adaptive PT." *MNRAS* 455(2):1919. arXiv:1501.05823
- Deb et al. (2002). "NSGA-II." *IEEE Trans. Evol. Comput.*
- Hansen (2016). "CMA-ES Tutorial." arXiv:1604.00772
- Suda et al. (2025). "O(N) QUBO constraints." arXiv:2601.18108
- IsingFormer (2025). "Transformer proposals for PT." arXiv:2509.23043
- Bagherbeik et al. (2020). "100× faster QAP via PT." Springer. DOI:10.1007/978-3-030-58112-1_22

---

## Architecture Overview

```
packages/
  physics-solvers/                — TypeScript orchestration + types
    src/
      types.ts                    — Energy, State, Config types for all solvers
      energy/
        layout-energy.ts          — Complete venue layout energy function
        schedule-energy.ts        — Event scheduling energy function
      orchestrator.ts             — Layered solver pipeline
      index.ts

  physics-solvers-wasm/           — Rust WASM (all compute-heavy solvers)
    src/
      lib.rs
      sa.rs                       — Simulated Annealing (adaptive + reheating)
      parallel_tempering.rs       — Replica Exchange Monte Carlo
      ising.rs                    — Ising spin model for binary decisions
      potts.rs                    — Potts model for multi-class assignment
      qubo.rs                     — QUBO formulation + penalty encoding
      simulated_bifurcation.rs    — Toshiba SB algorithm (ballistic + discrete)
      rbm.rs                      — Restricted Boltzmann Machine
      mcmc.rs                     — Metropolis-Hastings + HMC layout sampler
      nsga2.rs                    — NSGA-II multi-objective optimizer
      cmaes.rs                    — CMA-ES for continuous refinement
      pso.rs                      — Particle Swarm Optimization
      order_book.rs               — (from SP-5, shared)
    shaders/
      ising_metropolis.wgsl       — WebGPU Ising checkerboard Metropolis
      pt_swap.wgsl                — WebGPU parallel tempering swap kernel

apps/
  ml-api/
    src/
      solvers/
        diffusion_layout.py       — HouseDiffusion/DiffuScene layout generation
        layout_gpt.py             — LLM-based initial layout generation
        deep_mcmc.py              — NumPyro NUTS for high-dimensional layout sampling
        mfg_scheduling.py         — Mean-field game scheduling (from SP-4, shared)
      routes/
        solvers.py                — FastAPI endpoints

  web/
    src/
      components/
        solvers/
          EnergyLandscape.tsx     — 3D visualization of energy surface
          ParetoDashboard.tsx     — NSGA-II Pareto front explorer
          LayoutGallery.tsx       — MCMC-sampled diverse layout browser
          TemperatureViz.tsx      — Parallel tempering temperature ladder viz
          ScheduleGantt.tsx       — Gantt chart of optimized event schedule
          ConstraintPanel.tsx     — Interactive constraint weight controls
```

### Rust Crate Dependencies

```toml
# crates/physics-solvers-wasm/Cargo.toml
[dependencies]
wasm-bindgen = "0.2"
serde = { version = "1", features = ["derive"] }
serde-wasm-bindgen = "0.6"
rand = { version = "0.8", features = ["js"] }
rand_distr = "0.4"
nalgebra = "0.33"
argmin = "0.10"             # SA solver framework
ordered-float = "4"
```

### npm Dependencies

```json
{
  "highs": "^1.3.0",
  "lightweight-charts": "^5.1"
}
```

---

## PS-1: Simulated Annealing Engine (Adaptive + Reheating)

### What to Build

A production SA engine supporting three cooling schedules (geometric, Lam-Delosme
adaptive, Huang adaptive), systematic reheating, and configurable neighborhood
operators — all compiled to WASM for browser-speed optimization.

### Rust Implementation

```rust
// crates/physics-solvers-wasm/src/sa.rs

use wasm_bindgen::prelude::*;
use rand::prelude::*;

#[wasm_bindgen]
pub enum CoolingSchedule {
    /// T(k+1) = α·T(k), α typically 0.95-0.999
    Geometric,
    /// Targets 44% acceptance ratio, adjusts T using variance/mean of energy
    /// (Lam & Delosme, DAC 1988) — 2-24× speedup over fixed schedules
    LamDelosme,
    /// T_{k+1} = T_k · exp(-T_k · λ / σ_k)
    /// Decreases faster when landscape is flat, slower near phase transitions
    /// (Huang, Romeo & Sangiovanni-Vincentelli, ICCAD 1986)
    Huang,
}

#[wasm_bindgen]
pub struct SAConfig {
    pub initial_temp: f64,
    pub final_temp: f64,
    pub cooling: CoolingSchedule,
    pub alpha: f64,              // Geometric cooling rate (0.95-0.999)
    pub max_iterations: u32,
    pub reheat_interval: u32,    // 0 = no reheating, else reheat every N iterations
    pub reheat_temp_fraction: f64, // Fraction of initial temp to reheat to (0.3-0.7)
}

#[wasm_bindgen]
pub struct SAResult {
    pub best_energy: f64,
    pub best_state: Vec<f64>,    // Flattened best state
    pub iterations: u32,
    pub accepts: u32,
    pub rejects: u32,
    pub reheats: u32,
    pub energy_history: Vec<f64>, // For visualization
    pub temp_history: Vec<f64>,
}

/// Core SA loop with adaptive cooling and systematic reheating
#[wasm_bindgen]
pub fn simulated_annealing(
    initial_state: Vec<f64>,
    config: &SAConfig,
    // Energy function and neighbor function passed as indices into
    // a registered function table (WASM can't take closures directly)
    problem_type: ProblemType,
) -> SAResult {
    let mut rng = thread_rng();
    let mut state = initial_state.clone();
    let mut energy = compute_energy(&state, problem_type);
    let mut best_state = state.clone();
    let mut best_energy = energy;
    let mut temp = config.initial_temp;

    // Lam-Delosme adaptive state
    let mut energy_sum = 0.0;
    let mut energy_sq_sum = 0.0;
    let mut accept_count = 0u32;
    let mut window_size = 0u32;
    let lam_window = 100u32;  // τ for averaging

    let mut result = SAResult {
        energy_history: Vec::with_capacity(config.max_iterations as usize),
        temp_history: Vec::with_capacity(config.max_iterations as usize),
        ..Default::default()
    };

    for iter in 0..config.max_iterations {
        // Generate neighbor
        let neighbor = generate_neighbor(&state, problem_type, &mut rng);
        let neighbor_energy = compute_energy(&neighbor, problem_type);
        let delta_e = neighbor_energy - energy;

        // Metropolis acceptance: p = exp(-ΔE/T)
        let accept = if delta_e < 0.0 {
            true
        } else {
            let p = (-delta_e / temp).exp();
            rng.gen::<f64>() < p
        };

        if accept {
            state = neighbor;
            energy = neighbor_energy;
            accept_count += 1;
            if energy < best_energy {
                best_energy = energy;
                best_state = state.clone();
            }
        }

        // Track statistics for adaptive cooling
        energy_sum += energy;
        energy_sq_sum += energy * energy;
        window_size += 1;

        // Update temperature
        temp = match config.cooling {
            CoolingSchedule::Geometric => temp * config.alpha,

            CoolingSchedule::LamDelosme => {
                if window_size >= lam_window {
                    let mean = energy_sum / window_size as f64;
                    let variance = energy_sq_sum / window_size as f64 - mean * mean;
                    let acceptance_ratio = accept_count as f64 / window_size as f64;
                    let target_ratio = 0.44;

                    // Adjust temp to drive acceptance ratio toward 44%
                    if acceptance_ratio > target_ratio {
                        temp * (1.0 - 0.01 * (acceptance_ratio - target_ratio))
                    } else {
                        temp * (1.0 + 0.01 * (target_ratio - acceptance_ratio))
                    }
                    // Reset window
                    // (energy_sum, energy_sq_sum, accept_count, window_size = 0)
                } else {
                    temp
                }
            },

            CoolingSchedule::Huang => {
                if window_size >= lam_window {
                    let mean = energy_sum / window_size as f64;
                    let sigma = ((energy_sq_sum / window_size as f64) - mean * mean)
                        .max(1e-10).sqrt();
                    let lambda = 0.7; // Tuning parameter
                    temp * (-temp * lambda / sigma).exp()
                } else {
                    temp
                }
            },
        };

        // Systematic reheating
        if config.reheat_interval > 0 && iter % config.reheat_interval == 0 && iter > 0 {
            temp = config.initial_temp * config.reheat_temp_fraction;
            result.reheats += 1;
        }

        // Termination check
        if temp < config.final_temp { break; }

        result.energy_history.push(energy);
        result.temp_history.push(temp);
        result.iterations = iter + 1;
    }

    result.best_energy = best_energy;
    result.best_state = best_state;
    result.accepts = accept_count;
    result
}
```

---

## PS-2: Parallel Tempering / Replica Exchange Monte Carlo

### What to Build

Multiple SA chains at different temperatures, periodically swapping configurations.
High-T replicas explore freely, low-T replicas exploit basins. Swaps teleport
configurations across energy barriers that single-chain SA cannot cross.

### Rust Implementation

```rust
// crates/physics-solvers-wasm/src/parallel_tempering.rs

#[wasm_bindgen]
pub struct PTConfig {
    pub n_replicas: u32,         // 8-32 typical
    pub t_min: f64,              // Lowest temperature
    pub t_max: f64,              // Highest temperature
    pub spacing: TempSpacing,    // Geometric or Adaptive (Vousden et al.)
    pub sweeps_per_swap: u32,    // SA iterations between swap attempts
    pub total_swaps: u32,        // Total swap rounds
    pub sa_config: SAConfig,     // Per-replica SA settings
}

#[wasm_bindgen]
pub enum TempSpacing {
    /// T_i = T_min · (T_max/T_min)^{(i-1)/(N-1)}
    /// Standard, works well when heat capacity is roughly constant
    Geometric,
    /// Vousden, Farr & Mandel (arXiv:1501.05823)
    /// Dynamically adjusts temperatures to equalize swap acceptance rates
    /// 1.2-5× efficiency improvement over geometric
    Adaptive,
}

#[wasm_bindgen]
pub struct PTResult {
    pub best_energy: f64,
    pub best_state: Vec<f64>,
    pub replica_energies: Vec<f64>,    // Final energy per replica
    pub swap_acceptance_rates: Vec<f64>, // Between adjacent replicas
    pub energy_traces: Vec<Vec<f64>>,  // Per-replica energy history
}

/// Parallel tempering with geometric or adaptive temperature spacing
#[wasm_bindgen]
pub fn parallel_tempering(
    initial_state: Vec<f64>,
    config: &PTConfig,
    problem_type: ProblemType,
) -> PTResult {
    let n = config.n_replicas as usize;

    // Initialize temperature ladder
    let mut temps = match config.spacing {
        TempSpacing::Geometric => {
            (0..n).map(|i| {
                config.t_min * (config.t_max / config.t_min)
                    .powf(i as f64 / (n - 1) as f64)
            }).collect::<Vec<_>>()
        },
        TempSpacing::Adaptive => {
            // Start geometric, will adapt during run
            (0..n).map(|i| {
                config.t_min * (config.t_max / config.t_min)
                    .powf(i as f64 / (n - 1) as f64)
            }).collect::<Vec<_>>()
        },
    };

    // Initialize replicas (all start from same state)
    let mut states: Vec<Vec<f64>> = vec![initial_state.clone(); n];
    let mut energies: Vec<f64> = states.iter()
        .map(|s| compute_energy(s, problem_type)).collect();

    let mut best_energy = energies[0];
    let mut best_state = states[0].clone();
    let mut swap_accepts = vec![0u32; n - 1];
    let mut swap_attempts = vec![0u32; n - 1];
    let mut rng = thread_rng();

    for swap_round in 0..config.total_swaps {
        // Phase 1: Each replica runs sweeps_per_swap SA iterations at its temperature
        for r in 0..n {
            for _ in 0..config.sweeps_per_swap {
                let neighbor = generate_neighbor(&states[r], problem_type, &mut rng);
                let ne = compute_energy(&neighbor, problem_type);
                let delta = ne - energies[r];
                if delta < 0.0 || rng.gen::<f64>() < (-delta / temps[r]).exp() {
                    states[r] = neighbor;
                    energies[r] = ne;
                }
            }
            // Track global best
            if energies[r] < best_energy {
                best_energy = energies[r];
                best_state = states[r].clone();
            }
        }

        // Phase 2: Attempt swaps between adjacent replicas
        // Alternate even/odd pairs to avoid conflicts
        let start = if swap_round % 2 == 0 { 0 } else { 1 };
        for i in (start..n - 1).step_by(2) {
            let beta_i = 1.0 / temps[i];
            let beta_j = 1.0 / temps[i + 1];
            let delta_beta = beta_i - beta_j;
            let delta_e = energies[i] - energies[i + 1];

            // Metropolis swap criterion: p = min(1, exp((β_i - β_j)(E_i - E_j)))
            let p_swap = (delta_beta * delta_e).exp().min(1.0);
            swap_attempts[i] += 1;

            if rng.gen::<f64>() < p_swap {
                states.swap(i, i + 1);
                let tmp = energies[i];
                energies[i] = energies[i + 1];
                energies[i + 1] = tmp;
                swap_accepts[i] += 1;
            }
        }

        // Phase 3: Adaptive temperature adjustment (Vousden et al.)
        if matches!(config.spacing, TempSpacing::Adaptive) && swap_round % 100 == 99 {
            for i in 0..n - 1 {
                let rate = swap_accepts[i] as f64 / swap_attempts[i].max(1) as f64;
                let target = 0.234; // Optimal acceptance ~23.4%
                // Adjust temperature spacing to equalize rates
                if rate < target {
                    // Swap rate too low → bring temperatures closer
                    temps[i + 1] = temps[i + 1] * 0.95 + temps[i] * 0.05;
                } else {
                    // Swap rate too high → spread temperatures apart
                    temps[i + 1] = temps[i + 1] * 1.05;
                }
            }
            temps[n - 1] = temps[n - 1].min(config.t_max);
        }
    }

    PTResult {
        best_energy,
        best_state,
        replica_energies: energies,
        swap_acceptance_rates: swap_accepts.iter().zip(swap_attempts.iter())
            .map(|(&a, &t)| a as f64 / t.max(1) as f64).collect(),
        energy_traces: vec![], // Populated during run
    }
}
```

### Web Worker Parallelization

For true parallelism across CPU cores:

```typescript
// packages/physics-solvers/src/pt-workers.ts

/**
 * Parallel tempering with one Web Worker per replica.
 *
 * Architecture:
 * - Main thread: orchestrates swap attempts, holds temperature ladder
 * - Workers: each runs SA at assigned temperature, reports energy
 * - SharedArrayBuffer: energies[] and states[] shared across workers
 *   (requires CORS headers: COOP same-origin, COEP require-corp)
 * - Swap coordination: main thread reads energies via Atomics.load,
 *   evaluates swap criterion, swaps via Atomics.store + Atomics.notify
 *
 * Desktop browsers support 4-16 workers matching CPU cores.
 * Each worker loads the WASM module independently.
 */
export class ParallelTemperingWorkerPool {
  private workers: Worker[];
  private sharedBuffer: SharedArrayBuffer;
  private energies: Float64Array;

  constructor(nReplicas: number) { ... }
  async run(initialState: Float64Array, config: PTConfig): Promise<PTResult> { ... }
  private attemptSwaps(): void { ... }
  terminate(): void { ... }
}
```

---

## PS-3: Spin Glass Models — Ising & Potts for Scheduling

### What to Build

Map venue-event scheduling to spin glass Hamiltonians. Ising for binary decisions
(book/don't book), Potts for multi-class assignment (assign event to one of K rooms).

### QUBO Formulation for Venue Scheduling

```rust
// crates/physics-solvers-wasm/src/qubo.rs

/// Build the complete QUBO Hamiltonian for venue scheduling:
///
/// H = H_obj + λ₁·H_one_room + λ₂·H_no_conflict + λ₃·H_capacity
///
/// Variables: x_{e,r,t} = 1 if event e occupies room r at timeslot t
///
/// H_obj = -Σ w_{e,r,t} · x_{e,r,t}
///   (maximize preference scores — negative because we minimize H)
///
/// H_one_room = Σ_e P · (Σ_{r,t} x_{e,r,t} - 1)²
///   (each event assigned to exactly one room+timeslot)
///
/// H_no_conflict = Σ_{r,t} P · Σ_{e1<e2} x_{e1,r,t} · x_{e2,r,t}
///   (no two events in same room at same time)
///
/// H_capacity = Σ_{e,r,t} P · max(0, guests_e - capacity_r)² · x_{e,r,t}
///   (room must fit the event)
///
/// For 50 events × 10 rooms × 20 timeslots = 10,000 binary variables
/// before pruning, typically 1,000-3,000 active after removing infeasible.
#[wasm_bindgen]
pub fn build_scheduling_qubo(
    events: Vec<EventSpec>,       // [{id, guests, duration, preferences}]
    rooms: Vec<RoomSpec>,         // [{id, capacity, amenities}]
    timeslots: Vec<TimeslotSpec>, // [{id, start, end, day}]
    penalty_weight: f64,          // λ for constraint penalties (typically 10-1000)
) -> QUBOMatrix {
    // 1. Enumerate valid (event, room, timeslot) triples
    // 2. Build Q matrix (upper triangular, flattened)
    // 3. Diagonal: objective weights + one-hot linear terms
    // 4. Off-diagonal: conflict penalties + one-hot quadratic terms
    ...
}

/// Ising model for binary booking decisions
/// σ_i ∈ {-1, +1}, H = -½ Σ_{ij} J_{ij} σ_i σ_j - Σ_i h_i σ_i
///
/// Mapping: QUBO x ∈ {0,1} → Ising σ ∈ {-1,+1} via x = (σ+1)/2
/// J_{ij} = -Q_{ij}/4, h_i = -Q_{ii}/2 - Σ_j Q_{ij}/4
#[wasm_bindgen]
pub fn qubo_to_ising(qubo: &QUBOMatrix) -> IsingModel { ... }

/// Potts model for multi-class room assignment
/// σ_i ∈ {1,...,K}, H = -Σ_{⟨i,j⟩} J_{ij} δ(σ_i, σ_j)
///
/// Each event has a single Potts spin (which room?),
/// avoiding one-hot encoding overhead of Ising.
#[wasm_bindgen]
pub fn build_potts_scheduling(
    events: Vec<EventSpec>,
    rooms: Vec<RoomSpec>,
) -> PottsModel { ... }

/// Solve QUBO via SA on the Ising Hamiltonian
#[wasm_bindgen]
pub fn solve_qubo_sa(
    qubo: &QUBOMatrix,
    config: &SAConfig,
) -> Vec<u8> { ... }  // Binary solution vector

/// Solve QUBO via parallel tempering
#[wasm_bindgen]
pub fn solve_qubo_pt(
    qubo: &QUBOMatrix,
    config: &PTConfig,
) -> Vec<u8> { ... }
```

---

## PS-4: Simulated Bifurcation (Toshiba SB Algorithm)

### What to Build

Coupled nonlinear oscillators approaching bifurcation. GPU-acceleratable, scales
to 1,000,000 spins. Discrete SB shows "quasi-quantum tunneling" with superior
benchmarks over SA (Zeng et al. 2024).

### Rust Implementation

```rust
// crates/physics-solvers-wasm/src/simulated_bifurcation.rs

/// Simulated Bifurcation equations of motion:
///   dx_i/dt = y_i
///   dy_i/dt = -[Δ - p(t)]·x_i - K·x_i³ + ξ₀·Σ_j J_{ij}·f(x_j)
///
/// Ballistic SB: f(x_j) = x_j       (smooth, good for structured problems)
/// Discrete SB:  f(x_j) = sign(x_j)  (quasi-quantum tunneling, best overall)
///
/// As pumping p(t) increases from 0, each oscillator bifurcates to
/// x ≈ ±√(p/K), corresponding to Ising spin ±1.
/// Coupling J_{ij} biases which branch each oscillator takes.
#[wasm_bindgen]
pub enum SBVariant {
    Ballistic,  // f(x) = x
    Discrete,   // f(x) = sign(x) — superior overall performance
}

#[wasm_bindgen]
pub fn simulated_bifurcation(
    coupling_matrix: Vec<f64>,  // J_{ij} flattened [N×N]
    external_field: Vec<f64>,   // h_i [N]
    n_spins: u32,
    variant: SBVariant,
    n_steps: u32,               // Integration steps (typically 1000-10000)
    dt: f64,                    // Time step (0.01 typical)
    pump_rate: f64,             // How fast p(t) increases (0.01 typical)
    kerr: f64,                  // K nonlinearity coefficient (1.0 typical)
) -> Vec<i8> {  // Spin configuration {-1, +1}^N
    let n = n_spins as usize;
    let mut x = vec![0.0f64; n];  // Positions (initially at origin)
    let mut y = vec![0.0f64; n];  // Momenta (initially zero + small noise)
    let mut rng = thread_rng();

    // Small initial perturbation
    for i in 0..n {
        x[i] = rng.gen::<f64>() * 0.01 - 0.005;
        y[i] = rng.gen::<f64>() * 0.01 - 0.005;
    }

    for step in 0..n_steps {
        let p = pump_rate * step as f64 * dt; // Pumping amplitude

        for i in 0..n {
            // Coupling term: ξ₀ · Σ_j J_{ij} · f(x_j)
            let mut coupling = external_field[i];
            for j in 0..n {
                let fx_j = match variant {
                    SBVariant::Ballistic => x[j],
                    SBVariant::Discrete => if x[j] >= 0.0 { 1.0 } else { -1.0 },
                };
                coupling += coupling_matrix[i * n + j] * fx_j;
            }

            // Symplectic Euler integration
            y[i] += dt * (-(1.0 - p) * x[i] - kerr * x[i].powi(3) + coupling);
            x[i] += dt * y[i];
        }
    }

    // Read out spins from oscillator positions
    x.iter().map(|&xi| if xi >= 0.0 { 1i8 } else { -1i8 }).collect()
}
```

### Python Server-Side (GPU-Accelerated for Large Problems)

```python
# apps/ml-api/src/solvers/simulated_bifurcation_solver.py

"""
For problems >10,000 spins, use the GPU-accelerated Python package:
  pip install simulated-bifurcation

GPU PyTorch backend achieves 20,000× speedup over CPU SA for million-spin problems.
Median optimality gap < 1%.
"""
from simulated_bifurcation import build_model, optimize
```

---

## PS-5: The Complete Venue Layout Energy Function

### What to Build

This is the centrepiece. The mathematically complete energy function for venue
furniture layout, synthesizing Yu et al. (2011), Merrell et al. (2011), and
IBC/ADA building codes.

### Energy Function

```
E_venue = λ₁·E_overlap + λ₂·E_aisle + λ₃·E_egress + λ₄·E_sightline
        + λ₅·E_capacity + λ₆·E_ADA + λ₇·E_aesthetic + λ₈·E_service
```

Weight hierarchy: **fire code >> ADA >> overlap >> aisle width >> sightline >> aesthetic**
Hard constraints use λ = 10⁶, soft constraints use λ = 1–100.

### Rust Implementation

```rust
// crates/physics-solvers-wasm/src/layout_energy.rs

#[wasm_bindgen]
pub struct FurnitureItem {
    pub x: f64,           // Center x position (feet)
    pub y: f64,           // Center y position
    pub width: f64,       // Width (feet)
    pub depth: f64,       // Depth (feet)
    pub rotation: f64,    // Rotation (radians)
    pub item_type: ItemType,  // Table, Chair, Stage, Bar, etc.
    pub seats: u32,       // Number of seats (for tables/chairs)
}

#[wasm_bindgen]
pub struct RoomBoundary {
    pub vertices: Vec<f64>,   // Flattened [x1,y1, x2,y2, ...] polygon
    pub exits: Vec<f64>,      // Exit locations [x1,y1,w1, x2,y2,w2, ...]
    pub stage_area: Option<Vec<f64>>,  // Reserved stage zone
}

#[wasm_bindgen]
pub struct LayoutWeights {
    pub overlap: f64,      // 1e6 (hard constraint)
    pub aisle: f64,        // 1e4
    pub egress: f64,       // 1e6 (hard — fire code)
    pub sightline: f64,    // 100
    pub capacity: f64,     // 1e4
    pub ada: f64,          // 1e6 (hard — legal requirement)
    pub aesthetic: f64,    // 10
    pub service: f64,      // 50
}

/// Compute total layout energy — the objective function that SA/PT minimizes
#[wasm_bindgen]
pub fn compute_layout_energy(
    items: &[FurnitureItem],
    room: &RoomBoundary,
    weights: &LayoutWeights,
    target_capacity: u32,
) -> f64 {
    let mut energy = 0.0;

    // --- E_overlap: no furniture overlaps ---
    // E = Σ_{i<j} max(0, overlap_area(bbox_i, bbox_j))²
    energy += weights.overlap * e_overlap(items);

    // --- E_aisle: minimum aisle widths (IBC §1029.9.2) ---
    // Main aisles ≥ 54" (4.5ft)
    // Service aisles ≥ 60" (5ft)
    // Between table rows ≥ 36" (3ft)
    // Chair back to table edge ≥ 18" (1.5ft)
    // E = Σ_k max(0, w_min - w_k)²
    energy += weights.aisle * e_aisle(items);

    // --- E_egress: fire code egress paths (IBC §1017) ---
    // Shortest path from every seat to nearest exit ≤ 200ft
    // Path must be ≥ 44" wide for assembly occupancy
    // Compute via A* on 0.5ft discretized grid with furniture as obstacles
    // Blocked paths receive infinite penalty (1e10)
    energy += weights.egress * e_egress(items, room);

    // --- E_sightline: visibility to focal point ---
    // C-value formula (Green Guide): C = D(N+R)/(D+T) - R
    // Target C ≥ 60mm minimum (2.36")
    // For flat venues: ray-cast from each seat to focal point,
    // penalize occlusion ratio
    energy += weights.sightline * e_sightline(items, room);

    // --- E_capacity: target number of seats ---
    // Penalize both under-capacity and over-capacity
    let total_seats: u32 = items.iter().map(|i| i.seats).sum();
    let cap_diff = total_seats as f64 - target_capacity as f64;
    energy += weights.capacity * cap_diff * cap_diff;

    // --- E_ADA: accessibility requirements ---
    // Wheelchair spaces: minimum 36"×48" clear
    // Accessible paths: ≥ 36" clear width
    // Wheelchair spaces distributed across seating areas
    // At least 1 per 25 seats, or per ADA table 1002.4
    energy += weights.ada * e_ada(items, room);

    // --- E_aesthetic: visual quality ---
    // Alignment with walls (penalize items not parallel to nearest wall)
    // Symmetry (penalize deviation from room's axis of symmetry)
    // Visual balance (centroid of items near room center)
    // Table spacing uniformity
    energy += weights.aesthetic * e_aesthetic(items, room);

    // --- E_service: catering and AV paths ---
    // Catering: clear path from kitchen/service door to all tables ≥ 60"
    // AV: cable runs from AV booth to stage/screen ≤ max cable length
    // Service: staff circulation path exists around perimeter
    energy += weights.service * e_service(items, room);

    energy
}

/// Specific layout standards for common configurations:
///
/// Theater style:    6-8 sq ft/person, 36" between rows, 24" chair width
/// Banquet rounds:   10-12 sq ft/person, 60" round tables seat 8-10,
///                   60" between tables for service
/// Classroom:        18-20 sq ft/person
/// Cocktail/standing: 6 sq ft/person
/// Main aisles:      ≥ 54"
/// Service aisles:   ≥ 60"
/// Chair back to table edge: 18"
```

### Neighborhood Operators for Layout SA

```rust
/// Generate a neighbor layout by perturbing the current state.
/// Multiple operators, randomly selected:
pub fn generate_layout_neighbor(
    items: &mut [FurnitureItem],
    rng: &mut ThreadRng,
) -> Vec<FurnitureItem> {
    let mut neighbor = items.to_vec();
    let operator = rng.gen_range(0..6);
    match operator {
        0 => move_item(&mut neighbor, rng),      // Translate one item by small delta
        1 => rotate_item(&mut neighbor, rng),     // Rotate one item by small angle
        2 => swap_items(&mut neighbor, rng),      // Swap positions of two items
        3 => shift_row(&mut neighbor, rng),       // Translate entire row
        4 => adjust_spacing(&mut neighbor, rng),  // Uniform spacing adjustment
        5 => mirror_item(&mut neighbor, rng),     // Mirror one item across axis
        _ => unreachable!(),
    }
    neighbor
}
```

---

## PS-6: Restricted Boltzmann Machine for Layout Generation

### What to Build

RBMs learn the distribution of "good" layouts from training data, then generate
new configurations by sampling. Visible units = venue features + layout features,
hidden units = learned latent compatibility factors.

### Rust Implementation

```rust
// crates/physics-solvers-wasm/src/rbm.rs

/// RBM energy: E(v,h) = -aᵀv - bᵀh - vᵀWh
/// P(v,h) = (1/Z)exp(-E(v,h))
/// Block Gibbs: P(h_j=1|v) = σ(b_j + Σ_i W_{ij} v_i)
///              P(v_i=1|h) = σ(a_i + Σ_j W_{ij} h_j)
///
/// Training: Contrastive Divergence (Hinton 2002)
///   ΔW = ε(⟨v⁰h⁰ᵀ⟩_data - ⟨vᵏhᵏᵀ⟩_recon)
#[wasm_bindgen]
pub struct RBM {
    n_visible: usize,
    n_hidden: usize,
    weights: Vec<f64>,     // [n_visible × n_hidden]
    visible_bias: Vec<f64>,
    hidden_bias: Vec<f64>,
}

#[wasm_bindgen]
impl RBM {
    pub fn new(n_visible: u32, n_hidden: u32) -> Self { ... }

    /// Train with CD-k (Contrastive Divergence, k Gibbs steps)
    pub fn train_cd(
        &mut self,
        data: Vec<f64>,        // [n_samples × n_visible] training layouts
        n_samples: u32,
        k: u32,                // CD-k steps (1 for CD-1, 10+ for PCD)
        learning_rate: f64,
        epochs: u32,
        momentum: f64,
        weight_decay: f64,
    ) { ... }

    /// Generate a new layout by sampling from the learned distribution
    /// Start from random visible state, run Gibbs sampling for n_steps
    pub fn sample(&self, n_steps: u32) -> Vec<f64> { ... }

    /// Compute free energy F(v) = -aᵀv - Σ_j log(1 + exp(b_j + Σ_i W_{ij} v_i))
    /// Low free energy = high probability = layout matches learned distribution
    pub fn free_energy(&self, visible: &[f64]) -> f64 { ... }
}
```

---

## PS-7: MCMC Layout Sampling — Diverse High-Quality Alternatives

### What to Build

Rather than finding THE optimal layout, sample from the Boltzmann distribution
over good layouts: p(layout) ∝ exp(-E(layout)/T). This produces 20-50 diverse
high-quality configurations for planners to browse.

### Python Implementation (Server-Side, NumPyro for production quality)

```python
# apps/ml-api/src/solvers/deep_mcmc.py

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp

def sample_diverse_layouts(
    room: dict,
    n_items: int,
    item_specs: list[dict],
    n_samples: int = 50,
    n_warmup: int = 500,
    n_chains: int = 4,
) -> list[dict]:
    """
    Sample diverse high-quality layouts using NUTS (No-U-Turn Sampler).

    NUTS (Hoffman & Gelman, 2014) automatically determines leapfrog step count
    by detecting U-turns in Hamiltonian trajectory. No manual tuning needed
    beyond initial step size (auto-tuned during warmup).

    For N furniture items with (x, y, θ) each → 3N-dimensional state space.
    30 items = 90 dimensions — needs gradient-based methods (HMC/NUTS).

    Energy function must be differentiable → use smooth approximations:
    - Overlap: smooth max via softplus: softplus(overlap_amount)
    - Aisle width: smooth penalty: softplus(w_min - w)
    - Collision: sigmoid barrier functions

    Convergence: require rank-normalized R̂ < 1.01 (Vehtari et al. 2021),
    bulk-ESS and tail-ESS > 100 per chain.

    Returns a gallery of diverse high-quality layouts.
    """
    def model():
        # Prior: uniform over room dimensions
        positions = numpyro.sample('positions',
            dist.Uniform(0, jnp.array([room['width'], room['depth']])).expand([n_items, 2]))
        rotations = numpyro.sample('rotations',
            dist.Uniform(0, 2 * jnp.pi).expand([n_items]))

        # Energy as negative log-likelihood (smooth approximation)
        energy = compute_smooth_energy(positions, rotations, room, item_specs)
        numpyro.factor('layout_energy', -energy)

    kernel = NUTS(model, target_accept_prob=0.8)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                num_chains=n_chains)
    mcmc.run(jax.random.PRNGKey(0))

    # Extract samples, check convergence
    samples = mcmc.get_samples()
    diagnostics = numpyro.diagnostics.summary(samples)
    # Verify R̂ < 1.01 for all parameters

    return format_layout_gallery(samples, item_specs)
```

### WASM Implementation (Client-Side Preview)

```rust
// Simplified Metropolis-Hastings for quick client-side layout sampling
// (gradient-free, for <20 items where NUTS overhead isn't needed)
#[wasm_bindgen]
pub fn sample_layouts_mh(
    room: &RoomBoundary,
    items: &[FurnitureItem],
    weights: &LayoutWeights,
    temperature: f64,
    n_samples: u32,
    thin: u32,            // Keep every thin-th sample
) -> Vec<Vec<f64>> { ... }
```

---

## PS-8: NSGA-II Multi-Objective Optimization

### What to Build

Tri-objective venue optimization: minimize cost AND maximize attendee flow AND
satisfy fire code compliance. Returns a Pareto front of trade-off solutions.

### Rust Implementation

```rust
// crates/physics-solvers-wasm/src/nsga2.rs

/// NSGA-II (Deb et al. 2002):
/// 1. Non-dominated sorting → Pareto fronts F₁, F₂, ...
/// 2. Crowding distance for diversity within each front:
///    CD_i = Σ_m [(f_m(i+1) - f_m(i-1)) / (f_m_max - f_m_min)]
/// 3. Tournament selection: prefer lower front, then higher crowding distance
/// 4. Crossover + mutation → offspring
/// 5. Combine parent + offspring → select best N via fronts + crowding
///
/// For 4+ objectives, use NSGA-III (reference-point based) instead.
#[wasm_bindgen]
pub struct NSGA2Config {
    pub population_size: u32,     // 100-500 typical
    pub generations: u32,         // 200-1000
    pub crossover_rate: f64,      // 0.8-0.95
    pub mutation_rate: f64,       // 0.01-0.1
    pub crossover_type: CrossoverType, // PMX, OX, CX for permutation problems
}

/// Objectives for venue layout:
/// 1. Minimize total cost (setup labor, equipment rental)
/// 2. Maximize attendee flow (inverse of average path length between key zones)
/// 3. Maximize fire code compliance (inverse of E_egress + E_aisle)
/// Optional 4th: Maximize aesthetic score (inverse of E_aesthetic)
#[wasm_bindgen]
pub fn nsga2_venue_layout(
    initial_population: Vec<Vec<f64>>,  // N initial layouts
    room: &RoomBoundary,
    config: &NSGA2Config,
) -> Vec<ParetoSolution> { ... }

#[wasm_bindgen]
pub struct ParetoSolution {
    pub state: Vec<f64>,
    pub objectives: Vec<f64>,   // [cost, -flow, -compliance]
    pub front_rank: u32,        // 0 = Pareto optimal
    pub crowding_distance: f64,
}
```

---

## PS-9: CMA-ES for Continuous Layout Refinement

### What to Build

After discrete structure is fixed (which tables, what arrangement pattern),
use CMA-ES for fine-tuning continuous positions and rotations. State-of-the-art
for 3-100 dimensional continuous optimization.

```rust
// crates/physics-solvers-wasm/src/cmaes.rs

/// CMA-ES (Hansen 2016, arXiv:1604.00772)
/// Covariance update learns second-order landscape structure:
///   C_{g+1} = (1-c₁-c_μ)·C_g + c₁·p_c·p_cᵀ + c_μ·Σ w_i·(x_i-m)(x_i-m)ᵀ/σ²
///
/// Quasi-parameter-free: only needs initial σ₀ (step size).
/// Ideal for 3-100 dimensions → 1-33 furniture items × 3 params each.
#[wasm_bindgen]
pub fn cmaes_refine_layout(
    initial_mean: Vec<f64>,     // 3N params: [x1,y1,θ1, x2,y2,θ2, ...]
    initial_sigma: f64,         // Initial step size (e.g., 1.0 feet)
    room: &RoomBoundary,
    weights: &LayoutWeights,
    max_evaluations: u32,       // Budget: typically 1000 × dimension
) -> Vec<f64> { ... }
```

---

## PS-10: MIP Scheduling via HiGHS (In-Browser)

### What to Build

Exact optimal scheduling for hard constraints using the HiGHS MIP solver compiled
to WASM. Runs in the browser. Use for event-room-timeslot assignment where
optimality proof matters.

### TypeScript Implementation

```typescript
// packages/physics-solvers/src/mip-scheduler.ts

/**
 * MIP scheduling via HiGHS-js (WASM-compiled, MIT license).
 *
 * Use when: <10,000 binary variables with hard constraints where
 * you need provably optimal solutions.
 * Falls back to: SA/PT for larger instances or soft optimization.
 *
 * Formulation:
 *   min  -Σ w_{e,r,t} · x_{e,r,t}  (maximize preference scores)
 *   s.t. Σ_{r,t} x_{e,r,t} = 1     ∀ events (each event assigned once)
 *        Σ_e x_{e,r,t} ≤ 1          ∀ room-timeslots (no conflicts)
 *        guests_e · x_{e,r,t} ≤ capacity_r  ∀ (e,r,t)  (capacity)
 *        x_{e,r,t} ∈ {0, 1}
 */
import highs from 'highs';

export async function solveScheduleMIP(
  events: EventSpec[],
  rooms: RoomSpec[],
  timeslots: TimeslotSpec[],
): Promise<ScheduleResult> {
  const solver = await highs();

  // Build LP format string programmatically
  let lp = 'Minimize\n  obj: ';
  // ... generate objective, constraints, binary declarations ...

  const result = solver.solve(lp);
  return parseScheduleResult(result, events, rooms, timeslots);
}
```

---

## PS-11: Diffusion-Based Layout Generation (Server-Side)

### What to Build

Use HouseDiffusion/DiffuScene-style models to generate venue layouts from
text descriptions. Also use LayoutGPT for quick LLM-based initialization.

```python
# apps/ml-api/src/solvers/layout_generation.py

async def generate_layout_llm(
    description: str,   # "200-person banquet with dance floor and stage"
    room: dict,         # Room dimensions and constraints
) -> dict:
    """
    LayoutGPT-style generation (Feng et al. 2023, NeurIPS):
    Prompt the LLM with CSS-style spatial language:
      object {width: ?px; height: ?px; left: ?px; top: ?px;}

    20-40% better spatial correctness than text-to-image models.
    Provides a warm start for subsequent SA optimization.
    """
    ...

async def generate_layout_diffusion(
    room: dict,
    item_specs: list[dict],
    style: str,          # "banquet", "theater", "classroom", "cocktail"
    n_layouts: int = 5,
) -> list[dict]:
    """
    DiffuScene-style denoising diffusion (Tang et al. 2024, CVPR):
    Each object: (location, size, orientation, semantic attributes)
    Unordered set → denoising diffusion → physically plausible layout

    Superior physical plausibility. Supports:
    - Unconditional generation
    - Scene completion (partial layout → fill in rest)
    - Rearrangement (given items, optimize positions)
    - Text-conditioned synthesis
    """
    ...
```

---

## PS-12: The Layered Solver Pipeline (Orchestrator)

### What to Build

The production system combines ALL techniques in a layered pipeline:

```typescript
// packages/physics-solvers/src/orchestrator.ts

/**
 * Five-layer optimization pipeline:
 *
 * Layer 1 — Initial generation:
 *   LayoutGPT for warm start from natural language description
 *   OR template selection based on event type + capacity
 *
 * Layer 2 — Hard constraint enforcement:
 *   HiGHS-js (WASM MIP) for scheduling (event-room-timeslot)
 *   Column generation for instances > 10,000 variables
 *
 * Layer 3 — Soft optimization:
 *   Parallel tempering SA (Rust→WASM + Web Workers)
 *   8-16 replicas, geometric temperature spacing
 *   Energy function: PS-5 complete venue layout energy
 *
 * Layer 4 — Diversity sampling:
 *   MCMC at fixed temperature for 20-50 diverse layouts
 *   NumPyro NUTS server-side for production quality
 *   Simplified MH in WASM for client-side previews
 *
 * Layer 5 — Multi-objective trade-offs:
 *   NSGA-II for Pareto fronts (cost vs flow vs compliance)
 *   Interactive dashboard for stakeholder exploration
 */

export interface SolverPipeline {
  generateInitial(description: string, room: RoomSpec): Promise<Layout>;
  scheduleEvents(events: EventSpec[], rooms: RoomSpec[]): Promise<Schedule>;
  optimizeLayout(layout: Layout, weights: LayoutWeights): Promise<Layout>;
  sampleAlternatives(layout: Layout, n: number): Promise<Layout[]>;
  computeParetoFront(layout: Layout): Promise<ParetoSolution[]>;
  runFullPipeline(request: PlanningRequest): Promise<PlanningResult>;
}

export class VenueSolverPipeline implements SolverPipeline {
  private wasmModule: PhysicsSolversWasm;
  private ptPool: ParallelTemperingWorkerPool;
  private mipSolver: HiGHSSolver;

  async runFullPipeline(request: PlanningRequest): Promise<PlanningResult> {
    // 1. Generate initial layout from description
    const initial = await this.generateInitial(request.description, request.room);

    // 2. Schedule events to rooms/timeslots via MIP
    const schedule = await this.scheduleEvents(request.events, request.rooms);

    // 3. Optimize furniture layout via parallel tempering SA
    const optimized = await this.optimizeLayout(initial, request.weights);

    // 4. Sample 30 diverse alternatives via MCMC
    const alternatives = await this.sampleAlternatives(optimized, 30);

    // 5. Compute Pareto front for multi-objective trade-offs
    const pareto = await this.computeParetoFront(optimized);

    return { schedule, optimized, alternatives, pareto };
  }
}
```

---

## Visualization Suite

### Energy Landscape 3D Surface

```typescript
// apps/web/src/components/solvers/EnergyLandscape.tsx
/**
 * 3D surface plot (react-plotly.js) showing energy as a function of
 * two selected layout parameters (e.g., table spacing × rotation).
 * Shows the multimodal landscape that SA navigates.
 * Current state plotted as a moving point. Temperature shown as glow radius.
 * SA trajectory plotted as a path on the surface.
 */
```

### Pareto Front Dashboard

```typescript
// apps/web/src/components/solvers/ParetoDashboard.tsx
/**
 * Interactive scatter plot of Pareto-optimal solutions:
 * X = cost, Y = attendee flow, color = compliance score.
 * Click a point → preview that layout in the 2D/3D editor.
 * Drag slider to trade off between objectives.
 * Shows dominated vs non-dominated solutions clearly.
 */
```

### Layout Gallery (MCMC Samples)

```typescript
// apps/web/src/components/solvers/LayoutGallery.tsx
/**
 * Grid of 20-50 thumbnail previews of sampled layouts.
 * Each thumbnail: 2D top-down view of furniture arrangement.
 * Click to expand. Side panel shows metrics (capacity, flow, compliance).
 * Filter by: capacity range, compliance level, layout style.
 * Sort by: energy, diversity from current layout, specific objective.
 */
```

### Temperature Ladder Visualization

```typescript
// apps/web/src/components/solvers/TemperatureViz.tsx
/**
 * Real-time visualization of parallel tempering:
 * - Vertical axis: temperature (log scale)
 * - Horizontal axis: time/iteration
 * - Color traces: each replica's energy trajectory
 * - Swap events: horizontal lines when replicas exchange
 * - Acceptance rates: bar chart alongside
 * - Best-so-far energy: bold overlay line
 */
```

---

## Integration with Other Techniques

- **Category Theory** (CT): Solver pipeline is a composed morphism:
  `generate ∘ schedule ∘ optimize ∘ sample ∘ pareto`
- **Optimal Transport** (OT): Displacement interpolation (OT-5) animates between
  MCMC-sampled layout alternatives using the OT-optimal furniture path
- **TDA**: Dead space detection (TDA-3) provides alpha complex analysis AFTER
  layout optimization — checking for voids the energy function may have missed
- **Stochastic Pricing** (SP): Parallel tempering for the multi-room pricing
  problem where HJB alone is too slow for the full state space

---

## Session Management

1. **PS-1** (SA engine: 3 cooling schedules + reheating) — 1 session
2. **PS-2** (Parallel tempering + Web Worker pool) — 1-2 sessions
3. **PS-3** (QUBO + Ising + Potts for scheduling) — 1 session
4. **PS-4** (Simulated Bifurcation: ballistic + discrete) — 1 session
5. **PS-5** (Complete venue layout energy function — all 8 terms) — 1-2 sessions
6. **PS-6** (RBM training + sampling) — 1 session
7. **PS-7** (MCMC layout sampling: NUTS server + MH client) — 1-2 sessions
8. **PS-8** (NSGA-II multi-objective) — 1 session
9. **PS-9** (CMA-ES continuous refinement) — 1 session
10. **PS-10** (HiGHS MIP scheduling in-browser) — 1 session
11. **PS-11** (Diffusion + LayoutGPT generation stubs) — 1 session
12. **PS-12** (Orchestrator pipeline + visualization) — 1-2 sessions

Total: ~12-16 Claude Code sessions for the full physics-inspired solver suite.

Each session: implement in Rust (WASM) where compute-heavy, Python where ML-heavy,
TypeScript for orchestration and visualization. Write tests. Update PROGRESS.md.
Commit after each section.

---

## Industry Gap — The Key Insight

**No current venue planning tool does automated layout optimization.** Cvent Event
Diagramming, Prismm (formerly Allseated), Merri, 3D Event Designer — ALL are manual
drag-and-drop. Even a basic SA implementation with the energy function from PS-5
would represent a significant advance over the entire industry. The full pipeline
(LLM init → MIP scheduling → PT-SA optimization → MCMC diversity → NSGA-II Pareto)
is unprecedented.
