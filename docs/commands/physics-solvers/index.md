# Physics Solvers Command Track

Canonical command track: `docs/commands/physics-solvers/`

Execution authority for agent commands is `docs/commands/**`.

## Commands

- `PS-1` - Simulated Annealing Engine (Adaptive + Reheating) (depends_on: None)
- `PS-2` - Parallel Tempering / Replica Exchange Monte Carlo (depends_on: `PS-1`)
- `PS-3` - Spin Glass Models — Ising & Potts for Scheduling (depends_on: `PS-2`)
- `PS-4` - Simulated Bifurcation (Toshiba SB Algorithm) (depends_on: `PS-3`)
- `PS-5` - The Complete Venue Layout Energy Function (depends_on: `PS-INT-1`)
- `PS-6` - Restricted Boltzmann Machine for Layout Generation (depends_on: `PS-5`)
- `PS-7` - MCMC Layout Sampling — Diverse High-Quality Alternatives (depends_on: `PS-6`)
- `PS-8` - NSGA-II Multi-Objective Optimization (depends_on: `PS-7`)
- `PS-9` - CMA-ES for Continuous Layout Refinement (depends_on: `PS-INT-2`)
- `PS-10` - MIP Scheduling via HiGHS (In-Browser) (depends_on: `PS-9`)
- `PS-11` - Diffusion-Based Layout Generation (Server-Side) (depends_on: `PS-10`)
- `PS-12` - The Layered Solver Pipeline (Orchestrator) (depends_on: `PS-11`)
- `PS-INT-1` - Physics Solvers integration checkpoint 1 (depends_on: `PS-1`, `PS-2`, `PS-3`, `PS-4`)
- `PS-INT-2` - Physics Solvers integration checkpoint 2 (depends_on: `PS-5`, `PS-6`, `PS-7`, `PS-8`)
- `PS-INT-3` - Physics Solvers integration checkpoint 3 (depends_on: `PS-9`, `PS-10`, `PS-11`, `PS-12`)
