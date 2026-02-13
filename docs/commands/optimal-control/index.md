# Optimal Control Command Track

Canonical command track: `docs/commands/optimal-control/`

Execution authority for agent commands is `docs/commands/**`.

## Commands

- `OC-1` - LQR for Venue Resource Management (depends_on: None)
- `OC-2` - MPC Dynamic Pricing — The Central Workhorse (depends_on: `OC-1`)
- `OC-3` - Pontryagin's Maximum Principle for Price Trajectories (depends_on: `OC-2`)
- `OC-4` - Dynamic Programming / HJB for Booking Acceptance (depends_on: `OC-3`)
- `OC-5` - RL as Approximate Optimal Control (depends_on: `OC-INT-1`)
- `OC-6` - Feedback Linearization and Nonlinear Control (depends_on: `OC-5`)
- `OC-7` - Multi-Agent Control for Venue Networks (depends_on: `OC-6`)
- `OC-8` - Crowd Flow and Evacuation Control (depends_on: `OC-7`)
- `OC-9` - Stochastic Optimal Control (depends_on: `OC-INT-2`)
- `OC-10` - Optimal Experiment Design (Dual Control) (depends_on: `OC-9`)
- `OC-11` - Real-Time Control Architecture (depends_on: `OC-10`)
- `OC-12` - Edge / Browser Deployment (depends_on: `OC-11`)
- `OC-INT-1` - Optimal Control integration checkpoint 1 (depends_on: `OC-1`, `OC-2`, `OC-3`, `OC-4`)
- `OC-INT-2` - Optimal Control integration checkpoint 2 (depends_on: `OC-5`, `OC-6`, `OC-7`, `OC-8`)
- `OC-INT-3` - Optimal Control integration checkpoint 3 (depends_on: `OC-9`, `OC-10`, `OC-11`, `OC-12`)
