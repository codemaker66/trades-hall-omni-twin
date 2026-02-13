# Stochastic Pricing Command Track

Canonical command track: `docs/commands/stochastic-pricing/`

Execution authority for agent commands is `docs/commands/**`.

## Commands

- `SP-1` - Stochastic Price Models (GBM + OU + Seasonal + Jump-Diffusion) (depends_on: None)
- `SP-2` - Black-Scholes Booking Options (Hold Fees) (depends_on: `SP-1`)
- `SP-3` - Hamilton-Jacobi-Bellman Dynamic Pricing (depends_on: `SP-2`)
- `SP-4` - Mean-Field Games — Competing Planners (depends_on: `SP-3`)
- `SP-5` - Order Book for Venue Availability Market (depends_on: `SP-INT-1`)
- `SP-6` - Stochastic Demand Models (Poisson → Hawkes → Cox) (depends_on: `SP-5`)
- `SP-7` - Monte Carlo Engine with Variance Reduction + QMC (depends_on: `SP-6`)
- `SP-8` - Revenue Management (EMSRb → Bid-Price → Choice-Based → Deep RL) (depends_on: `SP-7`)
- `SP-9` - Portfolio Theory — Booking Mix Optimization (depends_on: `SP-INT-2`)
- `SP-10` - Visualization Suite (depends_on: `SP-9`)
- `SP-INT-1` - Stochastic Pricing integration checkpoint 1 (depends_on: `SP-1`, `SP-2`, `SP-3`, `SP-4`)
- `SP-INT-2` - Stochastic Pricing integration checkpoint 2 (depends_on: `SP-5`, `SP-6`, `SP-7`, `SP-8`)
- `SP-INT-3` - Stochastic Pricing integration checkpoint 3 (depends_on: `SP-9`, `SP-10`)
