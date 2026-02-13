# Statistical Learning Command Track

Canonical command track: `docs/commands/stat-learning/`

Execution authority for agent commands is `docs/commands/**`.

## Commands

- `SLT-1` - Conformal Prediction Suite (depends_on: None)
- `SLT-2` - PAC-Bayes Bounds and Model Selection (depends_on: `SLT-1`)
- `SLT-3` - VC Dimension and Sample Complexity (depends_on: `SLT-2`)
- `SLT-4` - Online Learning and Bandit Systems (depends_on: `SLT-3`)
- `SLT-5` - Gaussian Processes for Demand Forecasting (depends_on: `SLT-INT-1`)
- `SLT-6` - Tree Ensembles — The Correct Default for Venue Tabular Data (depends_on: `SLT-5`)
- `SLT-7` - Hierarchical Bayesian Models (Cold-Start Solution) (depends_on: `SLT-6`)
- `SLT-8` - Distribution Shift Detection and Robustness (depends_on: `SLT-7`)
- `SLT-9` - Causal Inference for Venue Optimization (depends_on: `SLT-INT-2`)
- `SLT-10` - Information-Theoretic Methods (depends_on: `SLT-9`)
- `SLT-11` - Calibration and Fairness (depends_on: `SLT-10`)
- `SLT-12` - Production API and Monitoring Architecture (depends_on: `SLT-11`)
- `SLT-INT-1` - Statistical Learning integration checkpoint 1 (depends_on: `SLT-1`, `SLT-2`, `SLT-3`, `SLT-4`)
- `SLT-INT-2` - Statistical Learning integration checkpoint 2 (depends_on: `SLT-5`, `SLT-6`, `SLT-7`, `SLT-8`)
- `SLT-INT-3` - Statistical Learning integration checkpoint 3 (depends_on: `SLT-9`, `SLT-10`, `SLT-11`, `SLT-12`)
