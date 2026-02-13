// ---------------------------------------------------------------------------
// OC-2: Model Predictive Control — Barrel re-exports
// ---------------------------------------------------------------------------

export {
  solveLinearMPC,
  buildCondensedQP,
  solveQP,
} from './linear-mpc.js';

export { solveEconomicMPC } from './economic-mpc.js';

export { solveNonlinearMPC, sqpStep } from './nonlinear-mpc.js';

export { solveTubeMPC, computeInvariantTube } from './robust-mpc.js';

export { solveStochasticMPC } from './stochastic-mpc.js';

export { explicitMPCLookup, buildExplicitMPCTable } from './explicit-mpc.js';
