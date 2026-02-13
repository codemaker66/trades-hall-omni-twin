// ---------------------------------------------------------------------------
// OC-9  Stochastic Optimal Control -- Barrel Re-export
// ---------------------------------------------------------------------------

export { stochasticValueIteration } from './stochastic-dp.js';
export { riskSensitiveVI, exponentialUtility } from './risk-sensitive.js';
export { computeCVaR, cvarOptimize } from './cvar.js';
export { wasserstein1D, wassersteinDRO } from './dro.js';
export { chanceConstrainedOptimize, solveScenarioMPC } from './chance-constrained.js';
