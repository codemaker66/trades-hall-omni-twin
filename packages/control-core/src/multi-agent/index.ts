// ---------------------------------------------------------------------------
// OC-7: Multi-Agent Control — Barrel Re-exports
// ---------------------------------------------------------------------------

export { solveDecentralizedMPC } from './decentralized-mpc.js';
export { findNashEquilibrium, bestResponse } from './nash.js';
export { solveStackelberg } from './stackelberg.js';
export { meanFieldStep, meanFieldEquilibrium } from './mean-field.js';
export {
  createMAPPOAgents,
  mappoSelectActions,
  mappoCriticValue,
} from './mappo.js';
export {
  createQMIX,
  qmixAgentQValues,
  qmixMix,
} from './qmix.js';
export type { QMIXMixerWeights } from './qmix.js';
