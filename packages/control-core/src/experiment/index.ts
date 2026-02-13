// ---------------------------------------------------------------------------
// OC-10: Optimal Experiment Design — Barrel Re-exports
// ---------------------------------------------------------------------------

export {
  dualControlStep,
  computeInfoGain,
  updateBelief,
} from './dual-control.js';

export {
  idsSelect,
  idsUpdate,
} from './ids.js';

export { solveActiveLearningMPC } from './active-learning-mpc.js';

export {
  thompsonPriceSelect,
  thompsonPriceUpdate,
} from './thompson-pricing.js';

export {
  safeBONext,
  safeExpectedImprovement,
} from './safe-bo.js';
