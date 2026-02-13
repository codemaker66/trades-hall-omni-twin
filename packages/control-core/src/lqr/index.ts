// ---------------------------------------------------------------------------
// OC-1: LQR / LQG — Barrel Re-exports
// ---------------------------------------------------------------------------

export {
  configToMatrices,
  solveDARE,
  computeLQRGain,
  discreteLQR,
  simulateLQR,
} from './dare.js';

export {
  inverseTolerance,
  venueDefaultQR,
} from './qr-tuning.js';

export {
  timeVaryingLQR,
  simulateTimeVaryingLQR,
} from './time-varying.js';

export {
  computeSteadyState,
  trackingLQR,
  simulateTracking,
} from './tracking.js';

export {
  designLQG,
  createLQGState,
  lqgStep,
  simulateLQG,
} from './lqg.js';
