// ---------------------------------------------------------------------------
// OC-11: Real-Time Control Architecture -- Barrel Re-exports
// ---------------------------------------------------------------------------

export { createControlLoop, controlLoopStep } from './control-loop.js';
export {
  createMultiRateScheduler,
  getActiveSubsystems,
} from './sample-rates.js';
export {
  multiSensorEstimate,
  movingHorizonEstimate,
} from './state-estimation.js';
export {
  createFaultTolerantController,
  computeDegradedGains,
} from './fault-tolerance.js';
