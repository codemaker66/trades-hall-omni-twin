// ---------------------------------------------------------------------------
// OC-11  Real-Time Control Architecture -- Control Loop
// ---------------------------------------------------------------------------

import type { ControlLoopConfig, ControlLoopState } from '../types.js';
import { vecClone } from '../types.js';

// ---------------------------------------------------------------------------
// createControlLoop
// ---------------------------------------------------------------------------

/**
 * Initialise a control loop state with zero vectors and timestamp 0.
 *
 * @param config  Timing configuration for sense / estimate / decide / actuate.
 * @param nx      State dimension.
 * @param nu      Control dimension.
 * @returns       A fresh {@link ControlLoopState}.
 */
export function createControlLoop(
  _config: ControlLoopConfig,
  nx: number,
  nu: number,
): ControlLoopState {
  return {
    timestamp: 0,
    sensorReadings: new Map<string, Float64Array>(),
    stateEstimate: new Float64Array(nx),
    currentAction: new Float64Array(nu),
    previousAction: new Float64Array(nu),
  };
}

// ---------------------------------------------------------------------------
// controlLoopStep
// ---------------------------------------------------------------------------

/**
 * Advance the control loop by one tick (minimum period).
 *
 * The tick duration equals the shortest sub-period in the config.  At each
 * tick the function checks whether each phase's period has elapsed since the
 * last execution and, if so, runs that phase:
 *
 *  1. **Sensing** -- update `sensorReadings` (readings are provided
 *     externally via the estimator's access to the map).
 *  2. **Estimation** -- run `estimator(xHat, readings)` to produce a new
 *     state estimate.
 *  3. **Decision** -- run `controller(xHat)` to compute a new action.
 *  4. **Actuation** -- copy `currentAction` to `previousAction` and apply
 *     the latest computed action.
 *
 * @param state       Current loop state (immutably consumed -- a new state is
 *                    returned).
 * @param config      Timing configuration.
 * @param controller  State-feedback controller: x -> u.
 * @param estimator   State estimator: (xHat, sensorReadings) -> xHat_new.
 * @returns           Updated {@link ControlLoopState}.
 */
export function controlLoopStep(
  state: ControlLoopState,
  config: ControlLoopConfig,
  controller: (x: Float64Array) => Float64Array,
  estimator: (
    xHat: Float64Array,
    readings: Map<string, Float64Array>,
  ) => Float64Array,
): ControlLoopState {
  const minPeriod = Math.min(
    config.sensingPeriodMs,
    config.estimationPeriodMs,
    config.decisionPeriodMs,
    config.actuationPeriodMs,
  );

  const newTimestamp = state.timestamp + minPeriod;

  // Clone mutable fields so we return a fresh state object.
  let sensorReadings = new Map<string, Float64Array>(state.sensorReadings);
  let stateEstimate = vecClone(state.stateEstimate);
  let currentAction = vecClone(state.currentAction);
  let previousAction = vecClone(state.previousAction);

  // (1) Sensing phase
  if (isPhaseElapsed(newTimestamp, config.sensingPeriodMs)) {
    // Sensor readings are expected to be populated externally (e.g. by a
    // hardware abstraction layer).  The control loop simply acknowledges
    // the phase fires; the estimator will consume whatever is in the map.
    sensorReadings = new Map<string, Float64Array>(sensorReadings);
  }

  // (2) Estimation phase
  if (isPhaseElapsed(newTimestamp, config.estimationPeriodMs)) {
    stateEstimate = estimator(stateEstimate, sensorReadings);
  }

  // (3) Decision phase
  if (isPhaseElapsed(newTimestamp, config.decisionPeriodMs)) {
    currentAction = controller(stateEstimate);
  }

  // (4) Actuation phase
  if (isPhaseElapsed(newTimestamp, config.actuationPeriodMs)) {
    previousAction = vecClone(currentAction);
  }

  return {
    timestamp: newTimestamp,
    sensorReadings,
    stateEstimate,
    currentAction,
    previousAction,
  };
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Returns `true` when `timestamp` is an integer multiple of `periodMs`
 * (within a small floating-point tolerance).
 */
function isPhaseElapsed(timestamp: number, periodMs: number): boolean {
  if (periodMs <= 0) return false;
  const remainder = timestamp % periodMs;
  const tolerance = 1e-9;
  return remainder < tolerance || periodMs - remainder < tolerance;
}
