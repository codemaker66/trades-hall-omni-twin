// ---------------------------------------------------------------------------
// OC-11  Real-Time Control Architecture -- Fault Tolerance
// ---------------------------------------------------------------------------

import type { FaultToleranceConfig } from '../types.js';
import { vecClone } from '../types.js';

// ---------------------------------------------------------------------------
// createFaultTolerantController
// ---------------------------------------------------------------------------

/**
 * Wrap a nominal state-feedback controller with fault-tolerant logic.
 *
 * The returned function checks the health of all known sensors before every
 * control computation.  Based on the configured {@link FaultToleranceConfig},
 * it selects one of four modes:
 *
 *  - **Nominal** -- all sensors healthy: delegates to `nominalController`.
 *  - **Degraded** -- at least one sensor faulted and `fallbackPolicy` is
 *    `'degraded'`: applies `u = -K_degraded * x` using the pre-configured
 *    degraded gains.
 *  - **Hold** -- `fallbackPolicy` is `'hold'`: returns the previous action
 *    (zeros on the very first call).
 *  - **Safe** -- `fallbackPolicy` is `'safe'`: returns zeros (safe shutdown).
 *
 * @param config              Fault tolerance parameters.
 * @param nominalController   Healthy-mode controller: x -> u.
 * @param nx                  State dimension.
 * @param nu                  Control dimension.
 * @returns                   A fault-aware controller function.
 */
export function createFaultTolerantController(
  config: FaultToleranceConfig,
  nominalController: (x: Float64Array) => Float64Array,
  nx: number,
  nu: number,
): (
  x: Float64Array,
  sensorHealth: Map<string, { lastUpdate: number; missedCount: number }>,
  currentTime: number,
) => Float64Array {
  // Persistent state: previous action for 'hold' policy.
  let previousAction = new Float64Array(nu);

  return (
    x: Float64Array,
    sensorHealth: Map<string, { lastUpdate: number; missedCount: number }>,
    currentTime: number,
  ): Float64Array => {
    // Determine whether any sensor is faulted.
    let anyFaulted = false;

    for (const [_name, health] of sensorHealth) {
      const timeSinceLast = currentTime - health.lastUpdate;
      if (
        health.missedCount > config.maxMissedReadings ||
        timeSinceLast > config.sensorTimeoutMs
      ) {
        anyFaulted = true;
        break;
      }
    }

    let u: Float64Array;

    if (!anyFaulted) {
      // All sensors healthy -- use nominal controller.
      u = nominalController(x);
    } else {
      switch (config.fallbackPolicy) {
        case 'degraded': {
          // u = -K_degraded * x
          // degradedGains is nu x nx, row-major.
          u = new Float64Array(nu);
          for (let i = 0; i < nu; i++) {
            let sum = 0;
            for (let j = 0; j < nx; j++) {
              sum += config.degradedGains[i * nx + j]! * x[j]!;
            }
            u[i] = -sum;
          }
          break;
        }
        case 'hold': {
          u = vecClone(previousAction);
          break;
        }
        case 'safe': {
          u = new Float64Array(nu);
          break;
        }
        default: {
          // Exhaustiveness guard -- treat unknown policy as safe shutdown.
          u = new Float64Array(nu);
          break;
        }
      }
    }

    previousAction = new Float64Array(u.buffer.slice(0)) as Float64Array<ArrayBuffer>;
    return u;
  };
}

// ---------------------------------------------------------------------------
// computeDegradedGains
// ---------------------------------------------------------------------------

/**
 * Compute a degraded gain matrix by zeroing out columns of the nominal gain
 * that correspond to states which are no longer observable.
 *
 * A state index `j` is considered unobservable when *all* sensors that
 * observe it have failed.  The `availableSensors` array acts as a simple
 * proxy: when `availableSensors[j]` is `false`, the column `j` of the gain
 * matrix is zeroed.
 *
 * @param nominalGains     Nominal gain matrix K (nu x nx, row-major flat).
 * @param availableSensors Boolean per state index indicating observability.
 * @param nx               State dimension.
 * @param nu               Control dimension.
 * @returns                Degraded gain matrix (nu x nx, row-major flat).
 */
export function computeDegradedGains(
  nominalGains: Float64Array,
  availableSensors: boolean[],
  nx: number,
  nu: number,
): Float64Array {
  const degraded = new Float64Array(nu * nx);

  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nx; j++) {
      const idx = i * nx + j;
      if (availableSensors[j]!) {
        degraded[idx] = nominalGains[idx]!;
      }
      // else leave as 0 -- unobservable state column is zeroed.
    }
  }

  return degraded;
}
