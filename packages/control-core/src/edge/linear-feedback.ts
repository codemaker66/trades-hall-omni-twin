// ---------------------------------------------------------------------------
// OC-12  Edge/Browser Deployment -- Linear Feedback Controller
// ---------------------------------------------------------------------------

import type { LinearFeedbackConfig } from '../types.js';

// ---------------------------------------------------------------------------
// linearFeedback
// ---------------------------------------------------------------------------

/**
 * Compute a linear feedback control action:
 *
 *   u = -K (x - xRef)
 *
 * The result is clipped component-wise to [uMin, uMax] when bounds are
 * provided in the config.
 *
 * @param config  Linear feedback gain and optional bounds.
 * @param x       Current state vector (nx).
 * @param xRef    Reference / setpoint state (nx). Defaults to zero.
 * @returns       Control action (nu).
 */
export function linearFeedback(
  config: LinearFeedbackConfig,
  x: Float64Array,
  xRef?: Float64Array,
): Float64Array {
  const { K, nx, nu, uMin, uMax } = config;

  // Compute error: e = x - xRef
  const e = new Float64Array(nx);
  for (let i = 0; i < nx; i++) {
    e[i] = xRef ? x[i]! - xRef[i]! : x[i]!;
  }

  // u = -K * e   (K is nu x nx, row-major)
  const u = new Float64Array(nu);
  for (let i = 0; i < nu; i++) {
    let sum = 0;
    for (let j = 0; j < nx; j++) {
      sum += K[i * nx + j]! * e[j]!;
    }
    u[i] = -sum;
  }

  // Clip to bounds
  if (uMin !== undefined && uMax !== undefined) {
    for (let i = 0; i < nu; i++) {
      if (u[i]! < uMin[i]!) u[i] = uMin[i]!;
      if (u[i]! > uMax[i]!) u[i] = uMax[i]!;
    }
  } else if (uMin !== undefined) {
    for (let i = 0; i < nu; i++) {
      if (u[i]! < uMin[i]!) u[i] = uMin[i]!;
    }
  } else if (uMax !== undefined) {
    for (let i = 0; i < nu; i++) {
      if (u[i]! > uMax[i]!) u[i] = uMax[i]!;
    }
  }

  return u;
}

// ---------------------------------------------------------------------------
// loadFeedbackGain
// ---------------------------------------------------------------------------

/**
 * Parse a JSON-serialised gain matrix into a `LinearFeedbackConfig`.
 *
 * Expected JSON shape:
 * ```json
 * { "K": [[k00, k01, ...], [k10, ...]], "nx": 4, "nu": 2 }
 * ```
 *
 * The 2-D array is flattened to a row-major Float64Array.
 *
 * @param json  Parsed JSON object with K, nx, and nu fields.
 * @returns     A LinearFeedbackConfig ready for `linearFeedback`.
 */
export function loadFeedbackGain(json: {
  K: number[][];
  nx: number;
  nu: number;
}): LinearFeedbackConfig {
  const { K: rows, nx, nu } = json;
  const K = new Float64Array(nu * nx);

  for (let i = 0; i < nu; i++) {
    const row = rows[i]!;
    for (let j = 0; j < nx; j++) {
      K[i * nx + j] = row[j]!;
    }
  }

  return { K, nx, nu };
}
