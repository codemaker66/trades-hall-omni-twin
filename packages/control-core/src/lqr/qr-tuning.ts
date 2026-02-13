// ---------------------------------------------------------------------------
// OC-1: Q/R Weight Tuning from Engineering Tolerances
// ---------------------------------------------------------------------------

import type { QRTuningConfig } from '../types.js';
import { matDiag } from '../types.js';

// ---------------------------------------------------------------------------
// Inverse-tolerance Q/R construction
// ---------------------------------------------------------------------------

/**
 * Build diagonal Q and R cost matrices from engineering tolerances using the
 * Bryson-style inverse-tolerance method.
 *
 * For each state variable i:   Q_ii = w_i / sigma_i^2
 * For each control variable j: R_jj = w_j / sigma_j^2
 *
 * Where sigma is the maximum acceptable deviation (tolerance) and w is a
 * priority weight. This converts physical units into consistent cost weights.
 *
 * @param config Tuning configuration with tolerances and weights
 * @returns Object containing diagonal Q and R as Float64Array (row-major)
 */
export function inverseTolerance(config: QRTuningConfig): {
  Q: Float64Array;
  R: Float64Array;
} {
  const { tolerances, weights, controlTolerances, controlWeights } = config;
  const nx = tolerances.length;
  const nu = controlTolerances.length;

  // Build diagonal Q: Q_ii = w_i / sigma_i^2
  const qDiag = new Float64Array(nx);
  for (let i = 0; i < nx; i++) {
    const sigma = tolerances[i]!;
    const w = weights[i]!;
    if (Math.abs(sigma) < 1e-15) {
      throw new Error(`State tolerance[${i}] is near zero; cannot invert.`);
    }
    qDiag[i] = w / (sigma * sigma);
  }

  // Build diagonal R: R_jj = w_j / sigma_j^2
  const rDiag = new Float64Array(nu);
  for (let j = 0; j < nu; j++) {
    const sigma = controlTolerances[j]!;
    const w = controlWeights[j]!;
    if (Math.abs(sigma) < 1e-15) {
      throw new Error(
        `Control tolerance[${j}] is near zero; cannot invert.`,
      );
    }
    rDiag[j] = w / (sigma * sigma);
  }

  // Return full row-major flat arrays (diagonal matrices)
  const Q = matDiag(qDiag);
  const R = matDiag(rDiag);

  return {
    Q: new Float64Array(Q.data),
    R: new Float64Array(R.data),
  };
}

// ---------------------------------------------------------------------------
// Venue-domain default Q/R
// ---------------------------------------------------------------------------

/**
 * Provide sensible default Q and R matrices for a venue operations system.
 *
 * When the dimensions match the canonical 4-state / 3-control venue model:
 *
 * States (nx = 4):
 *   0: occupancy     (people, tolerance ~50)
 *   1: staff_count   (people, tolerance ~5)
 *   2: inventory     (units, tolerance ~100)
 *   3: revenue       (dollars, tolerance ~500)
 *
 * Controls (nu = 3):
 *   0: price_adjust  (dollars, tolerance ~10)
 *   1: staff_alloc   (people, tolerance ~3)
 *   2: marketing     (spend, tolerance ~200)
 *
 * For other dimensions, default tolerances of 10.0 and weights of 1.0 are used
 * for each state and control variable.
 *
 * @param nx State dimension
 * @param nu Control dimension
 * @returns Object with Q (nx x nx row-major) and R (nu x nu row-major) Float64Arrays
 */
export function venueDefaultQR(
  nx: number,
  nu: number,
): { Q: Float64Array; R: Float64Array } {
  // Canonical 4-state, 3-control venue model
  if (nx === 4 && nu === 3) {
    return inverseTolerance({
      tolerances: new Float64Array([50, 5, 100, 500]),
      weights: new Float64Array([1, 1, 1, 1]),
      controlTolerances: new Float64Array([10, 3, 200]),
      controlWeights: new Float64Array([1, 1, 1]),
    });
  }

  // Generic fallback: tolerance = 10.0, weight = 1.0 for every dimension
  const tolerances = new Float64Array(nx);
  const weights = new Float64Array(nx);
  const controlTolerances = new Float64Array(nu);
  const controlWeights = new Float64Array(nu);

  for (let i = 0; i < nx; i++) {
    tolerances[i] = 10.0;
    weights[i] = 1.0;
  }
  for (let j = 0; j < nu; j++) {
    controlTolerances[j] = 10.0;
    controlWeights[j] = 1.0;
  }

  return inverseTolerance({
    tolerances,
    weights,
    controlTolerances,
    controlWeights,
  });
}
