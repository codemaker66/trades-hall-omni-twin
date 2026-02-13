// ---------------------------------------------------------------------------
// OC-10: Optimal Experiment Design -- Dual Control
// ---------------------------------------------------------------------------
//
// Dual control balances exploitation (driving the state to a target) with
// exploration (reducing parameter uncertainty).  The action is composed of
// an exploitation term (proportional feedback) and an exploration term
// (probing in the direction of greatest uncertainty).
//
// Also provides Kalman-style belief updates and information gain computation.
// ---------------------------------------------------------------------------

import type { PRNG, DualControlConfig, DualControlResult } from '../types.js';
import { vecAdd, vecScale, vecNorm } from '../types.js';

// ---------------------------------------------------------------------------
// Dual control step
// ---------------------------------------------------------------------------

/**
 * Compute a single dual-control action that trades off exploitation and
 * exploration.
 *
 * - **Exploitation**: proportional feedback  u_exploit = -K * x,  where K is a
 *   simple gain derived from the prior mean (diagonal approximation).
 * - **Exploration**: perturbation along the direction of greatest parameter
 *   uncertainty (the dimension with the largest diagonal covariance entry),
 *   scaled by a random sign from the PRNG so the probing signal is unbiased.
 * - **Action**: u = u_exploit + explorationWeight * u_explore.
 *
 * @param config  Dual control configuration (prior, exploration weight, etc.)
 * @param x       Current state vector (nx)
 * @param rng     Seedable PRNG
 * @returns DualControlResult with the composite action and its components
 */
export function dualControlStep(
  config: DualControlConfig,
  x: Float64Array,
  rng: PRNG,
): DualControlResult {
  const { priorMean, priorCov, nx, explorationWeight } = config;

  // -------------------------------------------------------------------------
  // Exploitation component: -K * x  (proportional to state deviation)
  // K is taken as the diagonal of the prior mean reshaped as a gain vector.
  // For a scalar-like proportional controller we use K_i = priorMean_i so
  // that the exploitation action drives x toward zero.
  // -------------------------------------------------------------------------
  const exploitationComponent = new Float64Array(nx);
  for (let i = 0; i < nx; i++) {
    exploitationComponent[i] = -priorMean[i]! * x[i]!;
  }

  // -------------------------------------------------------------------------
  // Exploration component: direction of maximum information gain.
  // Simplified as the standard basis direction with the largest diagonal
  // covariance entry, scaled by sqrt(variance) and a random sign.
  // -------------------------------------------------------------------------
  const explorationComponent = new Float64Array(nx);
  let maxVar = -Infinity;
  let maxIdx = 0;
  for (let i = 0; i < nx; i++) {
    const v = priorCov[i * nx + i]!;
    if (v > maxVar) {
      maxVar = v;
      maxIdx = i;
    }
  }

  // Random sign for unbiased probing
  const sign = rng() < 0.5 ? -1.0 : 1.0;
  explorationComponent[maxIdx] = sign * Math.sqrt(Math.max(0, maxVar));

  // -------------------------------------------------------------------------
  // Composite action
  // -------------------------------------------------------------------------
  const action = vecAdd(
    exploitationComponent,
    vecScale(explorationComponent, explorationWeight),
  );

  // -------------------------------------------------------------------------
  // Information gain from this action (approximate)
  // Use the expected variance reduction along the probed direction.
  // For a single probing action the information gain is approximately
  // 0.5 * ln(prior_var / (prior_var - probing_reduction)).
  // Here we approximate the reduction as proportional to the exploration
  // magnitude squared relative to the total variance.
  // -------------------------------------------------------------------------
  const exploreNorm = vecNorm(vecScale(explorationComponent, explorationWeight));
  const totalVar = maxVar > 0 ? maxVar : 1;
  const ratio = 1 + (exploreNorm * exploreNorm) / totalVar;
  const informationGain = 0.5 * Math.log(ratio);

  return {
    action,
    explorationComponent,
    exploitationComponent,
    informationGain,
  };
}

// ---------------------------------------------------------------------------
// Information gain between two covariances
// ---------------------------------------------------------------------------

/**
 * Compute the KL-divergence-based information gain between a prior and
 * posterior covariance.
 *
 * Simplified formula using diagonal entries:
 *
 *   IG = 0.5 * sum_i ln( priorCov_{ii} / posteriorCov_{ii} )
 *
 * This equals 0.5 * ln( det(prior) / det(posterior) ) when the covariance
 * matrices are diagonal.
 *
 * @param priorCov     Prior covariance (nx x nx, row-major)
 * @param posteriorCov Posterior covariance (nx x nx, row-major)
 * @param nx           Dimension
 * @returns Scalar information gain (nats)
 */
export function computeInfoGain(
  priorCov: Float64Array,
  posteriorCov: Float64Array,
  nx: number,
): number {
  let sum = 0;
  for (let i = 0; i < nx; i++) {
    const prii = priorCov[i * nx + i]!;
    const posti = posteriorCov[i * nx + i]!;
    if (posti > 0 && prii > 0) {
      sum += Math.log(prii / posti);
    }
  }
  return 0.5 * sum;
}

// ---------------------------------------------------------------------------
// Kalman-style belief update
// ---------------------------------------------------------------------------

/**
 * Update a Gaussian belief (mean, covariance) given a linear observation.
 *
 * Observation model:  z = H * theta + v,  v ~ N(0, R)
 *
 * Kalman update equations:
 *   S  = H P H^T + R            (innovation covariance, nz x nz)
 *   K  = P H^T S^{-1}           (Kalman gain, nx x nz)
 *   mean' = mean + K (z - H mean)
 *   P'    = (I - K H) P
 *
 * @param priorMean   Prior mean (nx)
 * @param priorCov    Prior covariance (nx x nx, row-major)
 * @param observation Observation vector z (nz)
 * @param H           Observation matrix (nz x nx, row-major)
 * @param R           Observation noise covariance (nz x nz, row-major)
 * @param nx          State / parameter dimension
 * @param nz          Observation dimension
 * @returns Updated mean and covariance
 */
export function updateBelief(
  priorMean: Float64Array,
  priorCov: Float64Array,
  observation: Float64Array,
  H: Float64Array,
  R: Float64Array,
  nx: number,
  nz: number,
): { mean: Float64Array; cov: Float64Array } {
  // S = H P H^T + R  (nz x nz)
  const S = new Float64Array(nz * nz);
  for (let i = 0; i < nz; i++) {
    for (let j = 0; j < nz; j++) {
      let hph = 0;
      for (let a = 0; a < nx; a++) {
        for (let b = 0; b < nx; b++) {
          hph += H[i * nx + a]! * priorCov[a * nx + b]! * H[j * nx + b]!;
        }
      }
      S[i * nz + j] = hph + R[i * nz + j]!;
    }
  }

  // Invert S via Gauss-Jordan (small matrix)
  const Sinv = invertSmall(S, nz);

  // K = P H^T S^{-1}  (nx x nz)
  // First compute P H^T  (nx x nz)
  const PHt = new Float64Array(nx * nz);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < nz; j++) {
      let val = 0;
      for (let k = 0; k < nx; k++) {
        val += priorCov[i * nx + k]! * H[j * nx + k]!;
      }
      PHt[i * nz + j] = val;
    }
  }

  // K = PHt * Sinv  (nx x nz)
  const K = new Float64Array(nx * nz);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < nz; j++) {
      let val = 0;
      for (let k = 0; k < nz; k++) {
        val += PHt[i * nz + k]! * Sinv[k * nz + j]!;
      }
      K[i * nz + j] = val;
    }
  }

  // Innovation: y = z - H * mean  (nz)
  const innovation = new Float64Array(nz);
  for (let i = 0; i < nz; i++) {
    let hm = 0;
    for (let j = 0; j < nx; j++) {
      hm += H[i * nx + j]! * priorMean[j]!;
    }
    innovation[i] = observation[i]! - hm;
  }

  // mean' = mean + K * innovation  (nx)
  const mean = new Float64Array(nx);
  for (let i = 0; i < nx; i++) {
    let kInnov = 0;
    for (let j = 0; j < nz; j++) {
      kInnov += K[i * nz + j]! * innovation[j]!;
    }
    mean[i] = priorMean[i]! + kInnov;
  }

  // P' = (I - K H) P  (nx x nx)
  // First compute K H  (nx x nx)
  const KH = new Float64Array(nx * nx);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < nx; j++) {
      let val = 0;
      for (let k = 0; k < nz; k++) {
        val += K[i * nz + k]! * H[k * nx + j]!;
      }
      KH[i * nx + j] = val;
    }
  }

  // (I - KH) P
  const cov = new Float64Array(nx * nx);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < nx; j++) {
      let val = 0;
      for (let k = 0; k < nx; k++) {
        const iMinusKH = (i === k ? 1 : 0) - KH[i * nx + k]!;
        val += iMinusKH * priorCov[k * nx + j]!;
      }
      cov[i * nx + j] = val;
    }
  }

  return { mean, cov };
}

// ---------------------------------------------------------------------------
// Internal: small matrix inversion via Gauss-Jordan with partial pivoting
// ---------------------------------------------------------------------------

function invertSmall(data: Float64Array, n: number): Float64Array {
  // Augmented matrix [A | I]
  const aug = new Float64Array(n * 2 * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      aug[i * 2 * n + j] = data[i * n + j]!;
    }
    aug[i * 2 * n + n + i] = 1;
  }

  for (let col = 0; col < n; col++) {
    // Partial pivoting
    let maxRow = col;
    let maxVal = Math.abs(aug[col * 2 * n + col]!);
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(aug[row * 2 * n + col]!);
      if (val > maxVal) {
        maxVal = val;
        maxRow = row;
      }
    }
    if (maxRow !== col) {
      for (let j = 0; j < 2 * n; j++) {
        const tmp = aug[col * 2 * n + j]!;
        aug[col * 2 * n + j] = aug[maxRow * 2 * n + j]!;
        aug[maxRow * 2 * n + j] = tmp;
      }
    }

    const pivot = aug[col * 2 * n + col]!;
    if (Math.abs(pivot) < 1e-15) {
      // Singular — return identity as fallback
      const fallback = new Float64Array(n * n);
      for (let i = 0; i < n; i++) {
        fallback[i * n + i] = 1;
      }
      return fallback;
    }

    // Scale pivot row
    for (let j = 0; j < 2 * n; j++) {
      aug[col * 2 * n + j] = aug[col * 2 * n + j]! / pivot;
    }

    // Eliminate column
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row * 2 * n + col]!;
      for (let j = 0; j < 2 * n; j++) {
        aug[row * 2 * n + j] = aug[row * 2 * n + j]! - factor * aug[col * 2 * n + j]!;
      }
    }
  }

  // Extract inverse
  const inv = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      inv[i * n + j] = aug[i * 2 * n + n + j]!;
    }
  }
  return inv;
}
