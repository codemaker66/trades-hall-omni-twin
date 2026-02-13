// ---------------------------------------------------------------------------
// OC-4  Dynamic Programming -- Approximate Dynamic Programming
// ---------------------------------------------------------------------------

import type { PRNG, ApproxDPConfig, ApproxDPResult } from '../types.js';
import { vecDot } from '../types.js';

// ---------------------------------------------------------------------------
// Fitted Value Iteration
// ---------------------------------------------------------------------------

/**
 * Fitted value iteration with linear function approximation.
 *
 * Approximates V(s) ~ phi(s)^T w where phi is a user-supplied feature map.
 * At each iteration:
 *   1. Sample states from the state space.
 *   2. Compute Bellman targets:  y_i = r(s_i, a_i) + gamma * phi(s'_i)^T w
 *      where s'_i ~ P(. | s_i, a_i).
 *   3. Solve the least-squares problem: min_w || Phi w - y ||^2.
 *
 * Convergence is declared when the max change in weights falls below
 * `config.tolerance`.
 */
export function fittedValueIteration(
  config: ApproxDPConfig,
  rng: PRNG,
): ApproxDPResult {
  const {
    nFeatures,
    nSamples,
    featureFn,
    transitionSample,
    rewardFn,
    discount,
    maxIter,
    tolerance,
  } = config;

  let weights: Float64Array = new Float64Array(nFeatures);
  let converged = false;
  let iterations = 0;

  // Pre-sample a set of representative states (sampled once, reused across
  // iterations).  We generate random states in [0,1]^nFeatures as a proxy
  // (the caller should embed domain-specific sampling into `featureFn` and
  // `transitionSample`).
  const sampleStates: Float64Array[] = [];
  for (let i = 0; i < nSamples; i++) {
    const s = new Float64Array(nFeatures);
    for (let d = 0; d < nFeatures; d++) {
      s[d] = rng();
    }
    sampleStates.push(s);
  }

  for (let iter = 0; iter < maxIter; iter++) {
    // Build feature matrix Phi (nSamples x nFeatures) and target vector y
    const Phi = new Float64Array(nSamples * nFeatures);
    const y = new Float64Array(nSamples);

    for (let i = 0; i < nSamples; i++) {
      const s = sampleStates[i]!;
      const phi = featureFn(s);

      // Store row i of Phi
      for (let f = 0; f < nFeatures; f++) {
        Phi[i * nFeatures + f] = phi[f]!;
      }

      // Use a zero action for simplicity in the single-action setting;
      // for multi-action, the caller encodes action selection into the
      // transitionSample / rewardFn.
      const a = new Float64Array(nFeatures); // zero action
      const r = rewardFn(s, a);
      const sPrime = transitionSample(s, a, rng);
      const phiPrime = featureFn(sPrime);

      y[i] = r + discount * vecDot(phiPrime, weights);
    }

    // Solve least squares:  w = (Phi^T Phi)^{-1} Phi^T y
    // Build Phi^T Phi (nFeatures x nFeatures) and Phi^T y (nFeatures)
    const PhiTPhi = new Float64Array(nFeatures * nFeatures);
    const PhiTy = new Float64Array(nFeatures);

    for (let i = 0; i < nSamples; i++) {
      for (let f1 = 0; f1 < nFeatures; f1++) {
        const phi_if1 = Phi[i * nFeatures + f1]!;
        PhiTy[f1] = PhiTy[f1]! + phi_if1 * y[i]!;
        for (let f2 = 0; f2 < nFeatures; f2++) {
          PhiTPhi[f1 * nFeatures + f2] = PhiTPhi[f1 * nFeatures + f2]! +
            phi_if1 * Phi[i * nFeatures + f2]!;
        }
      }
    }

    // Add small regularization for numerical stability
    for (let f = 0; f < nFeatures; f++) {
      PhiTPhi[f * nFeatures + f] = PhiTPhi[f * nFeatures + f]! + 1e-8;
    }

    // Solve via Cholesky-like direct method (Gauss elimination for small n)
    const newWeights = solveLinearSystem(PhiTPhi, PhiTy, nFeatures);

    // Check convergence
    let maxDelta = 0;
    for (let f = 0; f < nFeatures; f++) {
      const delta = Math.abs(newWeights[f]! - weights[f]!);
      if (delta > maxDelta) maxDelta = delta;
    }

    weights = newWeights as Float64Array<ArrayBuffer>;
    iterations = iter + 1;

    if (maxDelta < tolerance) {
      converged = true;
      break;
    }
  }

  return { weights, iterations, converged };
}

// ---------------------------------------------------------------------------
// evaluateApproxValue
// ---------------------------------------------------------------------------

/**
 * Evaluate the approximate value function at a single state:
 *   V(s) = w^T phi(s)
 *
 * where `weights` is the learned w and `features` is phi(s).
 */
export function evaluateApproxValue(
  weights: Float64Array,
  features: Float64Array,
): number {
  return vecDot(weights, features);
}

// ---------------------------------------------------------------------------
// Internal: small dense linear solve  (Gauss elimination with pivoting)
// ---------------------------------------------------------------------------

/**
 * Solve Ax = b for a small dense system using Gaussian elimination
 * with partial pivoting.
 */
function solveLinearSystem(
  A: Float64Array,
  b: Float64Array,
  n: number,
) {
  // Work on copies
  const M = A.slice();
  const rhs = b.slice();

  // Forward elimination
  for (let col = 0; col < n; col++) {
    // Partial pivoting
    let maxRow = col;
    let maxVal = Math.abs(M[col * n + col]!);
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(M[row * n + col]!);
      if (val > maxVal) {
        maxVal = val;
        maxRow = row;
      }
    }
    // Swap rows
    if (maxRow !== col) {
      for (let j = 0; j < n; j++) {
        const tmp = M[col * n + j]!;
        M[col * n + j] = M[maxRow * n + j]!;
        M[maxRow * n + j] = tmp;
      }
      const tmpB = rhs[col]!;
      rhs[col] = rhs[maxRow]!;
      rhs[maxRow] = tmpB;
    }

    const pivot = M[col * n + col]!;
    if (Math.abs(pivot) < 1e-15) continue;

    for (let row = col + 1; row < n; row++) {
      const factor = M[row * n + col]! / pivot;
      for (let j = col; j < n; j++) {
        M[row * n + j] = M[row * n + j]! - factor * M[col * n + j]!;
      }
      rhs[row] = rhs[row]! - factor * rhs[col]!;
    }
  }

  // Back substitution
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = rhs[i]!;
    for (let j = i + 1; j < n; j++) {
      sum -= M[i * n + j]! * x[j]!;
    }
    const diag = M[i * n + i]!;
    x[i] = Math.abs(diag) > 1e-15 ? sum / diag : 0;
  }

  return x;
}
