// ---------------------------------------------------------------------------
// SP-9: Orthogonal Matching Pursuit (OMP)
// ---------------------------------------------------------------------------
// Greedy sparse recovery: O(s²·M·N)
// At each iteration, select the column of A most correlated with the residual.
// If demand is s-sparse in frequency domain:
//   M = O(s·log(N/s)) measurements suffice.

import type { OMPConfig, SparseRecoveryResult } from '../types.js';

/**
 * Orthogonal Matching Pursuit for sparse signal recovery.
 *
 * Given y = A·x where x is s-sparse, recover x from y.
 * A is the measurement matrix (M × N).
 *
 * @param A Measurement matrix (row-major, M × N)
 * @param y Observation vector (M elements)
 * @param M Number of measurements
 * @param N Signal dimension
 * @param config OMP parameters
 */
export function omp(
  A: Float64Array,
  y: Float64Array,
  M: number,
  N: number,
  config: OMPConfig = { nComponents: 10 },
): SparseRecoveryResult {
  const { nComponents, tolerance = 1e-6 } = config;

  const residual = new Float64Array(y);
  const support: number[] = [];
  const coefficients = new Float64Array(N);

  for (let iter = 0; iter < nComponents; iter++) {
    // Find column most correlated with residual
    let bestCol = -1;
    let bestCorr = -1;

    for (let j = 0; j < N; j++) {
      if (support.includes(j)) continue;

      let corr = 0;
      for (let i = 0; i < M; i++) {
        corr += A[i * N + j]! * residual[i]!;
      }
      const absCorr = Math.abs(corr);
      if (absCorr > bestCorr) {
        bestCorr = absCorr;
        bestCol = j;
      }
    }

    if (bestCol < 0 || bestCorr < tolerance) break;
    support.push(bestCol);

    // Solve least squares on support set: A_S · x_S = y
    const s = support.length;
    const AS = new Float64Array(M * s);
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < s; j++) {
        AS[i * s + j] = A[i * N + support[j]!]!;
      }
    }

    // Normal equations: (ASᵀAS)·x = ASᵀy
    const AtA = new Float64Array(s * s);
    const Aty = new Float64Array(s);

    for (let i = 0; i < s; i++) {
      for (let j = 0; j < s; j++) {
        let sum = 0;
        for (let k = 0; k < M; k++) {
          sum += AS[k * s + i]! * AS[k * s + j]!;
        }
        AtA[i * s + j] = sum;
      }
      let sum = 0;
      for (let k = 0; k < M; k++) {
        sum += AS[k * s + i]! * y[k]!;
      }
      Aty[i] = sum;
    }

    // Solve via Gauss elimination
    const xS = solveLinearSystem(AtA, Aty, s);

    // Update residual: r = y - AS·xS
    for (let i = 0; i < M; i++) {
      residual[i] = y[i]!;
      for (let j = 0; j < s; j++) {
        residual[i] = residual[i]! - AS[i * s + j]! * xS[j]!;
      }
    }

    // Store coefficients
    coefficients.fill(0);
    for (let j = 0; j < s; j++) {
      coefficients[support[j]!] = xS[j]!;
    }

    // Check convergence
    let residualNorm = 0;
    for (let i = 0; i < M; i++) residualNorm += residual[i]! * residual[i]!;
    if (Math.sqrt(residualNorm) < tolerance) break;
  }

  // Reconstruct full signal (if using a basis like DCT)
  const signal = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    signal[i] = coefficients[i]!;
  }

  let residualNorm = 0;
  for (let i = 0; i < M; i++) residualNorm += residual[i]! * residual[i]!;

  return {
    signal,
    coefficients,
    support,
    residualNorm: Math.sqrt(residualNorm),
  };
}

/** Solve Ax = b via Gaussian elimination (small systems). */
function solveLinearSystem(A: Float64Array, b: Float64Array, n: number): Float64Array {
  // Augmented matrix [A | b]
  const aug = new Float64Array(n * (n + 1));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      aug[i * (n + 1) + j] = A[i * n + j]!;
    }
    aug[i * (n + 1) + n] = b[i]!;
  }

  // Forward elimination with partial pivoting
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    let maxVal = Math.abs(aug[col * (n + 1) + col]!);
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(aug[row * (n + 1) + col]!);
      if (val > maxVal) { maxVal = val; maxRow = row; }
    }

    if (maxRow !== col) {
      for (let j = 0; j <= n; j++) {
        const tmp = aug[col * (n + 1) + j]!;
        aug[col * (n + 1) + j] = aug[maxRow * (n + 1) + j]!;
        aug[maxRow * (n + 1) + j] = tmp;
      }
    }

    const pivot = aug[col * (n + 1) + col]!;
    if (Math.abs(pivot) < 1e-14) continue;

    for (let row = col + 1; row < n; row++) {
      const factor = aug[row * (n + 1) + col]! / pivot;
      for (let j = col; j <= n; j++) {
        aug[row * (n + 1) + j] = aug[row * (n + 1) + j]! - factor * aug[col * (n + 1) + j]!;
      }
    }
  }

  // Back substitution
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = aug[i * (n + 1) + n]!;
    for (let j = i + 1; j < n; j++) {
      sum -= aug[i * (n + 1) + j]! * x[j]!;
    }
    const diag = aug[i * (n + 1) + i]!;
    x[i] = Math.abs(diag) > 1e-14 ? sum / diag : 0;
  }

  return x;
}

/**
 * Recover a demand curve from sparse observations using OMP + DCT basis.
 * Uses Discrete Cosine Transform as sparsifying basis.
 *
 * @param observedDays Indices of observed days
 * @param observedValues Observed values at those days
 * @param totalDays Total number of days to recover
 * @param nComponents Number of DCT components to use
 */
export function recoverDemandCurve(
  observedDays: number[],
  observedValues: Float64Array,
  totalDays: number = 365,
  nComponents: number = 10,
): SparseRecoveryResult {
  const M = observedDays.length;
  const N = totalDays;

  // Build measurement matrix: Phi (row selection) × Psi (DCT basis)
  // A = Phi × Psi^T
  const A = new Float64Array(M * N);

  for (let i = 0; i < M; i++) {
    const day = observedDays[i]!;
    for (let k = 0; k < N; k++) {
      // DCT-II basis function
      A[i * N + k] = Math.cos((Math.PI * (2 * day + 1) * k) / (2 * N));
      if (k === 0) A[i * N + k] = A[i * N + k]! * (1 / Math.sqrt(N));
      else A[i * N + k] = A[i * N + k]! * Math.sqrt(2 / N);
    }
  }

  const result = omp(A, observedValues, M, N, { nComponents });

  // Reconstruct signal from DCT coefficients
  const signal = new Float64Array(N);
  for (let n = 0; n < N; n++) {
    for (let k = 0; k < N; k++) {
      let basis = Math.cos((Math.PI * (2 * n + 1) * k) / (2 * N));
      if (k === 0) basis *= 1 / Math.sqrt(N);
      else basis *= Math.sqrt(2 / N);
      signal[n] = signal[n]! + result.coefficients[k]! * basis;
    }
  }

  return { ...result, signal };
}
