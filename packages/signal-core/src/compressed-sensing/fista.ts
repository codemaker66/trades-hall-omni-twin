// ---------------------------------------------------------------------------
// SP-9: FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
// ---------------------------------------------------------------------------
// O(1/k²) convergence rate for L1-regularized problems.
// Solves: min_x (1/2)||Ax - y||² + λ||x||₁
// Uses Nesterov acceleration via momentum term.

import type { FISTAConfig, SparseRecoveryResult } from '../types.js';

/**
 * Soft-thresholding operator (proximal of L1 norm).
 * prox_{λ||·||₁}(x) = sign(x)·max(|x| - λ, 0)
 */
function softThreshold(x: Float64Array, lambda: number): Float64Array {
  const result = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const val = x[i]!;
    if (val > lambda) result[i] = val - lambda;
    else if (val < -lambda) result[i] = val + lambda;
    // else result[i] = 0 (already initialized)
  }
  return result;
}

/**
 * FISTA: Fast Iterative Shrinkage-Thresholding Algorithm.
 *
 * Solves: min_x (1/2)||Ax - y||² + λ||x||₁
 *
 * @param A Measurement matrix (M × N, row-major)
 * @param y Observation vector (M)
 * @param M Number of measurements
 * @param N Signal dimension
 * @param config FISTA parameters
 */
export function fista(
  A: Float64Array,
  y: Float64Array,
  M: number,
  N: number,
  config: FISTAConfig = { stepSize: 0.01, lambda: 0.1, maxIter: 500, tolerance: 1e-6 },
): SparseRecoveryResult {
  const { stepSize, lambda, maxIter, tolerance } = config;

  let x: Float64Array = new Float64Array(N);
  let z = new Float64Array(N); // Momentum term
  let tPrev = 1;

  for (let iter = 0; iter < maxIter; iter++) {
    // Gradient: ∇f(z) = Aᵀ(Az - y)
    // First compute Az
    const Az = new Float64Array(M);
    for (let i = 0; i < M; i++) {
      let sum = 0;
      for (let j = 0; j < N; j++) {
        sum += A[i * N + j]! * z[j]!;
      }
      Az[i] = sum;
    }

    // Residual: Az - y
    const residual = new Float64Array(M);
    for (let i = 0; i < M; i++) {
      residual[i] = Az[i]! - y[i]!;
    }

    // Gradient: Aᵀ·residual
    const gradient = new Float64Array(N);
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let i = 0; i < M; i++) {
        sum += A[i * N + j]! * residual[i]!;
      }
      gradient[j] = sum;
    }

    // Gradient step
    const stepped = new Float64Array(N);
    for (let j = 0; j < N; j++) {
      stepped[j] = z[j]! - stepSize * gradient[j]!;
    }

    // Proximal (soft-thresholding)
    const xNew = softThreshold(stepped, stepSize * lambda);

    // Nesterov momentum: t_{k+1} = (1 + √(1 + 4·t_k²)) / 2
    const tNew = (1 + Math.sqrt(1 + 4 * tPrev * tPrev)) / 2;
    const momentum = (tPrev - 1) / tNew;

    // z = x_new + momentum · (x_new - x)
    const zNew = new Float64Array(N);
    for (let j = 0; j < N; j++) {
      zNew[j] = xNew[j]! + momentum * (xNew[j]! - x[j]!);
    }

    // Check convergence
    let diffNorm = 0;
    for (let j = 0; j < N; j++) {
      const diff = xNew[j]! - x[j]!;
      diffNorm += diff * diff;
    }
    diffNorm = Math.sqrt(diffNorm);

    x = xNew;
    z = zNew;
    tPrev = tNew;

    if (diffNorm < tolerance) break;
  }

  // Compute residual norm
  const Ax = new Float64Array(M);
  for (let i = 0; i < M; i++) {
    let sum = 0;
    for (let j = 0; j < N; j++) sum += A[i * N + j]! * x[j]!;
    Ax[i] = sum;
  }
  let residualNorm = 0;
  for (let i = 0; i < M; i++) {
    const diff = Ax[i]! - y[i]!;
    residualNorm += diff * diff;
  }

  // Extract support
  const support: number[] = [];
  for (let j = 0; j < N; j++) {
    if (Math.abs(x[j]!) > 1e-10) support.push(j);
  }

  return {
    signal: x,
    coefficients: x,
    support,
    residualNorm: Math.sqrt(residualNorm),
  };
}
