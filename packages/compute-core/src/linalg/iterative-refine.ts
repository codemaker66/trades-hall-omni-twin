// ---------------------------------------------------------------------------
// @omni-twin/compute-core — HPC-12: Iterative Refinement Solver
// ---------------------------------------------------------------------------
// LU factorization with partial pivoting, forward/back substitution,
// iterative refinement, residual computation, and conjugate gradient
// solver for symmetric positive definite matrices.
// ---------------------------------------------------------------------------

import type { MixedPrecisionConfig, SolverResult } from '../types.js';
import { f64ToF32, f32ToF64 } from './mixed-precision.js';

// ---------------------------------------------------------------------------
// LU factorization with partial pivoting
// ---------------------------------------------------------------------------

/**
 * Computes the LU factorization of an n x n matrix A with partial pivoting.
 *
 * Returns:
 * - `L`: unit lower triangular matrix (n x n, row-major)
 * - `U`: upper triangular matrix (n x n, row-major)
 * - `P`: permutation vector of length n such that PA = LU
 *
 * Uses Doolittle's method with row pivoting for numerical stability.
 */
export function luFactorize(
  A: Float64Array,
  n: number,
): { L: Float64Array; U: Float64Array; P: Uint32Array } {
  // Working copy of A that becomes U
  const U = new Float64Array(A);
  const L = new Float64Array(n * n);
  const P = new Uint32Array(n);

  // Initialize permutation to identity and L diagonal to 1
  for (let i = 0; i < n; i++) {
    P[i] = i;
    L[i * n + i] = 1;
  }

  for (let k = 0; k < n; k++) {
    // Partial pivoting: find row with largest |U[i,k]| for i >= k
    let maxVal = Math.abs(U[k * n + k]!);
    let maxRow = k;
    for (let i = k + 1; i < n; i++) {
      const val = Math.abs(U[i * n + k]!);
      if (val > maxVal) {
        maxVal = val;
        maxRow = i;
      }
    }

    // Swap rows k and maxRow in U (full row)
    if (maxRow !== k) {
      for (let j = 0; j < n; j++) {
        const tmpU = U[k * n + j]!;
        U[k * n + j] = U[maxRow * n + j]!;
        U[maxRow * n + j] = tmpU;
      }
      // Swap already-computed L entries (columns 0..k-1)
      for (let j = 0; j < k; j++) {
        const tmpL = L[k * n + j]!;
        L[k * n + j] = L[maxRow * n + j]!;
        L[maxRow * n + j] = tmpL;
      }
      // Swap permutation entries
      const tmpP = P[k]!;
      P[k] = P[maxRow]!;
      P[maxRow] = tmpP;
    }

    const pivot = U[k * n + k]!;
    if (pivot === 0) {
      // Singular pivot — skip column (L entries remain 0)
      continue;
    }

    // Gaussian elimination below the pivot
    for (let i = k + 1; i < n; i++) {
      const factor = U[i * n + k]! / pivot;
      L[i * n + k] = factor;
      for (let j = k; j < n; j++) {
        U[i * n + j] = U[i * n + j]! - factor * U[k * n + j]!;
      }
    }
  }

  return { L, U, P };
}

// ---------------------------------------------------------------------------
// LU solve (forward + back substitution)
// ---------------------------------------------------------------------------

/**
 * Solves the system LUx = Pb using the precomputed LU factors and
 * permutation vector.
 *
 * 1. Apply permutation: y = Pb
 * 2. Forward substitution: Lz = y
 * 3. Back substitution: Ux = z
 */
export function luSolve(
  L: Float64Array,
  U: Float64Array,
  P: Uint32Array,
  b: Float64Array,
  n: number,
): Float64Array {
  // Apply permutation
  const pb = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    pb[i] = b[P[i]!]!;
  }

  // Forward substitution: Lz = pb  (L is unit lower triangular)
  const z = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = pb[i]!;
    for (let j = 0; j < i; j++) {
      sum -= L[i * n + j]! * z[j]!;
    }
    z[i] = sum;
  }

  // Back substitution: Ux = z
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = z[i]!;
    for (let j = i + 1; j < n; j++) {
      sum -= U[i * n + j]! * x[j]!;
    }
    const diag = U[i * n + i]!;
    x[i] = diag === 0 ? 0 : sum / diag;
  }

  return x;
}

// ---------------------------------------------------------------------------
// Iterative refinement
// ---------------------------------------------------------------------------

/**
 * Full iterative refinement solver using mixed-precision LU factorization.
 *
 * Algorithm:
 * 1. Optionally downcast A to factorPrecision for LU factorization.
 * 2. Compute LU(A) in factor precision.
 * 3. Solve for initial x.
 * 4. Loop (up to maxRefinements):
 *    a. Compute residual r = b - Ax in refinePrecision.
 *    b. Solve for correction d using LU factors.
 *    c. Update x = x + d.
 *    d. Check if ||r||_2 < targetResidual.
 */
export function iterativeRefinement(
  A: Float64Array,
  b: Float64Array,
  n: number,
  config: MixedPrecisionConfig,
): SolverResult {
  if (n <= 0) {
    return {
      solution: new Float64Array(0),
      iterations: 0,
      residual: 0,
      converged: true,
    };
  }

  // Optionally downcast for factorization
  let factorA: Float64Array;
  if (config.factorPrecision === 'f32') {
    factorA = f32ToF64(f64ToF32(A));
  } else {
    factorA = new Float64Array(A);
  }

  // LU factorize in factor precision
  const { L, U, P } = luFactorize(factorA, n);

  // Initial solve
  const x = luSolve(L, U, P, b, n);

  // Iterative refinement
  let r = computeResidual(A, x, b, n);
  let rNorm = residualNorm(r);
  let iterations = 0;

  for (let k = 0; k < config.maxRefinements; k++) {
    if (rNorm <= config.targetResidual) {
      break;
    }
    iterations++;

    // Solve for correction: use same LU factors
    const d = luSolve(L, U, P, r, n);

    // Update solution: x = x + d
    for (let i = 0; i < n; i++) {
      x[i] = x[i]! + d[i]!;
    }

    // Recompute residual
    r = computeResidual(A, x, b, n);
    rNorm = residualNorm(r);
  }

  return {
    solution: x,
    iterations,
    residual: rNorm,
    converged: rNorm <= config.targetResidual,
  };
}

// ---------------------------------------------------------------------------
// Residual computation
// ---------------------------------------------------------------------------

/**
 * Computes the residual vector r = b - Ax for an n x n system.
 */
export function computeResidual(
  A: Float64Array,
  x: Float64Array,
  b: Float64Array,
  n: number,
): Float64Array {
  const r = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let ax = 0;
    for (let j = 0; j < n; j++) {
      ax += A[i * n + j]! * x[j]!;
    }
    r[i] = b[i]! - ax;
  }
  return r;
}

/**
 * Computes the L2 (Euclidean) norm of a residual vector.
 */
export function residualNorm(r: Float64Array): number {
  let sumSq = 0;
  for (let i = 0; i < r.length; i++) {
    const val = r[i]!;
    sumSq += val * val;
  }
  return Math.sqrt(sumSq);
}

// ---------------------------------------------------------------------------
// Conjugate Gradient solver (for SPD matrices)
// ---------------------------------------------------------------------------

/**
 * Conjugate Gradient solver for symmetric positive definite (SPD) matrices.
 *
 * Solves Ax = b where A is n x n SPD, stored row-major in Float64Array.
 *
 * Algorithm:
 * 1. Initialize x = 0, r = b, p = r.
 * 2. Iterate: alpha = (r^T r) / (p^T A p),
 *    x += alpha * p, r -= alpha * Ap,
 *    beta = (r_new^T r_new) / (r_old^T r_old), p = r + beta * p.
 * 3. Converge when ||r||_2 < tol or maxIter reached.
 */
export function conjugateGradient(
  A: Float64Array,
  b: Float64Array,
  n: number,
  maxIter: number,
  tol: number,
): SolverResult {
  if (n <= 0) {
    return {
      solution: new Float64Array(0),
      iterations: 0,
      residual: 0,
      converged: true,
    };
  }

  // x = 0
  const x = new Float64Array(n);

  // r = b - Ax = b (since x=0)
  const r = new Float64Array(b);

  // p = r
  const p = new Float64Array(r);

  // rTr = r^T r
  let rTr = dot(r, r, n);
  let iterations = 0;

  const Ap = new Float64Array(n);

  for (let iter = 0; iter < maxIter; iter++) {
    const rNorm = Math.sqrt(rTr);
    if (rNorm <= tol) {
      return {
        solution: x,
        iterations,
        residual: rNorm,
        converged: true,
      };
    }

    iterations++;

    // Ap = A * p
    matVecMul(A, p, Ap, n);

    // alpha = rTr / (p^T Ap)
    const pAp = dot(p, Ap, n);
    if (pAp === 0) {
      // Breakdown: A is not SPD or p is in the null space
      return {
        solution: x,
        iterations,
        residual: Math.sqrt(rTr),
        converged: false,
      };
    }
    const alpha = rTr / pAp;

    // x = x + alpha * p
    for (let i = 0; i < n; i++) {
      x[i] = x[i]! + alpha * p[i]!;
    }

    // r = r - alpha * Ap
    for (let i = 0; i < n; i++) {
      r[i] = r[i]! - alpha * Ap[i]!;
    }

    // beta = rTr_new / rTr_old
    const rTrNew = dot(r, r, n);
    const beta = rTrNew / rTr;
    rTr = rTrNew;

    // p = r + beta * p
    for (let i = 0; i < n; i++) {
      p[i] = r[i]! + beta * p[i]!;
    }
  }

  return {
    solution: x,
    iterations,
    residual: Math.sqrt(rTr),
    converged: Math.sqrt(rTr) <= tol,
  };
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Dot product of two length-n vectors. */
function dot(a: Float64Array, b: Float64Array, n: number): number {
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += a[i]! * b[i]!;
  }
  return sum;
}

/** Matrix-vector multiply: out = A * v, where A is n x n row-major. */
function matVecMul(
  A: Float64Array,
  v: Float64Array,
  out: Float64Array,
  n: number,
): void {
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      sum += A[i * n + j]! * v[j]!;
    }
    out[i] = sum;
  }
}
