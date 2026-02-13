// ---------------------------------------------------------------------------
// @omni-twin/compute-core — HPC-12: Mixed Precision Utilities
// ---------------------------------------------------------------------------
// f32/f64 conversion, precision analysis, condition number estimation,
// and a mixed-precision iterative refinement solver.
// ---------------------------------------------------------------------------

import type { MixedPrecisionConfig, SolverResult } from '../types.js';

// ---------------------------------------------------------------------------
// Machine epsilon constants
// ---------------------------------------------------------------------------

/** Machine epsilon for float32 (~1.19e-7). */
const EPS_F32 = 1.1920928955078125e-7;

// ---------------------------------------------------------------------------
// Precision conversion
// ---------------------------------------------------------------------------

/**
 * Converts a Float64Array to Float32Array, potentially losing precision
 * for values that cannot be exactly represented in single precision.
 */
export function f64ToF32(data: Float64Array): Float32Array {
  const result = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    result[i] = data[i]!;
  }
  return result;
}

/**
 * Widens a Float32Array to Float64Array. This conversion is exact:
 * every float32 value is exactly representable as float64.
 */
export function f32ToF64(data: Float32Array): Float64Array {
  const result = new Float64Array(data.length);
  for (let i = 0; i < data.length; i++) {
    result[i] = data[i]!;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Precision analysis
// ---------------------------------------------------------------------------

/**
 * Computes the maximum absolute error between an original Float64Array
 * and a round-tripped version (f64 -> f32 -> f64).
 *
 * This quantifies the worst-case precision loss from using float32
 * representation for the given data.
 */
export function precisionLoss(
  original: Float64Array,
  roundTripped: Float64Array,
): number {
  let maxError = 0;
  const len = Math.min(original.length, roundTripped.length);
  for (let i = 0; i < len; i++) {
    const err = Math.abs(original[i]! - roundTripped[i]!);
    if (err > maxError) {
      maxError = err;
    }
  }
  return maxError;
}

/**
 * Checks whether all values in the array can be safely represented in
 * float32 without exceeding the given tolerance.
 *
 * Verifies two conditions:
 * 1. No value exceeds the float32 representable range (~3.4e38).
 * 2. The round-trip precision loss is within tolerance (default 1e-6).
 */
export function isSafeForF32(
  values: Float64Array,
  tolerance: number = 1e-6,
): boolean {
  const F32_MAX = 3.4028234663852886e38;

  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    if (Math.abs(v) > F32_MAX) {
      return false;
    }
  }

  // Check round-trip precision
  const f32 = f64ToF32(values);
  const roundTripped = f32ToF64(f32);
  const loss = precisionLoss(values, roundTripped);
  return loss <= tolerance;
}

// ---------------------------------------------------------------------------
// Condition number estimation
// ---------------------------------------------------------------------------

/**
 * Rough estimate of the condition number of an n x n matrix stored in
 * row-major Float64Array, using the ratio of the largest to smallest
 * singular value estimated via power iteration.
 *
 * This is a heuristic: for a more accurate estimate, a full SVD or
 * QR-based method would be needed. We run power iteration for the
 * largest singular value, then inverse iteration for the smallest.
 *
 * Returns `Infinity` if the matrix appears singular.
 */
export function estimateConditionNumber(
  matrix: Float64Array,
  n: number,
): number {
  if (n <= 0) return Infinity;
  if (n === 1) {
    const val = matrix[0]!;
    return val === 0 ? Infinity : 1;
  }

  const maxIter = 30;

  // Power iteration for largest singular value: compute A^T * A * v
  const v = new Float64Array(n);
  // Initialize with unit vector (1/sqrt(n), ..., 1/sqrt(n))
  const invSqrtN = 1 / Math.sqrt(n);
  for (let i = 0; i < n; i++) {
    v[i] = invSqrtN;
  }

  let sigmaMax = 0;
  for (let iter = 0; iter < maxIter; iter++) {
    // w = A * v
    const w = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < n; j++) {
        sum += matrix[i * n + j]! * v[j]!;
      }
      w[i] = sum;
    }

    // u = A^T * w
    const u = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < n; j++) {
        sum += matrix[j * n + i]! * w[j]!;
      }
      u[i] = sum;
    }

    // Compute norm of u
    let norm = 0;
    for (let i = 0; i < n; i++) {
      norm += u[i]! * u[i]!;
    }
    norm = Math.sqrt(norm);

    if (norm === 0) return Infinity;

    sigmaMax = Math.sqrt(norm);

    // Normalize: v = u / norm
    for (let i = 0; i < n; i++) {
      v[i] = u[i]! / norm;
    }
  }

  // Estimate smallest singular value using inverse power iteration
  // on (A^T A). We approximate by finding the minimum row norm as a
  // lower bound heuristic (true inverse iteration would need LU solve).
  let minRowNorm = Infinity;
  for (let i = 0; i < n; i++) {
    let rowNorm = 0;
    for (let j = 0; j < n; j++) {
      const val = matrix[i * n + j]!;
      rowNorm += val * val;
    }
    rowNorm = Math.sqrt(rowNorm);
    if (rowNorm < minRowNorm) {
      minRowNorm = rowNorm;
    }
  }

  if (minRowNorm === 0) return Infinity;

  return sigmaMax / minRowNorm;
}

// ---------------------------------------------------------------------------
// Precision decision
// ---------------------------------------------------------------------------

/**
 * Returns `true` if double precision is needed based on the condition
 * number and desired tolerance.
 *
 * The rationale: in float32 arithmetic, the achievable accuracy is
 * roughly `kappa * eps_f32`. If this exceeds the tolerance, we need
 * float64.
 */
export function needsDoublePrecision(
  conditionNumber: number,
  tolerance: number,
): boolean {
  return conditionNumber * EPS_F32 > tolerance;
}

// ---------------------------------------------------------------------------
// Mixed-precision solve
// ---------------------------------------------------------------------------

/**
 * Mixed-precision solver: factorizes in `config.factorPrecision`, then
 * iteratively refines in `config.refinePrecision`.
 *
 * Algorithm:
 * 1. Optionally downcast A to f32 for factorization.
 * 2. Compute LU factorization of A (in factor precision).
 * 3. Solve for initial x using LU factors.
 * 4. Iteratively refine: compute residual r = b - Ax in refine precision,
 *    solve for correction, and update x.
 * 5. Repeat until residual < targetResidual or maxRefinements reached.
 *
 * For simplicity, uses a basic LU decomposition with partial pivoting.
 */
export function mixedPrecisionSolve(
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

  // Optionally downcast to f32 for factorization
  let factorA: Float64Array;
  let factorB: Float64Array;

  if (config.factorPrecision === 'f32') {
    factorA = f32ToF64(f64ToF32(A));
    factorB = f32ToF64(f64ToF32(b));
  } else {
    factorA = new Float64Array(A);
    factorB = new Float64Array(b);
  }

  // LU factorization with partial pivoting
  const { L, U, P } = luDecompose(factorA, n);

  // Initial solve: Ly = Pb, Ux = y
  const x = luSolveInternal(L, U, P, factorB, n);

  // Iterative refinement in refine precision
  let residual = computeResidualNorm(A, x, b, n);
  let iterations = 0;

  for (let k = 0; k < config.maxRefinements; k++) {
    if (residual <= config.targetResidual) {
      break;
    }
    iterations++;

    // Compute residual r = b - Ax in full (refine) precision
    const r = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let ax = 0;
      for (let j = 0; j < n; j++) {
        ax += A[i * n + j]! * x[j]!;
      }
      r[i] = b[i]! - ax;
    }

    // Solve for correction: A * d = r (using same LU factors)
    const d = luSolveInternal(L, U, P, r, n);

    // Update: x = x + d
    for (let i = 0; i < n; i++) {
      x[i] = x[i]! + d[i]!;
    }

    residual = computeResidualNorm(A, x, b, n);
  }

  return {
    solution: x,
    iterations,
    residual,
    converged: residual <= config.targetResidual,
  };
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * LU decomposition with partial pivoting.
 * Returns L (unit lower triangular), U (upper triangular), P (permutation).
 */
function luDecompose(
  A: Float64Array,
  n: number,
): { L: Float64Array; U: Float64Array; P: Uint32Array } {
  const U = new Float64Array(A);
  const L = new Float64Array(n * n);
  const P = new Uint32Array(n);

  // Initialize permutation and L diagonal
  for (let i = 0; i < n; i++) {
    P[i] = i;
    L[i * n + i] = 1;
  }

  for (let k = 0; k < n; k++) {
    // Find pivot
    let maxVal = Math.abs(U[k * n + k]!);
    let maxRow = k;
    for (let i = k + 1; i < n; i++) {
      const val = Math.abs(U[i * n + k]!);
      if (val > maxVal) {
        maxVal = val;
        maxRow = i;
      }
    }

    // Swap rows in U
    if (maxRow !== k) {
      for (let j = 0; j < n; j++) {
        const tmp = U[k * n + j]!;
        U[k * n + j] = U[maxRow * n + j]!;
        U[maxRow * n + j] = tmp;
      }
      // Swap in L (only columns before k)
      for (let j = 0; j < k; j++) {
        const tmp = L[k * n + j]!;
        L[k * n + j] = L[maxRow * n + j]!;
        L[maxRow * n + j] = tmp;
      }
      // Swap permutation
      const tmp = P[k]!;
      P[k] = P[maxRow]!;
      P[maxRow] = tmp;
    }

    const pivot = U[k * n + k]!;
    if (pivot === 0) continue;

    // Eliminate below pivot
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

/**
 * Solves Ly = Pb (forward substitution), then Ux = y (back substitution).
 */
function luSolveInternal(
  L: Float64Array,
  U: Float64Array,
  P: Uint32Array,
  b: Float64Array,
  n: number,
): Float64Array {
  // Apply permutation: Pb
  const pb = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    pb[i] = b[P[i]!]!;
  }

  // Forward substitution: Ly = Pb
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = pb[i]!;
    for (let j = 0; j < i; j++) {
      sum -= L[i * n + j]! * y[j]!;
    }
    y[i] = sum; // L diagonal is 1
  }

  // Back substitution: Ux = y
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = y[i]!;
    for (let j = i + 1; j < n; j++) {
      sum -= U[i * n + j]! * x[j]!;
    }
    const diag = U[i * n + i]!;
    x[i] = diag === 0 ? 0 : sum / diag;
  }

  return x;
}

/**
 * Computes the L2 norm of the residual r = b - Ax.
 */
function computeResidualNorm(
  A: Float64Array,
  x: Float64Array,
  b: Float64Array,
  n: number,
): number {
  let normSq = 0;
  for (let i = 0; i < n; i++) {
    let ax = 0;
    for (let j = 0; j < n; j++) {
      ax += A[i * n + j]! * x[j]!;
    }
    const ri = b[i]! - ax;
    normSq += ri * ri;
  }
  return Math.sqrt(normSq);
}
