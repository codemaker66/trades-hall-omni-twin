// ---------------------------------------------------------------------------
// @omni-twin/compute-core — HPC-12: Solver Selection
// ---------------------------------------------------------------------------
// Analyzes matrix structural and numerical properties to recommend the
// best solver, precision, and estimated solve time.
// ---------------------------------------------------------------------------

import type { MatrixProperties, SolverRecommendation, Precision } from '../types.js';
import { estimateConditionNumber } from './mixed-precision.js';

// ---------------------------------------------------------------------------
// Machine epsilon
// ---------------------------------------------------------------------------

/** Machine epsilon for float32 (~1.19e-7). */
const EPS_F32 = 1.1920928955078125e-7;

// ---------------------------------------------------------------------------
// Matrix analysis
// ---------------------------------------------------------------------------

/**
 * Analyzes structural and numerical properties of an `rows x cols` matrix
 * stored in row-major Float64Array.
 *
 * Checks:
 * - **Symmetry**: |A[i,j] - A[j,i]| < eps for all i,j (square matrices only)
 * - **Positive definiteness**: all diagonal elements > 0 (necessary condition,
 *   simple heuristic — not sufficient in general)
 * - **Sparsity**: fraction of entries that are exactly zero > 0.5
 * - **Condition number**: estimated via power iteration (square matrices)
 * - **NNZ**: count of non-zero entries
 */
export function analyzeMatrix(
  data: Float64Array,
  rows: number,
  cols: number,
): MatrixProperties {
  const eps = 1e-12;
  const isSquare = rows === cols;
  const n = rows;

  // Count non-zeros
  let nnz = 0;
  for (let i = 0; i < data.length; i++) {
    if (data[i]! !== 0) {
      nnz++;
    }
  }

  const totalElements = rows * cols;
  const sparsity = totalElements > 0 ? 1 - nnz / totalElements : 0;
  const isSparse = sparsity > 0.5;

  // Symmetry check (only meaningful for square matrices)
  let isSymmetric = isSquare;
  if (isSquare) {
    for (let i = 0; i < n && isSymmetric; i++) {
      for (let j = i + 1; j < n && isSymmetric; j++) {
        const diff = Math.abs(data[i * n + j]! - data[j * n + i]!);
        if (diff > eps) {
          isSymmetric = false;
        }
      }
    }
  }

  // Positive definiteness heuristic: all diagonal elements > 0
  // (necessary condition for SPD; not sufficient in general)
  let isPD = isSymmetric;
  if (isSquare && isPD) {
    for (let i = 0; i < n; i++) {
      if (data[i * n + i]! <= 0) {
        isPD = false;
        break;
      }
    }
  }

  // Condition number estimate (only for square matrices)
  const kappa = isSquare ? estimateConditionNumber(data, n) : Infinity;

  return {
    rows,
    cols,
    symmetric: isSymmetric,
    positiveDefinite: isPD,
    sparse: isSparse,
    conditionNumber: kappa,
    nnz,
  };
}

// ---------------------------------------------------------------------------
// Solver recommendation
// ---------------------------------------------------------------------------

/**
 * Recommends the best solver based on detected matrix properties.
 *
 * Decision tree:
 * 1. Symmetric positive definite -> Cholesky (f64)
 * 2. Symmetric -> MINRES (f64)
 * 3. Well-conditioned (kappa < 1e4) -> LU (f32)
 * 4. Sparse + large (n >= 100) -> GMRES with preconditioner (f64)
 * 5. Dense + small (n < 100) -> Direct LU (f64)
 * 6. Default -> GMRES (f64)
 */
export function recommendSolver(props: MatrixProperties): SolverRecommendation {
  const n = Math.max(props.rows, props.cols);

  if (props.symmetric && props.positiveDefinite) {
    return {
      solver: 'direct_cholesky',
      precision: 'f64',
      reason: 'Matrix is symmetric positive definite; Cholesky is optimal',
      estimatedTimeMs: estimateSolveTimeMs(props, 'direct_cholesky'),
    };
  }

  if (props.symmetric) {
    return {
      solver: 'minres',
      precision: 'f64',
      reason: 'Matrix is symmetric; MINRES is well-suited',
      estimatedTimeMs: estimateSolveTimeMs(props, 'minres'),
    };
  }

  if (props.conditionNumber < 1e4) {
    return {
      solver: 'direct_lu',
      precision: 'f32',
      reason: `Well-conditioned (kappa=${props.conditionNumber.toExponential(2)}); f32 LU is sufficient`,
      estimatedTimeMs: estimateSolveTimeMs(props, 'direct_lu'),
    };
  }

  if (props.sparse && n >= 100) {
    return {
      solver: 'gmres',
      precision: 'f64',
      reason: 'Sparse and large; GMRES with preconditioner is efficient',
      estimatedTimeMs: estimateSolveTimeMs(props, 'gmres'),
    };
  }

  if (n < 100) {
    return {
      solver: 'direct_lu',
      precision: 'f64',
      reason: `Small matrix (n=${n}); direct LU in f64 is fast and reliable`,
      estimatedTimeMs: estimateSolveTimeMs(props, 'direct_lu'),
    };
  }

  // Default fallback
  return {
    solver: 'gmres',
    precision: 'f64',
    reason: 'General matrix; GMRES with f64 is a robust default',
    estimatedTimeMs: estimateSolveTimeMs(props, 'gmres'),
  };
}

// ---------------------------------------------------------------------------
// Solve time estimation
// ---------------------------------------------------------------------------

/**
 * Rough heuristic estimate of solve time in milliseconds.
 *
 * Models:
 * - **Direct solvers** (LU, Cholesky): O(n^3) with constant factor.
 *   Cholesky is ~2x faster than LU due to symmetry exploitation.
 * - **Iterative solvers** (CG, GMRES, BiCGSTAB, MINRES): O(nnz * iters).
 *   Default iteration count is sqrt(n), capped at 1000.
 *
 * Constant factors are calibrated for modern JS engines (V8/SpiderMonkey)
 * processing ~1e8 flops/second for dense linear algebra.
 */
export function estimateSolveTimeMs(
  props: MatrixProperties,
  solver: SolverRecommendation['solver'],
): number {
  const n = Math.max(props.rows, props.cols);
  // ~1e8 flops/s for JS linear algebra -> 1e-8 seconds per flop -> 1e-5 ms per flop
  const flopsToMs = 1e-5;

  switch (solver) {
    case 'direct_lu': {
      // O(2/3 n^3) flops
      const flops = (2 / 3) * n * n * n;
      return flops * flopsToMs;
    }
    case 'direct_cholesky': {
      // O(1/3 n^3) flops (half of LU due to symmetry)
      const flops = (1 / 3) * n * n * n;
      return flops * flopsToMs;
    }
    case 'cg':
    case 'gmres':
    case 'bicgstab':
    case 'minres': {
      // O(nnz * iters)
      const iters = Math.min(Math.ceil(Math.sqrt(n)), 1000);
      const flops = props.nnz * iters * 2; // 2 flops per multiply-add
      return flops * flopsToMs;
    }
  }
}

// ---------------------------------------------------------------------------
// Precision selection
// ---------------------------------------------------------------------------

/**
 * Selects the appropriate floating-point precision based on the matrix
 * condition number and the target residual.
 *
 * If `kappa * eps_f32 < targetResidual`, float32 is sufficient.
 * Otherwise, float64 is required.
 */
export function selectPrecision(
  props: MatrixProperties,
  targetResidual: number,
): Precision {
  if (props.conditionNumber * EPS_F32 < targetResidual) {
    return 'f32';
  }
  return 'f64';
}
