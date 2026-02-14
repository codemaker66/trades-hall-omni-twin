import { describe, it, expect } from 'vitest';
import {
  f64ToF32,
  f32ToF64,
  precisionLoss,
  isSafeForF32,
  estimateConditionNumber,
  needsDoublePrecision,
  mixedPrecisionSolve,
  analyzeMatrix,
  recommendSolver,
  estimateSolveTimeMs,
  selectPrecision,
  luFactorize,
  luSolve,
  iterativeRefinement,
  computeResidual,
  residualNorm,
  conjugateGradient,
} from '../linalg/index.js';

import type { MatrixProperties, MixedPrecisionConfig } from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build an n x n identity matrix in row-major Float64Array. */
function identityMatrix(n: number): Float64Array {
  const m = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    m[i * n + i] = 1;
  }
  return m;
}

/**
 * Build a symmetric positive definite (SPD) matrix for testing.
 * Uses A = I + alpha * ones(n,n) which is SPD for alpha > 0.
 */
function spdMatrix(n: number, alpha: number = 1): Float64Array {
  const m = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      m[i * n + j] = alpha;
    }
    m[i * n + i]! += 1;
  }
  return m;
}

// ---------------------------------------------------------------------------
// Mixed Precision
// ---------------------------------------------------------------------------

describe('Mixed Precision', () => {
  it('f64ToF32 roundtrip preserves approximate values for small integers', () => {
    const original = new Float64Array([1, 2, 3, 4, 5]);
    const f32 = f64ToF32(original);
    const back = f32ToF64(f32);
    for (let i = 0; i < original.length; i++) {
      expect(back[i]!).toBeCloseTo(original[i]!, 5);
    }
  });

  it('f64ToF32 produces a Float32Array of the same length', () => {
    const original = new Float64Array([1.5, 2.5, 3.5]);
    const f32 = f64ToF32(original);
    expect(f32).toBeInstanceOf(Float32Array);
    expect(f32.length).toBe(3);
  });

  it('precisionLoss computes max absolute error between arrays', () => {
    const a = new Float64Array([1.0, 2.0, 3.0]);
    const b = new Float64Array([1.1, 2.0, 2.8]);
    const loss = precisionLoss(a, b);
    expect(loss).toBeCloseTo(0.2, 10);
  });

  it('precisionLoss is 0 for identical arrays', () => {
    const a = new Float64Array([1, 2, 3]);
    expect(precisionLoss(a, a)).toBe(0);
  });

  it('isSafeForF32 returns true for small integers', () => {
    const values = new Float64Array([1, 2, 3, 100, -50]);
    expect(isSafeForF32(values)).toBe(true);
  });

  it('isSafeForF32 returns false for values exceeding f32 range', () => {
    const values = new Float64Array([1e39]);
    expect(isSafeForF32(values)).toBe(false);
  });

  it('estimateConditionNumber of identity matrix is 1', () => {
    const I = identityMatrix(3);
    const kappa = estimateConditionNumber(I, 3);
    expect(kappa).toBeCloseTo(1, 0);
  });

  it('estimateConditionNumber is positive for well-conditioned matrices', () => {
    const A = spdMatrix(3, 0.5);
    const kappa = estimateConditionNumber(A, 3);
    expect(kappa).toBeGreaterThan(0);
    expect(kappa).toBeLessThan(Infinity);
  });

  it('needsDoublePrecision returns true for ill-conditioned matrices', () => {
    // kappa = 1e12, eps_f32 ~ 1.19e-7 => kappa * eps > 1e-6
    expect(needsDoublePrecision(1e12, 1e-6)).toBe(true);
  });

  it('needsDoublePrecision returns false for well-conditioned matrices', () => {
    // kappa = 10, eps_f32 ~ 1.19e-7 => kappa * eps ~ 1.19e-6 < 1e-3
    expect(needsDoublePrecision(10, 1e-3)).toBe(false);
  });

  it('mixedPrecisionSolve solves a 2x2 system correctly', () => {
    // A = [[2, 1], [1, 3]], b = [5, 10] => x = [1, 3]
    const A = new Float64Array([2, 1, 1, 3]);
    const b = new Float64Array([5, 10]);
    const config: MixedPrecisionConfig = {
      factorPrecision: 'f64',
      refinePrecision: 'f64',
      maxRefinements: 5,
      targetResidual: 1e-10,
    };
    const result = mixedPrecisionSolve(A, b, 2, config);
    expect(result.solution[0]!).toBeCloseTo(1, 8);
    expect(result.solution[1]!).toBeCloseTo(3, 8);
    expect(result.converged).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// LU Factorize & Solve
// ---------------------------------------------------------------------------

describe('LU Factorize & Solve', () => {
  it('luFactorize produces L, U, P for a 2x2 matrix', () => {
    const A = new Float64Array([4, 3, 6, 3]);
    const { L, U, P } = luFactorize(A, 2);
    expect(L.length).toBe(4);
    expect(U.length).toBe(4);
    expect(P.length).toBe(2);
  });

  it('luSolve solves a 2x2 system Ax = b', () => {
    // A = [[2, 1], [1, 3]], b = [5, 10] => x = [1, 3]
    const A = new Float64Array([2, 1, 1, 3]);
    const { L, U, P } = luFactorize(A, 2);
    const b = new Float64Array([5, 10]);
    const x = luSolve(L, U, P, b, 2);
    expect(x[0]!).toBeCloseTo(1, 10);
    expect(x[1]!).toBeCloseTo(3, 10);
  });

  it('luSolve of identity matrix returns the rhs', () => {
    const I = identityMatrix(3);
    const { L, U, P } = luFactorize(I, 3);
    const b = new Float64Array([7, 8, 9]);
    const x = luSolve(L, U, P, b, 3);
    expect(x[0]!).toBeCloseTo(7, 10);
    expect(x[1]!).toBeCloseTo(8, 10);
    expect(x[2]!).toBeCloseTo(9, 10);
  });

  it('computeResidual is near zero for exact solution', () => {
    const A = new Float64Array([2, 1, 1, 3]);
    const x = new Float64Array([1, 3]);
    const b = new Float64Array([5, 10]);
    const r = computeResidual(A, x, b, 2);
    const norm = residualNorm(r);
    expect(norm).toBeLessThan(1e-10);
  });

  it('residualNorm of zero vector is 0', () => {
    expect(residualNorm(new Float64Array([0, 0, 0]))).toBe(0);
  });

  it('iterativeRefinement converges for well-conditioned system', () => {
    // A = [[4, 1], [1, 3]], b = [9, 10] => x = [17/11, 31/11]
    const A = new Float64Array([4, 1, 1, 3]);
    const b = new Float64Array([9, 10]);
    const config: MixedPrecisionConfig = {
      factorPrecision: 'f64',
      refinePrecision: 'f64',
      maxRefinements: 10,
      targetResidual: 1e-12,
    };
    const result = iterativeRefinement(A, b, 2, config);
    expect(result.converged).toBe(true);
    expect(result.residual).toBeLessThan(1e-12);
    expect(result.solution[0]!).toBeCloseTo(17 / 11, 8);
    expect(result.solution[1]!).toBeCloseTo(31 / 11, 8);
  });

  it('iterativeRefinement with f32 factorization still converges via refinement', () => {
    const A = new Float64Array([3, 1, 1, 2]);
    const b = new Float64Array([7, 5]);
    const config: MixedPrecisionConfig = {
      factorPrecision: 'f32',
      refinePrecision: 'f64',
      maxRefinements: 10,
      targetResidual: 1e-10,
    };
    const result = iterativeRefinement(A, b, 2, config);
    // x = [9/5, 8/5] = [1.8, 1.6]
    expect(result.solution[0]!).toBeCloseTo(1.8, 6);
    expect(result.solution[1]!).toBeCloseTo(1.6, 6);
  });
});

// ---------------------------------------------------------------------------
// Conjugate Gradient
// ---------------------------------------------------------------------------

describe('Conjugate Gradient', () => {
  it('solves a 3x3 SPD system', () => {
    // A = [[4, 1, 0], [1, 3, 1], [0, 1, 2]]
    const A = new Float64Array([4, 1, 0, 1, 3, 1, 0, 1, 2]);
    const b = new Float64Array([5, 5, 3]);
    const result = conjugateGradient(A, b, 3, 100, 1e-10);
    expect(result.converged).toBe(true);
    expect(result.residual).toBeLessThan(1e-10);
    // Verify Ax = b
    const r = computeResidual(A, result.solution, b, 3);
    expect(residualNorm(r)).toBeLessThan(1e-8);
  });

  it('converges within maxIter for well-conditioned system', () => {
    const A = spdMatrix(4, 0.1);
    const b = new Float64Array([1, 2, 3, 4]);
    const result = conjugateGradient(A, b, 4, 50, 1e-10);
    expect(result.converged).toBe(true);
    expect(result.iterations).toBeLessThanOrEqual(50);
  });

  it('returns n=0 result for empty system', () => {
    const result = conjugateGradient(new Float64Array(0), new Float64Array(0), 0, 10, 1e-8);
    expect(result.converged).toBe(true);
    expect(result.solution.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Solver Selection
// ---------------------------------------------------------------------------

describe('Solver Selection', () => {
  it('recommendSolver picks cholesky for SPD matrix', () => {
    const props: MatrixProperties = {
      rows: 10,
      cols: 10,
      symmetric: true,
      positiveDefinite: true,
      sparse: false,
      conditionNumber: 5,
      nnz: 100,
    };
    const rec = recommendSolver(props);
    expect(rec.solver).toBe('direct_cholesky');
    expect(rec.precision).toBe('f64');
  });

  it('recommendSolver picks minres for symmetric non-PD matrix', () => {
    const props: MatrixProperties = {
      rows: 10,
      cols: 10,
      symmetric: true,
      positiveDefinite: false,
      sparse: false,
      conditionNumber: 100,
      nnz: 100,
    };
    const rec = recommendSolver(props);
    expect(rec.solver).toBe('minres');
  });

  it('recommendSolver picks lu for well-conditioned non-symmetric matrix', () => {
    const props: MatrixProperties = {
      rows: 10,
      cols: 10,
      symmetric: false,
      positiveDefinite: false,
      sparse: false,
      conditionNumber: 50,
      nnz: 100,
    };
    const rec = recommendSolver(props);
    expect(rec.solver).toBe('direct_lu');
    expect(rec.precision).toBe('f32');
  });

  it('recommendSolver picks gmres for large sparse ill-conditioned matrix', () => {
    const props: MatrixProperties = {
      rows: 200,
      cols: 200,
      symmetric: false,
      positiveDefinite: false,
      sparse: true,
      conditionNumber: 1e8,
      nnz: 2000,
    };
    const rec = recommendSolver(props);
    expect(rec.solver).toBe('gmres');
    expect(rec.precision).toBe('f64');
  });

  it('estimateSolveTimeMs returns positive value', () => {
    const props: MatrixProperties = {
      rows: 50,
      cols: 50,
      symmetric: false,
      positiveDefinite: false,
      sparse: false,
      conditionNumber: 10,
      nnz: 2500,
    };
    const time = estimateSolveTimeMs(props, 'direct_lu');
    expect(time).toBeGreaterThan(0);
  });

  it('selectPrecision returns f32 for well-conditioned, f64 for ill-conditioned', () => {
    const wellConditioned: MatrixProperties = {
      rows: 10,
      cols: 10,
      symmetric: true,
      positiveDefinite: true,
      sparse: false,
      conditionNumber: 10,
      nnz: 100,
    };
    const illConditioned: MatrixProperties = {
      ...wellConditioned,
      conditionNumber: 1e12,
    };
    expect(selectPrecision(wellConditioned, 1e-4)).toBe('f32');
    expect(selectPrecision(illConditioned, 1e-4)).toBe('f64');
  });
});

// ---------------------------------------------------------------------------
// Matrix Analysis
// ---------------------------------------------------------------------------

describe('Matrix Analysis', () => {
  it('analyzeMatrix detects symmetric matrix', () => {
    // Symmetric 2x2: [[2, 1], [1, 3]]
    const data = new Float64Array([2, 1, 1, 3]);
    const props = analyzeMatrix(data, 2, 2);
    expect(props.symmetric).toBe(true);
    expect(props.rows).toBe(2);
    expect(props.cols).toBe(2);
  });

  it('analyzeMatrix detects non-symmetric matrix', () => {
    // Non-symmetric: [[1, 2], [3, 4]]
    const data = new Float64Array([1, 2, 3, 4]);
    const props = analyzeMatrix(data, 2, 2);
    expect(props.symmetric).toBe(false);
  });

  it('analyzeMatrix detects positive definiteness (heuristic)', () => {
    // SPD: [[4, 1], [1, 3]] -> symmetric, positive diag
    const data = new Float64Array([4, 1, 1, 3]);
    const props = analyzeMatrix(data, 2, 2);
    expect(props.positiveDefinite).toBe(true);
  });

  it('analyzeMatrix reports correct NNZ count', () => {
    // 2x2 with one zero: [[1, 0], [0, 3]]
    const data = new Float64Array([1, 0, 0, 3]);
    const props = analyzeMatrix(data, 2, 2);
    expect(props.nnz).toBe(2);
  });

  it('analyzeMatrix detects sparsity when > 50% zeros', () => {
    // 3x3 diagonal: 3 non-zero out of 9 => 66.7% zeros => sparse
    const data = new Float64Array([1, 0, 0, 0, 2, 0, 0, 0, 3]);
    const props = analyzeMatrix(data, 3, 3);
    expect(props.sparse).toBe(true);
    expect(props.nnz).toBe(3);
  });

  it('analyzeMatrix computes positive condition number', () => {
    const data = new Float64Array([4, 1, 1, 3]);
    const props = analyzeMatrix(data, 2, 2);
    expect(props.conditionNumber).toBeGreaterThan(0);
    expect(props.conditionNumber).toBeLessThan(Infinity);
  });
});
