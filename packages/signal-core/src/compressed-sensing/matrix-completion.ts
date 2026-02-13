// ---------------------------------------------------------------------------
// SP-9: Matrix Completion via Nuclear Norm Minimization
// ---------------------------------------------------------------------------
// Candès & Recht (2009): min ||X||_*  s.t. X_ij = M_ij for observed (i,j)
// Rank-r matrix: O(r·(n₁+n₂)·log²n) observations needed.
// Uses Alternating Least Squares (ALS) as a practical proxy.

import type { MatrixCompletionConfig, MatrixCompletionResult } from '../types.js';

/**
 * Matrix completion via Alternating Least Squares (ALS).
 * Factorizes X ≈ U·Vᵀ where U is (nRows × rank), V is (nCols × rank).
 *
 * @param observed Array of { row, col, value } for known entries
 * @param config Configuration
 * @param rank Target rank (default: estimated from data)
 */
export function matrixCompletionALS(
  observed: Array<{ row: number; col: number; value: number }>,
  config: MatrixCompletionConfig,
  rank?: number,
): MatrixCompletionResult {
  const { nRows, nCols, lambda, maxIter, tolerance } = config;
  const r = rank ?? Math.min(10, Math.floor(Math.min(nRows, nCols) / 2));

  // Initialize U and V with small random values
  const U = new Float64Array(nRows * r);
  const V = new Float64Array(nCols * r);
  for (let i = 0; i < U.length; i++) U[i] = 0.1 * (Math.random() - 0.5);
  for (let i = 0; i < V.length; i++) V[i] = 0.1 * (Math.random() - 0.5);

  // Build lookup: for each row, which observations exist
  const rowObs: Map<number, Array<{ col: number; value: number }>> = new Map();
  const colObs: Map<number, Array<{ row: number; value: number }>> = new Map();

  for (const { row, col, value } of observed) {
    if (!rowObs.has(row)) rowObs.set(row, []);
    rowObs.get(row)!.push({ col, value });
    if (!colObs.has(col)) colObs.set(col, []);
    colObs.get(col)!.push({ row, value });
  }

  let prevResidual = Infinity;

  for (let iter = 0; iter < maxIter; iter++) {
    // Fix V, solve for each row of U
    for (let i = 0; i < nRows; i++) {
      const obs = rowObs.get(i);
      if (!obs || obs.length === 0) continue;

      // Build VᵀV + λI for observed columns
      const VtV = new Float64Array(r * r);
      const Vty = new Float64Array(r);

      for (const { col, value } of obs) {
        for (let a = 0; a < r; a++) {
          Vty[a] = Vty[a]! + V[col * r + a]! * value;
          for (let b = 0; b < r; b++) {
            VtV[a * r + b] = VtV[a * r + b]! + V[col * r + a]! * V[col * r + b]!;
          }
        }
      }

      // Regularization
      for (let a = 0; a < r; a++) {
        VtV[a * r + a] = VtV[a * r + a]! + lambda;
      }

      // Solve (VᵀV + λI)·u_i = Vᵀy_i
      const ui = solveSmall(VtV, Vty, r);
      for (let a = 0; a < r; a++) {
        U[i * r + a] = ui[a]!;
      }
    }

    // Fix U, solve for each row of V
    for (let j = 0; j < nCols; j++) {
      const obs = colObs.get(j);
      if (!obs || obs.length === 0) continue;

      const UtU = new Float64Array(r * r);
      const Uty = new Float64Array(r);

      for (const { row, value } of obs) {
        for (let a = 0; a < r; a++) {
          Uty[a] = Uty[a]! + U[row * r + a]! * value;
          for (let b = 0; b < r; b++) {
            UtU[a * r + b] = UtU[a * r + b]! + U[row * r + a]! * U[row * r + b]!;
          }
        }
      }

      for (let a = 0; a < r; a++) {
        UtU[a * r + a] = UtU[a * r + a]! + lambda;
      }

      const vj = solveSmall(UtU, Uty, r);
      for (let a = 0; a < r; a++) {
        V[j * r + a] = vj[a]!;
      }
    }

    // Check convergence
    let residual = 0;
    for (const { row, col, value } of observed) {
      let pred = 0;
      for (let a = 0; a < r; a++) {
        pred += U[row * r + a]! * V[col * r + a]!;
      }
      const diff = pred - value;
      residual += diff * diff;
    }
    residual = Math.sqrt(residual / observed.length);

    if (Math.abs(prevResidual - residual) < tolerance) break;
    prevResidual = residual;
  }

  // Build completed matrix
  const completed = new Float64Array(nRows * nCols);
  for (let i = 0; i < nRows; i++) {
    for (let j = 0; j < nCols; j++) {
      let val = 0;
      for (let a = 0; a < r; a++) {
        val += U[i * r + a]! * V[j * r + a]!;
      }
      completed[i * nCols + j] = val;
    }
  }

  return {
    completed,
    rank: r,
    residual: prevResidual,
  };
}

/** Solve small linear system Ax = b. */
function solveSmall(A: Float64Array, b: Float64Array, n: number): Float64Array {
  const aug = new Float64Array(n * (n + 1));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) aug[i * (n + 1) + j] = A[i * n + j]!;
    aug[i * (n + 1) + n] = b[i]!;
  }

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

  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = aug[i * (n + 1) + n]!;
    for (let j = i + 1; j < n; j++) sum -= aug[i * (n + 1) + j]! * x[j]!;
    const diag = aug[i * (n + 1) + i]!;
    x[i] = Math.abs(diag) > 1e-14 ? sum / diag : 0;
  }
  return x;
}
