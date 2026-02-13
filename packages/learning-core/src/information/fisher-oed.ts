// ---------------------------------------------------------------------------
// Fisher Information & D-Optimal Experimental Design
// ---------------------------------------------------------------------------

import type { FisherOEDResult } from '../types.js';

/**
 * Compute the Fisher Information Matrix: I(θ) = J^T J
 *
 * The Fisher information matrix summarizes the amount of information
 * that an observable random variable carries about an unknown parameter.
 * For a model with Jacobian J (∂y/∂θ), the FIM is J^T * J.
 *
 * @param jacobian - Jacobian matrix, jacobian[i][j] = ∂y_i/∂θ_j
 *                   Shape: nObservations x nParams
 * @returns Fisher information matrix (nParams x nParams)
 */
export function fisherInformationMatrix(jacobian: number[][]): number[][] {
  const nObs = jacobian.length;
  if (nObs === 0) return [];

  const nParams = jacobian[0]?.length ?? 0;
  if (nParams === 0) return [];

  // Compute J^T * J
  const fim: number[][] = [];
  for (let i = 0; i < nParams; i++) {
    const row = new Array<number>(nParams);
    for (let j = 0; j < nParams; j++) {
      let sum = 0;
      for (let k = 0; k < nObs; k++) {
        const jRow = jacobian[k];
        if (!jRow) continue;
        sum += (jRow[i] ?? 0) * (jRow[j] ?? 0);
      }
      row[j] = sum;
    }
    fim[i] = row;
  }
  return fim;
}

/**
 * Compute the determinant of a square matrix using LU decomposition.
 * Used for D-optimality criterion.
 *
 * @param matrix - Square matrix
 * @returns Determinant
 */
function determinant(matrix: number[][]): number {
  const n = matrix.length;
  if (n === 0) return 1;
  if (n === 1) return matrix[0]![0] ?? 0;
  if (n === 2) {
    return ((matrix[0]![0] ?? 0) * (matrix[1]![1] ?? 0)) -
      ((matrix[0]![1] ?? 0) * (matrix[1]![0] ?? 0));
  }

  // LU decomposition with partial pivoting
  // Create a working copy
  const lu: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row = new Array<number>(n);
    for (let j = 0; j < n; j++) {
      row[j] = matrix[i]![j] ?? 0;
    }
    lu[i] = row;
  }

  let sign = 1;

  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxVal = Math.abs(lu[col]![col] ?? 0);
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(lu[row]![col] ?? 0);
      if (val > maxVal) {
        maxVal = val;
        maxRow = row;
      }
    }

    // Swap rows if needed
    if (maxRow !== col) {
      const tmp = lu[col]!;
      lu[col] = lu[maxRow]!;
      lu[maxRow] = tmp;
      sign = -sign;
    }

    const pivot = lu[col]![col] ?? 0;
    if (Math.abs(pivot) < 1e-300) return 0;

    // Eliminate below
    for (let row = col + 1; row < n; row++) {
      const factor = (lu[row]![col] ?? 0) / pivot;
      lu[row]![col] = factor;
      for (let j = col + 1; j < n; j++) {
        lu[row]![j] = (lu[row]![j] ?? 0) - factor * (lu[col]![j] ?? 0);
      }
    }
  }

  // Determinant is product of diagonal * sign
  let det = sign;
  for (let i = 0; i < n; i++) {
    det *= (lu[i]![i] ?? 0);
  }
  return det;
}

/**
 * Invert a square matrix using Gauss-Jordan elimination.
 *
 * @param matrix - Square matrix to invert
 * @returns Inverse matrix, or identity if singular
 */
function invertMatrix(matrix: number[][]): number[][] {
  const n = matrix.length;
  if (n === 0) return [];

  // Augmented matrix [A | I]
  const aug: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row = new Array<number>(2 * n);
    for (let j = 0; j < n; j++) {
      row[j] = matrix[i]![j] ?? 0;
    }
    for (let j = 0; j < n; j++) {
      row[n + j] = i === j ? 1 : 0;
    }
    aug[i] = row;
  }

  // Forward elimination with partial pivoting
  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxVal = Math.abs(aug[col]![col] ?? 0);
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(aug[row]![col] ?? 0);
      if (val > maxVal) {
        maxVal = val;
        maxRow = row;
      }
    }

    if (maxRow !== col) {
      const tmp = aug[col]!;
      aug[col] = aug[maxRow]!;
      aug[maxRow] = tmp;
    }

    const pivot = aug[col]![col] ?? 0;
    if (Math.abs(pivot) < 1e-300) {
      // Singular: return identity
      const identity: number[][] = [];
      for (let i = 0; i < n; i++) {
        const row = new Array<number>(n);
        for (let j = 0; j < n; j++) {
          row[j] = i === j ? 1 : 0;
        }
        identity[i] = row;
      }
      return identity;
    }

    // Scale pivot row
    for (let j = 0; j < 2 * n; j++) {
      aug[col]![j] = (aug[col]![j] ?? 0) / pivot;
    }

    // Eliminate all other rows
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row]![col] ?? 0;
      for (let j = 0; j < 2 * n; j++) {
        aug[row]![j] = (aug[row]![j] ?? 0) - factor * (aug[col]![j] ?? 0);
      }
    }
  }

  // Extract inverse from right half
  const inv: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row = new Array<number>(n);
    for (let j = 0; j < n; j++) {
      row[j] = aug[i]![n + j] ?? 0;
    }
    inv[i] = row;
  }
  return inv;
}

/**
 * D-optimality criterion: det(I(θ))
 *
 * D-optimal designs maximize the determinant of the Fisher information matrix,
 * which minimizes the volume of the confidence ellipsoid for parameter estimates.
 *
 * @param fisherInfo - Fisher information matrix (nParams x nParams)
 * @returns Determinant of the Fisher information matrix
 */
export function dOptimality(fisherInfo: number[][]): number {
  return determinant(fisherInfo);
}

/**
 * A-optimality criterion: trace(I(θ)^{-1})
 *
 * A-optimal designs minimize the trace of the inverse Fisher information matrix,
 * which minimizes the average variance of parameter estimates.
 *
 * @param fisherInfo - Fisher information matrix (nParams x nParams)
 * @returns Trace of the inverse Fisher information matrix
 */
export function aOptimality(fisherInfo: number[][]): number {
  const inv = invertMatrix(fisherInfo);
  const n = inv.length;
  let trace = 0;
  for (let i = 0; i < n; i++) {
    trace += (inv[i]![i] ?? 0);
  }
  return trace;
}

/**
 * Federov exchange algorithm for D-optimal experimental design.
 *
 * Greedy algorithm that iteratively builds an optimal design by:
 * 1. Starting with the first nSelect candidates
 * 2. For each iteration, trying to swap each selected point with each unselected
 *    candidate, keeping the swap that maximizes D-optimality
 * 3. Repeating until no swap improves D-optimality
 *
 * @param candidates - Array of candidate design points, each a vector of length nParams
 * @param nSelect - Number of design points to select
 * @param nParams - Number of model parameters
 * @param evalFisher - Function that computes the Fisher information matrix for a
 *                     given set of selected design points
 * @returns FisherOEDResult with optimal design indices, FIM, and D-optimality value
 */
export function federovExchange(
  candidates: number[][],
  nSelect: number,
  nParams: number,
  evalFisher: (selected: number[][]) => number[][],
): FisherOEDResult {
  const nCandidates = candidates.length;
  const nSel = Math.min(nSelect, nCandidates);

  if (nSel === 0 || nCandidates === 0) {
    return {
      optimalDesign: [],
      fisherInfo: [],
      dOptimality: 0,
    };
  }

  // Initialize: pick the first nSel candidates
  const selectedIndices: number[] = [];
  for (let i = 0; i < nSel; i++) {
    selectedIndices[i] = i;
  }

  // Build set for O(1) lookup
  const inDesign = new Set<number>(selectedIndices);

  // Compute initial D-optimality
  const getSelectedPoints = (): number[][] => {
    const pts: number[][] = [];
    for (let i = 0; i < selectedIndices.length; i++) {
      pts.push(candidates[selectedIndices[i]!]!);
    }
    return pts;
  };

  let currentFim = evalFisher(getSelectedPoints());
  let currentDOpt = dOptimality(currentFim);

  // Iterative exchange
  const maxIterations = 100; // Prevent infinite loops
  for (let iter = 0; iter < maxIterations; iter++) {
    let improved = false;

    for (let si = 0; si < nSel; si++) {
      const currentIdx = selectedIndices[si]!;

      let bestSwapCandidate = -1;
      let bestSwapDOpt = currentDOpt;

      // Try swapping with each unselected candidate
      for (let ci = 0; ci < nCandidates; ci++) {
        if (inDesign.has(ci)) continue;

        // Temporarily swap
        selectedIndices[si] = ci;
        const trialFim = evalFisher(getSelectedPoints());
        const trialDOpt = dOptimality(trialFim);

        if (trialDOpt > bestSwapDOpt) {
          bestSwapDOpt = trialDOpt;
          bestSwapCandidate = ci;
        }

        // Restore
        selectedIndices[si] = currentIdx;
      }

      // Apply the best swap if it improves D-optimality
      if (bestSwapCandidate >= 0 && bestSwapDOpt > currentDOpt * (1 + 1e-10)) {
        inDesign.delete(currentIdx);
        inDesign.add(bestSwapCandidate);
        selectedIndices[si] = bestSwapCandidate;

        currentFim = evalFisher(getSelectedPoints());
        currentDOpt = bestSwapDOpt;
        improved = true;
      }
    }

    if (!improved) break;
  }

  // Final evaluation
  const finalFim = evalFisher(getSelectedPoints());
  const finalDOpt = dOptimality(finalFim);

  return {
    optimalDesign: [...selectedIndices].sort((a, b) => a - b),
    fisherInfo: finalFim,
    dOptimality: finalDOpt,
  };
}
