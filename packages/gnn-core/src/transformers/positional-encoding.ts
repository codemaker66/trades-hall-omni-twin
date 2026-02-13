// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-7: Positional Encodings for Graph Transformers
// Laplacian PE (Dwivedi & Bresson 2021) and Random Walk PE (Li et al. 2020)
// ---------------------------------------------------------------------------

import type { Graph, PositionalEncodingResult } from '../types.js';
import { graphLaplacian } from '../graph.js';

// ---------------------------------------------------------------------------
// 1. laplacianPE — Laplacian Positional Encoding
// ---------------------------------------------------------------------------

/**
 * Compute Laplacian Positional Encoding: the k smallest non-trivial
 * eigenvectors of the graph Laplacian L = D - A.
 *
 * Algorithm:
 * 1. Compute the dense graph Laplacian L.
 * 2. Estimate the maximum eigenvalue via Gershgorin bound.
 * 3. Form shifted matrix M = lambda_max * I - L, so that the smallest
 *    eigenvectors of L become the largest eigenvectors of M.
 * 4. Use power iteration with deflation to extract the top k+1 eigenvectors
 *    of M (the first is the trivial constant eigenvector, which is skipped).
 * 5. Return the k non-trivial eigenvectors as positional encodings.
 *
 * Note: Eigenvectors have sign ambiguity. In practice, a sign-invariant
 * network (e.g., random sign flip during training) is used downstream.
 *
 * @param graph          - Input CSR graph.
 * @param k              - Number of positional encoding dimensions (eigenvectors).
 * @param maxIterations  - Power iteration steps (default 300).
 * @returns PositionalEncodingResult with pe (numNodes x k) and peDim = k.
 */
export function laplacianPE(
  graph: Graph,
  k: number,
  maxIterations = 300,
): PositionalEncodingResult {
  const n = graph.numNodes;

  // Edge case: fewer nodes than requested dimensions
  const effectiveK = Math.min(k, Math.max(n - 1, 0));

  if (n === 0 || effectiveK === 0) {
    return { pe: new Float64Array(0), peDim: effectiveK };
  }

  // Step 1: Dense Laplacian L = D - A
  const L = graphLaplacian(graph);

  // Step 2: Gershgorin bound for max eigenvalue
  let maxLambda = 0;
  for (let i = 0; i < n; i++) {
    let rowSum = 0;
    for (let j = 0; j < n; j++) {
      rowSum += Math.abs(L[i * n + j]!);
    }
    if (rowSum > maxLambda) maxLambda = rowSum;
  }

  // Ensure maxLambda > 0 for the shift to be meaningful
  if (maxLambda < 1e-12) {
    // Graph has no edges; all eigenvalues are 0
    return { pe: new Float64Array(n * effectiveK), peDim: effectiveK };
  }

  // Step 3: Shifted matrix M = maxLambda * I - L
  const M = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      M[i * n + j] = -L[i * n + j]!;
    }
    M[i * n + i] = M[i * n + i]! + maxLambda;
  }

  // Step 4: Power iteration with deflation
  // We need to find the top (effectiveK + 1) eigenvectors of M
  // (the first corresponds to the trivial constant eigvec of L, eigenvalue 0)
  const numToFind = effectiveK + 1;
  const eigenvectors: Float64Array[] = [];

  const tmp = new Float64Array(n);

  for (let ev = 0; ev < numToFind; ev++) {
    // Initialize random vector via simple deterministic seed
    const v = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      // Simple deterministic initialization (different per eigenvector)
      v[i] = Math.sin((ev + 1) * (i + 1) * 0.7853981633974483) + 0.1 * (i - n / 2);
    }

    // Deflate against all previously found eigenvectors
    const deflateAll = (vec: Float64Array): void => {
      for (const u of eigenvectors) {
        let dotProduct = 0;
        for (let i = 0; i < n; i++) {
          dotProduct += vec[i]! * u[i]!;
        }
        for (let i = 0; i < n; i++) {
          vec[i] = vec[i]! - dotProduct * u[i]!;
        }
      }
    };

    const normalizeVec = (vec: Float64Array): number => {
      let norm = 0;
      for (let i = 0; i < n; i++) {
        norm += vec[i]! * vec[i]!;
      }
      norm = Math.sqrt(norm);
      if (norm > 1e-15) {
        for (let i = 0; i < n; i++) {
          vec[i] = vec[i]! / norm;
        }
      }
      return norm;
    };

    deflateAll(v);
    normalizeVec(v);

    // Power iteration
    for (let iter = 0; iter < maxIterations; iter++) {
      // tmp = M * v
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) {
          s += M[i * n + j]! * v[j]!;
        }
        tmp[i] = s;
      }

      // Deflate against previously found eigenvectors
      deflateAll(tmp);

      // Normalize
      normalizeVec(tmp);

      // Copy tmp -> v
      v.set(tmp);
    }

    eigenvectors.push(new Float64Array(v));
  }

  // Step 5: Skip the first eigenvector (trivial constant eigenvector of L)
  // and pack the remaining k eigenvectors into the PE matrix.
  const pe = new Float64Array(n * effectiveK);
  for (let d = 0; d < effectiveK; d++) {
    const vec = eigenvectors[d + 1]!; // skip index 0 (trivial)
    for (let i = 0; i < n; i++) {
      pe[i * effectiveK + d] = vec[i]!;
    }
  }

  return { pe, peDim: effectiveK };
}

// ---------------------------------------------------------------------------
// 2. randomWalkPE — Random Walk Positional Encoding
// ---------------------------------------------------------------------------

/**
 * Compute Random Walk Positional Encoding: the diagonal of T^k for k=1..K,
 * where T = D^{-1} A is the row-stochastic transition matrix.
 *
 * PE(v) = [p_1(v), p_2(v), ..., p_K(v)] where p_k(v) = (T^k)[v, v]
 * is the probability of returning to node v after exactly k random walk steps.
 *
 * Algorithm:
 * 1. Build the dense transition matrix T = D^{-1} A.
 * 2. Iteratively compute T_power = T^k by matrix multiplication.
 * 3. At each step k, read the diagonal (T^k)[v, v] for each node v.
 *
 * Complexity: O(walkLength * N^2) for the dense matrix powers.
 * For large graphs, consider sparse matrix power or sampling-based approaches.
 *
 * @param graph      - Input CSR graph.
 * @param walkLength - Maximum walk length K (number of PE dimensions).
 * @returns PositionalEncodingResult with pe (numNodes x walkLength) and peDim = walkLength.
 */
export function randomWalkPE(
  graph: Graph,
  walkLength: number,
): PositionalEncodingResult {
  const n = graph.numNodes;

  if (n === 0 || walkLength === 0) {
    return { pe: new Float64Array(0), peDim: walkLength };
  }

  // Step 1: Build dense transition matrix T = D^{-1} A (row-stochastic)
  const T = new Float64Array(n * n);

  for (let i = 0; i < n; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;

    // Compute degree (sum of edge weights)
    let deg = 0;
    for (let e = start; e < end; e++) {
      deg += graph.edgeWeights ? graph.edgeWeights[e]! : 1.0;
    }

    if (deg > 0) {
      for (let e = start; e < end; e++) {
        const j = graph.colIdx[e]!;
        const w = graph.edgeWeights ? graph.edgeWeights[e]! : 1.0;
        T[i * n + j] = T[i * n + j]! + w / deg;
      }
    }
  }

  // Step 2: Iteratively compute T^k and extract diagonals.
  // T_power starts as T^1 = T itself.
  const pe = new Float64Array(n * walkLength);

  // Current power of T (dense n x n)
  let T_power = new Float64Array(T);

  for (let k = 0; k < walkLength; k++) {
    if (k === 0) {
      // T^1 = T, already in T_power
    } else {
      // T_power = T_power * T (dense matrix multiply)
      const next = new Float64Array(n * n);
      for (let i = 0; i < n; i++) {
        for (let p = 0; p < n; p++) {
          const val = T_power[i * n + p]!;
          if (val === 0) continue; // sparse skip
          for (let j = 0; j < n; j++) {
            next[i * n + j] = next[i * n + j]! + val * T[p * n + j]!;
          }
        }
      }
      T_power = next;
    }

    // Extract diagonal: p_k(v) = T_power[v, v]
    for (let v = 0; v < n; v++) {
      pe[v * walkLength + k] = T_power[v * n + v]!;
    }
  }

  return { pe, peDim: walkLength };
}
