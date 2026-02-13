// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — PinSage (Ying et al. 2018)
// Random-walk-based importance sampling for scalable graph recommendations.
// ---------------------------------------------------------------------------

import type { PRNG, Graph } from '../types.js';
import { matVecMul } from '../tensor.js';

// ---------------------------------------------------------------------------
// 1. pinSageSample — Random-walk importance sampling
// ---------------------------------------------------------------------------

/**
 * Perform random-walk-based importance sampling for a target node.
 *
 * Algorithm (PinSage):
 * 1. Perform `numWalks` random walks from `node`, each of `walkLen` steps.
 * 2. Count visit frequency for each neighbor encountered during walks.
 * 3. Normalize visit counts to importance scores.
 * 4. Return a map of neighbor -> importance score (all visited neighbors).
 *
 * The importance scores approximate the PPR (Personalized PageRank) vector,
 * which captures multi-hop relevance without expensive matrix operations.
 *
 * @param graph - Input CSR graph.
 * @param node - The target node to sample neighbors for.
 * @param walkLen - Length of each random walk (number of steps).
 * @param numWalks - Number of random walks to perform.
 * @param rng - Deterministic PRNG.
 * @returns Map from neighbor node ID to visit frequency (importance score).
 */
export function pinSageSample(
  graph: Graph,
  node: number,
  walkLen: number,
  numWalks: number,
  rng: PRNG,
): Map<number, number> {
  const visitCounts = new Map<number, number>();

  for (let w = 0; w < numWalks; w++) {
    let current = node;

    for (let step = 0; step < walkLen; step++) {
      const start = graph.rowPtr[current]!;
      const end = graph.rowPtr[current + 1]!;
      const deg = end - start;

      if (deg === 0) break; // Dead end, stop this walk

      // Uniformly sample a neighbor
      const idx = start + Math.floor(rng() * deg);
      current = graph.colIdx[idx]!;

      // Count visits (exclude the source node itself)
      if (current !== node) {
        const prev = visitCounts.get(current) ?? 0;
        visitCounts.set(current, prev + 1);
      }
    }
  }

  // Normalize visit counts to importance scores
  let totalVisits = 0;
  for (const count of visitCounts.values()) {
    totalVisits += count;
  }

  if (totalVisits > 0) {
    const normalized = new Map<number, number>();
    for (const [neighbor, count] of visitCounts) {
      normalized.set(neighbor, count / totalVisits);
    }
    return normalized;
  }

  return visitCounts;
}

// ---------------------------------------------------------------------------
// 2. pinSageConv — Importance-weighted neighbor aggregation
// ---------------------------------------------------------------------------

/**
 * PinSage convolution: importance-weighted aggregation for a single target node.
 *
 * Algorithm:
 * 1. Compute the weighted mean of neighbor features using importance scores:
 *    agg = SUM(importance[u] * x_u) / SUM(importance[u])
 * 2. Apply a learned linear transformation:
 *    h' = W * agg
 *
 * Unlike uniform aggregation (GCN/GraphSAGE), PinSage uses random-walk-derived
 * importance to weight neighbors, capturing multi-hop relevance.
 *
 * @param graph - Input CSR graph (used for feature lookup context).
 * @param X - Node feature matrix, row-major (numNodes x inDim).
 * @param sampledNeighbors - Map from neighbor node ID to importance score
 *                           (output of pinSageSample).
 * @param targetNode - The node to compute the convolution for.
 * @param W - Weight matrix, row-major (outDim x inDim).
 * @param inDim - Input feature dimension.
 * @param outDim - Output feature dimension.
 * @returns Output vector of length outDim for the target node.
 */
export function pinSageConv(
  graph: Graph,
  X: Float64Array,
  sampledNeighbors: Map<number, number>,
  targetNode: number,
  W: Float64Array,
  inDim: number,
  outDim: number,
): Float64Array {
  // Compute importance-weighted mean of neighbor features
  const aggFeatures = new Float64Array(inDim);
  let totalWeight = 0;

  for (const [neighbor, importance] of sampledNeighbors) {
    const nOff = neighbor * inDim;
    for (let d = 0; d < inDim; d++) {
      aggFeatures[d] = aggFeatures[d]! + importance * X[nOff + d]!;
    }
    totalWeight += importance;
  }

  // Normalize by total weight (weighted mean)
  if (totalWeight > 0) {
    for (let d = 0; d < inDim; d++) {
      aggFeatures[d] = aggFeatures[d]! / totalWeight;
    }
  }

  // Apply linear transformation: h' = W * agg
  // W is (outDim x inDim), aggFeatures is (inDim)
  const output = matVecMul(W, aggFeatures, outDim, inDim);

  return output;
}
