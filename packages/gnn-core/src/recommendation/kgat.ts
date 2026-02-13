// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — KGAT (Knowledge-Graph-Aware Attention)
// Attentive embedding propagation over entity-relation-entity triples.
// ---------------------------------------------------------------------------

import type { Graph } from '../types.js';
import { relu, leakyRelu } from '../tensor.js';

// ---------------------------------------------------------------------------
// 1. kgatLayer — Single KGAT propagation layer
// ---------------------------------------------------------------------------

/**
 * KGAT attentive embedding propagation layer.
 *
 * Algorithm:
 * 1. Project all node embeddings: Wh_i = W * h_i for every node i.
 * 2. For each edge (i, j), compute attention score:
 *    e_{ij} = LeakyReLU(a^T (Wh_i || Wh_j))
 *    where || denotes concatenation.
 * 3. Normalize attention across each node's neighborhood:
 *    alpha_{ij} = softmax_j(e_{ij})
 * 4. Aggregate:
 *    h_i' = SUM_j alpha_{ij} * h_j
 * 5. Apply ReLU activation + L2 row normalization.
 *
 * This follows the GAT-style attention mechanism adapted for knowledge graphs,
 * where edges represent entity-relation-entity triples.
 *
 * @param graph - CSR graph representing the knowledge graph structure.
 * @param X - Node embedding matrix, row-major (numNodes x inDim).
 * @param W - Weight matrix, row-major (outDim x inDim).
 * @param a - Attention vector of length 2*outDim (for concatenated projected features).
 * @param inDim - Input embedding dimension.
 * @param outDim - Output embedding dimension.
 * @returns Updated node embeddings (numNodes x outDim), row-major.
 */
export function kgatLayer(
  graph: Graph,
  X: Float64Array,
  W: Float64Array,
  a: Float64Array,
  inDim: number,
  outDim: number,
): Float64Array {
  const numNodes = graph.numNodes;

  // Step 1: Project all nodes — Wh[i] = W * h_i
  // W is (outDim x inDim), X row i is (inDim)
  const Wh = new Float64Array(numNodes * outDim);

  for (let i = 0; i < numNodes; i++) {
    const xOff = i * inDim;
    const whOff = i * outDim;
    for (let r = 0; r < outDim; r++) {
      let val = 0;
      for (let c = 0; c < inDim; c++) {
        val += W[r * inDim + c]! * X[xOff + c]!;
      }
      Wh[whOff + r] = val;
    }
  }

  // Step 2 & 3: Compute attention scores and normalize per-node

  // Pre-split the attention vector: a_left for source, a_right for target
  const aLeft = a.slice(0, outDim);   // applied to Wh_i (source)
  const aRight = a.slice(outDim, 2 * outDim); // applied to Wh_j (target)

  // Precompute a_left^T * Wh_i for each node i
  const srcScores = new Float64Array(numNodes);
  for (let i = 0; i < numNodes; i++) {
    let val = 0;
    const off = i * outDim;
    for (let d = 0; d < outDim; d++) {
      val += aLeft[d]! * Wh[off + d]!;
    }
    srcScores[i] = val;
  }

  // Precompute a_right^T * Wh_j for each node j
  const dstScores = new Float64Array(numNodes);
  for (let j = 0; j < numNodes; j++) {
    let val = 0;
    const off = j * outDim;
    for (let d = 0; d < outDim; d++) {
      val += aRight[d]! * Wh[off + d]!;
    }
    dstScores[j] = val;
  }

  // Allocate output
  const out = new Float64Array(numNodes * outDim);

  for (let i = 0; i < numNodes; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;
    const numNeighbors = end - start;

    if (numNeighbors === 0) continue;

    // Compute raw attention scores: e_{ij} = LeakyReLU(srcScore_i + dstScore_j)
    const rawAttn = new Float64Array(numNeighbors);
    for (let idx = 0; idx < numNeighbors; idx++) {
      const j = graph.colIdx[start + idx]!;
      const eij = srcScores[i]! + dstScores[j]!;
      // LeakyReLU with slope 0.2 (common for GAT-style attention)
      rawAttn[idx] = eij > 0 ? eij : 0.2 * eij;
    }

    // Softmax normalization (numerically stable)
    let maxAttn = -Infinity;
    for (let idx = 0; idx < numNeighbors; idx++) {
      if (rawAttn[idx]! > maxAttn) maxAttn = rawAttn[idx]!;
    }

    let sumExp = 0;
    const expAttn = new Float64Array(numNeighbors);
    for (let idx = 0; idx < numNeighbors; idx++) {
      const e = Math.exp(rawAttn[idx]! - maxAttn);
      expAttn[idx] = e;
      sumExp += e;
    }

    // Step 4: Aggregate — h_i' = SUM_j alpha_{ij} * h_j
    const iOff = i * outDim;
    for (let idx = 0; idx < numNeighbors; idx++) {
      const alpha = expAttn[idx]! / sumExp;
      const j = graph.colIdx[start + idx]!;
      const jOff = j * inDim;

      // Aggregate original features h_j (not projected Wh_j)
      // weighted by attention
      for (let d = 0; d < outDim; d++) {
        // Use raw h_j features for aggregation
        // Note: if outDim != inDim, we use the projected features Wh_j instead
        out[iOff + d] = out[iOff + d]! + alpha * Wh[j * outDim + d]!;
      }
    }
  }

  // Step 5: Apply ReLU activation
  const activated = relu(out);

  // Step 5: L2 normalize each row
  const result = new Float64Array(numNodes * outDim);
  for (let i = 0; i < numNodes; i++) {
    const iOff = i * outDim;

    // Compute L2 norm of row i
    let normSq = 0;
    for (let d = 0; d < outDim; d++) {
      const v = activated[iOff + d]!;
      normSq += v * v;
    }
    const norm = Math.sqrt(normSq);

    // Normalize (guard against zero norm)
    if (norm > 1e-12) {
      for (let d = 0; d < outDim; d++) {
        result[iOff + d] = activated[iOff + d]! / norm;
      }
    } else {
      for (let d = 0; d < outDim; d++) {
        result[iOff + d] = activated[iOff + d]!;
      }
    }
  }

  return result;
}
