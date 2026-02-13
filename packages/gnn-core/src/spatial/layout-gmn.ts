// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-4: Spatial Layout Understanding
// layout-gmn.ts — Graph Matching Network for layout comparison
//
// Implements cross-graph attention to compare two furniture layout graphs.
// Based on Li et al. "Graph Matching Networks for Learning the Similarity
// of Graph Structured Objects" (ICML 2019).
//
// For each node in G1, attends over all nodes in G2 to compute a
// cross-graph difference vector. Updated embeddings are then pooled
// and compared via cosine similarity.
// ---------------------------------------------------------------------------

import type { Graph, GraphMatchResult } from '../types.js';
import { dot } from '../tensor.js';

// ---------------------------------------------------------------------------
// 1. graphMatchingScore — Cross-graph attention and similarity
// ---------------------------------------------------------------------------

/**
 * Compute graph-level similarity between two layout graphs using a
 * Graph Matching Network (GMN).
 *
 * Algorithm:
 * 1. Cross-graph attention from G1 to G2:
 *    For each node i in G1:
 *      Compute attention weights: a_{ij} = softmax_j(h_i^T h_j) for all j in G2.
 *      Compute cross-graph message: mu_i = sum_j a_{ij} * (h_i - h_j).
 *    Updated embedding: h_i' = h_i + mu_i.
 *
 * 2. Cross-graph attention from G2 to G1 (symmetric):
 *    For each node j in G2:
 *      Compute attention weights: a_{ji} = softmax_i(h_j^T h_i) for all i in G1.
 *      Compute cross-graph message: mu_j = sum_i a_{ji} * (h_j - h_i).
 *    Updated embedding: h_j' = h_j + mu_j.
 *
 * 3. Pool each graph: g1 = mean(h_i'), g2 = mean(h_j').
 *
 * 4. Cosine similarity: similarity = dot(g1, g2) / (||g1|| * ||g2||).
 *
 * @param graph1  - First layout graph (CSR).
 * @param X1      - Node embeddings for graph1 (n1 x dim), row-major.
 * @param graph2  - Second layout graph (CSR).
 * @param X2      - Node embeddings for graph2 (n2 x dim), row-major.
 * @param W_match - Matching weight matrix (dim x dim), row-major.
 *                  Applied as linear projection before cross-attention.
 * @param dim     - Feature dimension of each node embedding.
 * @returns GraphMatchResult with similarity score and cross-attention matrices.
 */
export function graphMatchingScore(
  graph1: Graph,
  X1: Float64Array,
  graph2: Graph,
  X2: Float64Array,
  W_match: Float64Array,
  dim: number,
): GraphMatchResult {
  const n1 = graph1.numNodes;
  const n2 = graph2.numNodes;

  // Handle degenerate cases
  if (n1 === 0 || n2 === 0) {
    return {
      similarity: 0,
      crossAttention1: new Float64Array(0),
      crossAttention2: new Float64Array(0),
    };
  }

  // Project embeddings through W_match: Z1 = X1 * W_match, Z2 = X2 * W_match
  const Z1 = projectEmbeddings(X1, W_match, n1, dim);
  const Z2 = projectEmbeddings(X2, W_match, n2, dim);

  // --- Cross-graph attention: G1 nodes attending to G2 nodes ---
  // crossAttention1 is (n1 x n2): attention[i][j] = softmax_j(Z1_i^T Z2_j)
  const crossAttention1 = new Float64Array(n1 * n2);

  // Compute raw scores and softmax row-wise
  for (let i = 0; i < n1; i++) {
    const off1 = i * dim;

    // Find max for numerical stability
    let maxVal = -Infinity;
    for (let j = 0; j < n2; j++) {
      const off2 = j * dim;
      let score = 0;
      for (let d = 0; d < dim; d++) {
        score += Z1[off1 + d]! * Z2[off2 + d]!;
      }
      crossAttention1[i * n2 + j] = score;
      if (score > maxVal) maxVal = score;
    }

    // Softmax
    let sumExp = 0;
    for (let j = 0; j < n2; j++) {
      const expVal = Math.exp(crossAttention1[i * n2 + j]! - maxVal);
      crossAttention1[i * n2 + j] = expVal;
      sumExp += expVal;
    }
    if (sumExp > 0) {
      for (let j = 0; j < n2; j++) {
        crossAttention1[i * n2 + j] = crossAttention1[i * n2 + j]! / sumExp;
      }
    }
  }

  // Compute cross-graph messages for G1: mu_i = sum_j a_{ij} * (h_i - h_j)
  // Updated embeddings: h_i' = h_i + mu_i
  const H1_updated = new Float64Array(n1 * dim);
  for (let i = 0; i < n1; i++) {
    const off1 = i * dim;
    for (let d = 0; d < dim; d++) {
      let mu_d = 0;
      for (let j = 0; j < n2; j++) {
        const off2 = j * dim;
        const a_ij = crossAttention1[i * n2 + j]!;
        mu_d += a_ij * (Z1[off1 + d]! - Z2[off2 + d]!);
      }
      H1_updated[off1 + d] = Z1[off1 + d]! + mu_d;
    }
  }

  // --- Cross-graph attention: G2 nodes attending to G1 nodes ---
  // crossAttention2 is (n2 x n1): attention[j][i] = softmax_i(Z2_j^T Z1_i)
  const crossAttention2 = new Float64Array(n2 * n1);

  for (let j = 0; j < n2; j++) {
    const off2 = j * dim;

    let maxVal = -Infinity;
    for (let i = 0; i < n1; i++) {
      const off1 = i * dim;
      let score = 0;
      for (let d = 0; d < dim; d++) {
        score += Z2[off2 + d]! * Z1[off1 + d]!;
      }
      crossAttention2[j * n1 + i] = score;
      if (score > maxVal) maxVal = score;
    }

    let sumExp = 0;
    for (let i = 0; i < n1; i++) {
      const expVal = Math.exp(crossAttention2[j * n1 + i]! - maxVal);
      crossAttention2[j * n1 + i] = expVal;
      sumExp += expVal;
    }
    if (sumExp > 0) {
      for (let i = 0; i < n1; i++) {
        crossAttention2[j * n1 + i] = crossAttention2[j * n1 + i]! / sumExp;
      }
    }
  }

  // Compute cross-graph messages for G2: mu_j = sum_i a_{ji} * (h_j - h_i)
  const H2_updated = new Float64Array(n2 * dim);
  for (let j = 0; j < n2; j++) {
    const off2 = j * dim;
    for (let d = 0; d < dim; d++) {
      let mu_d = 0;
      for (let i = 0; i < n1; i++) {
        const off1 = i * dim;
        const a_ji = crossAttention2[j * n1 + i]!;
        mu_d += a_ji * (Z2[off2 + d]! - Z1[off1 + d]!);
      }
      H2_updated[off2 + d] = Z2[off2 + d]! + mu_d;
    }
  }

  // --- Graph-level pooling: mean of updated embeddings ---
  const g1 = new Float64Array(dim);
  for (let i = 0; i < n1; i++) {
    const off = i * dim;
    for (let d = 0; d < dim; d++) {
      g1[d] = g1[d]! + H1_updated[off + d]!;
    }
  }
  for (let d = 0; d < dim; d++) {
    g1[d] = g1[d]! / n1;
  }

  const g2 = new Float64Array(dim);
  for (let j = 0; j < n2; j++) {
    const off = j * dim;
    for (let d = 0; d < dim; d++) {
      g2[d] = g2[d]! + H2_updated[off + d]!;
    }
  }
  for (let d = 0; d < dim; d++) {
    g2[d] = g2[d]! / n2;
  }

  // --- Cosine similarity ---
  const dotProduct = dot(g1, g2);
  let norm1 = 0;
  let norm2 = 0;
  for (let d = 0; d < dim; d++) {
    norm1 += g1[d]! * g1[d]!;
    norm2 += g2[d]! * g2[d]!;
  }
  norm1 = Math.sqrt(norm1);
  norm2 = Math.sqrt(norm2);

  const similarity = norm1 > 0 && norm2 > 0
    ? dotProduct / (norm1 * norm2)
    : 0;

  return {
    similarity,
    crossAttention1,
    crossAttention2,
  };
}

// ---------------------------------------------------------------------------
// Internal helper — Project embeddings through a weight matrix
// ---------------------------------------------------------------------------

/**
 * Project embeddings: Z = X * W, where X is (n x dim) and W is (dim x dim).
 * Returns Z of shape (n x dim), row-major.
 */
function projectEmbeddings(
  X: Float64Array,
  W: Float64Array,
  n: number,
  dim: number,
): Float64Array {
  const Z = new Float64Array(n * dim);
  for (let i = 0; i < n; i++) {
    const xOff = i * dim;
    const zOff = i * dim;
    for (let k = 0; k < dim; k++) {
      const x_ik = X[xOff + k]!;
      for (let d = 0; d < dim; d++) {
        Z[zOff + d] = Z[zOff + d]! + x_ik * W[k * dim + d]!;
      }
    }
  }
  return Z;
}
