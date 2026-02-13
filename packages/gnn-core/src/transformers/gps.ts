// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-7: GPS Framework (General, Powerful, Scalable)
// Rampasek et al. 2022 — h' = MLP(LN(h + MPNN(h,G) + GlobalAttn(h)))
// ---------------------------------------------------------------------------

import type { Graph, GPSConfig, GPSWeights, GCNWeights, PRNG } from '../types.js';
import {
  matMul,
  layerNorm,
  relu,
  add,
  softmax,
} from '../tensor.js';
import { addSelfLoops, normalizeAdjacency } from '../graph.js';

// ---------------------------------------------------------------------------
// 1. globalSelfAttention — Multi-head self-attention O(N^2)
// ---------------------------------------------------------------------------

/**
 * Standard multi-head self-attention over all nodes.
 *
 * Algorithm:
 * 1. Project input to Q, K, V using weight matrices.
 * 2. Split Q, K, V into `heads` attention heads (headDim = dim / heads).
 * 3. For each head: compute scaled dot-product attention scores
 *    attn = softmax(Q_h K_h^T / sqrt(headDim)), output = attn * V_h.
 * 4. Concatenate all head outputs and project with W_O.
 *
 * @param X      - Node features, row-major (numNodes x dim).
 * @param W_Q    - Query projection (dim x dim).
 * @param W_K    - Key projection (dim x dim).
 * @param W_V    - Value projection (dim x dim).
 * @param W_O    - Output projection (dim x dim).
 * @param numNodes - Number of nodes.
 * @param dim    - Feature dimension.
 * @param heads  - Number of attention heads.
 * @returns Updated features (numNodes x dim).
 */
export function globalSelfAttention(
  X: Float64Array,
  W_Q: Float64Array,
  W_K: Float64Array,
  W_V: Float64Array,
  W_O: Float64Array,
  numNodes: number,
  dim: number,
  heads: number,
): Float64Array {
  const headDim = dim / heads;
  const scale = 1.0 / Math.sqrt(headDim);

  // Project: Q = X * W_Q, K = X * W_K, V = X * W_V  (all numNodes x dim)
  const Q = matMul(X, W_Q, numNodes, dim, dim);
  const K = matMul(X, W_K, numNodes, dim, dim);
  const V = matMul(X, W_V, numNodes, dim, dim);

  // Output buffer for concatenated head results (numNodes x dim)
  const multiHeadOut = new Float64Array(numNodes * dim);

  // Per-head attention
  for (let h = 0; h < heads; h++) {
    const hOffset = h * headDim;

    // Compute attention scores: scores[i][j] = Q_h[i] . K_h[j] / sqrt(headDim)
    const scores = new Float64Array(numNodes * numNodes);
    for (let i = 0; i < numNodes; i++) {
      for (let j = 0; j < numNodes; j++) {
        let dot = 0;
        for (let d = 0; d < headDim; d++) {
          dot += Q[i * dim + hOffset + d]! * K[j * dim + hOffset + d]!;
        }
        scores[i * numNodes + j] = dot * scale;
      }
    }

    // Softmax over keys for each query (row-wise softmax)
    const attn = softmax(scores, numNodes);

    // Weighted sum of values: out[i] = sum_j attn[i,j] * V_h[j]
    for (let i = 0; i < numNodes; i++) {
      for (let j = 0; j < numNodes; j++) {
        const a = attn[i * numNodes + j]!;
        for (let d = 0; d < headDim; d++) {
          multiHeadOut[i * dim + hOffset + d] = multiHeadOut[i * dim + hOffset + d]! + a * V[j * dim + hOffset + d]!;
        }
      }
    }
  }

  // Final projection: out = multiHeadOut * W_O (numNodes x dim)
  return matMul(multiHeadOut, W_O, numNodes, dim, dim);
}

// ---------------------------------------------------------------------------
// 2. gcnConvolution — Sparse GCN message passing (MPNN branch)
// ---------------------------------------------------------------------------

/**
 * GCN-style sparse convolution: h' = A_hat * X * W.
 *
 * Uses symmetric normalization D^{-1/2} A D^{-1/2} with self-loops added.
 *
 * @param graph   - Input CSR graph.
 * @param X       - Node features (numNodes x inDim), row-major.
 * @param W       - Weight matrix (inDim x outDim), row-major.
 * @param bias    - Optional bias (outDim).
 * @param inDim   - Input feature dimension.
 * @param outDim  - Output feature dimension.
 * @returns Updated features (numNodes x outDim).
 */
function gcnConvolution(
  graph: Graph,
  X: Float64Array,
  W: Float64Array,
  bias: Float64Array | undefined,
  inDim: number,
  outDim: number,
): Float64Array {
  const n = graph.numNodes;

  // Step 1: Transform features: H = X * W (numNodes x outDim)
  const H = matMul(X, W, n, inDim, outDim);

  // Step 2: Add self-loops and normalize
  const graphSL = addSelfLoops(graph);
  const normGraph = normalizeAdjacency(graphSL, 'symmetric');

  // Step 3: Sparse aggregation: out[i] = sum_j A_norm[i,j] * H[j]
  const out = new Float64Array(n * outDim);

  for (let i = 0; i < n; i++) {
    const start = normGraph.rowPtr[i]!;
    const end = normGraph.rowPtr[i + 1]!;

    for (let e = start; e < end; e++) {
      const j = normGraph.colIdx[e]!;
      const w = normGraph.edgeWeights ? normGraph.edgeWeights[e]! : 1.0;

      for (let d = 0; d < outDim; d++) {
        out[i * outDim + d] = out[i * outDim + d]! + w * H[j * outDim + d]!;
      }
    }

    // Add bias
    if (bias) {
      for (let d = 0; d < outDim; d++) {
        out[i * outDim + d] = out[i * outDim + d]! + bias[d]!;
      }
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// 3. gpsLayer — Full GPS layer
// ---------------------------------------------------------------------------

/**
 * GPS layer: h' = FFN(LN2( LN1(h + MPNN(h,G) + GlobalAttn(h)) )) with
 * residual around FFN and a second LayerNorm.
 *
 * Detailed computation:
 * 1. MPNN branch: GCN convolution with sparse aggregation.
 * 2. GlobalAttn branch: full multi-head self-attention.
 * 3. Residual sum: r = h + MPNN(h) + GlobalAttn(h).
 * 4. First LayerNorm: x = LN1(r).
 * 5. FFN: y = W2 * relu(W1 * x + B1) + B2.
 * 6. Residual + second LayerNorm: out = LN2(x + y).
 *
 * @param graph   - Input CSR graph.
 * @param X       - Node features (numNodes x dim), row-major.
 * @param weights - GPS layer weights.
 * @param config  - GPS configuration.
 * @param _rng    - PRNG (reserved for dropout; currently unused).
 * @returns Updated features (numNodes x dim).
 */
export function gpsLayer(
  graph: Graph,
  X: Float64Array,
  weights: GPSWeights,
  config: GPSConfig,
  _rng: PRNG,
): Float64Array {
  const { dim, heads, ffnDim } = config;
  const n = graph.numNodes;

  // ---- MPNN branch (GCN convolution) ----
  const gcnW = weights.mpnnWeights as GCNWeights;
  const mpnnOut = gcnConvolution(graph, X, gcnW.W, gcnW.bias, dim, dim);

  // ---- Global self-attention branch ----
  const attnOut = globalSelfAttention(
    X,
    weights.attnW_Q,
    weights.attnW_K,
    weights.attnW_V,
    weights.attnW_O,
    n,
    dim,
    heads,
  );

  // ---- Residual: h + MPNN(h) + GlobalAttn(h) ----
  const residual = add(X, add(mpnnOut, attnOut));

  // ---- LayerNorm 1 ----
  const normed1 = layerNorm(residual, dim, weights.norm1Gamma, weights.norm1Beta);

  // ---- FFN: W2 * relu(W1 * x + B1) + B2 ----
  // Step A: linear1 = X * W1 + B1  (numNodes x ffnDim)
  const linear1 = matMul(normed1, weights.ffnW1, n, dim, ffnDim);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < ffnDim; d++) {
      linear1[i * ffnDim + d] = linear1[i * ffnDim + d]! + weights.ffnB1[d]!;
    }
  }

  // Step B: activation
  const activated = relu(linear1);

  // Step C: linear2 = activated * W2 + B2  (numNodes x dim)
  const linear2 = matMul(activated, weights.ffnW2, n, ffnDim, dim);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < dim; d++) {
      linear2[i * dim + d] = linear2[i * dim + d]! + weights.ffnB2[d]!;
    }
  }

  // ---- Residual around FFN + LayerNorm 2 ----
  const ffnResidual = add(normed1, linear2);
  const normed2 = layerNorm(ffnResidual, dim, weights.norm2Gamma, weights.norm2Beta);

  return normed2;
}
