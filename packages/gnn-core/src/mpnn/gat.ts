// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GAT / GATv2 (Graph Attention Networks)
// GAT:   Velickovic et al. 2018 — static attention
// GATv2: Brody et al. 2022 — dynamic (query-dependent) attention
//
// GAT:   alpha_{ij} = softmax_j( LeakyReLU(a_src^T Wh_i + a_dst^T Wh_j) )
// GATv2: alpha_{ij} = softmax_j( a^T LeakyReLU(W_src h_i + W_dst h_j) )
// ---------------------------------------------------------------------------

import type { Graph, GATConfig, GATWeights, PRNG } from '../types.js';
import { leakyRelu } from '../tensor.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Compute Wh = X * W for the standard GAT (v1).
 *
 * X is (numNodes x inDim), W is (inDim x (outDim * heads)).
 * Result Wh is (numNodes x (outDim * heads)), row-major.
 */
function projectFeatures(
  X: Float64Array,
  W: Float64Array,
  numNodes: number,
  inDim: number,
  totalOutDim: number,
): Float64Array {
  const Wh = new Float64Array(numNodes * totalOutDim);
  for (let i = 0; i < numNodes; i++) {
    const xOff = i * inDim;
    const whOff = i * totalOutDim;
    for (let k = 0; k < inDim; k++) {
      const xik = X[xOff + k]!;
      for (let o = 0; o < totalOutDim; o++) {
        Wh[whOff + o]! += xik * W[k * totalOutDim + o]!;
      }
    }
  }
  return Wh;
}

/**
 * Per-node softmax over incoming edge attention scores.
 *
 * For each source node i, softmax is computed over all edges originating from i
 * (i.e., over all j such that (i,j) is an edge), independently per head.
 *
 * @param scores   - Raw attention scores (numEdges x heads), row-major.
 * @param graph    - CSR graph (rowPtr defines per-source grouping).
 * @param numEdges - Number of edges.
 * @param heads    - Number of attention heads.
 */
function edgeSoftmax(
  scores: Float64Array,
  graph: Graph,
  numEdges: number,
  heads: number,
): Float64Array {
  const attn = new Float64Array(numEdges * heads);
  const n = graph.numNodes;

  for (let i = 0; i < n; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;
    if (start === end) continue;

    // Per-head softmax
    for (let h = 0; h < heads; h++) {
      // Find max for numerical stability
      let maxVal = -Infinity;
      for (let e = start; e < end; e++) {
        const s = scores[e * heads + h]!;
        if (s > maxVal) maxVal = s;
      }

      // Compute exp and sum
      let sumExp = 0;
      for (let e = start; e < end; e++) {
        const expVal = Math.exp(scores[e * heads + h]! - maxVal);
        attn[e * heads + h] = expVal;
        sumExp += expVal;
      }

      // Normalize
      if (sumExp > 0) {
        for (let e = start; e < end; e++) {
          attn[e * heads + h] = attn[e * heads + h]! / sumExp;
        }
      }
    }
  }

  return attn;
}

/**
 * Apply dropout to attention weights. Zeros out each value with probability p,
 * and scales remaining values by 1/(1-p) to maintain expected value.
 */
function applyDropout(
  attn: Float64Array,
  p: number,
  rng: PRNG,
): Float64Array {
  if (p <= 0) return attn;
  const out = new Float64Array(attn.length);
  const scale = 1.0 / (1.0 - p);
  for (let i = 0; i < attn.length; i++) {
    out[i] = rng() >= p ? attn[i]! * scale : 0;
  }
  return out;
}

// ---------------------------------------------------------------------------
// gatLayer — GAT v1 (static attention)
// ---------------------------------------------------------------------------

/**
 * Single GAT (v1) layer forward pass.
 *
 * Algorithm:
 * 1. Project node features: Wh = X * W  (numNodes x (outDim * heads))
 * 2. Compute raw attention scores per edge per head:
 *    e_{ij}^k = LeakyReLU( a_src^k . Wh_i^k + a_dst^k . Wh_j^k )
 *    where ^k denotes the k-th head's slice of outDim.
 * 3. Softmax over neighbors per source node per head.
 * 4. Optionally apply dropout to attention coefficients.
 * 5. Weighted aggregation: h_i^k = sum_j alpha_{ij}^k * Wh_j^k
 * 6. If concat: output = [h_i^1 || h_i^2 || ... || h_i^K]  (outDim * heads)
 *    Else:      output = mean(h_i^1, ..., h_i^K)            (outDim)
 *
 * @param graph   - CSR Graph (edges represent attention connections).
 * @param X       - Node features (numNodes x inDim), row-major.
 * @param weights - { W, a_src, a_dst } — see GATWeights type.
 * @param config  - { inDim, outDim, heads, dropout, negativeSlope, concat, v2 }.
 * @param rng     - PRNG for dropout randomness.
 * @returns Float64Array of shape:
 *   - concat=true:  (numNodes x outDim*heads)
 *   - concat=false: (numNodes x outDim)
 */
export function gatLayer(
  graph: Graph,
  X: Float64Array,
  weights: GATWeights,
  config: GATConfig,
  rng: PRNG,
): Float64Array {
  const { inDim, outDim, heads, dropout, negativeSlope, concat } = config;
  const n = graph.numNodes;
  const totalOutDim = outDim * heads;

  // Step 1: Project features → Wh (numNodes x totalOutDim)
  const Wh = projectFeatures(X, weights.W, n, inDim, totalOutDim);

  // Step 2: Compute raw attention scores per edge per head
  // e_{ij}^k = LeakyReLU(a_src^k . Wh_i^k + a_dst^k . Wh_j^k)
  const numEdges = graph.numEdges;
  const rawScores = new Float64Array(numEdges * heads);

  // Precompute per-node attention scores: srcScore_i^k = a_src^k . Wh_i^k
  // and dstScore_j^k = a_dst^k . Wh_j^k
  const srcScores = new Float64Array(n * heads);
  const dstScores = new Float64Array(n * heads);

  for (let i = 0; i < n; i++) {
    const whOff = i * totalOutDim;
    for (let h = 0; h < heads; h++) {
      let srcS = 0;
      let dstS = 0;
      const headOff = h * outDim;
      for (let d = 0; d < outDim; d++) {
        srcS += weights.a_src[headOff + d]! * Wh[whOff + headOff + d]!;
        dstS += weights.a_dst[headOff + d]! * Wh[whOff + headOff + d]!;
      }
      srcScores[i * heads + h] = srcS;
      dstScores[i * heads + h] = dstS;
    }
  }

  // For each edge (i, j): raw_score = LeakyReLU(srcScore_i + dstScore_j)
  for (let i = 0; i < n; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;
    for (let e = start; e < end; e++) {
      const j = graph.colIdx[e]!;
      for (let h = 0; h < heads; h++) {
        const raw = srcScores[i * heads + h]! + dstScores[j * heads + h]!;
        // LeakyReLU
        rawScores[e * heads + h] = raw > 0 ? raw : negativeSlope * raw;
      }
    }
  }

  // Step 3: Softmax per source node per head
  let attn = edgeSoftmax(rawScores, graph, numEdges, heads);

  // Step 4: Dropout on attention weights
  if (dropout > 0) {
    attn = applyDropout(attn, dropout, rng);
  }

  // Step 5: Weighted aggregation
  // h_i^k = sum_j alpha_{ij}^k * Wh_j^k
  const finalOutDim = concat ? totalOutDim : outDim;
  const output = new Float64Array(n * finalOutDim);

  if (concat) {
    // Concat heads: output is (n x outDim*heads)
    for (let i = 0; i < n; i++) {
      const start = graph.rowPtr[i]!;
      const end = graph.rowPtr[i + 1]!;
      const outOff = i * totalOutDim;

      for (let e = start; e < end; e++) {
        const j = graph.colIdx[e]!;
        const whOff = j * totalOutDim;
        for (let h = 0; h < heads; h++) {
          const alpha = attn[e * heads + h]!;
          const headOff = h * outDim;
          for (let d = 0; d < outDim; d++) {
            output[outOff + headOff + d]! += alpha * Wh[whOff + headOff + d]!;
          }
        }
      }
    }
  } else {
    // Mean over heads: output is (n x outDim)
    for (let i = 0; i < n; i++) {
      const start = graph.rowPtr[i]!;
      const end = graph.rowPtr[i + 1]!;
      const outOff = i * outDim;

      for (let e = start; e < end; e++) {
        const j = graph.colIdx[e]!;
        const whOff = j * totalOutDim;
        for (let h = 0; h < heads; h++) {
          const alpha = attn[e * heads + h]!;
          const headOff = h * outDim;
          for (let d = 0; d < outDim; d++) {
            output[outOff + d]! += alpha * Wh[whOff + headOff + d]!;
          }
        }
      }

      // Divide by number of heads to get mean
      for (let d = 0; d < outDim; d++) {
        output[outOff + d] = output[outOff + d]! / heads;
      }
    }
  }

  return output;
}

// ---------------------------------------------------------------------------
// gatv2Layer — GATv2 (dynamic attention)
// ---------------------------------------------------------------------------

/**
 * Single GATv2 layer forward pass.
 *
 * Key difference from GATv1: the attention mechanism is "dynamic" — the
 * LeakyReLU is applied AFTER combining the source and destination projections,
 * which allows the attention ranking to depend on the query node.
 *
 * Algorithm:
 * 1. Compute per-node projections:
 *    src_i = X_i * W_src  (numNodes x (outDim * heads))
 *    dst_j = X_j * W_dst  (numNodes x (outDim * heads))
 * 2. For each edge (i, j), per head k:
 *    e_{ij}^k = a^k . LeakyReLU(src_i^k + dst_j^k)
 * 3. Softmax, dropout, weighted aggregation (same as GATv1).
 *
 * @param graph   - CSR Graph.
 * @param X       - Node features (numNodes x inDim), row-major.
 * @param weights - { W_src, W_dst, a } — GATv2 weight tensors.
 * @param config  - Same config as GAT.
 * @param rng     - PRNG for dropout.
 * @returns Float64Array.
 */
export function gatv2Layer(
  graph: Graph,
  X: Float64Array,
  weights: GATWeights,
  config: GATConfig,
  rng: PRNG,
): Float64Array {
  const { inDim, outDim, heads, dropout, negativeSlope, concat } = config;
  const n = graph.numNodes;
  const totalOutDim = outDim * heads;

  // Step 1: Project features
  // W_src and W_dst are (inDim x totalOutDim)
  const W_src = weights.W_src!;
  const W_dst = weights.W_dst!;
  const a = weights.a!;

  const srcProj = projectFeatures(X, W_src, n, inDim, totalOutDim);
  const dstProj = projectFeatures(X, W_dst, n, inDim, totalOutDim);

  // Step 2: Compute attention scores per edge per head
  const numEdges = graph.numEdges;
  const rawScores = new Float64Array(numEdges * heads);

  for (let i = 0; i < n; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;
    const srcOff = i * totalOutDim;

    for (let e = start; e < end; e++) {
      const j = graph.colIdx[e]!;
      const dstOff = j * totalOutDim;

      for (let h = 0; h < heads; h++) {
        const headOff = h * outDim;
        let score = 0;
        for (let d = 0; d < outDim; d++) {
          // LeakyReLU(src_i^k + dst_j^k)
          const combined = srcProj[srcOff + headOff + d]! + dstProj[dstOff + headOff + d]!;
          const activated = combined > 0 ? combined : negativeSlope * combined;
          // Dot with attention vector a^k
          score += a[headOff + d]! * activated;
        }
        rawScores[e * heads + h] = score;
      }
    }
  }

  // Step 3: Softmax per source node per head
  let attn = edgeSoftmax(rawScores, graph, numEdges, heads);

  // Step 4: Dropout
  if (dropout > 0) {
    attn = applyDropout(attn, dropout, rng);
  }

  // Step 5: Weighted aggregation using srcProj + dstProj combined values
  // In GATv2, the value is typically Wh_j (same as GATv1), so we use
  // the destination projection dstProj for the value, or equivalently W * X.
  // Standard GATv2 uses W_src * h_j as the value to aggregate.
  // We'll use (srcProj of j) + (dstProj of j) as combined message, but
  // more commonly just W * h_j. We project using W (shared) if available,
  // otherwise use dstProj as the value features.
  // Following the original GATv2 paper: value = W_src * h_j (the left branch).
  const finalOutDim = concat ? totalOutDim : outDim;
  const output = new Float64Array(n * finalOutDim);

  if (concat) {
    for (let i = 0; i < n; i++) {
      const start = graph.rowPtr[i]!;
      const end = graph.rowPtr[i + 1]!;
      const outOff = i * totalOutDim;

      for (let e = start; e < end; e++) {
        const j = graph.colIdx[e]!;
        const valOff = j * totalOutDim;
        for (let h = 0; h < heads; h++) {
          const alpha = attn[e * heads + h]!;
          const headOff = h * outDim;
          for (let d = 0; d < outDim; d++) {
            output[outOff + headOff + d]! += alpha * srcProj[valOff + headOff + d]!;
          }
        }
      }
    }
  } else {
    // Mean over heads
    for (let i = 0; i < n; i++) {
      const start = graph.rowPtr[i]!;
      const end = graph.rowPtr[i + 1]!;
      const outOff = i * outDim;

      for (let e = start; e < end; e++) {
        const j = graph.colIdx[e]!;
        const valOff = j * totalOutDim;
        for (let h = 0; h < heads; h++) {
          const alpha = attn[e * heads + h]!;
          const headOff = h * outDim;
          for (let d = 0; d < outDim; d++) {
            output[outOff + d]! += alpha * srcProj[valOff + headOff + d]!;
          }
        }
      }

      for (let d = 0; d < outDim; d++) {
        output[outOff + d] = output[outOff + d]! / heads;
      }
    }
  }

  return output;
}

// ---------------------------------------------------------------------------
// gatForward — Multi-layer GAT/GATv2 forward pass
// ---------------------------------------------------------------------------

/**
 * Multi-layer GAT forward pass.
 *
 * Sequentially applies each GAT layer. If config.v2 is true, uses GATv2
 * dynamic attention; otherwise uses standard GATv1 static attention.
 *
 * @param graph  - CSR Graph.
 * @param X      - Initial node features.
 * @param layers - Array of { weights, config } for each layer.
 * @param rng    - PRNG for dropout.
 * @returns Final node embeddings.
 */
export function gatForward(
  graph: Graph,
  X: Float64Array,
  layers: { weights: GATWeights; config: GATConfig }[],
  rng: PRNG,
): Float64Array {
  let H = X;
  for (let l = 0; l < layers.length; l++) {
    const layer = layers[l]!;
    if (layer.config.v2) {
      H = gatv2Layer(graph, H, layer.weights, layer.config, rng);
    } else {
      H = gatLayer(graph, H, layer.weights, layer.config, rng);
    }
  }
  return H;
}

// ---------------------------------------------------------------------------
// getAttentionWeights — Extract attention coefficients for interpretability
// ---------------------------------------------------------------------------

/**
 * Compute and return the attention weights for all edges across all heads.
 *
 * This is a read-only operation for interpretability; no dropout is applied.
 * Uses GATv1 attention by default; if config.v2 is true, uses GATv2.
 *
 * @param graph   - CSR Graph.
 * @param X       - Node features (numNodes x inDim).
 * @param weights - GAT weights.
 * @param config  - GAT config.
 * @returns Float64Array of shape (numEdges x heads), row-major.
 *          attn[e * heads + h] = attention weight for edge e, head h.
 */
export function getAttentionWeights(
  graph: Graph,
  X: Float64Array,
  weights: GATWeights,
  config: GATConfig,
): Float64Array {
  const { inDim, outDim, heads, negativeSlope, v2 } = config;
  const n = graph.numNodes;
  const totalOutDim = outDim * heads;
  const numEdges = graph.numEdges;

  if (v2) {
    // GATv2 attention
    const W_src = weights.W_src!;
    const W_dst = weights.W_dst!;
    const a = weights.a!;

    const srcProj = projectFeatures(X, W_src, n, inDim, totalOutDim);
    const dstProj = projectFeatures(X, W_dst, n, inDim, totalOutDim);

    const rawScores = new Float64Array(numEdges * heads);

    for (let i = 0; i < n; i++) {
      const start = graph.rowPtr[i]!;
      const end = graph.rowPtr[i + 1]!;
      const srcOff = i * totalOutDim;

      for (let e = start; e < end; e++) {
        const j = graph.colIdx[e]!;
        const dstOff = j * totalOutDim;
        for (let h = 0; h < heads; h++) {
          const headOff = h * outDim;
          let score = 0;
          for (let d = 0; d < outDim; d++) {
            const combined = srcProj[srcOff + headOff + d]! + dstProj[dstOff + headOff + d]!;
            const activated = combined > 0 ? combined : negativeSlope * combined;
            score += a[headOff + d]! * activated;
          }
          rawScores[e * heads + h] = score;
        }
      }
    }

    return edgeSoftmax(rawScores, graph, numEdges, heads);
  } else {
    // GATv1 attention
    const Wh = projectFeatures(X, weights.W, n, inDim, totalOutDim);

    const srcScores = new Float64Array(n * heads);
    const dstScores = new Float64Array(n * heads);

    for (let i = 0; i < n; i++) {
      const whOff = i * totalOutDim;
      for (let h = 0; h < heads; h++) {
        let srcS = 0;
        let dstS = 0;
        const headOff = h * outDim;
        for (let d = 0; d < outDim; d++) {
          srcS += weights.a_src[headOff + d]! * Wh[whOff + headOff + d]!;
          dstS += weights.a_dst[headOff + d]! * Wh[whOff + headOff + d]!;
        }
        srcScores[i * heads + h] = srcS;
        dstScores[i * heads + h] = dstS;
      }
    }

    const rawScores = new Float64Array(numEdges * heads);
    for (let i = 0; i < n; i++) {
      const start = graph.rowPtr[i]!;
      const end = graph.rowPtr[i + 1]!;
      for (let e = start; e < end; e++) {
        const j = graph.colIdx[e]!;
        for (let h = 0; h < heads; h++) {
          const raw = srcScores[i * heads + h]! + dstScores[j * heads + h]!;
          rawScores[e * heads + h] = raw > 0 ? raw : negativeSlope * raw;
        }
      }
    }

    return edgeSoftmax(rawScores, graph, numEdges, heads);
  }
}
