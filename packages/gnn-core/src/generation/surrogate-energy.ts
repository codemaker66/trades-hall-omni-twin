// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-5: Graph Generation
// Surrogate Energy Model — GNN as a differentiable layout energy function.
//
// Uses GAT layers with residual connections, global mean pooling, and a
// linear head to output a scalar energy. Numerical gradient descent
// optimises node positions to minimise this energy.
// ---------------------------------------------------------------------------

import type { Graph, SurrogateEnergyModel, GATWeights } from '../types.js';
import { matVecMul, add, relu, dot } from '../tensor.js';

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Numerically stable sigmoid for a single scalar.
 */
function sigmoidScalar(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  const ex = Math.exp(x);
  return ex / (1 + ex);
}

/**
 * Leaky ReLU for a single scalar (slope = 0.2 for GAT attention).
 */
function leakyReluScalar(x: number, slope: number): number {
  return x > 0 ? x : slope * x;
}

/**
 * Run a single GAT layer with residual connection.
 *
 * For each node i:
 *   1. Project features: h_i' = W * h_i
 *   2. Compute attention: e_ij = LeakyReLU( a_src^T h_i' + a_dst^T h_j' )
 *   3. Softmax attention over neighbors: alpha_ij = softmax_j(e_ij)
 *   4. Aggregate: out_i = sum_j alpha_ij * h_j'
 *   5. Residual: out_i += h_i  (when dims match)
 *
 * Single-head implementation (head=0 slice) for simplicity.
 *
 * @returns New node embeddings as Float64Array (numNodes * outDim), row-major.
 */
function gatLayerForward(
  graph: Graph,
  X: Float64Array,
  inDim: number,
  weights: GATWeights,
  outDim: number,
  negativeSlope: number,
): Float64Array {
  const n = graph.numNodes;

  // Step 1: project all nodes: H = X * W^T  (we store W as inDim x outDim)
  const H = new Float64Array(n * outDim);
  for (let i = 0; i < n; i++) {
    for (let o = 0; o < outDim; o++) {
      let val = 0;
      for (let d = 0; d < inDim; d++) {
        val += weights.W[d * outDim + o]! * X[i * inDim + d]!;
      }
      H[i * outDim + o] = val;
    }
  }

  // Step 2 & 3: compute attention scores and aggregate
  const out = new Float64Array(n * outDim);

  for (let i = 0; i < n; i++) {
    const rowStart = graph.rowPtr[i]!;
    const rowEnd = graph.rowPtr[i + 1]!;
    const degree = rowEnd - rowStart;

    if (degree === 0) {
      // No neighbors: just keep projected features
      for (let o = 0; o < outDim; o++) {
        out[i * outDim + o] = H[i * outDim + o]!;
      }
      continue;
    }

    // Compute attention logits for each neighbor
    // e_ij = LeakyReLU( a_src . h_i + a_dst . h_j )
    const attnLogits = new Float64Array(degree);
    let srcScore = 0;
    for (let o = 0; o < outDim; o++) {
      srcScore += weights.a_src[o]! * H[i * outDim + o]!;
    }

    let maxLogit = -Infinity;
    for (let eIdx = 0; eIdx < degree; eIdx++) {
      const j = graph.colIdx[rowStart + eIdx]!;
      let dstScore = 0;
      for (let o = 0; o < outDim; o++) {
        dstScore += weights.a_dst[o]! * H[j * outDim + o]!;
      }
      const raw = leakyReluScalar(srcScore + dstScore, negativeSlope);
      attnLogits[eIdx] = raw;
      if (raw > maxLogit) maxLogit = raw;
    }

    // Softmax
    let sumExp = 0;
    const attnWeights = new Float64Array(degree);
    for (let eIdx = 0; eIdx < degree; eIdx++) {
      const e = Math.exp(attnLogits[eIdx]! - maxLogit);
      attnWeights[eIdx] = e;
      sumExp += e;
    }
    for (let eIdx = 0; eIdx < degree; eIdx++) {
      attnWeights[eIdx] = attnWeights[eIdx]! / sumExp;
    }

    // Weighted aggregation
    for (let eIdx = 0; eIdx < degree; eIdx++) {
      const j = graph.colIdx[rowStart + eIdx]!;
      const alpha = attnWeights[eIdx]!;
      for (let o = 0; o < outDim; o++) {
        out[i * outDim + o] = out[i * outDim + o]! + alpha * H[j * outDim + o]!;
      }
    }
  }

  // Residual connection (add input if dimensions match)
  if (inDim === outDim) {
    for (let i = 0; i < n * outDim; i++) {
      out[i] = out[i]! + X[i]!;
    }
  }

  // ReLU activation
  for (let i = 0; i < out.length; i++) {
    if (out[i]! < 0) out[i] = 0;
  }

  return out;
}

/**
 * Global mean pooling: average all node embeddings into a single vector.
 */
function globalMeanPool(
  X: Float64Array,
  numNodes: number,
  dim: number,
): Float64Array {
  const pooled = new Float64Array(dim);
  for (let i = 0; i < numNodes; i++) {
    for (let d = 0; d < dim; d++) {
      pooled[d] = pooled[d]! + X[i * dim + d]!;
    }
  }
  if (numNodes > 0) {
    for (let d = 0; d < dim; d++) {
      pooled[d] = pooled[d]! / numNodes;
    }
  }
  return pooled;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Forward pass of the surrogate energy model.
 *
 * Architecture:
 *   1. Stack of GAT layers (with residual connections and ReLU).
 *   2. Global mean pooling over all nodes.
 *   3. Linear projection: poolingW maps pooled vector to intermediate.
 *   4. Linear head: headW * intermediate + headBias → scalar energy.
 *
 * @param graph - Input CSR Graph with node features set.
 * @param model - Surrogate energy model (GAT weights, pooling, head).
 * @returns     - Scalar energy value (lower = better layout).
 */
export function surrogateEnergyForward(
  graph: Graph,
  model: SurrogateEnergyModel,
): number {
  const n = graph.numNodes;
  const { gatWeights, poolingW, headW, headBias, config: gatConfig } = model;

  let X = graph.nodeFeatures;
  let currentDim = graph.featureDim;

  // Run GAT layers
  for (let l = 0; l < gatWeights.length; l++) {
    const outDim = gatConfig.outDim;
    X = gatLayerForward(
      graph,
      X,
      currentDim,
      gatWeights[l]!,
      outDim,
      gatConfig.negativeSlope,
    );
    currentDim = outDim;
  }

  // Global mean pool
  const pooled = globalMeanPool(X, n, currentDim);

  // Projection: intermediate = ReLU(poolingW * pooled)
  // poolingW is (intermediateDim × currentDim)
  const intermediateDim = poolingW.length / currentDim;
  const intermediate = new Float64Array(intermediateDim);
  for (let i = 0; i < intermediateDim; i++) {
    let val = 0;
    for (let d = 0; d < currentDim; d++) {
      val += poolingW[i * currentDim + d]! * pooled[d]!;
    }
    intermediate[i] = val > 0 ? val : 0; // ReLU
  }

  // Head: energy = headW . intermediate + headBias
  // headW is (1 × intermediateDim), headBias is (1)
  let energy = headBias[0]!;
  for (let i = 0; i < intermediateDim; i++) {
    energy += headW[i]! * intermediate[i]!;
  }

  return energy;
}

/**
 * Optimise node positions to minimise the surrogate energy via numerical
 * gradient descent.
 *
 * Algorithm:
 * For each step:
 *   1. For each coordinate (2 per node: x and y), compute the numerical
 *      gradient using central differences:
 *        dE/dx_i ~ (E(x_i + eps) - E(x_i - eps)) / (2 * eps)
 *   2. Update: pos_i -= lr * gradient_i
 *
 * The positions are injected into the graph's node features at indices
 * [0..1] of each node's feature vector (first two dimensions are x, y).
 *
 * @param graph      - Input CSR Graph (template — features will be modified per eval).
 * @param positions  - Initial positions, Float64Array of length numNodes * 2.
 * @param model      - Surrogate energy model.
 * @param lr         - Learning rate for gradient descent.
 * @param steps      - Number of optimisation steps.
 * @param featureDim - Full feature dimension per node (positions occupy first 2 slots).
 * @returns          - Optimised positions, Float64Array of length numNodes * 2.
 */
export function surrogateGradientDescent(
  graph: Graph,
  positions: Float64Array,
  model: SurrogateEnergyModel,
  lr: number,
  steps: number,
  featureDim: number,
): Float64Array {
  const n = graph.numNodes;
  const pos = new Float64Array(positions); // working copy
  const eps = 1e-4;

  // Base feature buffer (we only modify the first two dims per node)
  const features = new Float64Array(graph.nodeFeatures);

  /**
   * Build a graph snapshot with positions injected into features.
   */
  function makeGraph(p: Float64Array): Graph {
    const f = new Float64Array(features);
    for (let i = 0; i < n; i++) {
      f[i * featureDim] = p[i * 2]!;
      f[i * featureDim + 1] = p[i * 2 + 1]!;
    }
    return {
      numNodes: graph.numNodes,
      numEdges: graph.numEdges,
      rowPtr: graph.rowPtr,
      colIdx: graph.colIdx,
      edgeWeights: graph.edgeWeights,
      nodeFeatures: f,
      featureDim,
    };
  }

  for (let step = 0; step < steps; step++) {
    const grad = new Float64Array(n * 2);

    // Central difference for each position coordinate
    for (let i = 0; i < n * 2; i++) {
      const orig = pos[i]!;

      // E(pos + eps)
      pos[i] = orig + eps;
      const ePlus = surrogateEnergyForward(makeGraph(pos), model);

      // E(pos - eps)
      pos[i] = orig - eps;
      const eMinus = surrogateEnergyForward(makeGraph(pos), model);

      // Restore
      pos[i] = orig;

      grad[i] = (ePlus - eMinus) / (2 * eps);
    }

    // Gradient descent update
    for (let i = 0; i < n * 2; i++) {
      pos[i] = pos[i]! - lr * grad[i]!;
    }
  }

  return pos;
}
