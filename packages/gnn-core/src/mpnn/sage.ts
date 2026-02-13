// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GraphSAGE (SAmple and agGrEgate)
// Hamilton, Ying & Leskovec 2017: Inductive Representation Learning
//
// Implements mean-aggregator and max-aggregator variants.
// h_v = activation(W_self * x_v + W_neigh * AGG(x_u : u in N(v)) + bias)
// ---------------------------------------------------------------------------

import type { Graph, SAGEConfig, SAGEWeights, ActivationFn } from '../types.js';
import { relu, elu, leakyRelu, tanhActivation, sigmoid, l2Norm } from '../tensor.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Apply an activation function by name to the given array.
 */
function applyActivation(x: Float64Array, activation: ActivationFn): Float64Array {
  switch (activation) {
    case 'relu':
      return relu(x);
    case 'elu':
      return elu(x, 1.0);
    case 'leaky_relu':
      return leakyRelu(x, 0.01);
    case 'tanh':
      return tanhActivation(x);
    case 'sigmoid':
      return sigmoid(x);
    case 'none':
      return x;
  }
}

/**
 * L2-normalize each row of a (numNodes x dim) matrix in-place.
 */
function l2NormalizeRows(H: Float64Array, numNodes: number, dim: number): Float64Array {
  const out = new Float64Array(H.length);
  for (let i = 0; i < numNodes; i++) {
    const offset = i * dim;
    // Compute row norm
    let normSq = 0;
    for (let d = 0; d < dim; d++) {
      normSq += H[offset + d]! * H[offset + d]!;
    }
    const norm = Math.sqrt(normSq);
    const invNorm = norm > 0 ? 1.0 / norm : 0;
    for (let d = 0; d < dim; d++) {
      out[offset + d] = H[offset + d]! * invNorm;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// sageMeanLayer — Mean aggregation
// ---------------------------------------------------------------------------

/**
 * GraphSAGE layer with MEAN aggregation.
 *
 * Algorithm:
 * For each node v:
 *   1. Compute agg_v = MEAN(x_u for u in N(v))
 *      If v has no neighbors, agg_v = zero vector.
 *   2. h_v = activation(W_self * x_v + W_neigh * agg_v + bias)
 *   3. Optionally L2-normalize h_v.
 *
 * @param graph   - CSR Graph.
 * @param X       - Node features (numNodes x inDim), row-major.
 * @param weights - { W_self: inDim x outDim, W_neigh: inDim x outDim, bias?: outDim }
 * @param config  - { inDim, outDim, aggregator, normalize, activation }
 * @returns Float64Array of shape (numNodes x outDim).
 */
export function sageMeanLayer(
  graph: Graph,
  X: Float64Array,
  weights: SAGEWeights,
  config: SAGEConfig,
): Float64Array {
  const { inDim, outDim, activation, normalize: doNormalize } = config;
  const n = graph.numNodes;
  const output = new Float64Array(n * outDim);

  for (let i = 0; i < n; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;
    const deg = end - start;

    // Aggregate neighbor features: mean
    const aggNeigh = new Float64Array(inDim);
    if (deg > 0) {
      for (let e = start; e < end; e++) {
        const j = graph.colIdx[e]!;
        const jOffset = j * inDim;
        for (let f = 0; f < inDim; f++) {
          aggNeigh[f]! += X[jOffset + f]!;
        }
      }
      const invDeg = 1.0 / deg;
      for (let f = 0; f < inDim; f++) {
        aggNeigh[f] = aggNeigh[f]! * invDeg;
      }
    }

    // h_i = W_self * x_i + W_neigh * aggNeigh + bias
    const outOffset = i * outDim;
    const xOffset = i * inDim;

    for (let o = 0; o < outDim; o++) {
      let val = 0;
      // W_self * x_i
      for (let k = 0; k < inDim; k++) {
        val += weights.W_self[k * outDim + o]! * X[xOffset + k]!;
      }
      // W_neigh * aggNeigh
      for (let k = 0; k < inDim; k++) {
        val += weights.W_neigh[k * outDim + o]! * aggNeigh[k]!;
      }
      // bias
      if (weights.bias) {
        val += weights.bias[o]!;
      }
      output[outOffset + o] = val;
    }
  }

  // Apply activation
  const activated = applyActivation(output, activation);

  // Optional L2 normalization
  if (doNormalize) {
    return l2NormalizeRows(activated, n, outDim);
  }

  return activated;
}

// ---------------------------------------------------------------------------
// sageMaxLayer — Max-pooling aggregation
// ---------------------------------------------------------------------------

/**
 * GraphSAGE layer with MAX aggregation.
 *
 * Algorithm:
 * For each node v:
 *   1. Compute agg_v = element-wise MAX(x_u for u in N(v))
 *      If v has no neighbors, agg_v = zero vector.
 *   2. h_v = activation(W_self * x_v + W_neigh * agg_v + bias)
 *   3. Optionally L2-normalize h_v.
 *
 * @param graph   - CSR Graph.
 * @param X       - Node features (numNodes x inDim), row-major.
 * @param weights - { W_self: inDim x outDim, W_neigh: inDim x outDim, bias?: outDim }
 * @param config  - { inDim, outDim, aggregator, normalize, activation }
 * @returns Float64Array of shape (numNodes x outDim).
 */
export function sageMaxLayer(
  graph: Graph,
  X: Float64Array,
  weights: SAGEWeights,
  config: SAGEConfig,
): Float64Array {
  const { inDim, outDim, activation, normalize: doNormalize } = config;
  const n = graph.numNodes;
  const output = new Float64Array(n * outDim);

  for (let i = 0; i < n; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;
    const deg = end - start;

    // Aggregate neighbor features: element-wise max
    const aggNeigh = new Float64Array(inDim);
    if (deg > 0) {
      // Initialize with first neighbor's features
      const firstNeighbor = graph.colIdx[start]!;
      const firstOffset = firstNeighbor * inDim;
      for (let f = 0; f < inDim; f++) {
        aggNeigh[f] = X[firstOffset + f]!;
      }
      // Take element-wise max over remaining neighbors
      for (let e = start + 1; e < end; e++) {
        const j = graph.colIdx[e]!;
        const jOffset = j * inDim;
        for (let f = 0; f < inDim; f++) {
          const v = X[jOffset + f]!;
          if (v > aggNeigh[f]!) {
            aggNeigh[f] = v;
          }
        }
      }
    }

    // h_i = W_self * x_i + W_neigh * aggNeigh + bias
    const outOffset = i * outDim;
    const xOffset = i * inDim;

    for (let o = 0; o < outDim; o++) {
      let val = 0;
      // W_self * x_i
      for (let k = 0; k < inDim; k++) {
        val += weights.W_self[k * outDim + o]! * X[xOffset + k]!;
      }
      // W_neigh * aggNeigh
      for (let k = 0; k < inDim; k++) {
        val += weights.W_neigh[k * outDim + o]! * aggNeigh[k]!;
      }
      // bias
      if (weights.bias) {
        val += weights.bias[o]!;
      }
      output[outOffset + o] = val;
    }
  }

  // Apply activation
  const activated = applyActivation(output, activation);

  // Optional L2 normalization
  if (doNormalize) {
    return l2NormalizeRows(activated, n, outDim);
  }

  return activated;
}

// ---------------------------------------------------------------------------
// sageForward — Multi-layer GraphSAGE forward pass
// ---------------------------------------------------------------------------

/**
 * Multi-layer GraphSAGE forward pass.
 *
 * Sequentially applies each SAGE layer. The aggregator type in the config
 * determines which aggregation function is used (mean or max).
 *
 * @param graph  - CSR Graph.
 * @param X      - Initial node features (numNodes x layers[0].config.inDim).
 * @param layers - Array of { weights, config } for each layer.
 * @returns Float64Array of shape (numNodes x layers[-1].config.outDim).
 */
export function sageForward(
  graph: Graph,
  X: Float64Array,
  layers: { weights: SAGEWeights; config: SAGEConfig }[],
): Float64Array {
  let H = X;
  for (let l = 0; l < layers.length; l++) {
    const layer = layers[l]!;
    if (layer.config.aggregator === 'max') {
      H = sageMaxLayer(graph, H, layer.weights, layer.config);
    } else {
      // 'mean' and 'gcn' both use mean aggregation
      H = sageMeanLayer(graph, H, layer.weights, layer.config);
    }
  }
  return H;
}
