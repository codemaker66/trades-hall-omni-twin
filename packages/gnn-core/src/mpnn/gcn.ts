// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GCN (Graph Convolutional Network)
// Kipf & Welling 2017: Semi-Supervised Classification with GCNs
//
// Implements: H' = activation(D^{-1/2}(A+I)D^{-1/2} * X * W + bias)
// Uses CSR sparse-dense multiply for efficient neighborhood aggregation.
// ---------------------------------------------------------------------------

import type { Graph, GCNConfig, GCNWeights, ActivationFn } from '../types.js';
import { addSelfLoops } from '../graph.js';
import { relu, elu, leakyRelu, tanhActivation, sigmoid } from '../tensor.js';

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
 * Compute the symmetric normalization coefficients for a graph with self-loops.
 *
 * For each edge (i, j) in the CSR, the coefficient is:
 *   1 / sqrt(deg(i) * deg(j))
 *
 * where deg(i) counts all edges from node i (including self-loop).
 *
 * @returns Float64Array of length numEdges with normalization coefficients.
 */
function symmetricNormCoeffs(graph: Graph): Float64Array {
  const n = graph.numNodes;
  const coeffs = new Float64Array(graph.numEdges);

  // Compute degree for each node (count outgoing edges in CSR)
  const deg = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    deg[i] = graph.rowPtr[i + 1]! - graph.rowPtr[i]!;
  }

  // Compute D^{-1/2}_{ii} = 1/sqrt(deg_i)
  const invSqrtDeg = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    invSqrtDeg[i] = deg[i]! > 0 ? 1.0 / Math.sqrt(deg[i]!) : 0;
  }

  // For each edge (i, j): coeff = invSqrtDeg[i] * invSqrtDeg[j]
  for (let i = 0; i < n; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;
    for (let e = start; e < end; e++) {
      const j = graph.colIdx[e]!;
      coeffs[e] = invSqrtDeg[i]! * invSqrtDeg[j]!;
    }
  }

  return coeffs;
}

// ---------------------------------------------------------------------------
// gcnLayer — Single GCN layer
// ---------------------------------------------------------------------------

/**
 * Single GCN layer forward pass.
 *
 * Implements: H' = activation(hat{A} * X * W + bias)
 * where hat{A} = D^{-1/2}(A+I)D^{-1/2}
 *
 * Algorithm:
 * 1. Add self-loops to get A+I.
 * 2. Compute symmetric normalization coefficients for hat{A}.
 * 3. Sparse-dense multiply: for each node i, accumulate hat{A}_{ij} * X_j over
 *    neighbors j (including self), then multiply the aggregated features by W.
 * 4. Add bias (if present) and apply activation.
 *
 * The operation is equivalent to hat{A} * X * W, but we fuse the sparse matmul
 * with the dense matmul for better locality:
 *   For each node i: h_i = sum_j(hat{A}_{ij} * x_j) then h_i = h_i * W + bias
 *
 * @param graph  - CSR Graph (without self-loops; they are added internally).
 * @param X      - Node features, row-major Float64Array of shape (numNodes x inDim).
 * @param weights - GCN layer weights { W: inDim x outDim, bias?: outDim }.
 * @param config  - Layer configuration { inDim, outDim, bias, activation }.
 * @returns Float64Array of shape (numNodes x outDim), the layer output.
 */
export function gcnLayer(
  graph: Graph,
  X: Float64Array,
  weights: GCNWeights,
  config: GCNConfig,
): Float64Array {
  const { inDim, outDim, activation } = config;
  const n = graph.numNodes;

  // Step 1: Add self-loops → A + I
  const augGraph = addSelfLoops(graph);

  // Step 2: Compute symmetric normalization coefficients
  const normCoeffs = symmetricNormCoeffs(augGraph);

  // Step 3: Sparse aggregation → aggregated (n x inDim)
  // For each node i: agg_i[f] = sum_{j in N(i)} normCoeff_{ij} * X_j[f]
  const agg = new Float64Array(n * inDim);

  for (let i = 0; i < n; i++) {
    const start = augGraph.rowPtr[i]!;
    const end = augGraph.rowPtr[i + 1]!;
    const aggOffset = i * inDim;

    for (let e = start; e < end; e++) {
      const j = augGraph.colIdx[e]!;
      const coeff = normCoeffs[e]!;
      const xOffset = j * inDim;

      for (let f = 0; f < inDim; f++) {
        agg[aggOffset + f]! += coeff * X[xOffset + f]!;
      }
    }
  }

  // Step 4: Dense matmul with W → output (n x outDim)
  // output_i = agg_i * W + bias
  const output = new Float64Array(n * outDim);

  for (let i = 0; i < n; i++) {
    const aggOffset = i * inDim;
    const outOffset = i * outDim;

    for (let k = 0; k < inDim; k++) {
      const aik = agg[aggOffset + k]!;
      for (let o = 0; o < outDim; o++) {
        output[outOffset + o]! += aik * weights.W[k * outDim + o]!;
      }
    }

    // Add bias
    if (config.bias && weights.bias) {
      for (let o = 0; o < outDim; o++) {
        output[outOffset + o] = output[outOffset + o]! + weights.bias[o]!;
      }
    }
  }

  // Step 5: Apply activation
  return applyActivation(output, activation);
}

// ---------------------------------------------------------------------------
// gcnForward — Multi-layer GCN forward pass
// ---------------------------------------------------------------------------

/**
 * Multi-layer GCN forward pass.
 *
 * Sequentially applies each GCN layer, feeding the output of layer l as the
 * input to layer l+1.
 *
 * @param graph  - CSR Graph.
 * @param X      - Initial node features (numNodes x layers[0].config.inDim).
 * @param layers - Array of { weights, config } for each layer.
 * @returns Float64Array of shape (numNodes x layers[-1].config.outDim).
 */
export function gcnForward(
  graph: Graph,
  X: Float64Array,
  layers: { weights: GCNWeights; config: GCNConfig }[],
): Float64Array {
  let H = X;
  for (let l = 0; l < layers.length; l++) {
    const layer = layers[l]!;
    H = gcnLayer(graph, H, layer.weights, layer.config);
  }
  return H;
}
