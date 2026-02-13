// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GIN (Graph Isomorphism Network)
// Xu et al. 2019: How Powerful are Graph Neural Networks?
//
// Implements: h_v = MLP((1 + epsilon) * h_v + SUM_{u in N(v)} h_u)
// The MLP is a 2-layer feedforward network: Linear(inDim, hiddenDim) -> ReLU
// -> Linear(hiddenDim, outDim).
// ---------------------------------------------------------------------------

import type { Graph, GINConfig, GINWeights, MLPWeights } from '../types.js';
import { relu } from '../tensor.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Apply a 2-layer MLP to a batch of vectors.
 *
 * Layer 1: z = ReLU(x * W1 + b1)   (inDim -> hiddenDim)
 * Layer 2: out = z * W2 + b2        (hiddenDim -> outDim)
 *
 * @param X       - Input (numVectors x inDim), row-major.
 * @param mlp     - MLPWeights with exactly 2 layers.
 * @param numVecs - Number of vectors.
 * @returns Float64Array of shape (numVectors x outDim).
 */
function applyMLP(
  X: Float64Array,
  mlp: MLPWeights,
  numVecs: number,
): Float64Array {
  let current = X;

  for (let l = 0; l < mlp.layers.length; l++) {
    const layer = mlp.layers[l]!;
    const { W, bias, inDim: lIn, outDim: lOut } = layer;
    const next = new Float64Array(numVecs * lOut);

    for (let i = 0; i < numVecs; i++) {
      const inOff = i * lIn;
      const outOff = i * lOut;

      for (let k = 0; k < lIn; k++) {
        const xik = current[inOff + k]!;
        for (let o = 0; o < lOut; o++) {
          next[outOff + o]! += xik * W[k * lOut + o]!;
        }
      }

      // Add bias
      for (let o = 0; o < lOut; o++) {
        next[outOff + o] = next[outOff + o]! + bias[o]!;
      }
    }

    // Apply ReLU after all layers except the last
    if (l < mlp.layers.length - 1) {
      current = relu(next);
    } else {
      current = next;
    }
  }

  return current;
}

// ---------------------------------------------------------------------------
// ginLayer — Single GIN layer
// ---------------------------------------------------------------------------

/**
 * Single GIN layer forward pass.
 *
 * Algorithm:
 * For each node v:
 *   1. aggV = (1 + epsilon) * h_v + SUM_{u in N(v)} h_u
 *   2. h_v' = MLP(aggV)
 *
 * The MLP is a 2-layer network defined by weights.mlp:
 *   Layer 1: Linear(inDim -> hiddenDim) + ReLU
 *   Layer 2: Linear(hiddenDim -> outDim)
 *
 * @param graph   - CSR Graph.
 * @param X       - Node features (numNodes x inDim), row-major.
 * @param weights - { mlp: MLPWeights, epsilon: number }.
 * @param config  - { inDim, hiddenDim, outDim, epsilon, trainEpsilon }.
 * @returns Float64Array of shape (numNodes x outDim).
 */
export function ginLayer(
  graph: Graph,
  X: Float64Array,
  weights: GINWeights,
  config: GINConfig,
): Float64Array {
  const { inDim } = config;
  const n = graph.numNodes;
  const eps = weights.epsilon;

  // Step 1: Compute aggregated features for each node
  // agg_v = (1 + eps) * x_v + sum_{u in N(v)} x_u
  const agg = new Float64Array(n * inDim);

  for (let i = 0; i < n; i++) {
    const iOff = i * inDim;

    // Self-contribution: (1 + eps) * x_i
    for (let f = 0; f < inDim; f++) {
      agg[iOff + f] = (1.0 + eps) * X[iOff + f]!;
    }

    // Neighbor sum
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;
    for (let e = start; e < end; e++) {
      const j = graph.colIdx[e]!;
      const jOff = j * inDim;
      for (let f = 0; f < inDim; f++) {
        agg[iOff + f] = agg[iOff + f]! + X[jOff + f]!;
      }
    }
  }

  // Step 2: Apply MLP
  return applyMLP(agg, weights.mlp, n);
}

// ---------------------------------------------------------------------------
// ginGraphReadout — Sum pooling per graph in a batch
// ---------------------------------------------------------------------------

/**
 * Graph-level readout via SUM pooling.
 *
 * For graph-level tasks (e.g., graph classification), aggregate all node
 * features belonging to each graph in the batch by summation.
 *
 * Algorithm:
 * For each graph g in [0, numGraphs):
 *   readout_g = SUM(h_v for all v where batchIndex[v] == g)
 *
 * @param X          - Node features (totalNodes x featureDim), row-major.
 * @param batchIndex - Uint32Array mapping each node to its graph index.
 * @param numGraphs  - Total number of graphs in the batch.
 * @param featureDim - Dimension of each node's feature vector.
 * @returns Float64Array of shape (numGraphs x featureDim).
 */
export function ginGraphReadout(
  X: Float64Array,
  batchIndex: Uint32Array,
  numGraphs: number,
  featureDim: number,
): Float64Array {
  const totalNodes = batchIndex.length;
  const output = new Float64Array(numGraphs * featureDim);

  for (let v = 0; v < totalNodes; v++) {
    const g = batchIndex[v]!;
    const vOff = v * featureDim;
    const gOff = g * featureDim;
    for (let f = 0; f < featureDim; f++) {
      output[gOff + f] = output[gOff + f]! + X[vOff + f]!;
    }
  }

  return output;
}

// ---------------------------------------------------------------------------
// ginForward — Multi-layer GIN forward pass
// ---------------------------------------------------------------------------

/**
 * Multi-layer GIN forward pass.
 *
 * Sequentially applies each GIN layer, feeding the output of layer l as the
 * input to layer l+1.
 *
 * @param graph  - CSR Graph.
 * @param X      - Initial node features (numNodes x layers[0].config.inDim).
 * @param layers - Array of { weights, config } for each layer.
 * @returns Float64Array of shape (numNodes x layers[-1].config.outDim).
 */
export function ginForward(
  graph: Graph,
  X: Float64Array,
  layers: { weights: GINWeights; config: GINConfig }[],
): Float64Array {
  let H = X;
  for (let l = 0; l < layers.length; l++) {
    const layer = layers[l]!;
    H = ginLayer(graph, H, layer.weights, layer.config);
  }
  return H;
}
