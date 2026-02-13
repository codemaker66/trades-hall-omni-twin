// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — Over-smoothing Mitigations
//
// Deep GNNs suffer from over-smoothing where node representations converge
// to indistinguishable vectors. This module implements three countermeasures:
//
// 1. Residual connections (He et al. 2016, adapted for GNNs)
// 2. JK-Net (Jumping Knowledge) combination (Xu et al. 2018)
// 3. DropEdge (Rong et al. 2020)
// ---------------------------------------------------------------------------

import type { Graph, PRNG } from '../types.js';
import { buildCSR } from '../graph.js';

// ---------------------------------------------------------------------------
// residualConnection — Element-wise skip connection
// ---------------------------------------------------------------------------

/**
 * Residual (skip) connection: h' = input + output.
 *
 * Both arrays MUST have the same length (same number of nodes x same dimension).
 * This is the standard ResNet-style shortcut adapted for GNN layers.
 *
 * When the input and output dimensions differ, the caller should apply a
 * linear projection to one of them before calling this function.
 *
 * @param input  - The layer's input features (numNodes x dim).
 * @param output - The layer's output features (numNodes x dim).
 * @returns Float64Array of the same shape, with element-wise sum.
 * @throws Error if input and output lengths differ.
 */
export function residualConnection(
  input: Float64Array,
  output: Float64Array,
): Float64Array {
  if (input.length !== output.length) {
    throw new Error(
      `residualConnection: input length (${input.length}) must match output length (${output.length})`,
    );
  }

  const result = new Float64Array(input.length);
  for (let i = 0; i < input.length; i++) {
    result[i] = input[i]! + output[i]!;
  }
  return result;
}

// ---------------------------------------------------------------------------
// jkNetCombine — Jumping Knowledge aggregation across layers
// ---------------------------------------------------------------------------

/**
 * Jumping Knowledge (JK-Net) combination of per-layer representations.
 *
 * Instead of using only the final layer's output, JK-Net aggregates the
 * outputs of ALL intermediate layers. This preserves information from
 * different neighborhood radii and counteracts over-smoothing.
 *
 * Modes:
 * - **concat**: Horizontal concatenation of all L layers.
 *   Output shape: (numNodes x (L * featureDim)).
 *   Most expressive but increases dimensionality.
 *
 * - **max**: Element-wise maximum across L layers.
 *   Output shape: (numNodes x featureDim).
 *   All layers must have the same featureDim.
 *
 * - **mean**: Element-wise mean across L layers.
 *   Output shape: (numNodes x featureDim).
 *   All layers must have the same featureDim.
 *
 * @param layerOutputs - Array of L Float64Arrays, each (numNodes x featureDim).
 * @param mode         - Combination strategy: 'concat', 'max', or 'mean'.
 * @param featureDim   - Feature dimension per node per layer.
 * @returns Combined representation as Float64Array.
 * @throws Error if layerOutputs is empty.
 */
export function jkNetCombine(
  layerOutputs: Float64Array[],
  mode: 'concat' | 'max' | 'mean',
  featureDim: number,
): Float64Array {
  const L = layerOutputs.length;
  if (L === 0) {
    throw new Error('jkNetCombine: layerOutputs must not be empty');
  }

  const numElements = layerOutputs[0]!.length;
  const numNodes = numElements / featureDim;

  switch (mode) {
    case 'concat': {
      // Horizontally concatenate: for each node, place layer 0's features,
      // then layer 1's features, etc.
      // Output: (numNodes x (L * featureDim))
      const outDim = L * featureDim;
      const result = new Float64Array(numNodes * outDim);

      for (let l = 0; l < L; l++) {
        const layerOut = layerOutputs[l]!;
        for (let i = 0; i < numNodes; i++) {
          const srcOff = i * featureDim;
          const dstOff = i * outDim + l * featureDim;
          for (let f = 0; f < featureDim; f++) {
            result[dstOff + f] = layerOut[srcOff + f]!;
          }
        }
      }

      return result;
    }

    case 'max': {
      // Element-wise max across all layers
      const result = new Float64Array(numElements);

      // Initialize with first layer
      const first = layerOutputs[0]!;
      for (let i = 0; i < numElements; i++) {
        result[i] = first[i]!;
      }

      // Max with remaining layers
      for (let l = 1; l < L; l++) {
        const layerOut = layerOutputs[l]!;
        for (let i = 0; i < numElements; i++) {
          const v = layerOut[i]!;
          if (v > result[i]!) {
            result[i] = v;
          }
        }
      }

      return result;
    }

    case 'mean': {
      // Element-wise mean across all layers
      const result = new Float64Array(numElements);

      for (let l = 0; l < L; l++) {
        const layerOut = layerOutputs[l]!;
        for (let i = 0; i < numElements; i++) {
          result[i] = result[i]! + layerOut[i]!;
        }
      }

      const invL = 1.0 / L;
      for (let i = 0; i < numElements; i++) {
        result[i] = result[i]! * invL;
      }

      return result;
    }
  }
}

// ---------------------------------------------------------------------------
// dropEdge — Random edge removal for regularization
// ---------------------------------------------------------------------------

/**
 * DropEdge: randomly remove edges from the graph with probability p.
 *
 * During training, randomly dropping edges acts as a data augmentation that
 * reduces the convergence speed of over-smoothing, allowing deeper GNNs to
 * retain more discriminative node features.
 *
 * Algorithm:
 * 1. Iterate over all edges in the CSR graph.
 * 2. For each edge, keep it with probability (1 - p).
 * 3. Rebuild the CSR graph from the surviving edges.
 * 4. Preserve node features and feature dimension from the original graph.
 *
 * @param graph - Input CSR Graph.
 * @param p     - Drop probability in [0, 1). Each edge is independently removed
 *                with this probability.
 * @param rng   - Deterministic PRNG for reproducibility.
 * @returns A new Graph with a subset of the original edges.
 */
export function dropEdge(graph: Graph, p: number, rng: PRNG): Graph {
  // Edge case: no dropping
  if (p <= 0) {
    return graph;
  }

  // Collect surviving edges
  const edges: [number, number][] = [];
  const weights: number[] = [];
  const hasWeights = graph.edgeWeights !== undefined;

  for (let i = 0; i < graph.numNodes; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;
    for (let e = start; e < end; e++) {
      // Keep edge with probability (1 - p)
      if (rng() >= p) {
        const j = graph.colIdx[e]!;
        edges.push([i, j]);
        if (hasWeights) {
          weights.push(graph.edgeWeights![e]!);
        }
      }
    }
  }

  // Rebuild CSR from surviving edges
  const result = buildCSR(
    edges,
    graph.numNodes,
    hasWeights ? weights : undefined,
  );

  // Preserve node features
  return {
    ...result,
    nodeFeatures: graph.nodeFeatures,
    featureDim: graph.featureDim,
  };
}
