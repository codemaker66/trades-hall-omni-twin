// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-5: Graph Generation
// GraphRNN — Sequential graph generation (You et al., 2018).
//
// Generates graphs one node at a time using two GRU levels:
//   1. Node-level GRU decides whether to add a new node.
//   2. Edge-level GRU decides edges from the new node to prior nodes.
// Uses BFS ordering to reduce the edge horizon, so each new node only
// considers connecting to the last `edgeHorizon` nodes.
// ---------------------------------------------------------------------------

import type { PRNG, GraphRNNWeights, GraphRNNConfig, GeneratedGraph, GRUWeights } from '../types.js';
import { gruCell, sigmoid, matVecMul, add } from '../tensor.js';

/**
 * Helper: run a single GRU step using the named-weight struct.
 */
function gruStep(
  x: Float64Array,
  hPrev: Float64Array,
  w: GRUWeights,
): Float64Array {
  return gruCell(
    x, hPrev,
    w.W_z, w.U_z, w.b_z,
    w.W_r, w.U_r, w.b_r,
    w.W_h, w.U_h, w.b_h,
  );
}

/**
 * Generate a graph sequentially using GraphRNN.
 *
 * Algorithm:
 * 1. Initialize node-level hidden state h_node to zeros.
 * 2. For each potential new node (up to maxNodes):
 *    a. Feed a start-of-sequence token (ones vector) into the node GRU.
 *    b. Compute a "continue" probability via linear head + sigmoid.
 *       If < 0.5, stop generating (the graph is done).
 *    c. Otherwise add the node. Initialize edge-level hidden state h_edge.
 *    d. For each of the last `edgeHorizon` existing nodes (in reverse BFS order):
 *       - Feed the current edge context into the edge GRU.
 *       - Compute edge probability via linear head + sigmoid.
 *       - Sample the edge from Bernoulli(p).
 *       - Feed the sampled bit back as the next input to the edge GRU.
 * 3. Return the generated adjacency as a flattened n x n Uint8Array.
 *
 * @param weights - Learned weights for the node- and edge-level GRUs and output heads.
 * @param config  - Generation hyper-parameters (maxNodes, hiddenDim, edgeHorizon).
 * @param rng     - Deterministic seeded PRNG returning values in [0, 1).
 * @returns       - GeneratedGraph { adjacency, numNodes }.
 */
export function graphRNNGenerate(
  weights: GraphRNNWeights,
  config: GraphRNNConfig,
  rng: PRNG,
): GeneratedGraph {
  const { maxNodes, hiddenDim, edgeHorizon } = config;

  // Track adjacency as a dynamic list; we'll flatten at the end.
  const adjRows: Uint8Array[] = [];

  // Node-level GRU hidden state
  let hNode = new Float64Array(hiddenDim);

  // Start-of-sequence token for the node GRU (ones vector, dim = 1)
  const sosNode = new Float64Array(1);
  sosNode[0] = 1.0;

  let numNodes = 0;

  for (let nodeIdx = 0; nodeIdx < maxNodes; nodeIdx++) {
    // --- Node-level step ---
    // Build node input: if first node, use SOS; otherwise use the edge
    // sequence output (encoded as a single scalar = fraction of edges added).
    let nodeInput: Float64Array;
    if (nodeIdx === 0) {
      nodeInput = sosNode;
    } else {
      // Summarise previous edge decisions as a single scalar: mean of adjacency row
      const prevRow = adjRows[nodeIdx - 1]!;
      let edgeSum = 0;
      for (let k = 0; k < prevRow.length; k++) {
        edgeSum += prevRow[k]!;
      }
      nodeInput = new Float64Array(1);
      nodeInput[0] = prevRow.length > 0 ? edgeSum / prevRow.length : 0;
    }

    hNode = gruStep(nodeInput, hNode, weights.nodeGRU) as Float64Array<ArrayBuffer>;

    // Output probability: sigmoid( W_out * h_node + b_out )
    const nodeLogits = add(
      matVecMul(weights.nodeOutputW, hNode, 1, hiddenDim),
      weights.nodeOutputBias,
    );
    const nodeProb = sigmoid(nodeLogits)[0]!;

    // Stop condition (skip for very first node — always generate at least 1)
    if (nodeIdx > 0 && nodeProb < 0.5) {
      break;
    }

    // --- This node is accepted ---
    numNodes++;
    const row = new Uint8Array(nodeIdx); // edges to prior nodes

    if (nodeIdx > 0) {
      // Edge-level GRU: generate adjacency to previous nodes in the BFS window.
      const windowStart = Math.max(0, nodeIdx - edgeHorizon);
      let hEdge = new Float64Array(hiddenDim);

      // SOS token for edge GRU
      const sosEdge = new Float64Array(1);
      sosEdge[0] = 1.0;
      let edgeInput = sosEdge;

      for (let prev = nodeIdx - 1; prev >= windowStart; prev--) {
        hEdge = gruStep(edgeInput, hEdge, weights.edgeGRU) as Float64Array<ArrayBuffer>;

        // Edge probability
        const edgeLogits = add(
          matVecMul(weights.edgeOutputW, hEdge, 1, hiddenDim),
          weights.edgeOutputBias,
        );
        const edgeProb = sigmoid(edgeLogits)[0]!;

        // Sample
        const hasEdge = rng() < edgeProb ? 1 : 0;
        row[prev] = hasEdge;

        // Feed sampled bit back as next input
        edgeInput = new Float64Array(1);
        edgeInput[0] = hasEdge;
      }
    }

    adjRows.push(row);
  }

  // Build flat n x n adjacency (symmetric undirected)
  const n = numNodes;
  const adjacency = new Uint8Array(n * n);

  for (let i = 0; i < n; i++) {
    const row = adjRows[i]!;
    for (let j = 0; j < row.length; j++) {
      if (row[j]) {
        adjacency[i * n + j] = 1;
        adjacency[j * n + i] = 1; // symmetric
      }
    }
  }

  return { adjacency, numNodes: n };
}
