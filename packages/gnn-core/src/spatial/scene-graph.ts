// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-4: Spatial Layout Understanding
// scene-graph.ts — SceneGraphNet with multi-edge-type message functions
//
// Implements a scene-graph neural network where different edge types
// (e.g., "next_to", "facing", "on_top_of") use different message
// transformation weights. Node states are updated via GRU cells to
// maintain persistent representations across message-passing rounds.
// ---------------------------------------------------------------------------

import type { Graph, GRUWeights } from '../types.js';
import { gruCell } from '../tensor.js';

// ---------------------------------------------------------------------------
// 1. sceneGraphForward — Multi-edge-type message passing with GRU updates
// ---------------------------------------------------------------------------

/**
 * SceneGraphNet forward pass: typed message passing with GRU node updates.
 *
 * Algorithm:
 * 1. For each edge (i -> j) with edge type t:
 *    msg_{i,j} = W_msg[t] * h_j
 *    where W_msg[t] is (featureDim x featureDim), specific to edge type t.
 *
 * 2. Aggregate messages per node (sum):
 *    agg_i = sum_{j in N(i)} msg_{i,j}
 *
 * 3. Update each node state via GRU:
 *    h_i' = GRU(agg_i, h_i)
 *    where agg_i is the "input" and h_i is the previous hidden state.
 *
 * The GRU ensures smooth state transitions and prevents information loss
 * across multiple rounds of message passing.
 *
 * @param graph        - CSR Graph encoding the scene graph structure.
 * @param X            - Node features (numNodes x featureDim), row-major.
 * @param edgeTypes    - Per-edge type labels (Uint8Array of length numEdges).
 *                       Each value in [0, numEdgeTypes).
 * @param W_msg        - Array of message weight matrices, one per edge type.
 *                       Each is Float64Array of shape (featureDim x featureDim), row-major.
 * @param W_gru        - GRU weight matrices for node state updates.
 *                       W_z, W_r, W_h are (featureDim x featureDim) — input-to-hidden.
 *                       U_z, U_r, U_h are (featureDim x featureDim) — hidden-to-hidden.
 *                       b_z, b_r, b_h are (featureDim) — biases.
 * @param featureDim   - Dimension of node feature vectors.
 * @param numEdgeTypes - Number of distinct edge types.
 * @returns Updated node features (numNodes x featureDim), row-major.
 */
export function sceneGraphForward(
  graph: Graph,
  X: Float64Array,
  edgeTypes: Uint8Array,
  W_msg: Float64Array[],
  W_gru: GRUWeights,
  featureDim: number,
  numEdgeTypes: number,
): Float64Array {
  const n = graph.numNodes;

  if (n === 0) {
    return new Float64Array(0);
  }

  // Step 1 & 2: Compute typed messages and aggregate per node
  const aggregated = new Float64Array(n * featureDim);

  for (let i = 0; i < n; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;

    for (let e = start; e < end; e++) {
      const j = graph.colIdx[e]!;
      const edgeType = edgeTypes[e]!;

      // Get the weight matrix for this edge type
      // W_msg[edgeType] is (featureDim x featureDim)
      const W_t = W_msg[edgeType]!;

      // Compute message: msg = W_t * h_j
      // Then accumulate into aggregated[i]
      const jOff = j * featureDim;
      const iOff = i * featureDim;

      for (let row = 0; row < featureDim; row++) {
        let val = 0;
        for (let col = 0; col < featureDim; col++) {
          val += W_t[row * featureDim + col]! * X[jOff + col]!;
        }
        aggregated[iOff + row] = aggregated[iOff + row]! + val;
      }
    }
  }

  // Step 3: Update each node state via GRU cell
  // GRU input = aggregated message, hidden state = current node features
  const output = new Float64Array(n * featureDim);

  for (let i = 0; i < n; i++) {
    const iOff = i * featureDim;

    // Extract the aggregated message for node i (GRU input)
    const aggMsg = aggregated.slice(iOff, iOff + featureDim);

    // Extract current hidden state for node i
    const hPrev = X.slice(iOff, iOff + featureDim);

    // GRU cell: h_new = GRU(aggMsg, hPrev)
    const hNew = gruCell(
      aggMsg,
      hPrev,
      W_gru.W_z,
      W_gru.U_z,
      W_gru.b_z,
      W_gru.W_r,
      W_gru.U_r,
      W_gru.b_r,
      W_gru.W_h,
      W_gru.U_h,
      W_gru.b_h,
    );

    // Copy new hidden state to output
    output.set(hNew, iOff);
  }

  return output;
}
