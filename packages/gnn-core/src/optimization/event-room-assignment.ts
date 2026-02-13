// ---------------------------------------------------------------------------
// GNN-8: Combinatorial Optimization — Event-Room Assignment
// Bipartite GNN embedding + Sinkhorn normalization + Hungarian algorithm
// for optimal event-to-room assignment in venue management.
// ---------------------------------------------------------------------------

import type { AssignmentResult, SinkhornConfig } from '../types.js';
import { matMul, dot } from '../tensor.js';

// ---------------------------------------------------------------------------
// 1. sinkhornAssignment — Differentiable soft assignment via Sinkhorn
// ---------------------------------------------------------------------------

/**
 * Compute a doubly-stochastic matrix from a cost matrix using Sinkhorn
 * normalization (Sinkhorn-Knopp algorithm).
 *
 * This provides a differentiable relaxation of the assignment problem:
 *   1. Initialize M_ij = exp(-C_ij / temperature)
 *   2. Iterate:
 *      a. Row normalization: M_ij = M_ij / sum_k M_ik
 *      b. Column normalization: M_ij = M_ij / sum_k M_kj
 *   3. After convergence, M approximates a doubly-stochastic (permutation) matrix.
 *
 * @param costMatrix - Cost matrix (numRows x numCols), row-major Float64Array.
 * @param numRows - Number of rows (events).
 * @param numCols - Number of columns (rooms).
 * @param config - Sinkhorn parameters (iterations, temperature, epsilon).
 * @returns Float64Array doubly-stochastic matrix (numRows x numCols), row-major.
 */
export function sinkhornAssignment(
  costMatrix: Float64Array,
  numRows: number,
  numCols: number,
  config: SinkhornConfig,
): Float64Array {
  const { iterations, temperature, epsilon } = config;

  // Initialize: M_ij = exp(-C_ij / temperature)
  const M = new Float64Array(numRows * numCols);
  for (let i = 0; i < numRows; i++) {
    for (let j = 0; j < numCols; j++) {
      M[i * numCols + j] = Math.exp(-costMatrix[i * numCols + j]! / temperature);
    }
  }

  // Sinkhorn iterations: alternating row and column normalization
  for (let iter = 0; iter < iterations; iter++) {
    // Row normalization: divide each element by its row sum
    for (let i = 0; i < numRows; i++) {
      let rowSum = 0;
      for (let j = 0; j < numCols; j++) {
        rowSum += M[i * numCols + j]!;
      }
      const invRowSum = rowSum > epsilon ? 1 / rowSum : 0;
      for (let j = 0; j < numCols; j++) {
        M[i * numCols + j] = M[i * numCols + j]! * invRowSum;
      }
    }

    // Column normalization: divide each element by its column sum
    for (let j = 0; j < numCols; j++) {
      let colSum = 0;
      for (let i = 0; i < numRows; i++) {
        colSum += M[i * numCols + j]!;
      }
      const invColSum = colSum > epsilon ? 1 / colSum : 0;
      for (let i = 0; i < numRows; i++) {
        M[i * numCols + j] = M[i * numCols + j]! * invColSum;
      }
    }
  }

  return M;
}

// ---------------------------------------------------------------------------
// 2. hungarianAlgorithm — Exact O(n^3) optimal assignment
// ---------------------------------------------------------------------------

/**
 * Solve the linear assignment problem for an n x n cost matrix using the
 * Hungarian algorithm (Kuhn-Munkres).
 *
 * Algorithm (O(n^3)):
 *   1. Row reduction: subtract row minimum from each row.
 *   2. Column reduction: subtract column minimum from each column.
 *   3. Cover all zeros with minimum number of lines.
 *   4. If n lines needed, optimal assignment found.
 *   5. Otherwise, find minimum uncovered value, subtract from uncovered,
 *      add to doubly-covered, repeat from step 3.
 *
 * Uses the sequential shortest augmenting path variant for O(n^3) complexity.
 *
 * @param costMatrix - Square cost matrix (n x n), row-major Float64Array.
 * @param n - Matrix dimension.
 * @returns AssignmentResult with optimal assignment, total cost, and feasibility.
 */
export function hungarianAlgorithm(
  costMatrix: Float64Array,
  n: number,
): AssignmentResult {
  if (n === 0) {
    return { assignment: new Uint32Array(0), cost: 0, feasible: true };
  }

  if (n === 1) {
    return {
      assignment: new Uint32Array([0]),
      cost: costMatrix[0]!,
      feasible: true,
    };
  }

  // Use the shortest augmenting path variant (Jonker-Volgenant style)
  // u[i] = potential for row i, v[j] = potential for column j
  // Reduced cost: c_ij - u_i - v_j >= 0 at optimality
  const INF = 1e18;

  // Potentials
  const u = new Float64Array(n + 1); // 1-indexed for convenience
  const v = new Float64Array(n + 1);

  // p[j] = row assigned to column j (0 = unassigned, 1-indexed rows)
  const p = new Int32Array(n + 1);

  // way[j] = previous column in the augmenting path to column j
  const way = new Int32Array(n + 1);

  for (let i = 1; i <= n; i++) {
    // Start augmenting path from row i
    p[0] = i;
    let j0 = 0; // virtual column 0 is always "assigned" to current row

    const minv = new Float64Array(n + 1);
    minv.fill(INF);
    const used = new Uint8Array(n + 1);

    // Find augmenting path using Dijkstra-like shortest path
    do {
      used[j0] = 1;
      let i0 = p[j0]!;
      let delta = INF;
      let j1 = -1;

      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;

        // Reduced cost: c[i0-1][j-1] - u[i0] - v[j]
        const cur = costMatrix[(i0 - 1) * n + (j - 1)]! - u[i0]! - v[j]!;

        if (cur < minv[j]!) {
          minv[j] = cur;
          way[j] = j0;
        }

        if (minv[j]! < delta) {
          delta = minv[j]!;
          j1 = j;
        }
      }

      // Update potentials
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]!] = u[p[j]!]! + delta;
          v[j] = v[j]! - delta;
        } else {
          minv[j] = minv[j]! - delta;
        }
      }

      j0 = j1;
    } while (p[j0]! !== 0);

    // Trace back the augmenting path and update assignment
    do {
      const j1 = way[j0]!;
      p[j0] = p[j1]!;
      j0 = j1;
    } while (j0 !== 0);
  }

  // Extract assignment: row i is assigned to column assignment[i]
  const assignment = new Uint32Array(n);
  let totalCost = 0;

  for (let j = 1; j <= n; j++) {
    if (p[j]! > 0) {
      const row = p[j]! - 1;
      const col = j - 1;
      assignment[row] = col;
      totalCost += costMatrix[row * n + col]!;
    }
  }

  return { assignment, cost: totalCost, feasible: true };
}

// ---------------------------------------------------------------------------
// 3. bipartiteGNNAssignment — GNN-based cost matrix from bipartite embeddings
// ---------------------------------------------------------------------------

/**
 * Compute a cost matrix for event-room assignment using bipartite GNN embeddings.
 *
 * The approach:
 *   1. Project event features through a learned weight matrix: e_i = W_event * x_event_i
 *   2. Project room features through a learned weight matrix: r_j = W_room * x_room_j
 *   3. Cost matrix: C_ij = -dot(e_i, r_j) (negative dot product as cost,
 *      so higher similarity = lower cost).
 *
 * The weight matrix W is laid out as [W_event (featureDim x outDim) | W_room (featureDim x outDim)],
 * i.e., W has shape (2 * featureDim) x outDim, with the first featureDim rows for events
 * and the next featureDim rows for rooms.
 *
 * @param eventFeatures - Event feature matrix (numEvents x featureDim), row-major.
 * @param roomFeatures - Room feature matrix (numRooms x featureDim), row-major.
 * @param numEvents - Number of events.
 * @param numRooms - Number of rooms.
 * @param featureDim - Input feature dimension for both events and rooms.
 * @param W - Combined weight matrix [(featureDim x outDim) for events, (featureDim x outDim) for rooms].
 * @param outDim - Output embedding dimension.
 * @returns Float64Array cost matrix (numEvents x numRooms), row-major.
 */
export function bipartiteGNNAssignment(
  eventFeatures: Float64Array,
  roomFeatures: Float64Array,
  numEvents: number,
  numRooms: number,
  featureDim: number,
  W: Float64Array,
  outDim: number,
): Float64Array {
  // Split W into W_event and W_room
  // W_event: first featureDim * outDim elements (featureDim x outDim)
  // W_room: next featureDim * outDim elements (featureDim x outDim)
  const wSize = featureDim * outDim;
  const W_event = W.slice(0, wSize);
  const W_room = W.slice(wSize, 2 * wSize);

  // Compute event embeddings: E = eventFeatures * W_event
  // (numEvents x featureDim) * (featureDim x outDim) -> (numEvents x outDim)
  const eventEmbeddings = matMul(eventFeatures, W_event, numEvents, featureDim, outDim);

  // Compute room embeddings: R = roomFeatures * W_room
  // (numRooms x featureDim) * (featureDim x outDim) -> (numRooms x outDim)
  const roomEmbeddings = matMul(roomFeatures, W_room, numRooms, featureDim, outDim);

  // Cost matrix: C_ij = -dot(e_i, r_j)
  const costMatrix = new Float64Array(numEvents * numRooms);

  for (let i = 0; i < numEvents; i++) {
    for (let j = 0; j < numRooms; j++) {
      let d = 0;
      for (let k = 0; k < outDim; k++) {
        d += eventEmbeddings[i * outDim + k]! * roomEmbeddings[j * outDim + k]!;
      }
      costMatrix[i * numRooms + j] = -d;
    }
  }

  return costMatrix;
}
