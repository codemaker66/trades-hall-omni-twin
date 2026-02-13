// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-5: Graph Generation
// Graph-to-Layout — Convert graph topology into spatial coordinates.
//
// Three complementary approaches:
// 1. Force-directed layout (Fruchterman-Reingold) for classical positioning.
// 2. GNN coordinate decoder — MLP that predicts (x, y, theta) from features.
// 3. Constrained optimisation — project positions to satisfy spatial constraints.
// ---------------------------------------------------------------------------

import type { PRNG, Graph, ForceDirectedConfig } from '../types.js';
import { matVecMul, add, relu } from '../tensor.js';

// ---------------------------------------------------------------------------
// 1. forceDirectedLayout — Fruchterman-Reingold
// ---------------------------------------------------------------------------

/**
 * Compute a 2D force-directed layout using the Fruchterman-Reingold algorithm.
 *
 * Algorithm:
 * 1. Initialise node positions randomly within a unit square.
 * 2. For each iteration:
 *    a. Compute repulsive forces between all pairs of nodes:
 *       F_rep = -k_rep / d^2  (directed away from the other node).
 *    b. Compute attractive forces along edges:
 *       F_att = d^2 / k_att  (directed toward the other node).
 *    c. Update positions: pos += lr * displacement (clamped to temperature).
 *    d. Anneal temperature: lr *= 0.95.
 * 3. Return final positions as Float64Array (numNodes * 2), row-major [x, y].
 *
 * @param graph  - Input CSR Graph (edges define attraction).
 * @param config - Force-directed hyper-parameters.
 * @param rng    - Deterministic PRNG for initial positions.
 * @returns      - Float64Array of length numNodes * 2 (x, y per node).
 */
export function forceDirectedLayout(
  graph: Graph,
  config: ForceDirectedConfig,
  rng: PRNG,
): Float64Array {
  const n = graph.numNodes;
  const {
    iterations,
    learningRate: initialLR,
    attractionStrength: kAtt,
    repulsionStrength: kRep,
    idealEdgeLength,
  } = config;

  // Initialise positions randomly in [0, idealEdgeLength * sqrt(n)]
  const spread = idealEdgeLength * Math.sqrt(n);
  const pos = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) {
    pos[i * 2] = rng() * spread;
    pos[i * 2 + 1] = rng() * spread;
  }

  let lr = initialLR;

  // Displacement accumulators
  const disp = new Float64Array(n * 2);

  for (let iter = 0; iter < iterations; iter++) {
    // Reset displacements
    disp.fill(0);

    // --- Repulsive forces: all pairs ---
    for (let i = 0; i < n; i++) {
      const ix = pos[i * 2]!;
      const iy = pos[i * 2 + 1]!;

      for (let j = i + 1; j < n; j++) {
        const dx = ix - pos[j * 2]!;
        const dy = iy - pos[j * 2 + 1]!;
        const distSq = dx * dx + dy * dy;
        const dist = Math.sqrt(distSq) || 1e-10;

        // Repulsion magnitude: k_rep / d^2
        const force = kRep / (distSq || 1e-10);
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;

        disp[i * 2] = disp[i * 2]! + fx;
        disp[i * 2 + 1] = disp[i * 2 + 1]! + fy;
        disp[j * 2] = disp[j * 2]! - fx;
        disp[j * 2 + 1] = disp[j * 2 + 1]! - fy;
      }
    }

    // --- Attractive forces: along edges ---
    for (let i = 0; i < n; i++) {
      const start = graph.rowPtr[i]!;
      const end = graph.rowPtr[i + 1]!;
      const ix = pos[i * 2]!;
      const iy = pos[i * 2 + 1]!;

      for (let e = start; e < end; e++) {
        const j = graph.colIdx[e]!;
        // Only process each undirected edge once (i < j)
        if (j <= i) continue;

        const dx = ix - pos[j * 2]!;
        const dy = iy - pos[j * 2 + 1]!;
        const distSq = dx * dx + dy * dy;
        const dist = Math.sqrt(distSq) || 1e-10;

        // Attraction magnitude: d^2 / k_att
        const force = distSq / (kAtt || 1e-10);
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;

        // Attractive: pull i toward j and j toward i
        disp[i * 2] = disp[i * 2]! - fx;
        disp[i * 2 + 1] = disp[i * 2 + 1]! - fy;
        disp[j * 2] = disp[j * 2]! + fx;
        disp[j * 2 + 1] = disp[j * 2 + 1]! + fy;
      }
    }

    // --- Update positions with temperature-limited step ---
    for (let i = 0; i < n; i++) {
      const dx = disp[i * 2]!;
      const dy = disp[i * 2 + 1]!;
      const mag = Math.sqrt(dx * dx + dy * dy) || 1e-10;
      // Clamp displacement to temperature (lr acts as temperature)
      const scale = Math.min(lr, mag) / mag;
      pos[i * 2] = pos[i * 2]! + dx * scale;
      pos[i * 2 + 1] = pos[i * 2 + 1]! + dy * scale;
    }

    // --- Anneal temperature ---
    lr *= 0.95;
  }

  return pos;
}

// ---------------------------------------------------------------------------
// 2. gnnCoordinateDecoder — MLP predicting (x, y, theta) per node
// ---------------------------------------------------------------------------

/**
 * Decode spatial coordinates from node feature embeddings using a simple MLP.
 *
 * Architecture:
 *   hidden = ReLU(W_1 * X_i + b_1)
 *   (x, y, theta)_i = W_2 * hidden + b_2
 *
 * The weight matrix W is packed as:
 *   [W_1 (hiddenDim × inDim), b_1 (hiddenDim), W_2 (3 × hiddenDim), b_2 (3)]
 * where hiddenDim is inferred from the total weight size.
 *
 * @param graph  - Input CSR Graph (uses numNodes).
 * @param X      - Node feature matrix, flat (numNodes × inDim), row-major.
 * @param W      - Packed MLP weights [W_1, b_1, W_2, b_2].
 * @param inDim  - Input feature dimension per node.
 * @returns      - Float64Array of length numNodes × 3 (x, y, rotation per node).
 */
export function gnnCoordinateDecoder(
  graph: Graph,
  X: Float64Array,
  W: Float64Array,
  inDim: number,
): Float64Array {
  const n = graph.numNodes;
  const outDim = 3; // x, y, theta

  // Infer hidden dimension from total weight count:
  // total = hiddenDim * inDim + hiddenDim + 3 * hiddenDim + 3
  // total = hiddenDim * (inDim + 1 + 3) + 3
  // hiddenDim = (total - 3) / (inDim + 4)
  const hiddenDim = (W.length - outDim) / (inDim + 1 + outDim);

  // Parse weight offsets
  let offset = 0;
  const W1 = W.subarray(offset, offset + hiddenDim * inDim);
  offset += hiddenDim * inDim;
  const b1 = W.subarray(offset, offset + hiddenDim);
  offset += hiddenDim;
  const W2 = W.subarray(offset, offset + outDim * hiddenDim);
  offset += outDim * hiddenDim;
  const b2 = W.subarray(offset, offset + outDim);

  const result = new Float64Array(n * outDim);

  for (let i = 0; i < n; i++) {
    // Extract node features
    const xi = X.subarray(i * inDim, (i + 1) * inDim);

    // Layer 1: hidden = ReLU(W1 * xi + b1)
    const hidden = new Float64Array(hiddenDim);
    for (let h = 0; h < hiddenDim; h++) {
      let val = b1[h]!;
      for (let d = 0; d < inDim; d++) {
        val += W1[h * inDim + d]! * xi[d]!;
      }
      hidden[h] = val > 0 ? val : 0; // ReLU
    }

    // Layer 2: output = W2 * hidden + b2
    for (let o = 0; o < outDim; o++) {
      let val = b2[o]!;
      for (let h = 0; h < hiddenDim; h++) {
        val += W2[o * hiddenDim + h]! * hidden[h]!;
      }
      result[i * outDim + o] = val;
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// 3. constrainedOptimization — Project positions to satisfy constraints
// ---------------------------------------------------------------------------

/**
 * Project 2D positions to satisfy spatial constraints through iterative projection.
 *
 * Constraints:
 * 1. Bounding box: keep all nodes within [xMin, yMin, xMax, yMax].
 * 2. Minimum distance: push apart any pair of nodes closer than minDist.
 *
 * Algorithm:
 * For `iterations` rounds:
 *   a. Clamp all positions to the bounding box.
 *   b. For every pair (i, j) with distance < minDist, push both nodes apart
 *      symmetrically along their connecting line so they are exactly minDist apart.
 *
 * @param positions   - Flat Float64Array of length numNodes * 2 (x, y per node).
 * @param constraints - { minDist, bounds: [xMin, yMin, xMax, yMax] }.
 * @param iterations  - Number of projection iterations.
 * @returns           - Adjusted positions (numNodes * 2), new Float64Array.
 */
export function constrainedOptimization(
  positions: Float64Array,
  constraints: { minDist: number; bounds: [number, number, number, number] },
  iterations: number,
): Float64Array {
  const n = positions.length / 2;
  const pos = new Float64Array(positions); // copy
  const { minDist, bounds } = constraints;
  const [xMin, yMin, xMax, yMax] = bounds;
  const minDistSq = minDist * minDist;

  for (let iter = 0; iter < iterations; iter++) {
    // --- Step 1: Clamp to bounding box ---
    for (let i = 0; i < n; i++) {
      let x = pos[i * 2]!;
      let y = pos[i * 2 + 1]!;
      if (x < xMin) x = xMin;
      if (x > xMax) x = xMax;
      if (y < yMin) y = yMin;
      if (y > yMax) y = yMax;
      pos[i * 2] = x;
      pos[i * 2 + 1] = y;
    }

    // --- Step 2: Push apart overlapping pairs ---
    for (let i = 0; i < n; i++) {
      const ix = pos[i * 2]!;
      const iy = pos[i * 2 + 1]!;

      for (let j = i + 1; j < n; j++) {
        const jx = pos[j * 2]!;
        const jy = pos[j * 2 + 1]!;

        const dx = ix - jx;
        const dy = iy - jy;
        const distSq = dx * dx + dy * dy;

        if (distSq < minDistSq && distSq > 0) {
          const dist = Math.sqrt(distSq);
          const overlap = (minDist - dist) / 2;
          const nx = dx / dist;
          const ny = dy / dist;

          // Push apart symmetrically
          pos[i * 2] = pos[i * 2]! + nx * overlap;
          pos[i * 2 + 1] = pos[i * 2 + 1]! + ny * overlap;
          pos[j * 2] = pos[j * 2]! - nx * overlap;
          pos[j * 2 + 1] = pos[j * 2 + 1]! - ny * overlap;
        } else if (distSq === 0) {
          // Coincident points: nudge randomly
          const angle = iter * 2.399 + i * 0.618; // deterministic spiral
          const nudge = minDist / 2;
          pos[i * 2] = pos[i * 2]! + Math.cos(angle) * nudge;
          pos[i * 2 + 1] = pos[i * 2 + 1]! + Math.sin(angle) * nudge;
          pos[j * 2] = pos[j * 2]! - Math.cos(angle) * nudge;
          pos[j * 2 + 1] = pos[j * 2 + 1]! - Math.sin(angle) * nudge;
        }
      }
    }
  }

  // Final bounding box clamp
  for (let i = 0; i < n; i++) {
    let x = pos[i * 2]!;
    let y = pos[i * 2 + 1]!;
    if (x < xMin) x = xMin;
    if (x > xMax) x = xMax;
    if (y < yMin) y = yMin;
    if (y > yMax) y = yMax;
    pos[i * 2] = x;
    pos[i * 2 + 1] = y;
  }

  return pos;
}
