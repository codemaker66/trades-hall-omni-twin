// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-7: Exphormer — O(N+E) Sparse Graph Attention
// Shirzad et al. 2023 — Sparse attention via local + expander + virtual nodes
// ---------------------------------------------------------------------------

import type { Graph, PRNG } from '../types.js';
import { getNeighbors } from '../graph.js';
import { softmax } from '../tensor.js';

// ---------------------------------------------------------------------------
// 1. generateExpanderEdges — Random regular expander graph
// ---------------------------------------------------------------------------

/**
 * Generate edges for a random regular expander graph.
 *
 * Algorithm:
 * For each node, connect it to `degree` other nodes chosen uniformly at random
 * (avoiding self-loops). This approximates a random d-regular graph which is
 * an expander with high probability (Friedman 2008).
 *
 * The resulting graph is directed (each node has exactly `degree` outgoing edges).
 * To make it undirected, the caller can union (src,dst) with (dst,src).
 *
 * @param numNodes - Number of nodes in the graph.
 * @param degree   - Number of random connections per node.
 * @param rng      - Deterministic PRNG.
 * @returns Tuple [src, dst] of Uint32Arrays, each of length numNodes * degree.
 */
export function generateExpanderEdges(
  numNodes: number,
  degree: number,
  rng: PRNG,
): [Uint32Array, Uint32Array] {
  const totalEdges = numNodes * degree;
  const src = new Uint32Array(totalEdges);
  const dst = new Uint32Array(totalEdges);

  let idx = 0;
  for (let i = 0; i < numNodes; i++) {
    // For each node, pick `degree` random distinct neighbors (no self-loops)
    const chosen = new Set<number>();
    let attempts = 0;
    const maxAttempts = degree * 10; // safety bound for small graphs

    while (chosen.size < degree && attempts < maxAttempts) {
      const j = Math.floor(rng() * numNodes);
      if (j !== i && !chosen.has(j)) {
        chosen.add(j);
      }
      attempts++;
    }

    for (const j of chosen) {
      src[idx] = i;
      dst[idx] = j;
      idx++;
    }
  }

  // If some nodes couldn't fill all `degree` slots (very small graph),
  // truncate the arrays to the actual number of edges produced.
  if (idx < totalEdges) {
    return [src.slice(0, idx), dst.slice(0, idx)];
  }

  return [src, dst];
}

// ---------------------------------------------------------------------------
// 2. exphormerAttention — Sparse attention with local + expander + virtual
// ---------------------------------------------------------------------------

/**
 * Exphormer sparse attention layer.
 *
 * Combines three types of attention connections into a single sparse
 * multi-head attention mechanism:
 *
 * 1. **Local edges**: edges from the original graph (captures local structure).
 * 2. **Expander edges**: edges from a random expander graph (provides global
 *    reach in O(1) hops with high probability).
 * 3. **Virtual nodes**: a small set of global nodes connected to all real nodes
 *    (provides a global information bottleneck).
 *
 * For each real node i, the attention neighborhood N(i) is the union of:
 *   - Local neighbors from the graph
 *   - Expander neighbors
 *   - All virtual nodes
 *
 * For each virtual node v, N(v) = all real nodes + all other virtual nodes.
 *
 * Attention is computed only over these sparse neighborhoods, yielding
 * O(N + E_local + E_expander + N * numVirtual) complexity instead of O(N^2).
 *
 * @param graph           - Original CSR graph (local edges).
 * @param X               - Node features (numNodes x dim), row-major.
 * @param expanderEdges   - [src, dst] arrays from generateExpanderEdges.
 * @param numVirtualNodes - Number of global virtual nodes.
 * @param W_Q             - Query weight (dim x dim).
 * @param W_K             - Key weight (dim x dim).
 * @param W_V             - Value weight (dim x dim).
 * @param dim             - Feature dimension.
 * @param heads           - Number of attention heads.
 * @returns Updated features for real nodes only (numNodes x dim).
 */
export function exphormerAttention(
  graph: Graph,
  X: Float64Array,
  expanderEdges: [Uint32Array, Uint32Array],
  numVirtualNodes: number,
  W_Q: Float64Array,
  W_K: Float64Array,
  W_V: Float64Array,
  dim: number,
  heads: number,
): Float64Array {
  const n = graph.numNodes;
  const totalNodes = n + numVirtualNodes;
  const headDim = dim / heads;
  const scale = 1.0 / Math.sqrt(headDim);

  // Extend feature matrix to include virtual nodes (initialized to zero)
  const Xext = new Float64Array(totalNodes * dim);
  Xext.set(X);
  // Virtual node features remain zero-initialized (learnable in a full model)

  // Project all nodes (real + virtual) to Q, K, V
  // Q = Xext * W_Q, K = Xext * W_K, V = Xext * W_V  (totalNodes x dim)
  const Q = new Float64Array(totalNodes * dim);
  const K = new Float64Array(totalNodes * dim);
  const V = new Float64Array(totalNodes * dim);

  for (let i = 0; i < totalNodes; i++) {
    for (let od = 0; od < dim; od++) {
      let qVal = 0;
      let kVal = 0;
      let vVal = 0;
      for (let id = 0; id < dim; id++) {
        const x = Xext[i * dim + id]!;
        qVal += x * W_Q[id * dim + od]!;
        kVal += x * W_K[id * dim + od]!;
        vVal += x * W_V[id * dim + od]!;
      }
      Q[i * dim + od] = qVal;
      K[i * dim + od] = kVal;
      V[i * dim + od] = vVal;
    }
  }

  // Build per-node neighbor sets (union of local + expander + virtual)
  // For efficiency, use adjacency lists stored as arrays.
  const neighbors: number[][] = new Array(totalNodes);
  for (let i = 0; i < totalNodes; i++) {
    neighbors[i] = [];
  }

  // Local edges (from the original graph)
  for (let i = 0; i < n; i++) {
    const { indices } = getNeighbors(graph, i);
    for (let e = 0; e < indices.length; e++) {
      neighbors[i]!.push(indices[e]!);
    }
  }

  // Expander edges
  const [expSrc, expDst] = expanderEdges;
  for (let e = 0; e < expSrc.length; e++) {
    const s = expSrc[e]!;
    const d = expDst[e]!;
    if (s < n && d < n) {
      neighbors[s]!.push(d);
    }
  }

  // Virtual node connections
  // Each real node is connected to all virtual nodes
  for (let i = 0; i < n; i++) {
    for (let v = 0; v < numVirtualNodes; v++) {
      neighbors[i]!.push(n + v);   // real -> virtual
      neighbors[n + v]!.push(i);   // virtual -> real
    }
  }

  // Virtual nodes also connect to each other
  for (let v1 = 0; v1 < numVirtualNodes; v1++) {
    for (let v2 = 0; v2 < numVirtualNodes; v2++) {
      if (v1 !== v2) {
        neighbors[n + v1]!.push(n + v2);
      }
    }
  }

  // Deduplicate neighbor lists (union semantics)
  for (let i = 0; i < totalNodes; i++) {
    neighbors[i] = Array.from(new Set(neighbors[i]!));
  }

  // Compute sparse multi-head attention
  const output = new Float64Array(totalNodes * dim);

  for (let h = 0; h < heads; h++) {
    const hOffset = h * headDim;

    for (let i = 0; i < totalNodes; i++) {
      const nbrs = neighbors[i]!;
      const numNbrs = nbrs.length;

      if (numNbrs === 0) continue;

      // Compute attention scores for this node's neighborhood
      const scores = new Float64Array(numNbrs);

      for (let ni = 0; ni < numNbrs; ni++) {
        const j = nbrs[ni]!;
        let dot = 0;
        for (let d = 0; d < headDim; d++) {
          dot += Q[i * dim + hOffset + d]! * K[j * dim + hOffset + d]!;
        }
        scores[ni] = dot * scale;
      }

      // Softmax over the neighborhood
      const attnWeights = softmax(scores, numNbrs);

      // Weighted aggregation of values
      for (let ni = 0; ni < numNbrs; ni++) {
        const j = nbrs[ni]!;
        const a = attnWeights[ni]!;
        for (let d = 0; d < headDim; d++) {
          output[i * dim + hOffset + d] = output[i * dim + hOffset + d]! + a * V[j * dim + hOffset + d]!;
        }
      }
    }
  }

  // Return only real node features (exclude virtual nodes)
  return output.slice(0, n * dim);
}
