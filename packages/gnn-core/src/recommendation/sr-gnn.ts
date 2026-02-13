// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — SR-GNN (Wu et al. 2019)
// Session-based Recommendation with Graph Neural Networks.
// ---------------------------------------------------------------------------

import type { Graph, GRUWeights, SessionGraphData } from '../types.js';
import { gruCell, dot, softmax } from '../tensor.js';
import { buildCSR } from '../graph.js';

// ---------------------------------------------------------------------------
// 1. buildSessionGraph — Convert click sequence to directed session graph
// ---------------------------------------------------------------------------

/**
 * Build a directed session graph from a click sequence.
 *
 * Algorithm:
 * Given a click sequence [a, b, c, a, d]:
 * 1. Extract unique items and assign contiguous local IDs.
 * 2. Create directed edges from consecutive clicks:
 *    (a->b), (b->c), (c->a), (a->d)
 * 3. Deduplicate edges (keep unique directed pairs).
 * 4. Build CSR graph with local IDs.
 *
 * @param session - Array of item IDs in click order.
 * @returns SessionGraphData with the directed graph, mappings, and session items.
 */
export function buildSessionGraph(session: number[]): SessionGraphData {
  // Extract unique items preserving first-occurrence order
  const nodeMapping = new Map<number, number>(); // original -> local
  const reverseMapping: number[] = [];           // local -> original

  for (const item of session) {
    if (!nodeMapping.has(item)) {
      const localId = reverseMapping.length;
      nodeMapping.set(item, localId);
      reverseMapping.push(item);
    }
  }

  const numNodes = reverseMapping.length;

  // Build directed edges from consecutive clicks, deduplicating
  const edgeSet = new Set<string>();
  const edges: [number, number][] = [];

  for (let i = 0; i < session.length - 1; i++) {
    const src = nodeMapping.get(session[i]!)!;
    const dst = nodeMapping.get(session[i + 1]!)!;
    const key = `${src},${dst}`;

    if (!edgeSet.has(key)) {
      edgeSet.add(key);
      edges.push([src, dst]);
    }
  }

  // Build CSR graph from directed edges
  const graph = buildCSR(edges, numNodes);

  // Session items in local ID order (preserving sequence order)
  const sessionItems = session.map((item) => nodeMapping.get(item)!);

  return {
    graph,
    nodeMapping,
    reverseMapping,
    sessionItems,
  };
}

// ---------------------------------------------------------------------------
// 2. srGNNForward — Gated GNN + hybrid attention session embedding
// ---------------------------------------------------------------------------

/**
 * SR-GNN forward pass: 1-step Gated GNN with hybrid attention.
 *
 * Algorithm:
 * 1. **Gated GNN propagation** (1 step):
 *    - For each node, aggregate messages from directed neighbors.
 *    - Update node hidden state using a GRU cell:
 *      message_u = SUM_{v in N(u)} h_v   (incoming messages)
 *      h_u' = GRU(message_u, h_u)
 *
 * 2. **Hybrid attention** for session embedding:
 *    - Compute attention scores for each session node relative to the last-clicked item:
 *      alpha_i = softmax(q^T * sigma(W_1 * h_i + W_2 * h_last + c))
 *    - Global session preference: s_g = SUM alpha_i * h_i
 *    - Session embedding: s = s_g + h_last (global + local interest)
 *
 * W_attn layout (packed):
 *   [W_1 (featureDim x featureDim) | W_2 (featureDim x featureDim) | c (featureDim) | q (featureDim)]
 *   Total: 2 * featureDim^2 + 2 * featureDim
 *
 * @param sessionGraph - Directed session graph in CSR format.
 * @param X - Node feature/embedding matrix, row-major (numNodes x featureDim).
 * @param W_gru - GRU weight matrices for the gated update.
 * @param W_attn - Packed attention weights [W_1 | W_2 | c | q].
 * @param featureDim - Dimension of node features/embeddings.
 * @param lastItemIdx - Local index of the last-clicked item in the session.
 * @returns Session embedding vector of length featureDim.
 */
export function srGNNForward(
  sessionGraph: Graph,
  X: Float64Array,
  W_gru: GRUWeights,
  W_attn: Float64Array,
  featureDim: number,
  lastItemIdx: number,
): Float64Array {
  const numNodes = sessionGraph.numNodes;

  // ---- Step 1: Gated GNN propagation (1 step) ----

  // Copy initial hidden states from X
  const H = new Float64Array(X);

  // For each node, aggregate incoming messages from neighbors
  for (let u = 0; u < numNodes; u++) {
    // Compute message for node u: sum of neighbor hidden states
    const message = new Float64Array(featureDim);
    const start = sessionGraph.rowPtr[u]!;
    const end = sessionGraph.rowPtr[u + 1]!;

    for (let e = start; e < end; e++) {
      const v = sessionGraph.colIdx[e]!;
      const vOff = v * featureDim;
      for (let d = 0; d < featureDim; d++) {
        message[d] = message[d]! + X[vOff + d]!;
      }
    }

    // GRU update: h_u' = GRU(message, h_u_prev)
    const hPrev = X.slice(u * featureDim, (u + 1) * featureDim);
    const hNew = gruCell(
      message,
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

    // Store updated hidden state
    const uOff = u * featureDim;
    for (let d = 0; d < featureDim; d++) {
      H[uOff + d] = hNew[d]!;
    }
  }

  // ---- Step 2: Hybrid attention for session embedding ----

  // Unpack attention weights:
  // W_1: featureDim x featureDim (at offset 0)
  // W_2: featureDim x featureDim (at offset featureDim^2)
  // c:   featureDim              (at offset 2 * featureDim^2)
  // q:   featureDim              (at offset 2 * featureDim^2 + featureDim)
  const fd2 = featureDim * featureDim;
  const W_1 = W_attn.slice(0, fd2);
  const W_2 = W_attn.slice(fd2, 2 * fd2);
  const c = W_attn.slice(2 * fd2, 2 * fd2 + featureDim);
  const q = W_attn.slice(2 * fd2 + featureDim, 2 * fd2 + 2 * featureDim);

  // Get last item hidden state
  const hLast = H.slice(lastItemIdx * featureDim, (lastItemIdx + 1) * featureDim);

  // Precompute W_2 * h_last (featureDim vector)
  const W2hLast = new Float64Array(featureDim);
  for (let i = 0; i < featureDim; i++) {
    let val = 0;
    for (let j = 0; j < featureDim; j++) {
      val += W_2[i * featureDim + j]! * hLast[j]!;
    }
    W2hLast[i] = val;
  }

  // Compute attention scores for each session node
  const attnScores = new Float64Array(numNodes);

  for (let i = 0; i < numNodes; i++) {
    // W_1 * h_i
    const hi = H.slice(i * featureDim, (i + 1) * featureDim);
    const W1hi = new Float64Array(featureDim);
    for (let r = 0; r < featureDim; r++) {
      let val = 0;
      for (let j = 0; j < featureDim; j++) {
        val += W_1[r * featureDim + j]! * hi[j]!;
      }
      W1hi[r] = val;
    }

    // sigma(W_1 * h_i + W_2 * h_last + c) — using sigmoid as the activation
    const activated = new Float64Array(featureDim);
    for (let d = 0; d < featureDim; d++) {
      const v = W1hi[d]! + W2hLast[d]! + c[d]!;
      // Numerically stable sigmoid
      if (v >= 0) {
        activated[d] = 1.0 / (1.0 + Math.exp(-v));
      } else {
        const ev = Math.exp(v);
        activated[d] = ev / (1.0 + ev);
      }
    }

    // alpha_i = q^T * activated
    attnScores[i] = dot(q, activated);
  }

  // Softmax over attention scores
  const attnWeights = softmax(attnScores, numNodes);

  // Compute global session preference: s_g = SUM alpha_i * h_i
  const sGlobal = new Float64Array(featureDim);
  for (let i = 0; i < numNodes; i++) {
    const weight = attnWeights[i]!;
    const iOff = i * featureDim;
    for (let d = 0; d < featureDim; d++) {
      sGlobal[d] = sGlobal[d]! + weight * H[iOff + d]!;
    }
  }

  // Session embedding: s = s_g + h_last (global preference + current interest)
  const sessionEmbedding = new Float64Array(featureDim);
  for (let d = 0; d < featureDim; d++) {
    sessionEmbedding[d] = sGlobal[d]! + hLast[d]!;
  }

  return sessionEmbedding;
}
