// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-5: Graph Generation
// GRAN — Graph Recurrent Attention Network (Liao et al., 2019).
//
// Generates graphs block-wise: at each step a block of `blockSize` new nodes
// is added and edges are sampled between new nodes and all existing nodes
// using a mixture-of-Bernoulli parameterisation.
// ---------------------------------------------------------------------------

import type { PRNG, GRANConfig, GeneratedGraph } from '../types.js';

/**
 * Numerically stable sigmoid for a single scalar value.
 */
function sigmoidScalar(x: number): number {
  if (x >= 0) {
    const ez = Math.exp(-x);
    return 1 / (1 + ez);
  }
  const ez = Math.exp(x);
  return ez / (1 + ez);
}

/**
 * Concatenate two Float64Arrays into a new one.
 */
function concat(a: Float64Array, b: Float64Array): Float64Array {
  const out = new Float64Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

/**
 * Generate a graph block-wise using the GRAN procedure.
 *
 * Algorithm:
 * 1. Start with an empty node set.
 * 2. At each step, add `blockSize` new nodes (or fewer if hitting maxNodes).
 * 3. For each new node, sample a feature vector h_i ~ N(0, 1) projected through
 *    W_node (hiddenDim × hiddenDim).
 * 4. For each (new_node, existing_node) pair, compute edge probability using
 *    a mixture of K Bernoulli components:
 *      p(edge) = sum_k  pi_k * sigmoid( W_edge_k^T [h_i || h_j] )
 *    where W_edge is sliced into K mixture components and pi_k = 1/K (uniform).
 * 5. Accept an edge when the probability exceeds 0.5.
 * 6. Also sample intra-block edges among new nodes the same way.
 * 7. Continue until maxNodes is reached.
 *
 * @param config  - GRAN hyper-parameters { maxNodes, blockSize, hiddenDim, numMixtures }.
 * @param W_node  - Node projection weights, hiddenDim × hiddenDim row-major.
 * @param W_edge  - Edge scoring weights, numMixtures × (2 * hiddenDim) row-major.
 * @param rng     - Deterministic seeded PRNG returning values in [0, 1).
 * @returns       - GeneratedGraph { adjacency, numNodes }.
 */
export function granGenerate(
  config: GRANConfig,
  W_node: Float64Array,
  W_edge: Float64Array,
  rng: PRNG,
): GeneratedGraph {
  const { maxNodes, blockSize, hiddenDim, numMixtures } = config;

  // Node features stored per node (hiddenDim each)
  const nodeFeats: Float64Array[] = [];
  const n = maxNodes;
  const adjacency = new Uint8Array(n * n);
  let currentCount = 0;

  while (currentCount < maxNodes) {
    const toAdd = Math.min(blockSize, maxNodes - currentCount);

    // --- Sample new node features ---
    const newNodeFeats: Float64Array[] = [];
    for (let b = 0; b < toAdd; b++) {
      // Sample raw features ~ N(0,1) via Box-Muller, then project through W_node
      const raw = new Float64Array(hiddenDim);
      for (let d = 0; d < hiddenDim; d += 2) {
        const u1 = rng() || 1e-15;
        const u2 = rng();
        const r = Math.sqrt(-2 * Math.log(u1));
        const theta = 2 * Math.PI * u2;
        raw[d] = r * Math.cos(theta);
        if (d + 1 < hiddenDim) {
          raw[d + 1] = r * Math.sin(theta);
        }
      }

      // h = tanh(W_node * raw)   (W_node is hiddenDim x hiddenDim)
      const h = new Float64Array(hiddenDim);
      for (let i = 0; i < hiddenDim; i++) {
        let val = 0;
        for (let j = 0; j < hiddenDim; j++) {
          val += W_node[i * hiddenDim + j]! * raw[j]!;
        }
        h[i] = Math.tanh(val);
      }

      newNodeFeats.push(h);
    }

    // --- Sample edges between new nodes and existing nodes ---
    const pairDim = 2 * hiddenDim;
    // W_edge layout: numMixtures rows of (2*hiddenDim) cols
    const mixWeight = 1.0 / numMixtures;

    for (let b = 0; b < toAdd; b++) {
      const newIdx = currentCount + b;
      const hNew = newNodeFeats[b]!;

      // Edges to existing nodes
      for (let j = 0; j < currentCount; j++) {
        const hExist = nodeFeats[j]!;
        const pair = concat(hNew, hExist);

        // Mixture of Bernoulli
        let prob = 0;
        for (let k = 0; k < numMixtures; k++) {
          let dot = 0;
          for (let d = 0; d < pairDim; d++) {
            dot += W_edge[k * pairDim + d]! * pair[d]!;
          }
          prob += mixWeight * sigmoidScalar(dot);
        }

        if (prob > 0.5) {
          adjacency[newIdx * n + j] = 1;
          adjacency[j * n + newIdx] = 1;
        }
      }

      // Intra-block edges: between this new node and previously added nodes in the block
      for (let b2 = 0; b2 < b; b2++) {
        const otherIdx = currentCount + b2;
        const hOther = newNodeFeats[b2]!;
        const pair = concat(hNew, hOther);

        let prob = 0;
        for (let k = 0; k < numMixtures; k++) {
          let dot = 0;
          for (let d = 0; d < pairDim; d++) {
            dot += W_edge[k * pairDim + d]! * pair[d]!;
          }
          prob += mixWeight * sigmoidScalar(dot);
        }

        if (prob > 0.5) {
          adjacency[newIdx * n + otherIdx] = 1;
          adjacency[otherIdx * n + newIdx] = 1;
        }
      }
    }

    // Commit new nodes
    for (let b = 0; b < toAdd; b++) {
      nodeFeats.push(newNodeFeats[b]!);
    }
    currentCount += toAdd;
  }

  // Trim adjacency to actual size (currentCount may equal maxNodes)
  const finalN = currentCount;
  if (finalN === maxNodes) {
    return { adjacency, numNodes: finalN };
  }

  // Resize if we stopped early (shouldn't happen with while condition, but safe)
  const trimmed = new Uint8Array(finalN * finalN);
  for (let i = 0; i < finalN; i++) {
    for (let j = 0; j < finalN; j++) {
      trimmed[i * finalN + j] = adjacency[i * n + j]!;
    }
  }
  return { adjacency: trimmed, numNodes: finalN };
}
