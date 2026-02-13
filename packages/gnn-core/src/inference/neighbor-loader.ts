// ---------------------------------------------------------------------------
// GNN-9 Scalable Inference — Neighbor Loader (GraphSAGE Mini-Batch Sampling)
// ---------------------------------------------------------------------------
//
// Implements multi-hop neighbor sampling for GraphSAGE-style mini-batch
// training. Instead of computing embeddings for all nodes, we sample a fixed
// number of neighbors per hop to keep computation bounded.
//
// References:
//   Hamilton et al., "Inductive Representation Learning on Large Graphs"
//   (NeurIPS 2017) — GraphSAGE
// ---------------------------------------------------------------------------

import type { Graph, PRNG, NeighborSampleResult, MiniBatch } from '../types.js';
import { sampleNeighbors, subgraph, buildCSR } from '../graph.js';

// ---------------------------------------------------------------------------
// 1. neighborSample — Multi-hop neighbor sampling from seed nodes
// ---------------------------------------------------------------------------

/**
 * Multi-hop neighbor sampling starting from seed nodes.
 *
 * Algorithm (GraphSAGE mini-batch construction):
 * 1. Start with the seed nodes as the "frontier" for the deepest layer.
 * 2. For each layer l (from last to first in `fanout`):
 *    a. For each node in the frontier, sample min(fanout[l], degree) neighbors.
 *    b. Add all sampled neighbors to the frontier for the next outer layer.
 * 3. Collect all sampled nodes (union of seed + all sampled neighbors).
 * 4. Build the induced subgraph containing only those nodes.
 * 5. Return the subgraph, the indices of seed nodes within the subgraph,
 *    and a mapping from subgraph node IDs to original graph node IDs.
 *
 * The fanout array is ordered from the innermost (closest to seed) layer
 * to the outermost. For example, fanout = [10, 25] means: first sample
 * 25 neighbors (outer hop), then sample 10 neighbors (inner hop) from those.
 *
 * @param graph - Input CSR Graph.
 * @param seedNodes - Uint32Array of seed node indices to sample around.
 * @param fanout - Array of fanout sizes per layer, from inner to outer.
 * @param rng - Deterministic PRNG function.
 * @returns NeighborSampleResult with subgraph, targetNodes, and originalIds.
 */
export function neighborSample(
  graph: Graph,
  seedNodes: Uint32Array,
  fanout: number[],
  rng: PRNG,
): NeighborSampleResult {
  const numLayers = fanout.length;

  // Maintain the set of all nodes we need in the subgraph
  const allNodes = new Set<number>();
  for (let i = 0; i < seedNodes.length; i++) {
    allNodes.add(seedNodes[i]!);
  }

  // Current frontier: starts as the seed nodes
  let frontier = new Uint32Array(seedNodes);

  // Sample from outer layer (last in fanout) to inner layer (first in fanout)
  // This builds the computation graph from the target nodes outward
  for (let l = numLayers - 1; l >= 0; l--) {
    const layerFanout = fanout[l]!;
    const sampled = sampleNeighbors(graph, frontier, layerFanout, rng);

    // Collect all newly sampled neighbors into the next frontier
    const nextFrontierSet = new Set<number>();
    for (const [, neighbors] of sampled) {
      for (let j = 0; j < neighbors.length; j++) {
        const neighbor = neighbors[j]!;
        nextFrontierSet.add(neighbor);
        allNodes.add(neighbor);
      }
    }

    // The next frontier includes both existing frontier nodes and new neighbors
    // so that the next outer layer samples around all of them
    const nextFrontierArr: number[] = [];
    for (const node of nextFrontierSet) {
      nextFrontierArr.push(node);
    }
    // Also include current frontier nodes for the next layer's sampling
    for (let i = 0; i < frontier.length; i++) {
      if (!nextFrontierSet.has(frontier[i]!)) {
        nextFrontierArr.push(frontier[i]!);
      }
    }
    frontier = new Uint32Array(nextFrontierArr);
  }

  // Build the induced subgraph from all collected nodes
  const sub = subgraph(graph, allNodes);

  // Build mapping: sorted original IDs
  const sortedOriginalIds = Array.from(allNodes).sort((a, b) => a - b);
  const originalIds = new Uint32Array(sortedOriginalIds);

  // Build reverse mapping: original node ID -> subgraph node index
  const originalToSubgraph = new Map<number, number>();
  for (let i = 0; i < sortedOriginalIds.length; i++) {
    originalToSubgraph.set(sortedOriginalIds[i]!, i);
  }

  // Find the seed nodes' indices within the subgraph
  const targetNodes = new Uint32Array(seedNodes.length);
  for (let i = 0; i < seedNodes.length; i++) {
    targetNodes[i] = originalToSubgraph.get(seedNodes[i]!)!;
  }

  return {
    subgraph: sub,
    targetNodes,
    originalIds,
  };
}

// ---------------------------------------------------------------------------
// 2. createMiniBatchIterator — Iterator over node mini-batches
// ---------------------------------------------------------------------------

/**
 * Create an iterator that yields mini-batches of nodes with their sampled
 * neighborhoods, suitable for GraphSAGE-style training.
 *
 * Algorithm:
 * 1. Shuffle all node indices using Fisher-Yates with the provided PRNG.
 * 2. On each call to next(), take the next `batchSize` nodes as seeds.
 * 3. Call neighborSample to build the mini-batch subgraph.
 * 4. Return null when all nodes have been visited.
 * 5. reset() reshuffles the node ordering and resets the cursor.
 *
 * @param graph - Input CSR Graph.
 * @param batchSize - Number of seed nodes per mini-batch.
 * @param fanout - Fanout sizes per layer for neighborSample.
 * @param rng - Deterministic PRNG function.
 * @returns Object with next() and reset() methods.
 */
export function createMiniBatchIterator(
  graph: Graph,
  batchSize: number,
  fanout: number[],
  rng: PRNG,
): { next(): MiniBatch | null; reset(): void } {
  const n = graph.numNodes;

  // Permutation array for shuffled node order
  const perm = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    perm[i] = i;
  }

  let cursor = 0;

  /** Fisher-Yates shuffle of the permutation array. */
  function shuffle(): void {
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = perm[i]!;
      perm[i] = perm[j]!;
      perm[j] = tmp;
    }
  }

  // Initial shuffle
  shuffle();

  function next(): MiniBatch | null {
    if (cursor >= n) return null;

    const end = Math.min(cursor + batchSize, n);
    const seedNodes = perm.slice(cursor, end);
    cursor = end;

    // Sample neighbors around the seed nodes
    const sampleResult = neighborSample(graph, seedNodes, fanout, rng);
    const subgraphNodes = sampleResult.subgraph.numNodes;

    // Build target mask: 1 for seed nodes, 0 for context nodes
    const targetMask = new Uint8Array(subgraphNodes);
    for (let i = 0; i < sampleResult.targetNodes.length; i++) {
      targetMask[sampleResult.targetNodes[i]!] = 1;
    }

    return {
      subgraph: sampleResult.subgraph,
      targetMask,
      originalIds: sampleResult.originalIds,
    };
  }

  function reset(): void {
    cursor = 0;
    shuffle();
  }

  return { next, reset };
}
