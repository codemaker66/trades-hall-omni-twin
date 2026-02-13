// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-3: Recommendation System Tests
// Tests for LightGCN, PinSage, SR-GNN, and Cold-Start strategies.
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';

// Types and PRNG
import {
  createPRNG,
  type Graph,
} from '../types.js';

// Graph utilities
import { buildCSR } from '../graph.js';

// Tensor utilities
import { xavierInit } from '../tensor.js';

// Recommendation modules under test
import {
  lightGCNPropagate,
  bprLossCompute,
  pinSageSample,
  buildSessionGraph,
  contentBasedInit,
} from '../recommendation/index.js';

// ---------------------------------------------------------------------------
// Helper: Build a small bipartite user-item graph
// 3 users (nodes 0,1,2), 4 items (nodes 3,4,5,6)
// Interactions: user0->item3, user0->item4, user1->item4, user1->item5, user2->item6
// Bipartite edges in both directions for LightGCN.
// ---------------------------------------------------------------------------

function makeBipartiteGraph(): { graph: Graph; numUsers: number; numItems: number } {
  const numUsers = 3;
  const numItems = 4;
  const numNodes = numUsers + numItems; // 7

  // Forward + reverse edges for undirected bipartite graph
  const edges: [number, number][] = [
    // user -> item
    [0, 3], [0, 4],
    [1, 4], [1, 5],
    [2, 6],
    // item -> user (reverse)
    [3, 0], [4, 0],
    [4, 1], [5, 1],
    [6, 2],
  ];

  const graph = buildCSR(edges, numNodes);

  return { graph, numUsers, numItems };
}

// ---------------------------------------------------------------------------
// LightGCN Tests
// ---------------------------------------------------------------------------

describe('lightGCNPropagate', () => {
  it('returns embeddings of correct dimensions (numNodes x embDim)', () => {
    const { graph, numUsers, numItems } = makeBipartiteGraph();
    const numNodes = numUsers + numItems;
    const embDim = 8;
    const numLayers = 2;

    const rng = createPRNG(42);
    const embeddings = new Float64Array(numNodes * embDim);
    for (let i = 0; i < embeddings.length; i++) embeddings[i] = rng() - 0.5;

    const result = lightGCNPropagate(graph, embeddings, numLayers, embDim);

    // Output should be numNodes x embDim
    expect(result.length).toBe(numNodes * embDim);
  });

  it('produces embeddings that differ from initial (aggregation happened) without nonlinearity', () => {
    const { graph, numUsers, numItems } = makeBipartiteGraph();
    const numNodes = numUsers + numItems;
    const embDim = 4;
    const numLayers = 1;

    const rng = createPRNG(42);
    const embeddings = new Float64Array(numNodes * embDim);
    for (let i = 0; i < embeddings.length; i++) embeddings[i] = rng() - 0.5;

    const result = lightGCNPropagate(graph, embeddings, numLayers, embDim);

    // The result should differ from the initial embeddings
    // (because aggregation averages in layer-0 and layer-1 contributions)
    let allSame = true;
    for (let i = 0; i < result.length; i++) {
      if (Math.abs(result[i]! - embeddings[i]!) > 1e-12) {
        allSame = false;
        break;
      }
    }
    expect(allSame).toBe(false);

    // LightGCN has NO nonlinearity: values can be negative
    // This verifies that no ReLU/sigmoid was applied
    let hasNegative = false;
    for (let i = 0; i < result.length; i++) {
      if (result[i]! < 0) {
        hasNegative = true;
        break;
      }
    }
    // With random initialization, some values should be negative (no activation)
    expect(hasNegative).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// BPR Loss Tests
// ---------------------------------------------------------------------------

describe('bprLossCompute', () => {
  it('returns a positive loss for known pos/neg scores', () => {
    // Positive scores higher than negative scores
    const posScores = new Float64Array([2.0, 1.5, 3.0]);
    const negScores = new Float64Array([0.5, 0.3, 1.0]);

    const loss = bprLossCompute(posScores, negScores);

    // BPR loss = -mean(ln(sigmoid(pos - neg)))
    // All (pos - neg) > 0, so sigmoid > 0.5, ln > -0.69, loss is positive
    expect(loss).toBeGreaterThan(0);
  });

  it('returns higher loss when neg scores approach pos scores', () => {
    const posScores = new Float64Array([2.0, 1.5, 3.0]);
    const negScoresEasy = new Float64Array([0.0, 0.0, 0.0]);
    const negScoresHard = new Float64Array([1.9, 1.4, 2.9]);

    const lossEasy = bprLossCompute(posScores, negScoresEasy);
    const lossHard = bprLossCompute(posScores, negScoresHard);

    // Harder negatives -> higher loss
    expect(lossHard).toBeGreaterThan(lossEasy);
  });
});

// ---------------------------------------------------------------------------
// PinSage Sampling Tests
// ---------------------------------------------------------------------------

describe('pinSageSample', () => {
  it('returns neighbor importance scores that sum to ~1', () => {
    // Build a small graph: 0-1, 0-2, 1-2, 2-3, 3-4
    const edges: [number, number][] = [
      [0, 1], [1, 0],
      [0, 2], [2, 0],
      [1, 2], [2, 1],
      [2, 3], [3, 2],
      [3, 4], [4, 3],
    ];
    const graph = buildCSR(edges, 5);

    const rng = createPRNG(101);
    const walkLen = 4;
    const numWalks = 100;

    const importanceMap = pinSageSample(graph, 0, walkLen, numWalks, rng);

    // Should have discovered some neighbors
    expect(importanceMap.size).toBeGreaterThan(0);

    // Importance scores should sum to ~1 (normalized visit frequencies)
    let sum = 0;
    for (const score of importanceMap.values()) {
      sum += score;
    }
    expect(sum).toBeCloseTo(1.0, 5);

    // Immediate neighbors (1 and 2) should have higher importance than distant nodes
    const score1 = importanceMap.get(1) ?? 0;
    const score2 = importanceMap.get(2) ?? 0;
    const score4 = importanceMap.get(4) ?? 0;
    // Nodes 1 and 2 are direct neighbors, so they should each have higher
    // importance than node 4 which is 2 hops away
    expect(score1 + score2).toBeGreaterThan(score4);
  });
});

// ---------------------------------------------------------------------------
// SR-GNN Session Graph Tests
// ---------------------------------------------------------------------------

describe('buildSessionGraph', () => {
  it('builds correct directed edges and node mapping from session [0,1,2,0,3]', () => {
    const session = [0, 1, 2, 0, 3];

    const result = buildSessionGraph(session);

    // Unique items: 0, 1, 2, 3 -> local IDs 0, 1, 2, 3
    expect(result.nodeMapping.size).toBe(4);
    expect(result.reverseMapping.length).toBe(4);

    // Verify mapping: first occurrence order
    expect(result.nodeMapping.get(0)).toBe(0);
    expect(result.nodeMapping.get(1)).toBe(1);
    expect(result.nodeMapping.get(2)).toBe(2);
    expect(result.nodeMapping.get(3)).toBe(3);

    // Reverse mapping
    expect(result.reverseMapping[0]).toBe(0);
    expect(result.reverseMapping[1]).toBe(1);
    expect(result.reverseMapping[2]).toBe(2);
    expect(result.reverseMapping[3]).toBe(3);

    // Session items in local IDs: [0, 1, 2, 0, 3]
    expect(result.sessionItems).toEqual([0, 1, 2, 0, 3]);

    // Expected directed edges (in local IDs): 0->1, 1->2, 2->0, 0->3
    // Deduplicated: {0->1, 1->2, 2->0, 0->3}
    const graph = result.graph;
    expect(graph.numNodes).toBe(4);
    expect(graph.numEdges).toBe(4);

    // Verify specific edges exist in the CSR
    // Node 0 should have outgoing edges to 1 and 3
    const start0 = graph.rowPtr[0]!;
    const end0 = graph.rowPtr[1]!;
    const neighbors0 = Array.from(graph.colIdx.slice(start0, end0)).sort();
    expect(neighbors0).toEqual([1, 3]);

    // Node 1 should have outgoing edge to 2
    const start1 = graph.rowPtr[1]!;
    const end1 = graph.rowPtr[2]!;
    const neighbors1 = Array.from(graph.colIdx.slice(start1, end1));
    expect(neighbors1).toEqual([2]);

    // Node 2 should have outgoing edge to 0
    const start2 = graph.rowPtr[2]!;
    const end2 = graph.rowPtr[3]!;
    const neighbors2 = Array.from(graph.colIdx.slice(start2, end2));
    expect(neighbors2).toEqual([0]);

    // Node 3 should have no outgoing edges
    const start3 = graph.rowPtr[3]!;
    const end3 = graph.rowPtr[4]!;
    expect(end3 - start3).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Cold-Start k-NN Tests
// ---------------------------------------------------------------------------

describe('contentBasedInit', () => {
  it('returns k nearest neighbor indices', () => {
    const featureDim = 3;
    const numItems = 5;
    const k = 3;

    // New item features: [1, 0, 0]
    const newItemFeatures = new Float64Array([1, 0, 0]);

    // Existing items:
    // Item 0: [0.9, 0.1, 0]   -> close to new item
    // Item 1: [0, 1, 0]       -> far
    // Item 2: [1.1, 0, 0.1]   -> close to new item
    // Item 3: [0, 0, 1]       -> far
    // Item 4: [0.8, 0.2, 0]   -> close to new item
    const existingItems = new Float64Array([
      0.9, 0.1, 0,
      0, 1, 0,
      1.1, 0, 0.1,
      0, 0, 1,
      0.8, 0.2, 0,
    ]);

    const result = contentBasedInit(
      newItemFeatures,
      existingItems,
      numItems,
      featureDim,
      k,
    );

    // Should return exactly k indices
    expect(result.length).toBe(k);

    // The 3 closest items should be 0, 2, and 4 (in some order by distance)
    const indices = Array.from(result).sort();
    expect(indices).toEqual([0, 2, 4]);
  });

  it('returns fewer than k when numItems < k', () => {
    const featureDim = 2;
    const numItems = 2;
    const k = 5;

    const newItemFeatures = new Float64Array([1, 0]);
    const existingItems = new Float64Array([0.9, 0.1, 0.5, 0.5]);

    const result = contentBasedInit(
      newItemFeatures,
      existingItems,
      numItems,
      featureDim,
      k,
    );

    // Should return min(k, numItems) = 2
    expect(result.length).toBe(2);
  });
});
