// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-9: Scalable Inference Tests
// Tests for neighbor sampling, mini-batch iterator, Cluster-GCN partitioning,
// GLNN knowledge distillation, and IVF/exact-kNN embedding search.
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';

// Types and PRNG
import { createPRNG, type Graph, type GNNForwardFn } from '../types.js';

// Graph helpers
import { buildCSR } from '../graph.js';

// Modules under test
import {
  neighborSample,
  createMiniBatchIterator,
} from '../inference/neighbor-loader.js';

import {
  spectralBipartition,
  getClusterSubgraph,
  recursivePartition,
} from '../inference/cluster-gcn.js';

import {
  distillGNNToMLP,
  mlpInference,
} from '../inference/glnn-distill.js';

import {
  buildIVFIndex,
  searchIVF,
  exactKNN,
} from '../inference/embedding-search.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Build a small undirected graph for testing:
 *
 *  0 -- 1 -- 2
 *  |         |
 *  3 -- 4 -- 5
 *
 * 6 nodes, 14 directed edges (7 undirected edges x2).
 * Each node has a 4-dimensional feature vector.
 */
function makeTestGraph(): Graph {
  const edges: [number, number][] = [
    [0, 1], [1, 0],
    [1, 2], [2, 1],
    [0, 3], [3, 0],
    [2, 5], [5, 2],
    [3, 4], [4, 3],
    [4, 5], [5, 4],
    [1, 4], [4, 1],
  ];
  const numNodes = 6;
  const featureDim = 4;

  const base = buildCSR(edges, numNodes);

  // Assign deterministic features
  const nodeFeatures = new Float64Array(numNodes * featureDim);
  const rng = createPRNG(42);
  for (let i = 0; i < nodeFeatures.length; i++) {
    nodeFeatures[i] = rng();
  }

  return {
    ...base,
    nodeFeatures,
    featureDim,
  };
}

/**
 * Create a simple mock GNN forward function.
 * Outputs numClasses logits per node, derived from sum of features.
 */
function makeMockGNN(numClasses: number): GNNForwardFn {
  return (graph: Graph, features: Float64Array): Float64Array => {
    const numNodes = graph.numNodes;
    const featureDim = graph.featureDim;
    const output = new Float64Array(numNodes * numClasses);

    for (let i = 0; i < numNodes; i++) {
      // Sum the features for this node
      let featureSum = 0;
      for (let f = 0; f < featureDim; f++) {
        featureSum += features[i * featureDim + f]!;
      }
      // Spread across classes with some variation
      for (let c = 0; c < numClasses; c++) {
        output[i * numClasses + c] = featureSum * (c + 1) * 0.1;
      }
    }

    return output;
  };
}

// ---------------------------------------------------------------------------
// 1. Neighbor Sampling (neighbor-loader.ts)
// ---------------------------------------------------------------------------

describe('neighborSample', () => {
  it('returns correct subgraph with expected fanout', () => {
    const graph = makeTestGraph();
    const rng = createPRNG(123);
    const seedNodes = new Uint32Array([0]);
    const fanout = [2]; // 1-layer, sample up to 2 neighbors

    const result = neighborSample(graph, seedNodes, fanout, rng);

    // Subgraph should include seed node + sampled neighbors
    expect(result.subgraph.numNodes).toBeGreaterThanOrEqual(seedNodes.length);
    // The original IDs should contain the seed node
    const originalIdsSet = new Set(result.originalIds);
    expect(originalIdsSet.has(0)).toBe(true);
    // Target nodes in the subgraph should map back to seed nodes
    expect(result.targetNodes.length).toBe(1);
    // The subgraph should have edges (the induced subgraph is non-trivial)
    expect(result.subgraph.numNodes).toBeLessThanOrEqual(graph.numNodes);
    // originalIds should be sorted
    for (let i = 1; i < result.originalIds.length; i++) {
      expect(result.originalIds[i]!).toBeGreaterThan(result.originalIds[i - 1]!);
    }
  });
});

// ---------------------------------------------------------------------------
// 2. Mini-batch Iterator (neighbor-loader.ts)
// ---------------------------------------------------------------------------

describe('createMiniBatchIterator', () => {
  it('yields all nodes across batches', () => {
    const graph = makeTestGraph();
    const rng = createPRNG(456);
    const batchSize = 2;
    const fanout = [2];

    const iter = createMiniBatchIterator(graph, batchSize, fanout, rng);

    const seenNodes = new Set<number>();
    let batchCount = 0;

    let batch = iter.next();
    while (batch !== null) {
      batchCount++;
      // Each batch should have originalIds that map subgraph nodes to original graph
      for (let i = 0; i < batch.originalIds.length; i++) {
        seenNodes.add(batch.originalIds[i]!);
      }
      // targetMask should have same length as subgraph nodes
      expect(batch.targetMask.length).toBe(batch.subgraph.numNodes);
      // At least some entries in targetMask should be 1 (the seed nodes)
      let targetCount = 0;
      for (let i = 0; i < batch.targetMask.length; i++) {
        if (batch.targetMask[i] === 1) targetCount++;
      }
      expect(targetCount).toBeGreaterThan(0);
      expect(targetCount).toBeLessThanOrEqual(batchSize);

      batch = iter.next();
    }

    // With 6 nodes and batchSize=2, we should get ceil(6/2)=3 batches
    expect(batchCount).toBe(3);
    // All nodes should have been visited as targets across all batches
    expect(seenNodes.size).toBe(graph.numNodes);
  });
});

// ---------------------------------------------------------------------------
// 3. Spectral Bipartition (cluster-gcn.ts)
// ---------------------------------------------------------------------------

describe('spectralBipartition', () => {
  it('produces two non-empty partitions', () => {
    const graph = makeTestGraph();
    const rng = createPRNG(789);

    const assignment = spectralBipartition(graph, rng);

    expect(assignment.length).toBe(graph.numNodes);

    // Count nodes in each partition
    let count0 = 0;
    let count1 = 0;
    for (let i = 0; i < assignment.length; i++) {
      if (assignment[i] === 0) count0++;
      else if (assignment[i] === 1) count1++;
    }

    // Both partitions should be non-empty
    expect(count0).toBeGreaterThan(0);
    expect(count1).toBeGreaterThan(0);
    // All nodes should be assigned to 0 or 1
    expect(count0 + count1).toBe(graph.numNodes);
  });
});

// ---------------------------------------------------------------------------
// 4. getClusterSubgraph (cluster-gcn.ts)
// ---------------------------------------------------------------------------

describe('getClusterSubgraph', () => {
  it('returns valid subgraph for a partition cluster', () => {
    const graph = makeTestGraph();
    const rng = createPRNG(101);

    // Partition into 2 clusters
    const partition = recursivePartition(graph, 2, rng);

    expect(partition.numClusters).toBe(2);
    expect(partition.clusterSizes.length).toBe(2);

    // Extract subgraph for cluster 0
    const sub0 = getClusterSubgraph(graph, partition, 0);

    // Subgraph node count should match cluster size
    expect(sub0.numNodes).toBe(partition.clusterSizes[0]);
    // Subgraph should have valid CSR structure
    expect(sub0.rowPtr.length).toBe(sub0.numNodes + 1);
    // First rowPtr entry should be 0
    expect(sub0.rowPtr[0]).toBe(0);
    // Last rowPtr entry should be numEdges
    expect(sub0.rowPtr[sub0.numNodes]).toBe(sub0.numEdges);
    // All colIdx entries should be valid node indices
    for (let e = 0; e < sub0.numEdges; e++) {
      expect(sub0.colIdx[e]).toBeLessThan(sub0.numNodes);
    }
  });
});

// ---------------------------------------------------------------------------
// 5. distillGNNToMLP (glnn-distill.ts)
// ---------------------------------------------------------------------------

describe('distillGNNToMLP', () => {
  it('returns MLP weights with correct dimensions', () => {
    const graph = makeTestGraph();
    const numClasses = 3;
    const teacherFn = makeMockGNN(numClasses);
    const rng = createPRNG(202);

    const result = distillGNNToMLP(teacherFn, graph, graph.nodeFeatures, {
      hiddenDims: [8],
      lambda: 0.5,
      temperature: 2.0,
      epochs: 3,
      learningRate: 0.01,
    }, rng);

    // MLP should have 2 layers: input(4)->hidden(8) and hidden(8)->output(3)
    expect(result.mlpWeights.layers.length).toBe(2);

    const layer0 = result.mlpWeights.layers[0]!;
    expect(layer0.inDim).toBe(graph.featureDim);
    expect(layer0.outDim).toBe(8);
    expect(layer0.W.length).toBe(graph.featureDim * 8);
    expect(layer0.bias.length).toBe(8);

    const layer1 = result.mlpWeights.layers[1]!;
    expect(layer1.inDim).toBe(8);
    expect(layer1.outDim).toBe(numClasses);
    expect(layer1.W.length).toBe(8 * numClasses);
    expect(layer1.bias.length).toBe(numClasses);

    // Should have losses for each epoch
    expect(result.losses.length).toBe(3);

    // Accuracy should be in [0, 1]
    expect(result.accuracy).toBeGreaterThanOrEqual(0);
    expect(result.accuracy).toBeLessThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// 6. mlpInference (glnn-distill.ts)
// ---------------------------------------------------------------------------

describe('mlpInference', () => {
  it('output shape matches expectations', () => {
    const graph = makeTestGraph();
    const numClasses = 3;
    const teacherFn = makeMockGNN(numClasses);
    const rng = createPRNG(303);

    // First distill to get MLP weights
    const distillResult = distillGNNToMLP(teacherFn, graph, graph.nodeFeatures, {
      hiddenDims: [8],
      lambda: 0.5,
      temperature: 2.0,
      epochs: 2,
      learningRate: 0.01,
    }, rng);

    // Run MLP inference
    const output = mlpInference(
      graph.nodeFeatures,
      distillResult.mlpWeights,
      graph.numNodes,
      graph.featureDim,
    );

    // Output should be numNodes * outDim (numClasses)
    expect(output.length).toBe(graph.numNodes * numClasses);

    // All output values should be finite
    for (let i = 0; i < output.length; i++) {
      expect(Number.isFinite(output[i]!)).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// 7. buildIVFIndex (embedding-search.ts)
// ---------------------------------------------------------------------------

describe('buildIVFIndex', () => {
  it('creates correct number of clusters', () => {
    const rng = createPRNG(404);
    const dim = 4;
    const numEmbeddings = 20;
    const nClusters = 4;

    // Generate random embeddings
    const embeddings = new Float64Array(numEmbeddings * dim);
    for (let i = 0; i < embeddings.length; i++) {
      embeddings[i] = rng();
    }

    const index = buildIVFIndex(embeddings, numEmbeddings, dim, nClusters, rng);

    expect(index.nClusters).toBe(nClusters);
    expect(index.dim).toBe(dim);
    expect(index.numEmbeddings).toBe(numEmbeddings);
    expect(index.centroids.length).toBe(nClusters * dim);
    expect(index.assignments.length).toBe(numEmbeddings);

    // Every assignment should be a valid cluster ID
    for (let i = 0; i < numEmbeddings; i++) {
      expect(index.assignments[i]).toBeLessThan(nClusters);
    }
  });
});

// ---------------------------------------------------------------------------
// 8. searchIVF (embedding-search.ts)
// ---------------------------------------------------------------------------

describe('searchIVF', () => {
  it('returns top-k results', () => {
    const rng = createPRNG(505);
    const dim = 4;
    const numEmbeddings = 30;
    const nClusters = 3;
    const k = 5;
    const nProbes = 2;

    // Generate random embeddings
    const embeddings = new Float64Array(numEmbeddings * dim);
    for (let i = 0; i < embeddings.length; i++) {
      embeddings[i] = rng();
    }

    const index = buildIVFIndex(embeddings, numEmbeddings, dim, nClusters, rng);

    // Use the first embedding as the query
    const query = embeddings.slice(0, dim);

    const result = searchIVF(index, query, k, nProbes);

    // Should return at most k results
    expect(result.indices.length).toBeLessThanOrEqual(k);
    expect(result.distances.length).toBe(result.indices.length);

    // Distances should be non-negative and sorted ascending
    for (let i = 0; i < result.distances.length; i++) {
      expect(result.distances[i]).toBeGreaterThanOrEqual(0);
      if (i > 0) {
        expect(result.distances[i]).toBeGreaterThanOrEqual(result.distances[i - 1]!);
      }
    }

    // All returned indices should be valid embedding indices
    for (let i = 0; i < result.indices.length; i++) {
      expect(result.indices[i]).toBeLessThan(numEmbeddings);
    }
  });
});

// ---------------------------------------------------------------------------
// 9. exactKNN (embedding-search.ts)
// ---------------------------------------------------------------------------

describe('exactKNN', () => {
  it('returns correct nearest neighbors for simple case', () => {
    const dim = 2;
    // Place embeddings at known positions
    // Embedding 0: [0, 0]
    // Embedding 1: [1, 0]
    // Embedding 2: [0, 1]
    // Embedding 3: [10, 10]  (far away)
    // Embedding 4: [0.5, 0.5]
    const numEmbeddings = 5;
    const embeddings = new Float64Array([
      0, 0,       // index 0
      1, 0,       // index 1
      0, 1,       // index 2
      10, 10,     // index 3
      0.5, 0.5,   // index 4
    ]);

    // Query at origin [0, 0]
    const query = new Float64Array([0, 0]);
    const k = 3;

    const result = exactKNN(embeddings, numEmbeddings, dim, query, k);

    expect(result.indices.length).toBe(k);
    expect(result.distances.length).toBe(k);

    // Nearest to origin should be embedding 0 (distance 0)
    expect(result.indices[0]).toBe(0);
    expect(result.distances[0]).toBeCloseTo(0, 10);

    // Second nearest should be embedding 4 at (0.5, 0.5), distance ~0.707
    expect(result.indices[1]).toBe(4);
    expect(result.distances[1]).toBeCloseTo(Math.sqrt(0.5), 5);

    // Third nearest should be either embedding 1 or 2 (both distance 1.0)
    expect([1, 2]).toContain(result.indices[2]);
    expect(result.distances[2]).toBeCloseTo(1.0, 5);

    // Embedding 3 (far away) should NOT be in top-3
    const indicesSet = new Set(result.indices);
    expect(indicesSet.has(3)).toBe(false);
  });
});
