// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-4: Spatial Layout Understanding Tests
// Tests for buildLayoutGraph, layoutQualityGNN, graphMatchingScore,
// and sceneGraphForward.
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';

// Types and PRNG
import {
  createPRNG,
  type LayoutItem,
  type LayoutGraphConfig,
  type GATConfig,
  type GATWeights,
  type SurrogateEnergyModel,
  type GRUWeights,
  type Graph,
} from '../types.js';

// Graph utilities
import { buildCSR } from '../graph.js';

// Tensor utilities
import { xavierInit } from '../tensor.js';

// Spatial module under test
import { buildLayoutGraph, layoutItemsToFeatures } from '../spatial/layout-gnn.js';
import { layoutQualityGNN, globalMeanPool, globalSumPool } from '../spatial/layout-quality.js';
import { graphMatchingScore } from '../spatial/layout-gmn.js';
import { sceneGraphForward } from '../spatial/scene-graph.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Create a list of 4 LayoutItems in a rough square arrangement. */
function makeLayoutItems(): LayoutItem[] {
  return [
    { id: 0, type: 0, numTypes: 3, width: 0.5, depth: 0.5, height: 0.8, x: 0, y: 0, rotation: 0 },
    { id: 1, type: 1, numTypes: 3, width: 1.2, depth: 1.2, height: 0.75, x: 2, y: 0, rotation: 0 },
    { id: 2, type: 0, numTypes: 3, width: 0.5, depth: 0.5, height: 0.8, x: 2, y: 2, rotation: Math.PI / 2 },
    { id: 3, type: 2, numTypes: 3, width: 0.4, depth: 0.4, height: 1.0, x: 0, y: 2, rotation: Math.PI },
  ];
}

/** Check that every element in a Float64Array is finite. */
function allFinite(arr: Float64Array): boolean {
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i]!)) return false;
  }
  return true;
}

/** Create a small SurrogateEnergyModel for testing layoutQualityGNN. */
function makeSurrogateModel(inDim: number): SurrogateEnergyModel {
  const rng = createPRNG(42);
  const outDim = inDim; // keep same dim for residual
  const heads = 1;

  const gatWeights: GATWeights[] = [
    {
      W: xavierInit(inDim, outDim * heads, rng),
      a_src: xavierInit(1, outDim * heads, rng),
      a_dst: xavierInit(1, outDim * heads, rng),
    },
  ];

  const gatConfig: GATConfig = {
    inDim,
    outDim,
    heads,
    dropout: 0,
    negativeSlope: 0.2,
    concat: true,
    v2: false,
  };

  // poolingW: outDim -> outDim (identity-like projection)
  const poolingW = xavierInit(outDim, outDim, rng);
  // headW: outDim -> 1 scalar
  const headW = xavierInit(1, outDim, rng);
  const headBias = new Float64Array([0.0]);

  return {
    gatWeights,
    poolingW,
    headW,
    headBias,
    config: gatConfig,
  };
}

/** Create GRU weights for sceneGraphForward test. */
function makeGRUWeights(featureDim: number, seed: number): GRUWeights {
  const rng = createPRNG(seed);
  return {
    W_z: xavierInit(featureDim, featureDim, rng),
    U_z: xavierInit(featureDim, featureDim, rng),
    b_z: new Float64Array(featureDim),
    W_r: xavierInit(featureDim, featureDim, rng),
    U_r: xavierInit(featureDim, featureDim, rng),
    b_r: new Float64Array(featureDim),
    W_h: xavierInit(featureDim, featureDim, rng),
    U_h: xavierInit(featureDim, featureDim, rng),
    b_h: new Float64Array(featureDim),
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('GNN-4: Spatial Layout Understanding', () => {
  // =========================================================================
  // buildLayoutGraph
  // =========================================================================
  describe('buildLayoutGraph', () => {
    it('builds a graph with correct node features and edges from 4 LayoutItems', () => {
      const items = makeLayoutItems();
      const config: LayoutGraphConfig = {
        distanceThreshold: 5.0, // large enough to connect nearby items
        functionalEdges: false,
        wallEdges: false,
      };

      const graph = buildLayoutGraph(items, config);

      // 4 items => 4 nodes
      expect(graph.numNodes).toBe(4);

      // featureDim = numTypes + 7 = 3 + 7 = 10
      expect(graph.featureDim).toBe(10);

      // nodeFeatures length = 4 nodes * 10 dims = 40
      expect(graph.nodeFeatures.length).toBe(40);
      expect(allFinite(graph.nodeFeatures)).toBe(true);

      // All items within 5.0 distance => should have edges
      // Distance from (0,0) to (2,0) = 2, (0,0) to (2,2) = 2.83, (0,0) to (0,2) = 2,
      // (2,0) to (2,2) = 2, (2,0) to (0,2) = 2.83, (2,2) to (0,2) = 2
      // All pairs within 5.0 => 6 undirected pairs = 12 directed edges
      expect(graph.numEdges).toBe(12);

      // Edge features should be present with edgeFeatureDim = 5
      expect(graph.edgeFeatureDim).toBe(5);
      expect(graph.edgeFeatures).toBeDefined();
      expect(graph.edgeFeatures!.length).toBe(12 * 5);
    });

    it('returns empty graph for empty items', () => {
      const config: LayoutGraphConfig = {
        distanceThreshold: 5.0,
        functionalEdges: false,
        wallEdges: false,
      };

      const graph = buildLayoutGraph([], config);
      expect(graph.numNodes).toBe(0);
      expect(graph.numEdges).toBe(0);
    });

    it('includes functional edges from chairs to tables', () => {
      // item 0 is a chair (type=0), item 1 is a table (type=1), close together
      const items: LayoutItem[] = [
        { id: 0, type: 0, numTypes: 3, width: 0.5, depth: 0.5, height: 0.8, x: 0, y: 0, rotation: 0 },
        { id: 1, type: 1, numTypes: 3, width: 1.2, depth: 1.2, height: 0.75, x: 1, y: 0, rotation: 0 },
      ];

      const configWithFunctional: LayoutGraphConfig = {
        distanceThreshold: 3.0,
        functionalEdges: true,
        wallEdges: false,
      };

      const graph = buildLayoutGraph(items, configWithFunctional);
      expect(graph.numNodes).toBe(2);
      // At least 2 directed edges (spatial adjacency): 0->1 and 1->0
      expect(graph.numEdges).toBeGreaterThanOrEqual(2);
    });
  });

  // =========================================================================
  // layoutItemsToFeatures
  // =========================================================================
  describe('layoutItemsToFeatures', () => {
    it('produces correctly shaped feature matrix', () => {
      const items = makeLayoutItems();
      const features = layoutItemsToFeatures(items);

      // featureDim = 3 + 7 = 10, 4 items => 40
      expect(features.length).toBe(40);
      expect(allFinite(features)).toBe(true);

      // Check one-hot encoding: item 0 has type=0, so features[0] should be 1
      expect(features[0]).toBe(1.0);
      expect(features[1]).toBe(0.0);
      expect(features[2]).toBe(0.0);
    });
  });

  // =========================================================================
  // layoutQualityGNN
  // =========================================================================
  describe('layoutQualityGNN', () => {
    it('produces a score in [0, 1] and finite', () => {
      const items = makeLayoutItems();
      const config: LayoutGraphConfig = {
        distanceThreshold: 5.0,
        functionalEdges: false,
        wallEdges: false,
      };

      const graph = buildLayoutGraph(items, config);

      // featureDim = 10
      const model = makeSurrogateModel(graph.featureDim);

      const result = layoutQualityGNN(graph, model);

      expect(Number.isFinite(result.score)).toBe(true);
      expect(result.score).toBeGreaterThanOrEqual(0);
      expect(result.score).toBeLessThanOrEqual(1);
    });

    it('returns 0.5 for an empty graph', () => {
      const emptyGraph: Graph = {
        numNodes: 0,
        numEdges: 0,
        rowPtr: new Uint32Array(1),
        colIdx: new Uint32Array(0),
        nodeFeatures: new Float64Array(0),
        featureDim: 0,
      };

      const model = makeSurrogateModel(4);
      const result = layoutQualityGNN(emptyGraph, model);
      expect(result.score).toBe(0.5);
    });
  });

  // =========================================================================
  // graphMatchingScore
  // =========================================================================
  describe('graphMatchingScore', () => {
    it('computes similarity in [-1, 1] for two small graphs', () => {
      const rng = createPRNG(123);
      const dim = 4;
      const numNodes1 = 3;
      const numNodes2 = 3;

      // Build two small graphs
      const edges1: [number, number][] = [[0, 1], [1, 0], [1, 2], [2, 1]];
      const edges2: [number, number][] = [[0, 1], [1, 0], [0, 2], [2, 0]];

      const graph1: Graph = {
        ...buildCSR(edges1, numNodes1),
        nodeFeatures: new Float64Array(numNodes1 * dim),
        featureDim: dim,
      };
      const graph2: Graph = {
        ...buildCSR(edges2, numNodes2),
        nodeFeatures: new Float64Array(numNodes2 * dim),
        featureDim: dim,
      };

      // Fill features
      for (let i = 0; i < graph1.nodeFeatures.length; i++) {
        (graph1.nodeFeatures as Float64Array)[i] = rng() * 2 - 1;
      }
      for (let i = 0; i < graph2.nodeFeatures.length; i++) {
        (graph2.nodeFeatures as Float64Array)[i] = rng() * 2 - 1;
      }

      // W_match: dim x dim identity-like
      const W_match = xavierInit(dim, dim, rng);

      const result = graphMatchingScore(graph1, graph1.nodeFeatures, graph2, graph2.nodeFeatures, W_match, dim);

      expect(Number.isFinite(result.similarity)).toBe(true);
      expect(result.similarity).toBeGreaterThanOrEqual(-1);
      expect(result.similarity).toBeLessThanOrEqual(1);

      // Cross attention matrices should have correct dimensions
      expect(result.crossAttention1.length).toBe(numNodes1 * numNodes2);
      expect(result.crossAttention2.length).toBe(numNodes2 * numNodes1);
    });

    it('returns similarity=0 when one graph is empty', () => {
      const emptyGraph: Graph = {
        numNodes: 0,
        numEdges: 0,
        rowPtr: new Uint32Array(1),
        colIdx: new Uint32Array(0),
        nodeFeatures: new Float64Array(0),
        featureDim: 4,
      };
      const graph2: Graph = {
        ...buildCSR([[0, 1], [1, 0]], 2),
        nodeFeatures: new Float64Array(8),
        featureDim: 4,
      };

      const W_match = new Float64Array(16);
      const result = graphMatchingScore(emptyGraph, emptyGraph.nodeFeatures, graph2, graph2.nodeFeatures, W_match, 4);
      expect(result.similarity).toBe(0);
    });
  });

  // =========================================================================
  // sceneGraphForward
  // =========================================================================
  describe('sceneGraphForward', () => {
    it('produces output with same dimensions as input', () => {
      const numNodes = 4;
      const featureDim = 6;
      const numEdgeTypes = 2;

      // Build a small graph
      const edges: [number, number][] = [
        [0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2],
      ];
      const graph: Graph = {
        ...buildCSR(edges, numNodes),
        nodeFeatures: new Float64Array(numNodes * featureDim),
        featureDim,
      };

      // Fill node features
      const rng = createPRNG(99);
      for (let i = 0; i < graph.nodeFeatures.length; i++) {
        (graph.nodeFeatures as Float64Array)[i] = rng() * 2 - 1;
      }

      // Edge types: alternating 0 and 1
      const edgeTypes = new Uint8Array(graph.numEdges);
      for (let i = 0; i < edgeTypes.length; i++) {
        edgeTypes[i] = i % numEdgeTypes;
      }

      // Message weight matrices (one per edge type)
      const W_msg: Float64Array[] = [];
      for (let t = 0; t < numEdgeTypes; t++) {
        W_msg.push(xavierInit(featureDim, featureDim, rng));
      }

      // GRU weights
      const W_gru = makeGRUWeights(featureDim, 200);

      const output = sceneGraphForward(graph, graph.nodeFeatures, edgeTypes, W_msg, W_gru, featureDim, numEdgeTypes);

      // Output should be numNodes * featureDim
      expect(output.length).toBe(numNodes * featureDim);
      expect(allFinite(output)).toBe(true);
    });

    it('returns empty array for empty graph', () => {
      const emptyGraph: Graph = {
        numNodes: 0,
        numEdges: 0,
        rowPtr: new Uint32Array(1),
        colIdx: new Uint32Array(0),
        nodeFeatures: new Float64Array(0),
        featureDim: 4,
      };

      const output = sceneGraphForward(
        emptyGraph,
        emptyGraph.nodeFeatures,
        new Uint8Array(0),
        [],
        makeGRUWeights(4, 300),
        4,
        0,
      );
      expect(output.length).toBe(0);
    });
  });

  // =========================================================================
  // globalMeanPool and globalSumPool
  // =========================================================================
  describe('pooling operations', () => {
    it('globalMeanPool averages node features correctly', () => {
      const X = new Float64Array([1, 2, 3, 4, 5, 6]); // 3 nodes, dim=2
      const pooled = globalMeanPool(X, 3, 2);

      expect(pooled.length).toBe(2);
      expect(pooled[0]).toBeCloseTo((1 + 3 + 5) / 3, 10);
      expect(pooled[1]).toBeCloseTo((2 + 4 + 6) / 3, 10);
    });

    it('globalSumPool sums node features correctly', () => {
      const X = new Float64Array([1, 2, 3, 4, 5, 6]); // 3 nodes, dim=2
      const pooled = globalSumPool(X, 3, 2);

      expect(pooled.length).toBe(2);
      expect(pooled[0]).toBeCloseTo(1 + 3 + 5, 10);
      expect(pooled[1]).toBeCloseTo(2 + 4 + 6, 10);
    });
  });
});
