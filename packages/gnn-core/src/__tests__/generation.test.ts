// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-5: Graph Generation Tests
// Tests for graphRNNGenerate, granGenerate, forceDirectedLayout,
// and constrainedOptimization.
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';

// Types and PRNG
import {
  createPRNG,
  type GraphRNNConfig,
  type GraphRNNWeights,
  type GRUWeights,
  type GRANConfig,
  type ForceDirectedConfig,
  type Graph,
} from '../types.js';

// Graph utilities
import { buildCSR } from '../graph.js';

// Tensor utilities
import { xavierInit } from '../tensor.js';

// Generation module under test
import { graphRNNGenerate } from '../generation/graph-rnn.js';
import { granGenerate } from '../generation/gran.js';
import { forceDirectedLayout, constrainedOptimization } from '../generation/graph-to-layout.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Create GRU weights for GraphRNN. */
function makeGRUWeights(inputDim: number, hiddenDim: number, seed: number): GRUWeights {
  const rng = createPRNG(seed);
  return {
    W_z: xavierInit(hiddenDim, inputDim, rng),
    U_z: xavierInit(hiddenDim, hiddenDim, rng),
    b_z: new Float64Array(hiddenDim),
    W_r: xavierInit(hiddenDim, inputDim, rng),
    U_r: xavierInit(hiddenDim, hiddenDim, rng),
    b_r: new Float64Array(hiddenDim),
    W_h: xavierInit(hiddenDim, inputDim, rng),
    U_h: xavierInit(hiddenDim, hiddenDim, rng),
    b_h: new Float64Array(hiddenDim),
  };
}

/** Create GraphRNN weights for testing. */
function makeGraphRNNWeights(hiddenDim: number, seed: number): GraphRNNWeights {
  const rng = createPRNG(seed);
  // Node GRU input dim = 1 (scalar summary of edges or SOS token)
  const nodeGRU = makeGRUWeights(1, hiddenDim, seed + 1);
  // Edge GRU input dim = 1 (scalar: sampled edge bit or SOS)
  const edgeGRU = makeGRUWeights(1, hiddenDim, seed + 2);

  // nodeOutputW: 1 x hiddenDim, nodeOutputBias: 1
  const nodeOutputW = xavierInit(1, hiddenDim, rng);
  const nodeOutputBias = new Float64Array(1);

  // edgeOutputW: 1 x hiddenDim, edgeOutputBias: 1
  const edgeOutputW = xavierInit(1, hiddenDim, rng);
  const edgeOutputBias = new Float64Array(1);

  return {
    nodeGRU,
    edgeGRU,
    nodeOutputW,
    nodeOutputBias,
    edgeOutputW,
    edgeOutputBias,
  };
}

/** Check that every element in a Float64Array is finite. */
function allFinite(arr: Float64Array): boolean {
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i]!)) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('GNN-5: Graph Generation', () => {
  // =========================================================================
  // graphRNNGenerate
  // =========================================================================
  describe('graphRNNGenerate', () => {
    it('generates a graph with valid adjacency and numNodes > 0', () => {
      const hiddenDim = 8;
      const config: GraphRNNConfig = {
        maxNodes: 6,
        hiddenDim,
        edgeHorizon: 3,
      };
      const weights = makeGraphRNNWeights(hiddenDim, 42);
      const rng = createPRNG(100);

      const result = graphRNNGenerate(weights, config, rng);

      // Must have at least 1 node (always generates first node)
      expect(result.numNodes).toBeGreaterThanOrEqual(1);
      expect(result.numNodes).toBeLessThanOrEqual(config.maxNodes);

      // Adjacency should be n x n
      const n = result.numNodes;
      expect(result.adjacency.length).toBe(n * n);

      // Adjacency should be symmetric (undirected)
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          expect(result.adjacency[i * n + j]).toBe(result.adjacency[j * n + i]);
        }
      }

      // Diagonal should be 0 (no self-loops)
      for (let i = 0; i < n; i++) {
        expect(result.adjacency[i * n + i]).toBe(0);
      }
    });

    it('respects maxNodes limit', () => {
      const hiddenDim = 8;
      const config: GraphRNNConfig = {
        maxNodes: 3,
        hiddenDim,
        edgeHorizon: 2,
      };
      const weights = makeGraphRNNWeights(hiddenDim, 55);
      const rng = createPRNG(200);

      const result = graphRNNGenerate(weights, config, rng);
      expect(result.numNodes).toBeLessThanOrEqual(3);
    });
  });

  // =========================================================================
  // granGenerate
  // =========================================================================
  describe('granGenerate', () => {
    it('generates a graph with correct adjacency size', () => {
      const hiddenDim = 8;
      const config: GRANConfig = {
        maxNodes: 5,
        blockSize: 2,
        hiddenDim,
        numMixtures: 2,
      };

      const rng = createPRNG(300);
      // W_node: hiddenDim x hiddenDim
      const W_node = xavierInit(hiddenDim, hiddenDim, createPRNG(301));
      // W_edge: numMixtures x (2 * hiddenDim)
      const W_edge = xavierInit(config.numMixtures, 2 * hiddenDim, createPRNG(302));

      const result = granGenerate(config, W_node, W_edge, rng);

      // Should produce exactly maxNodes nodes (GRAN always fills to maxNodes)
      expect(result.numNodes).toBe(config.maxNodes);

      // Adjacency should be maxNodes x maxNodes
      expect(result.adjacency.length).toBe(config.maxNodes * config.maxNodes);

      // Adjacency should be symmetric
      const n = result.numNodes;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          expect(result.adjacency[i * n + j]).toBe(result.adjacency[j * n + i]);
        }
      }
    });

    it('handles blockSize=1 gracefully', () => {
      const hiddenDim = 4;
      const config: GRANConfig = {
        maxNodes: 3,
        blockSize: 1,
        hiddenDim,
        numMixtures: 1,
      };

      const rng = createPRNG(400);
      const W_node = xavierInit(hiddenDim, hiddenDim, createPRNG(401));
      const W_edge = xavierInit(config.numMixtures, 2 * hiddenDim, createPRNG(402));

      const result = granGenerate(config, W_node, W_edge, rng);
      expect(result.numNodes).toBe(3);
      expect(result.adjacency.length).toBe(9);
    });
  });

  // =========================================================================
  // forceDirectedLayout
  // =========================================================================
  describe('forceDirectedLayout', () => {
    it('produces positions that are all finite and converge within bounds', () => {
      // Create a small connected graph
      const edges: [number, number][] = [
        [0, 1], [1, 0],
        [1, 2], [2, 1],
        [2, 3], [3, 2],
        [3, 0], [0, 3],
      ];
      const numNodes = 4;
      const graph: Graph = {
        ...buildCSR(edges, numNodes),
        nodeFeatures: new Float64Array(0),
        featureDim: 0,
      };

      const config: ForceDirectedConfig = {
        iterations: 50,
        learningRate: 1.0,
        attractionStrength: 1.0,
        repulsionStrength: 1.0,
        idealEdgeLength: 1.0,
      };

      const rng = createPRNG(500);
      const positions = forceDirectedLayout(graph, config, rng);

      // Should produce numNodes * 2 values (x, y per node)
      expect(positions.length).toBe(numNodes * 2);
      expect(allFinite(positions)).toBe(true);

      // After convergence, positions should be reasonable (not NaN/Inf)
      for (let i = 0; i < positions.length; i++) {
        expect(Math.abs(positions[i]!)).toBeLessThan(1e6);
      }
    });

    it('produces different positions for disconnected vs connected graphs', () => {
      // Connected graph
      const edgesConnected: [number, number][] = [
        [0, 1], [1, 0], [1, 2], [2, 1],
      ];
      const graphConnected: Graph = {
        ...buildCSR(edgesConnected, 3),
        nodeFeatures: new Float64Array(0),
        featureDim: 0,
      };

      // Disconnected graph (no edges)
      const graphDisconnected: Graph = {
        ...buildCSR([], 3),
        nodeFeatures: new Float64Array(0),
        featureDim: 0,
      };

      const config: ForceDirectedConfig = {
        iterations: 30,
        learningRate: 1.0,
        attractionStrength: 1.0,
        repulsionStrength: 1.0,
        idealEdgeLength: 1.0,
      };

      const pos1 = forceDirectedLayout(graphConnected, config, createPRNG(600));
      const pos2 = forceDirectedLayout(graphDisconnected, config, createPRNG(600));

      // Both should be finite
      expect(allFinite(pos1)).toBe(true);
      expect(allFinite(pos2)).toBe(true);

      // They should differ (attraction vs no attraction changes layout)
      let allSame = true;
      for (let i = 0; i < pos1.length; i++) {
        if (Math.abs(pos1[i]! - pos2[i]!) > 1e-6) {
          allSame = false;
          break;
        }
      }
      expect(allSame).toBe(false);
    });
  });

  // =========================================================================
  // constrainedOptimization
  // =========================================================================
  describe('constrainedOptimization', () => {
    it('keeps positions within specified bounds', () => {
      // Start with positions outside bounds
      const positions = new Float64Array([
        -5, -5,   // node 0: way outside
        15, 15,   // node 1: way outside
        5, 5,     // node 2: inside
        0, 10,    // node 3: partially outside
      ]);

      const constraints = {
        minDist: 0.5,
        bounds: [0, 0, 10, 10] as [number, number, number, number],
      };

      const result = constrainedOptimization(positions, constraints, 10);

      expect(result.length).toBe(8);
      expect(allFinite(result)).toBe(true);

      // All positions should be within bounds [0, 0, 10, 10]
      for (let i = 0; i < 4; i++) {
        const x = result[i * 2]!;
        const y = result[i * 2 + 1]!;
        expect(x).toBeGreaterThanOrEqual(0);
        expect(x).toBeLessThanOrEqual(10);
        expect(y).toBeGreaterThanOrEqual(0);
        expect(y).toBeLessThanOrEqual(10);
      }
    });

    it('pushes coincident points apart to minDist', () => {
      // Two coincident points at (5, 5)
      const positions = new Float64Array([
        5, 5,
        5, 5,
      ]);

      const constraints = {
        minDist: 2.0,
        bounds: [0, 0, 10, 10] as [number, number, number, number],
      };

      const result = constrainedOptimization(positions, constraints, 20);

      // Nodes should be pushed apart by at least minDist
      const dx = result[0]! - result[2]!;
      const dy = result[1]! - result[3]!;
      const dist = Math.sqrt(dx * dx + dy * dy);

      // After many iterations, distance should approach minDist
      expect(dist).toBeGreaterThanOrEqual(constraints.minDist * 0.9);
    });

    it('does not modify a single point within bounds', () => {
      const positions = new Float64Array([5, 5]);
      const constraints = {
        minDist: 1.0,
        bounds: [0, 0, 10, 10] as [number, number, number, number],
      };

      const result = constrainedOptimization(positions, constraints, 10);
      expect(result[0]).toBeCloseTo(5, 10);
      expect(result[1]).toBeCloseTo(5, 10);
    });
  });
});
