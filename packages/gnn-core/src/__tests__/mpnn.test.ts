// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — MPNN Foundation Tests (GNN-1)
// Tests for GCN, GraphSAGE, GAT, GIN, and over-smoothing mitigations.
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';

// Types and PRNG
import {
  createPRNG,
  type Graph,
  type GCNWeights,
  type GCNConfig,
  type SAGEWeights,
  type SAGEConfig,
  type GATWeights,
  type GATConfig,
  type GINWeights,
  type GINConfig,
  type MLPWeights,
} from '../types.js';

// Graph utilities
import { buildCSR, addSelfLoops } from '../graph.js';

// Tensor utilities
import { xavierInit } from '../tensor.js';

// MPNN layers under test
import { gcnLayer, gcnForward } from '../mpnn/gcn.js';
import { sageMeanLayer, sageMaxLayer, sageForward } from '../mpnn/sage.js';
import { gatLayer, gatv2Layer, gatForward, getAttentionWeights } from '../mpnn/gat.js';
import { ginLayer, ginGraphReadout, ginForward } from '../mpnn/gin.js';
import { residualConnection, jkNetCombine, dropEdge } from '../mpnn/over-smoothing.js';

// ---------------------------------------------------------------------------
// Helper: Build a small test graph (4 nodes, undirected edges: 0-1, 1-2, 2-3, 0-3)
// with 3-dimensional node features.
// ---------------------------------------------------------------------------

function makeTestGraph(): Graph {
  const numNodes = 4;
  const featureDim = 3;

  // Undirected edges: each undirected edge becomes two directed edges
  const edges: [number, number][] = [
    [0, 1], [1, 0],
    [1, 2], [2, 1],
    [2, 3], [3, 2],
    [0, 3], [3, 0],
  ];

  const base = buildCSR(edges, numNodes);

  // Deterministic node features
  const rng = createPRNG(42);
  const nodeFeatures = new Float64Array(numNodes * featureDim);
  for (let i = 0; i < nodeFeatures.length; i++) {
    nodeFeatures[i] = rng() * 2 - 1; // values in [-1, 1]
  }

  return {
    ...base,
    nodeFeatures,
    featureDim,
  };
}

/** Create GCN weights with Xavier initialization. */
function makeGCNWeights(inDim: number, outDim: number, seed: number, bias: boolean): GCNWeights {
  const rng = createPRNG(seed);
  const W = xavierInit(inDim, outDim, rng);
  return bias
    ? { W, bias: new Float64Array(outDim) }
    : { W };
}

/** Create SAGEWeights with Xavier initialization. */
function makeSAGEWeights(inDim: number, outDim: number, seed: number, bias: boolean): SAGEWeights {
  const rng = createPRNG(seed);
  const W_self = xavierInit(inDim, outDim, rng);
  const W_neigh = xavierInit(inDim, outDim, rng);
  return bias
    ? { W_self, W_neigh, bias: new Float64Array(outDim) }
    : { W_self, W_neigh };
}

/** Create GATWeights with Xavier initialization. */
function makeGATWeights(
  inDim: number,
  outDim: number,
  heads: number,
  seed: number,
  v2: boolean,
): GATWeights {
  const rng = createPRNG(seed);
  const W = xavierInit(inDim, outDim * heads, rng);
  const a_src = xavierInit(1, outDim * heads, rng);
  const a_dst = xavierInit(1, outDim * heads, rng);

  if (v2) {
    const W_src = xavierInit(inDim, outDim * heads, rng);
    const W_dst = xavierInit(inDim, outDim * heads, rng);
    const a = xavierInit(1, outDim, rng);
    return { W, a_src, a_dst, W_src, W_dst, a };
  }

  return { W, a_src, a_dst };
}

/** Create a 2-layer MLP weights for GIN. */
function makeMLPWeights(inDim: number, hiddenDim: number, outDim: number, seed: number): MLPWeights {
  const rng = createPRNG(seed);
  return {
    layers: [
      {
        W: xavierInit(inDim, hiddenDim, rng),
        bias: new Float64Array(hiddenDim),
        inDim,
        outDim: hiddenDim,
      },
      {
        W: xavierInit(hiddenDim, outDim, rng),
        bias: new Float64Array(outDim),
        inDim: hiddenDim,
        outDim,
      },
    ],
  };
}

/** Create GIN weights. */
function makeGINWeights(
  inDim: number,
  hiddenDim: number,
  outDim: number,
  epsilon: number,
  seed: number,
): GINWeights {
  return {
    mlp: makeMLPWeights(inDim, hiddenDim, outDim, seed),
    epsilon,
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

describe('GNN-1: MPNN Foundation', () => {
  // Shared test graph: 4 nodes, featureDim=3, undirected edges 0-1, 1-2, 2-3, 0-3
  const graph = makeTestGraph();
  const graphWithLoops = addSelfLoops(graph);

  // =========================================================================
  // GCN
  // =========================================================================
  describe('GCN', () => {
    const inDim = 3;
    const outDim = 2;

    it('produces correct output dimensions', () => {
      const weights = makeGCNWeights(inDim, outDim, 100, true);
      const config: GCNConfig = { inDim, outDim, bias: true, activation: 'relu' };

      const output = gcnLayer(graphWithLoops, graph.nodeFeatures, weights, config);

      // 4 nodes * outDim=2 => length 8
      expect(output).toBeInstanceOf(Float64Array);
      expect(output.length).toBe(4 * outDim);
    });

    it('outputs finite values with random weights', () => {
      const weights = makeGCNWeights(inDim, outDim, 200, true);
      const config: GCNConfig = { inDim, outDim, bias: true, activation: 'relu' };

      const output = gcnLayer(graphWithLoops, graph.nodeFeatures, weights, config);

      expect(allFinite(output)).toBe(true);
    });

    it('multi-layer forward preserves dimensions', () => {
      const hiddenDim = 4;
      const finalDim = 2;

      const weights1 = makeGCNWeights(inDim, hiddenDim, 301, true);
      const weights2 = makeGCNWeights(hiddenDim, finalDim, 302, true);

      const configs: GCNConfig[] = [
        { inDim, outDim: hiddenDim, bias: true, activation: 'relu' },
        { inDim: hiddenDim, outDim: finalDim, bias: true, activation: 'none' },
      ];

      const output = gcnForward(graphWithLoops, graph.nodeFeatures, [
        { weights: weights1, config: configs[0]! },
        { weights: weights2, config: configs[1]! },
      ]);

      // 4 nodes * finalDim=2 => length 8
      expect(output).toBeInstanceOf(Float64Array);
      expect(output.length).toBe(4 * finalDim);
      expect(allFinite(output)).toBe(true);
    });
  });

  // =========================================================================
  // GraphSAGE
  // =========================================================================
  describe('GraphSAGE', () => {
    const inDim = 3;
    const outDim = 2;

    it('mean aggregation produces correct dimensions', () => {
      const weights = makeSAGEWeights(inDim, outDim, 400, true);
      const config: SAGEConfig = {
        inDim,
        outDim,
        aggregator: 'mean',
        normalize: false,
        activation: 'relu',
      };

      const output = sageMeanLayer(graph, graph.nodeFeatures, weights, config);

      expect(output).toBeInstanceOf(Float64Array);
      expect(output.length).toBe(4 * outDim);
      expect(allFinite(output)).toBe(true);
    });

    it('max aggregation produces correct dimensions', () => {
      const weights = makeSAGEWeights(inDim, outDim, 500, true);
      const config: SAGEConfig = {
        inDim,
        outDim,
        aggregator: 'max',
        normalize: false,
        activation: 'relu',
      };

      const output = sageMaxLayer(graph, graph.nodeFeatures, weights, config);

      expect(output).toBeInstanceOf(Float64Array);
      expect(output.length).toBe(4 * outDim);
      expect(allFinite(output)).toBe(true);
    });

    it('L2 normalization makes unit vectors', () => {
      const weights = makeSAGEWeights(inDim, outDim, 600, true);
      const config: SAGEConfig = {
        inDim,
        outDim,
        aggregator: 'mean',
        normalize: true,
        activation: 'relu',
      };

      const output = sageForward(graph, graph.nodeFeatures, [{ weights, config }]);

      // Each node's output vector (length outDim) should have L2 norm approximately 1
      for (let node = 0; node < 4; node++) {
        let normSq = 0;
        for (let d = 0; d < outDim; d++) {
          const val = output[node * outDim + d]!;
          normSq += val * val;
        }
        const norm = Math.sqrt(normSq);
        // Norm should be approximately 1 (unless the vector is all-zero from ReLU)
        if (norm > 1e-8) {
          expect(norm).toBeCloseTo(1.0, 4);
        }
      }
    });
  });

  // =========================================================================
  // GAT
  // =========================================================================
  describe('GAT', () => {
    const inDim = 3;
    const outDim = 3;

    it('attention weights sum to 1 per node', () => {
      const heads = 1;
      const weights = makeGATWeights(inDim, outDim, heads, 700, false);
      const config: GATConfig = {
        inDim,
        outDim,
        heads,
        dropout: 0,
        negativeSlope: 0.2,
        concat: true,
        v2: false,
      };

      const attnWeights = getAttentionWeights(graphWithLoops, graph.nodeFeatures, weights, config);

      // attnWeights should contain attention coefficients per node
      // For each node, the attention weights over its neighbors should sum to ~1
      for (let node = 0; node < graph.numNodes; node++) {
        const start = graphWithLoops.rowPtr[node]!;
        const end = graphWithLoops.rowPtr[node + 1]!;
        let attnSum = 0;
        for (let e = start; e < end; e++) {
          attnSum += attnWeights[e]!;
        }
        expect(attnSum).toBeCloseTo(1.0, 5);
      }
    });

    it('GATv2 produces different attention than GAT', () => {
      const heads = 1;
      const seed = 800;
      const weightsV1 = makeGATWeights(inDim, outDim, heads, seed, false);
      const weightsV2 = makeGATWeights(inDim, outDim, heads, seed, true);

      const configV1: GATConfig = {
        inDim,
        outDim,
        heads,
        dropout: 0,
        negativeSlope: 0.2,
        concat: true,
        v2: false,
      };

      const configV2: GATConfig = {
        inDim,
        outDim,
        heads,
        dropout: 0,
        negativeSlope: 0.2,
        concat: true,
        v2: true,
      };

      const rng1 = createPRNG(111);
      const rng2 = createPRNG(111);
      const outputV1 = gatLayer(graphWithLoops, graph.nodeFeatures, weightsV1, configV1, rng1);
      const outputV2 = gatv2Layer(graphWithLoops, graph.nodeFeatures, weightsV2, configV2, rng2);

      // Both outputs should be valid
      expect(allFinite(outputV1)).toBe(true);
      expect(allFinite(outputV2)).toBe(true);

      // They should produce different results (different attention mechanisms)
      let allEqual = true;
      for (let i = 0; i < outputV1.length; i++) {
        if (Math.abs(outputV1[i]! - outputV2[i]!) > 1e-10) {
          allEqual = false;
          break;
        }
      }
      expect(allEqual).toBe(false);
    });

    it('multi-head concat output dimension is heads * outDim', () => {
      const heads = 2;
      const perHeadOut = 3;
      const weights = makeGATWeights(inDim, perHeadOut, heads, 900, false);
      const config: GATConfig = {
        inDim,
        outDim: perHeadOut,
        heads,
        dropout: 0,
        negativeSlope: 0.2,
        concat: true,
        v2: false,
      };

      const rng = createPRNG(901);
      const output = gatForward(graphWithLoops, graph.nodeFeatures, [{ weights, config }], rng);

      // concat=true: output dim = heads * outDim = 2 * 3 = 6
      const expectedDim = heads * perHeadOut;
      expect(output.length).toBe(4 * expectedDim);
      expect(allFinite(output)).toBe(true);
    });

    it('multi-head average output dimension is outDim', () => {
      const heads = 2;
      const perHeadOut = 3;
      const weights = makeGATWeights(inDim, perHeadOut, heads, 1000, false);
      const config: GATConfig = {
        inDim,
        outDim: perHeadOut,
        heads,
        dropout: 0,
        negativeSlope: 0.2,
        concat: false,
        v2: false,
      };

      const rng = createPRNG(1001);
      const output = gatForward(graphWithLoops, graph.nodeFeatures, [{ weights, config }], rng);

      // concat=false: output dim = outDim = 3 (averaged across heads)
      expect(output.length).toBe(4 * perHeadOut);
      expect(allFinite(output)).toBe(true);
    });
  });

  // =========================================================================
  // GIN
  // =========================================================================
  describe('GIN', () => {
    const inDim = 3;
    const hiddenDim = 4;
    const outDim = 2;

    it('produces correct output dimensions', () => {
      const weights = makeGINWeights(inDim, hiddenDim, outDim, 0.0, 1100);
      const config: GINConfig = {
        inDim,
        hiddenDim,
        outDim,
        epsilon: 0.0,
        trainEpsilon: false,
      };

      const output = ginLayer(graph, graph.nodeFeatures, weights, config);

      // 4 nodes * outDim=2 => length 8
      expect(output).toBeInstanceOf(Float64Array);
      expect(output.length).toBe(4 * outDim);
      expect(allFinite(output)).toBe(true);
    });

    it('graph readout produces per-graph embeddings', () => {
      // Simulate a batch of 2 graphs: nodes 0,1 belong to graph 0; nodes 2,3 belong to graph 1
      const batchIndex = new Uint32Array([0, 0, 1, 1]);
      const numGraphs = 2;
      const featureDim = 3;

      // Use the node features from our test graph (4 nodes * 3 dims)
      const output = ginGraphReadout(graph.nodeFeatures, batchIndex, numGraphs, featureDim);

      // Output should be numGraphs * featureDim = 2 * 3 = 6
      expect(output).toBeInstanceOf(Float64Array);
      expect(output.length).toBe(numGraphs * featureDim);
      expect(allFinite(output)).toBe(true);
    });

    it('epsilon=0 reduces to sum aggregation', () => {
      const weights = makeGINWeights(inDim, hiddenDim, outDim, 0.0, 1200);
      const config: GINConfig = {
        inDim,
        hiddenDim,
        outDim,
        epsilon: 0.0,
        trainEpsilon: false,
      };

      const output = ginForward(graph, graph.nodeFeatures, [{ weights, config }]);

      // With epsilon=0, the layer computes MLP((1+0)*h_v + sum_neighbors(h_u)) = MLP(h_v + sum(h_u))
      expect(output).toBeInstanceOf(Float64Array);
      expect(output.length).toBe(4 * outDim);
      expect(allFinite(output)).toBe(true);
    });
  });

  // =========================================================================
  // Over-smoothing Mitigations
  // =========================================================================
  describe('Over-smoothing mitigations', () => {
    it('residual connection adds input to output', () => {
      const input = new Float64Array([1.0, 2.0, 3.0, 4.0]);
      const output = new Float64Array([0.5, -1.0, 2.0, 0.0]);

      const result = residualConnection(input, output);

      // result[i] = input[i] + output[i]
      expect(result.length).toBe(4);
      expect(result[0]).toBeCloseTo(1.5, 10);
      expect(result[1]).toBeCloseTo(1.0, 10);
      expect(result[2]).toBeCloseTo(5.0, 10);
      expect(result[3]).toBeCloseTo(4.0, 10);
    });

    it('JKNet concat produces L*dim features', () => {
      const dim = 2;
      const numLayers = 3;

      // 3 layers of output, each with 4 nodes * 2 dims
      const layerOutputs: Float64Array[] = [];
      for (let l = 0; l < numLayers; l++) {
        const arr = new Float64Array(4 * dim);
        for (let i = 0; i < arr.length; i++) {
          arr[i] = (l + 1) * 0.1 * (i + 1);
        }
        layerOutputs.push(arr);
      }

      const result = jkNetCombine(layerOutputs, 'concat', dim);

      // concat: each node gets L * dim features = 3 * 2 = 6
      const expectedDim = numLayers * dim;
      expect(result.length).toBe(4 * expectedDim);
      expect(allFinite(result)).toBe(true);
    });

    it('JKNet max produces same dim as input', () => {
      const dim = 2;
      const numLayers = 3;

      const layerOutputs: Float64Array[] = [];
      for (let l = 0; l < numLayers; l++) {
        const arr = new Float64Array(4 * dim);
        for (let i = 0; i < arr.length; i++) {
          arr[i] = (l + 1) * 0.1 * (i + 1);
        }
        layerOutputs.push(arr);
      }

      const result = jkNetCombine(layerOutputs, 'max', dim);

      // max: element-wise max across layers => same dim as input
      expect(result.length).toBe(4 * dim);
      expect(allFinite(result)).toBe(true);

      // The max should be from the last layer (highest multiplier: 0.3 * (i+1))
      for (let i = 0; i < 4 * dim; i++) {
        const expectedMax = numLayers * 0.1 * (i + 1);
        expect(result[i]).toBeCloseTo(expectedMax, 10);
      }
    });

    it('dropEdge removes approximately p fraction of edges', () => {
      const rng = createPRNG(1337);
      const p = 0.5;
      const originalEdges = graph.numEdges; // 8 directed edges

      const dropped = dropEdge(graph, p, rng);

      // The dropped graph should have fewer edges
      expect(dropped.numEdges).toBeLessThanOrEqual(originalEdges);
      expect(dropped.numEdges).toBeGreaterThanOrEqual(0);

      // With p=0.5, expect roughly half removed. Allow wide tolerance due to small sample.
      // For 8 edges with p=0.5, expect ~4 remaining, but accept [1, 7]
      expect(dropped.numEdges).toBeGreaterThanOrEqual(1);
      expect(dropped.numEdges).toBeLessThanOrEqual(7);

      // numNodes should be unchanged
      expect(dropped.numNodes).toBe(graph.numNodes);

      // Node features and featureDim should be preserved
      expect(dropped.featureDim).toBe(graph.featureDim);
      expect(dropped.nodeFeatures.length).toBe(graph.nodeFeatures.length);
    });
  });
});
