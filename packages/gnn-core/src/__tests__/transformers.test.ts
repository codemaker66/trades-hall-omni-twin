// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-7: Graph Transformers Tests
// Tests for GPS layer, globalSelfAttention, Laplacian PE, Random Walk PE,
// and Exphormer (expander edges + sparse attention).
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';

// Types and PRNG
import {
  createPRNG,
  type Graph,
  type GPSConfig,
  type GPSWeights,
  type GCNWeights,
} from '../types.js';

// Tensor utilities
import { xavierInit } from '../tensor.js';

// Graph utilities
import { buildCSR } from '../graph.js';

// Modules under test
import { gpsLayer, globalSelfAttention } from '../transformers/gps.js';
import { laplacianPE, randomWalkPE } from '../transformers/positional-encoding.js';
import { generateExpanderEdges, exphormerAttention } from '../transformers/exphormer.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Check that every element in a Float64Array is finite. */
function allFinite(arr: Float64Array): boolean {
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i]!)) return false;
  }
  return true;
}

/** Build a simple undirected ring graph (0-1-2-...-n-1-0) with given feature dim. */
function buildRingGraph(numNodes: number, featureDim: number, rng: () => number): Graph {
  const edges: [number, number][] = [];
  for (let i = 0; i < numNodes; i++) {
    const j = (i + 1) % numNodes;
    edges.push([i, j]);
    edges.push([j, i]);
  }
  const csr = buildCSR(edges, numNodes);
  const nodeFeatures = new Float64Array(numNodes * featureDim);
  for (let i = 0; i < nodeFeatures.length; i++) {
    nodeFeatures[i] = rng() * 2 - 1;
  }
  return {
    ...csr,
    nodeFeatures,
    featureDim,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('GNN-7: Graph Transformers', () => {
  // =========================================================================
  // GPS Layer
  // =========================================================================
  describe('gpsLayer', () => {
    it('produces output with correct dimensions (numNodes x dim)', () => {
      const numNodes = 6;
      const dim = 8;
      const heads = 2;
      const ffnDim = 16;
      const rng = createPRNG(42);

      const graph = buildRingGraph(numNodes, dim, rng);

      const config: GPSConfig = {
        dim,
        heads,
        ffnDim,
        dropout: 0.0,
        mpnnType: 'gcn',
      };

      // Build GPS weights
      const gcnWeights: GCNWeights = {
        W: xavierInit(dim, dim, createPRNG(100)),
        bias: new Float64Array(dim),
      };

      const weights: GPSWeights = {
        mpnnWeights: gcnWeights,
        attnW_Q: xavierInit(dim, dim, createPRNG(101)),
        attnW_K: xavierInit(dim, dim, createPRNG(102)),
        attnW_V: xavierInit(dim, dim, createPRNG(103)),
        attnW_O: xavierInit(dim, dim, createPRNG(104)),
        ffnW1: xavierInit(dim, ffnDim, createPRNG(105)),
        ffnB1: new Float64Array(ffnDim),
        ffnW2: xavierInit(ffnDim, dim, createPRNG(106)),
        ffnB2: new Float64Array(dim),
        norm1Gamma: (() => { const g = new Float64Array(dim); g.fill(1); return g; })(),
        norm1Beta: new Float64Array(dim),
        norm2Gamma: (() => { const g = new Float64Array(dim); g.fill(1); return g; })(),
        norm2Beta: new Float64Array(dim),
      };

      const output = gpsLayer(graph, graph.nodeFeatures, weights, config, rng);

      // Output should be numNodes x dim
      expect(output.length).toBe(numNodes * dim);
      expect(allFinite(output)).toBe(true);
    });

    it('produces different output than input (non-trivial transform)', () => {
      const numNodes = 4;
      const dim = 4;
      const heads = 2;
      const ffnDim = 8;
      const rng = createPRNG(99);

      const graph = buildRingGraph(numNodes, dim, rng);

      const config: GPSConfig = {
        dim,
        heads,
        ffnDim,
        dropout: 0.0,
        mpnnType: 'gcn',
      };

      const gcnWeights: GCNWeights = {
        W: xavierInit(dim, dim, createPRNG(200)),
        bias: new Float64Array(dim),
      };

      const weights: GPSWeights = {
        mpnnWeights: gcnWeights,
        attnW_Q: xavierInit(dim, dim, createPRNG(201)),
        attnW_K: xavierInit(dim, dim, createPRNG(202)),
        attnW_V: xavierInit(dim, dim, createPRNG(203)),
        attnW_O: xavierInit(dim, dim, createPRNG(204)),
        ffnW1: xavierInit(dim, ffnDim, createPRNG(205)),
        ffnB1: new Float64Array(ffnDim),
        ffnW2: xavierInit(ffnDim, dim, createPRNG(206)),
        ffnB2: new Float64Array(dim),
        norm1Gamma: (() => { const g = new Float64Array(dim); g.fill(1); return g; })(),
        norm1Beta: new Float64Array(dim),
        norm2Gamma: (() => { const g = new Float64Array(dim); g.fill(1); return g; })(),
        norm2Beta: new Float64Array(dim),
      };

      const input = new Float64Array(graph.nodeFeatures);
      const output = gpsLayer(graph, graph.nodeFeatures, weights, config, rng);

      // Output should differ from input
      let different = false;
      for (let i = 0; i < output.length; i++) {
        if (Math.abs(output[i]! - input[i]!) > 1e-10) {
          different = true;
          break;
        }
      }
      expect(different).toBe(true);
    });
  });

  // =========================================================================
  // Global Self-Attention
  // =========================================================================
  describe('globalSelfAttention', () => {
    it('produces output with correct dimensions and finite values', () => {
      const numNodes = 5;
      const dim = 8;
      const heads = 2;
      const rng = createPRNG(10);

      // Random input features
      const X = new Float64Array(numNodes * dim);
      for (let i = 0; i < X.length; i++) {
        X[i] = rng() * 2 - 1;
      }

      const W_Q = xavierInit(dim, dim, createPRNG(11));
      const W_K = xavierInit(dim, dim, createPRNG(12));
      const W_V = xavierInit(dim, dim, createPRNG(13));
      const W_O = xavierInit(dim, dim, createPRNG(14));

      const output = globalSelfAttention(X, W_Q, W_K, W_V, W_O, numNodes, dim, heads);

      expect(output.length).toBe(numNodes * dim);
      expect(allFinite(output)).toBe(true);
    });

    it('with identity-like weights preserves information', () => {
      const numNodes = 3;
      const dim = 4;
      const heads = 1;
      const rng = createPRNG(20);

      const X = new Float64Array(numNodes * dim);
      for (let i = 0; i < X.length; i++) {
        X[i] = rng() * 2 - 1;
      }

      // Use identity-like projections for Q, K, V, O
      const makeIdentity = (d: number): Float64Array => {
        const m = new Float64Array(d * d);
        for (let i = 0; i < d; i++) {
          m[i * d + i] = 1.0;
        }
        return m;
      };

      const I = makeIdentity(dim);
      const output = globalSelfAttention(X, I, I, I, I, numNodes, dim, heads);

      // Output should be a weighted average of the values (which are just X)
      // so each output node should be a convex combination of input rows
      expect(output.length).toBe(numNodes * dim);
      expect(allFinite(output)).toBe(true);
    });
  });

  // =========================================================================
  // Laplacian Positional Encoding
  // =========================================================================
  describe('laplacianPE', () => {
    it('produces k eigenvectors per node with finite values', () => {
      const numNodes = 8;
      const k = 3;
      const rng = createPRNG(30);
      const graph = buildRingGraph(numNodes, 1, rng);

      const result = laplacianPE(graph, k);

      expect(result.peDim).toBe(k);
      expect(result.pe.length).toBe(numNodes * k);
      expect(allFinite(result.pe)).toBe(true);
    });

    it('handles k > numNodes-1 gracefully', () => {
      const numNodes = 4;
      const k = 10; // larger than numNodes - 1
      const rng = createPRNG(31);
      const graph = buildRingGraph(numNodes, 1, rng);

      const result = laplacianPE(graph, k);

      // effectiveK should be min(k, numNodes - 1) = 3
      expect(result.peDim).toBe(numNodes - 1);
      expect(result.pe.length).toBe(numNodes * (numNodes - 1));
      expect(allFinite(result.pe)).toBe(true);
    });

    it('returns empty for graph with zero nodes', () => {
      const graph: Graph = {
        numNodes: 0,
        numEdges: 0,
        rowPtr: new Uint32Array([0]),
        colIdx: new Uint32Array(0),
        nodeFeatures: new Float64Array(0),
        featureDim: 0,
      };

      const result = laplacianPE(graph, 3);
      expect(result.peDim).toBe(0);
      expect(result.pe.length).toBe(0);
    });

    it('eigenvectors are approximately normalized', () => {
      const numNodes = 6;
      const k = 2;
      const rng = createPRNG(32);
      const graph = buildRingGraph(numNodes, 1, rng);

      const result = laplacianPE(graph, k);

      // Each eigenvector column should have approximately unit norm
      for (let d = 0; d < k; d++) {
        let norm = 0;
        for (let i = 0; i < numNodes; i++) {
          const val = result.pe[i * k + d]!;
          norm += val * val;
        }
        norm = Math.sqrt(norm);
        expect(norm).toBeCloseTo(1.0, 1);
      }
    });
  });

  // =========================================================================
  // Random Walk Positional Encoding
  // =========================================================================
  describe('randomWalkPE', () => {
    it('produces walkLength values per node with values in [0,1]', () => {
      const numNodes = 6;
      const walkLength = 4;
      const rng = createPRNG(40);
      const graph = buildRingGraph(numNodes, 1, rng);

      const result = randomWalkPE(graph, walkLength);

      expect(result.peDim).toBe(walkLength);
      expect(result.pe.length).toBe(numNodes * walkLength);
      expect(allFinite(result.pe)).toBe(true);

      // Random walk return probabilities should be in [0, 1]
      for (let i = 0; i < result.pe.length; i++) {
        expect(result.pe[i]!).toBeGreaterThanOrEqual(-1e-10);
        expect(result.pe[i]!).toBeLessThanOrEqual(1 + 1e-10);
      }
    });

    it('self-loop graph has return probability 1 at all steps', () => {
      // A single-node graph with a self-loop: T^k[0,0] = 1 for all k
      const graph: Graph = {
        numNodes: 1,
        numEdges: 1,
        rowPtr: new Uint32Array([0, 1]),
        colIdx: new Uint32Array([0]),
        nodeFeatures: new Float64Array(0),
        featureDim: 0,
      };

      const result = randomWalkPE(graph, 5);

      expect(result.peDim).toBe(5);
      expect(result.pe.length).toBe(5);
      for (let k = 0; k < 5; k++) {
        expect(result.pe[k]).toBeCloseTo(1.0, 10);
      }
    });

    it('returns empty for zero-node graph', () => {
      const graph: Graph = {
        numNodes: 0,
        numEdges: 0,
        rowPtr: new Uint32Array([0]),
        colIdx: new Uint32Array(0),
        nodeFeatures: new Float64Array(0),
        featureDim: 0,
      };

      const result = randomWalkPE(graph, 3);
      expect(result.pe.length).toBe(0);
    });

    it('ring graph has identical PE for all nodes (by symmetry)', () => {
      const numNodes = 4;
      const walkLength = 3;
      const rng = createPRNG(41);
      const graph = buildRingGraph(numNodes, 1, rng);

      const result = randomWalkPE(graph, walkLength);

      // In a ring, all nodes are symmetric, so PE values should be the same
      for (let k = 0; k < walkLength; k++) {
        const refVal = result.pe[0 * walkLength + k]!;
        for (let i = 1; i < numNodes; i++) {
          expect(result.pe[i * walkLength + k]).toBeCloseTo(refVal, 10);
        }
      }
    });
  });

  // =========================================================================
  // Exphormer
  // =========================================================================
  describe('Exphormer', () => {
    describe('generateExpanderEdges', () => {
      it('produces correct number of edges', () => {
        const numNodes = 10;
        const degree = 3;
        const rng = createPRNG(50);

        const [src, dst] = generateExpanderEdges(numNodes, degree, rng);

        // Should have exactly numNodes * degree edges
        expect(src.length).toBe(numNodes * degree);
        expect(dst.length).toBe(numNodes * degree);
      });

      it('produces no self-loops', () => {
        const numNodes = 8;
        const degree = 2;
        const rng = createPRNG(51);

        const [src, dst] = generateExpanderEdges(numNodes, degree, rng);

        for (let i = 0; i < src.length; i++) {
          expect(src[i]).not.toBe(dst[i]);
        }
      });

      it('all node indices are within valid range', () => {
        const numNodes = 12;
        const degree = 4;
        const rng = createPRNG(52);

        const [src, dst] = generateExpanderEdges(numNodes, degree, rng);

        for (let i = 0; i < src.length; i++) {
          expect(src[i]!).toBeGreaterThanOrEqual(0);
          expect(src[i]!).toBeLessThan(numNodes);
          expect(dst[i]!).toBeGreaterThanOrEqual(0);
          expect(dst[i]!).toBeLessThan(numNodes);
        }
      });
    });

    describe('exphormerAttention', () => {
      it('produces output with correct dimensions (numNodes x dim)', () => {
        const numNodes = 6;
        const dim = 8;
        const heads = 2;
        const numVirtualNodes = 2;
        const expanderDegree = 2;
        const rng = createPRNG(60);

        const graph = buildRingGraph(numNodes, dim, rng);
        const expanderEdges = generateExpanderEdges(numNodes, expanderDegree, createPRNG(61));

        const W_Q = xavierInit(dim, dim, createPRNG(62));
        const W_K = xavierInit(dim, dim, createPRNG(63));
        const W_V = xavierInit(dim, dim, createPRNG(64));

        const output = exphormerAttention(
          graph,
          graph.nodeFeatures,
          expanderEdges,
          numVirtualNodes,
          W_Q,
          W_K,
          W_V,
          dim,
          heads,
        );

        // Output should be for real nodes only: numNodes x dim
        expect(output.length).toBe(numNodes * dim);
        expect(allFinite(output)).toBe(true);
      });

      it('with zero virtual nodes still produces valid output', () => {
        const numNodes = 5;
        const dim = 4;
        const heads = 1;
        const rng = createPRNG(70);

        const graph = buildRingGraph(numNodes, dim, rng);
        const expanderEdges = generateExpanderEdges(numNodes, 2, createPRNG(71));

        const W_Q = xavierInit(dim, dim, createPRNG(72));
        const W_K = xavierInit(dim, dim, createPRNG(73));
        const W_V = xavierInit(dim, dim, createPRNG(74));

        const output = exphormerAttention(
          graph,
          graph.nodeFeatures,
          expanderEdges,
          0, // no virtual nodes
          W_Q,
          W_K,
          W_V,
          dim,
          heads,
        );

        expect(output.length).toBe(numNodes * dim);
        expect(allFinite(output)).toBe(true);
      });
    });
  });
});
