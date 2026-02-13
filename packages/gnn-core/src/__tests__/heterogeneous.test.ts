// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-2: Heterogeneous GNN Tests
// Tests for R-GCN, HAN, HGT, Simple-HGN, and venue graph builder.
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';

// Types
import type {
  Graph,
  HeteroGraph,
  HeteroNodeStore,
  HeteroEdgeStore,
  RGCNConfig,
  RGCNWeights,
  HANWeights,
  HGTConfig,
  HGTWeights,
  SimpleHGNWeights,
} from '../types.js';

// Graph utilities
import { buildCSR } from '../graph.js';

// Tensor utilities
import { xavierInit } from '../tensor.js';

// Heterogeneous GNN layers under test
import {
  rgcnLayer,
  hanNodeAttention,
  hanSemanticAttention,
  hgtLayer,
  simpleHGNLayer,
  buildVenueHeteroGraph,
  addReverseEdges,
} from '../heterogeneous/index.js';

import { createPRNG } from '../types.js';

// ---------------------------------------------------------------------------
// Helper: Build a small HeteroGraph for testing
// 3 venues (featureDim=4), 2 planners (featureDim=4)
// Edge type: planner --books--> venue
// ---------------------------------------------------------------------------

function makeSmallHeteroGraph(): {
  heteroGraph: HeteroGraph;
  nodeFeatures: Map<string, Float64Array>;
} {
  const rng = createPRNG(42);
  const venueCount = 3;
  const plannerCount = 2;
  const featureDim = 4;

  const venueFeatures = new Float64Array(venueCount * featureDim);
  const plannerFeatures = new Float64Array(plannerCount * featureDim);

  for (let i = 0; i < venueFeatures.length; i++) venueFeatures[i] = rng();
  for (let i = 0; i < plannerFeatures.length; i++) plannerFeatures[i] = rng();

  const nodes = new Map<string, HeteroNodeStore>();
  nodes.set('venue', { features: venueFeatures, count: venueCount, featureDim });
  nodes.set('planner', { features: plannerFeatures, count: plannerCount, featureDim });

  // Edges: planner 0 -> venue 0, planner 0 -> venue 1, planner 1 -> venue 2
  // CSR is indexed by destination (venue), colIdx = source (planner)
  // venue 0: [planner 0], venue 1: [planner 0], venue 2: [planner 1]
  const rowPtr = new Uint32Array([0, 1, 2, 3]);
  const colIdx = new Uint32Array([0, 0, 1]);

  const edges = new Map<string, HeteroEdgeStore>();
  edges.set('planner/books/venue', {
    rowPtr,
    colIdx,
    numEdges: 3,
  });

  const edgeTypes: [string, string, string][] = [
    ['planner', 'books', 'venue'],
  ];

  const heteroGraph: HeteroGraph = {
    nodeTypes: ['venue', 'planner'],
    edgeTypes,
    nodes,
    edges,
  };

  const nodeFeatures = new Map<string, Float64Array>();
  nodeFeatures.set('venue', venueFeatures);
  nodeFeatures.set('planner', plannerFeatures);

  return { heteroGraph, nodeFeatures };
}

// ---------------------------------------------------------------------------
// Helper: Build a small CSR graph for HAN node attention tests
// 4 nodes, edges: 0->1, 0->2, 1->2, 2->3 (directed)
// ---------------------------------------------------------------------------

function makeSmallGraph(): Graph {
  const edges: [number, number][] = [
    [0, 1], [0, 2],
    [1, 2],
    [2, 3],
  ];
  const numNodes = 4;
  const featureDim = 4;
  const graph = buildCSR(edges, numNodes);

  // Add node features
  const rng = createPRNG(123);
  const nodeFeatures = new Float64Array(numNodes * featureDim);
  for (let i = 0; i < nodeFeatures.length; i++) nodeFeatures[i] = rng();

  return {
    ...graph,
    nodeFeatures,
    featureDim,
  };
}

// ---------------------------------------------------------------------------
// R-GCN Tests
// ---------------------------------------------------------------------------

describe('rgcnLayer', () => {
  it('returns features for each node type with correct dimensions', () => {
    const { heteroGraph, nodeFeatures } = makeSmallHeteroGraph();
    const rng = createPRNG(7);
    const inDim = 4;
    const outDim = 3;
    const numRelations = 1;
    const numBases = 2;

    // Create basis matrices: numBases arrays, each inDim x outDim
    const bases: Float64Array[] = [];
    for (let b = 0; b < numBases; b++) {
      bases.push(xavierInit(inDim, outDim, rng));
    }

    // Coefficients: numRelations x numBases
    const coeffs = new Float64Array(numRelations * numBases);
    for (let i = 0; i < coeffs.length; i++) coeffs[i] = rng();

    // Bias
    const bias = new Float64Array(outDim);
    for (let i = 0; i < outDim; i++) bias[i] = 0.01;

    const config: RGCNConfig = { inDim, outDim, numRelations, numBases, bias: true };
    const weights: RGCNWeights = { bases, coeffs, bias };

    const result = rgcnLayer(heteroGraph, nodeFeatures, weights, config);

    // Should return features for both node types
    expect(result.has('venue')).toBe(true);
    expect(result.has('planner')).toBe(true);

    // Venue features: 3 venues x outDim=3
    const venueOut = result.get('venue')!;
    expect(venueOut.length).toBe(3 * outDim);

    // Planner features: 2 planners x outDim=3
    const plannerOut = result.get('planner')!;
    expect(plannerOut.length).toBe(2 * outDim);

    // All values should be >= 0 (ReLU activation)
    for (let i = 0; i < venueOut.length; i++) {
      expect(venueOut[i]!).toBeGreaterThanOrEqual(0);
    }
  });
});

// ---------------------------------------------------------------------------
// HAN Tests
// ---------------------------------------------------------------------------

describe('hanNodeAttention', () => {
  it('returns output with correct dimensions and attention weights summing to ~1 per node', () => {
    const graph = makeSmallGraph();
    const numNodes = graph.numNodes;
    const inDim = graph.featureDim; // 4
    const outDim = 3;

    const rng = createPRNG(99);

    // W: inDim x outDim
    const W = xavierInit(inDim, outDim, rng);

    // a: attention vector of length 2 * outDim
    const a = new Float64Array(2 * outDim);
    for (let i = 0; i < a.length; i++) a[i] = rng() - 0.5;

    const { output, weights } = hanNodeAttention(
      graph,
      graph.nodeFeatures,
      W,
      a,
      inDim,
      outDim,
    );

    // Output should be numNodes x outDim
    expect(output.length).toBe(numNodes * outDim);

    // Attention weights should have length = numEdges
    expect(weights.length).toBe(graph.numEdges);

    // For each node with neighbors, attention weights should sum to ~1
    for (let i = 0; i < numNodes; i++) {
      const start = graph.rowPtr[i]!;
      const end = graph.rowPtr[i + 1]!;
      const degree = end - start;

      if (degree > 0) {
        let sum = 0;
        for (let e = start; e < end; e++) {
          sum += weights[e]!;
        }
        expect(sum).toBeCloseTo(1.0, 5);
      }
    }
  });
});

describe('hanSemanticAttention', () => {
  it('combines 2 meta-path outputs into a single output of correct dimensions', () => {
    const rng = createPRNG(55);
    const numNodes = 4;
    const dim = 3;

    // Two meta-path outputs, each numNodes x dim
    const mp1 = new Float64Array(numNodes * dim);
    const mp2 = new Float64Array(numNodes * dim);
    for (let i = 0; i < mp1.length; i++) {
      mp1[i] = rng();
      mp2[i] = rng();
    }

    // W: dim x dim
    const W = xavierInit(dim, dim, rng);

    // q: length dim
    const q = new Float64Array(dim);
    for (let i = 0; i < dim; i++) q[i] = rng() - 0.5;

    const result = hanSemanticAttention([mp1, mp2], W, q, numNodes, dim);

    // Result should be numNodes x dim
    expect(result.length).toBe(numNodes * dim);

    // Result should be a weighted combination of mp1 and mp2
    // so it should not be all zeros (given non-zero inputs)
    let hasNonZero = false;
    for (let i = 0; i < result.length; i++) {
      if (result[i] !== 0) { hasNonZero = true; break; }
    }
    expect(hasNonZero).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// HGT Tests
// ---------------------------------------------------------------------------

describe('hgtLayer', () => {
  it('returns features per node type with correct dimensions', () => {
    const { heteroGraph, nodeFeatures } = makeSmallHeteroGraph();
    const rng = createPRNG(77);
    const inDim = 4;
    const outDim = 4; // must be divisible by heads
    const heads = 2;
    const numNodeTypes = 2; // venue, planner
    const numEdgeTypes = 1; // planner/books/venue

    const config: HGTConfig = {
      inDim,
      outDim,
      heads,
      numNodeTypes,
      numEdgeTypes,
    };

    // Per node type weight matrices (inDim x outDim)
    const W_Q: Float64Array[] = [];
    const W_K: Float64Array[] = [];
    const W_V: Float64Array[] = [];
    for (let t = 0; t < numNodeTypes; t++) {
      W_Q.push(xavierInit(inDim, outDim, rng));
      W_K.push(xavierInit(inDim, outDim, rng));
      W_V.push(xavierInit(inDim, outDim, rng));
    }

    // Per edge type: W_ATT (dHead x dHead), W_MSG (outDim x outDim), mu (scalar)
    const dHead = outDim / heads;
    const W_ATT: Float64Array[] = [xavierInit(dHead, dHead, rng)];
    const W_MSG: Float64Array[] = [xavierInit(outDim, outDim, rng)];
    const mu = new Float64Array([1.0]);

    const weights: HGTWeights = { W_Q, W_K, W_V, W_ATT, W_MSG, mu };

    const result = hgtLayer(heteroGraph, nodeFeatures, weights, config);

    // Should return features for both node types
    expect(result.has('venue')).toBe(true);
    expect(result.has('planner')).toBe(true);

    // Venue: 3 venues x outDim=4
    expect(result.get('venue')!.length).toBe(3 * outDim);

    // Planner: 2 planners x outDim=4
    expect(result.get('planner')!.length).toBe(2 * outDim);
  });
});

// ---------------------------------------------------------------------------
// Simple-HGN Tests
// ---------------------------------------------------------------------------

describe('simpleHGNLayer', () => {
  it('returns features per node type with L2-normalized rows', () => {
    const { heteroGraph, nodeFeatures } = makeSmallHeteroGraph();
    const rng = createPRNG(33);
    const inDim = 4;
    const outDim = 3;

    // W: inDim x outDim
    const W = xavierInit(inDim, outDim, rng);

    // a: 2 * outDim (attention vector split into a_left, a_right)
    const a = new Float64Array(2 * outDim);
    for (let i = 0; i < a.length; i++) a[i] = rng() - 0.5;

    // Edge type embeddings: 1 edge type, each of length outDim
    const edgeTypeEmb: Float64Array[] = [new Float64Array(outDim)];
    for (let i = 0; i < outDim; i++) edgeTypeEmb[0]![i] = rng() - 0.5;

    const weights: SimpleHGNWeights = { W, a, edgeTypeEmb };

    const result = simpleHGNLayer(heteroGraph, nodeFeatures, weights, inDim, outDim);

    // Should return features for both node types
    expect(result.has('venue')).toBe(true);
    expect(result.has('planner')).toBe(true);

    // Venue: 3 venues x outDim=3
    const venueOut = result.get('venue')!;
    expect(venueOut.length).toBe(3 * outDim);

    // Each row should be L2-normalized (norm ~1) for nodes that received messages
    for (let i = 0; i < 3; i++) {
      let normSq = 0;
      for (let d = 0; d < outDim; d++) {
        const val = venueOut[i * outDim + d]!;
        normSq += val * val;
      }
      const norm = Math.sqrt(normSq);
      // Node should either be zero (no incoming edges) or L2-normalized
      if (norm > 1e-10) {
        expect(norm).toBeCloseTo(1.0, 4);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// Venue Graph Builder Tests
// ---------------------------------------------------------------------------

describe('buildVenueHeteroGraph', () => {
  it('creates correct node types, edge types, and includes reverse edges', () => {
    const featureDim = 3;
    const venues = {
      features: new Float64Array(2 * featureDim),
      count: 2,
      featureDim,
    };
    const planners = {
      features: new Float64Array(2 * featureDim),
      count: 2,
      featureDim,
    };
    const events = {
      features: new Float64Array(1 * featureDim),
      count: 1,
      featureDim,
    };

    // planner 0 books venue 0, planner 1 books venue 1
    const bookingEdges: [number, number][] = [[0, 0], [1, 1]];
    // event 0 held at venue 0
    const eventVenueEdges: [number, number][] = [[0, 0]];

    const graph = buildVenueHeteroGraph(
      venues,
      planners,
      events,
      bookingEdges,
      eventVenueEdges,
    );

    // Node types
    expect(graph.nodeTypes).toContain('venue');
    expect(graph.nodeTypes).toContain('planner');
    expect(graph.nodeTypes).toContain('event');

    // Forward edge types
    expect(graph.edges.has('planner/books/venue')).toBe(true);
    expect(graph.edges.has('event/held_at/venue')).toBe(true);

    // Reverse edge types
    expect(graph.edges.has('venue/booked_by/planner')).toBe(true);
    expect(graph.edges.has('venue/hosts/event')).toBe(true);

    // Check edge counts
    expect(graph.edges.get('planner/books/venue')!.numEdges).toBe(2);
    expect(graph.edges.get('event/held_at/venue')!.numEdges).toBe(1);
    expect(graph.edges.get('venue/booked_by/planner')!.numEdges).toBe(2);
    expect(graph.edges.get('venue/hosts/event')!.numEdges).toBe(1);

    // Check node stores
    expect(graph.nodes.get('venue')!.count).toBe(2);
    expect(graph.nodes.get('planner')!.count).toBe(2);
    expect(graph.nodes.get('event')!.count).toBe(1);

    // Edge type triplets should include both forward and reverse
    const edgeTypeStrs = graph.edgeTypes.map(
      ([s, r, d]) => `${s}/${r}/${d}`,
    );
    expect(edgeTypeStrs).toContain('planner/books/venue');
    expect(edgeTypeStrs).toContain('event/held_at/venue');
    expect(edgeTypeStrs).toContain('venue/booked_by/planner');
    expect(edgeTypeStrs).toContain('venue/hosts/event');
  });
});

describe('addReverseEdges', () => {
  it('adds reverse edges for a simple HeteroGraph', () => {
    const rng = createPRNG(10);
    const featureDim = 2;

    const nodes = new Map<string, HeteroNodeStore>();
    nodes.set('A', { features: new Float64Array(3 * featureDim), count: 3, featureDim });
    nodes.set('B', { features: new Float64Array(2 * featureDim), count: 2, featureDim });

    // A -> B edges: A0->B0, A1->B1
    // CSR indexed by destination (B). B0: [A0], B1: [A1]
    const rowPtr = new Uint32Array([0, 1, 2]);
    const colIdx = new Uint32Array([0, 1]);

    const edges = new Map<string, HeteroEdgeStore>();
    edges.set('A/likes/B', { rowPtr, colIdx, numEdges: 2 });

    const graph: HeteroGraph = {
      nodeTypes: ['A', 'B'],
      edgeTypes: [['A', 'likes', 'B']],
      nodes,
      edges,
    };

    const withReverse = addReverseEdges(graph);

    // Should now have the reverse edge type
    expect(withReverse.edges.has('B/rev_likes/A')).toBe(true);

    const revStore = withReverse.edges.get('B/rev_likes/A')!;
    expect(revStore.numEdges).toBe(2);

    // Edge type triplets should include original + reverse
    expect(withReverse.edgeTypes.length).toBe(2);
    const tripletStrs = withReverse.edgeTypes.map(
      ([s, r, d]) => `${s}/${r}/${d}`,
    );
    expect(tripletStrs).toContain('A/likes/B');
    expect(tripletStrs).toContain('B/rev_likes/A');
  });
});
