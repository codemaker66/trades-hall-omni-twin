// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-11: Integration Tests
// Tests for OT-GNN, TOGL, Sheaf Neural Networks, Demand-GNN, and Pricing.
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';

// Types and PRNG
import { createPRNG, type Graph } from '../types.js';

// Graph and tensor helpers
import { buildCSR } from '../graph.js';
import { xavierInit } from '../tensor.js';

// Modules under test — OT-GNN
import { wassersteinReadout, fgwDistance } from '../integration/ot-gnn.js';

// Modules under test — TOGL
import {
  computePersistenceDiagram,
  persistenceImage,
  toglLayer,
} from '../integration/togl.js';

// Modules under test — Sheaf
import { buildRestrictionMaps, neuralSheafDiffusion } from '../integration/sheaf.js';

// Modules under test — Demand-GNN
import { demandPredictorGNN, stochasticPricingOptimizer } from '../integration/demand-gnn.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Build a small undirected triangle graph: 0--1--2--0
 * 3 nodes, 6 directed edges (3 undirected x2).
 */
function makeTriangleGraph(featureDim: number): Graph {
  const rng = createPRNG(42);
  const edges: [number, number][] = [
    [0, 1], [1, 0],
    [1, 2], [2, 1],
    [0, 2], [2, 0],
  ];
  const numNodes = 3;
  const base = buildCSR(edges, numNodes);

  const nodeFeatures = new Float64Array(numNodes * featureDim);
  for (let i = 0; i < nodeFeatures.length; i++) {
    nodeFeatures[i] = rng() * 2 - 1;
  }

  return {
    ...base,
    nodeFeatures,
    featureDim,
  };
}

/**
 * Build a small path graph: 0--1--2--3
 * 4 nodes, 6 directed edges (3 undirected x2).
 */
function makePathGraph(featureDim: number): Graph {
  const rng = createPRNG(99);
  const edges: [number, number][] = [
    [0, 1], [1, 0],
    [1, 2], [2, 1],
    [2, 3], [3, 2],
  ];
  const numNodes = 4;
  const base = buildCSR(edges, numNodes);

  const nodeFeatures = new Float64Array(numNodes * featureDim);
  for (let i = 0; i < nodeFeatures.length; i++) {
    nodeFeatures[i] = rng() * 2 - 1;
  }

  return {
    ...base,
    nodeFeatures,
    featureDim,
  };
}

// ---------------------------------------------------------------------------
// OT-GNN Tests
// ---------------------------------------------------------------------------

describe('OT-GNN: wassersteinReadout', () => {
  it('returns correct number of distances', () => {
    const numNodes = 3;
    const embDim = 4;
    const numPrototypes = 5;
    const rng = createPRNG(1);

    const nodeEmbeddings = new Float64Array(numNodes * embDim);
    for (let i = 0; i < nodeEmbeddings.length; i++) {
      nodeEmbeddings[i] = rng();
    }

    const prototypes = new Float64Array(numPrototypes * embDim);
    for (let i = 0; i < prototypes.length; i++) {
      prototypes[i] = rng();
    }

    const result = wassersteinReadout(
      nodeEmbeddings,
      numNodes,
      embDim,
      prototypes,
      numPrototypes,
      {
        numPrototypes,
        prototypeDim: embDim,
        sinkhornIterations: 20,
        epsilon: 0.1,
      },
    );

    expect(result.distances.length).toBe(numPrototypes);
    expect(result.transportPlans.length).toBe(numPrototypes);
  });

  it('distances are non-negative', () => {
    const numNodes = 4;
    const embDim = 3;
    const numPrototypes = 3;
    const rng = createPRNG(7);

    const nodeEmbeddings = new Float64Array(numNodes * embDim);
    for (let i = 0; i < nodeEmbeddings.length; i++) {
      nodeEmbeddings[i] = rng() * 10 - 5;
    }

    const prototypes = new Float64Array(numPrototypes * embDim);
    for (let i = 0; i < prototypes.length; i++) {
      prototypes[i] = rng() * 10 - 5;
    }

    const result = wassersteinReadout(
      nodeEmbeddings,
      numNodes,
      embDim,
      prototypes,
      numPrototypes,
      {
        numPrototypes,
        prototypeDim: embDim,
        sinkhornIterations: 30,
        epsilon: 0.05,
      },
    );

    for (let i = 0; i < numPrototypes; i++) {
      expect(result.distances[i]).toBeGreaterThanOrEqual(0);
    }
  });
});

describe('OT-GNN: fgwDistance', () => {
  it('returns a non-negative scalar', () => {
    const featureDim = 3;
    const g1 = makeTriangleGraph(featureDim);
    const g2 = makePathGraph(featureDim);

    const dist = fgwDistance(
      g1,
      g1.nodeFeatures,
      g2,
      g2.nodeFeatures,
      0.5,
      3,
    );

    expect(typeof dist).toBe('number');
    expect(dist).toBeGreaterThanOrEqual(0);
    expect(Number.isFinite(dist)).toBe(true);
  });

  it('distance of graph with itself is approximately 0', () => {
    const featureDim = 3;
    const g = makeTriangleGraph(featureDim);

    const dist = fgwDistance(
      g,
      g.nodeFeatures,
      g,
      g.nodeFeatures,
      0.5,
      5,
    );

    expect(dist).toBeLessThan(1e-6);
  });
});

// ---------------------------------------------------------------------------
// TOGL Tests
// ---------------------------------------------------------------------------

describe('TOGL: computePersistenceDiagram', () => {
  it('returns births and deaths arrays', () => {
    // Build a path graph 0--1--2--3 with distinct filtration values
    // so that edge filtration values strictly exceed component birth times.
    const edges: [number, number][] = [
      [0, 1], [1, 0],
      [1, 2], [2, 1],
      [2, 3], [3, 2],
    ];
    const numNodes = 4;
    const base = buildCSR(edges, numNodes);
    const graph: Graph = {
      ...base,
      nodeFeatures: new Float64Array(numNodes * 2),
      featureDim: 2,
    };

    // Filtration: node values chosen so that edge max values exceed births
    // Node 0 born at 0.0, Node 1 born at 0.1, Node 2 born at 0.2, Node 3 born at 0.3
    // Edge (0,1): filtration = max(0.0, 0.1) = 0.1 > birth of younger node (0.1)? No, equal.
    // To get persistence pairs we need edge filtration > component birth.
    // Use: [0.0, 0.5, 0.1, 0.8]
    // Edge (0,1): max(0.0, 0.5)=0.5 -- merges {0}(birth=0.0) and {1}(birth=0.5) -- younger birth 0.5, death 0.5 => equal, no pair
    // Use: [0.0, 0.1, 0.2, 0.3]
    // Edge (0,1): max(0.0,0.1)=0.1, sorted first. Merges {0}(0.0) and {1}(0.1). Younger={1}(birth 0.1). Death=0.1, birth=0.1 => no pair.
    // The issue: max(filtration[src], filtration[dst]) often equals the younger component's birth.
    // Need node filtration where max of pair > both birth values.
    // E.g., [0.0, 0.0, 0.0, 0.0] => all edges have filtration 0, all births 0 => no persistence.
    // Better: use different values that ensure edge filtration exceeds at least one birth.
    // Actually the "younger" component birth is max(birth_a, birth_b), and edge filtration is
    // max(filtration[src], filtration[dst]). Since initially birthTime[v] = filtration[v],
    // the edge filtration equals max(f[i], f[j]) which is exactly the younger node's birth
    // (the one with the higher filtration). So all persistence is zero unless union-by-rank
    // causes the "older" label to have a higher birth due to rank ties.

    // With 4 nodes in a path, all starting with rank 0:
    // f = [0.0, 0.3, 0.1, 0.6]
    // Edges sorted by filtration:
    //   (1,2): max(0.3, 0.1) = 0.3
    //   (0,1): max(0.0, 0.3) = 0.3
    //   (2,3): max(0.1, 0.6) = 0.6
    // Process (1,2) at 0.3: root1=1(birth=0.3), root2=2(birth=0.1). older=2(0.1), younger=1(0.3). death=0.3, birth=0.3 => no pair.
    // Process (0,1) at 0.3: find(0)=0(birth=0.0), find(1) -> after union with 2, find(1)=2(birth=0.1). older=0(0.0), younger=2(0.1). death=0.3 > birth=0.1 => pair!
    // Process (2,3) at 0.6: find(2) -> chain -> root with birth=0.0, find(3)=3(birth=0.6). older=root(0.0), younger=3(0.6). death=0.6, birth=0.6 => no pair.

    // So we expect exactly 1 persistence pair with these values.
    const filtration = new Float64Array([0.0, 0.3, 0.1, 0.6]);

    const diagram = computePersistenceDiagram(graph, filtration);

    expect(diagram.births).toBeInstanceOf(Float64Array);
    expect(diagram.deaths).toBeInstanceOf(Float64Array);
    expect(diagram.births.length).toBe(diagram.deaths.length);
    expect(diagram.dim).toBe(0);

    // At least one persistence pair should be produced
    expect(diagram.births.length).toBeGreaterThanOrEqual(1);

    // All deaths should be >= births
    for (let i = 0; i < diagram.births.length; i++) {
      expect(diagram.deaths[i]).toBeGreaterThan(diagram.births[i]!);
    }
  });
});

describe('TOGL: persistenceImage', () => {
  it('returns correct-sized vector (resolution^2)', () => {
    // Manually construct a persistence diagram with known births/deaths
    const diagram = {
      births: new Float64Array([0.0, 0.1]),
      deaths: new Float64Array([0.5, 0.8]),
      dim: 0 as const,
    };

    const resolution = 5;
    const sigma = 0.1;
    const image = persistenceImage(diagram, resolution, sigma);

    expect(image.length).toBe(resolution * resolution);
    expect(image).toBeInstanceOf(Float64Array);

    // Since we have non-trivial persistence pairs, at least some pixels
    // should be non-zero
    let hasNonZero = false;
    for (let i = 0; i < image.length; i++) {
      if (image[i]! > 0) {
        hasNonZero = true;
        break;
      }
    }
    expect(hasNonZero).toBe(true);
  });
});

describe('TOGL: toglLayer', () => {
  it('output has augmented feature dimension', () => {
    const featureDim = 3;
    const numNodes = 3;
    const graph = makeTriangleGraph(featureDim);
    const resolution = 4;
    const sigma = 0.5;

    // Filtration weights: one weight per feature dimension
    const rng = createPRNG(55);
    const filtrationWeights = new Float64Array(featureDim);
    for (let i = 0; i < featureDim; i++) {
      filtrationWeights[i] = rng() * 2 - 1;
    }

    const augmented = toglLayer(
      graph,
      graph.nodeFeatures,
      numNodes,
      featureDim,
      filtrationWeights,
      resolution,
      sigma,
    );

    const expectedDim = featureDim + resolution * resolution;
    expect(augmented.length).toBe(numNodes * expectedDim);
    expect(augmented).toBeInstanceOf(Float64Array);
  });
});

// ---------------------------------------------------------------------------
// Sheaf Tests
// ---------------------------------------------------------------------------

describe('Sheaf: buildRestrictionMaps', () => {
  it('returns correct-sized output', () => {
    const featureDim = 3;
    const numNodes = 3;
    const stalkDim = 2;
    const graph = makeTriangleGraph(featureDim);
    const numEdges = graph.numEdges; // 6 directed edges

    const rng = createPRNG(12);
    const inputDim = 2 * featureDim;   // 6
    const hiddenDim = 8;
    const outputDim = 2 * stalkDim * stalkDim; // 8

    const W1 = xavierInit(hiddenDim, inputDim, rng);  // hiddenDim x inputDim
    const b1 = new Float64Array(hiddenDim);
    const W2 = xavierInit(outputDim, hiddenDim, rng);  // outputDim x hiddenDim
    const b2 = new Float64Array(outputDim);

    const maps = buildRestrictionMaps(
      graph,
      graph.nodeFeatures,
      numNodes,
      featureDim,
      stalkDim,
      { W1, b1, W2, b2 },
    );

    const expectedLength = numEdges * 2 * stalkDim * stalkDim;
    expect(maps.length).toBe(expectedLength);
    expect(maps).toBeInstanceOf(Float64Array);
  });
});

describe('Sheaf: neuralSheafDiffusion', () => {
  it('output shape matches (numNodes * stalkDim)', () => {
    const featureDim = 3;
    const numNodes = 3;
    const stalkDim = 2;
    const graph = makeTriangleGraph(featureDim);

    const rng = createPRNG(77);
    const inputDim = 2 * featureDim; // 6
    const hiddenDim = 8;
    const outputPerEdge = 2 * stalkDim * stalkDim; // 8

    // Build MLP weights for the restriction MLP
    const W1 = xavierInit(hiddenDim, inputDim, rng);
    const b1 = new Float64Array(hiddenDim);
    const W2 = xavierInit(outputPerEdge, hiddenDim, rng);
    const b2 = new Float64Array(outputPerEdge);

    const result = neuralSheafDiffusion(
      graph,
      graph.nodeFeatures,
      numNodes,
      featureDim,
      {
        stalkDim,
        diffusionSteps: 3,
        learningRate: 0.01,
      },
      {
        restrictionMLP: {
          layers: [
            { W: W1, bias: b1, inDim: inputDim, outDim: hiddenDim },
            { W: W2, bias: b2, inDim: hiddenDim, outDim: outputPerEdge },
          ],
        },
        stalkDim,
      },
    );

    expect(result.length).toBe(numNodes * stalkDim);
    expect(result).toBeInstanceOf(Float64Array);
  });
});

// ---------------------------------------------------------------------------
// Demand-GNN Tests
// ---------------------------------------------------------------------------

describe('Demand-GNN: demandPredictorGNN', () => {
  it('returns means and variances with correct length', () => {
    const featureDim = 3;
    const timeDim = 2;
    const numNodes = 3;
    const graph = makeTriangleGraph(featureDim);

    const rng = createPRNG(33);

    // Time features
    const timeFeatures = new Float64Array(numNodes * timeDim);
    for (let i = 0; i < timeFeatures.length; i++) {
      timeFeatures[i] = rng();
    }

    const inputDim = featureDim + timeDim; // 5
    const gcn1OutDim = 4;
    const gcn2OutDim = 4;
    const mlp1OutDim = 4;
    const mlp2OutDim = 2; // mean + log_var

    // GCN layer weights
    const gnnWeights = [
      {
        W: xavierInit(gcn1OutDim, inputDim, rng),
        b: new Float64Array(gcn1OutDim),
      },
      {
        W: xavierInit(gcn2OutDim, gcn1OutDim, rng),
        b: new Float64Array(gcn2OutDim),
      },
    ];

    // MLP head weights
    const mlpWeights = [
      {
        W: xavierInit(mlp1OutDim, gcn2OutDim, rng),
        b: new Float64Array(mlp1OutDim),
      },
      {
        W: xavierInit(mlp2OutDim, mlp1OutDim, rng),
        b: new Float64Array(mlp2OutDim),
      },
    ];

    const forecast = demandPredictorGNN(
      graph,
      graph.nodeFeatures,
      numNodes,
      featureDim,
      timeFeatures,
      timeDim,
      { gnnWeights, mlpWeights },
    );

    expect(forecast.mean.length).toBe(numNodes);
    expect(forecast.variance.length).toBe(numNodes);
    expect(forecast.timestamps.length).toBe(numNodes);

    // Variances must be positive (exp of log-var)
    for (let i = 0; i < numNodes; i++) {
      expect(forecast.variance[i]).toBeGreaterThan(0);
      expect(Number.isFinite(forecast.mean[i])).toBe(true);
      expect(Number.isFinite(forecast.variance[i])).toBe(true);
    }
  });
});

describe('Demand-GNN: stochasticPricingOptimizer', () => {
  it('returns prices in valid range', () => {
    const rng = createPRNG(88);
    const numNodes = 3;

    const demand = {
      mean: new Float64Array([10, 20, 15]),
      variance: new Float64Array([2, 3, 1]),
      timestamps: new Float64Array([0, 1, 2]),
    };

    const minPrice = 50;
    const maxPrice = 200;

    const result = stochasticPricingOptimizer(
      demand,
      [minPrice, maxPrice],
      100,
      rng,
    );

    expect(result.optimalPrice).toBeGreaterThanOrEqual(minPrice);
    expect(result.optimalPrice).toBeLessThanOrEqual(maxPrice);
    expect(result.expectedRevenue).toBeGreaterThan(0);
    expect(result.demandAtPrice).toBeGreaterThanOrEqual(0);
    expect(Number.isFinite(result.optimalPrice)).toBe(true);
    expect(Number.isFinite(result.expectedRevenue)).toBe(true);
  });
});
