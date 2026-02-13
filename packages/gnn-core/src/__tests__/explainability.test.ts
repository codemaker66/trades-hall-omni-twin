// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-10: Explainability Tests
// Tests for GNNExplainer, PGExplainer, CF-GNNExplainer, and template mapper.
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';

// Types and PRNG
import {
  createPRNG,
  type Graph,
  type GNNForwardFn,
  type ExplainerConfig,
  type MLPWeights,
} from '../types.js';

// Graph and tensor helpers
import { buildCSR } from '../graph.js';
import { xavierInit } from '../tensor.js';

// Modules under test
import { gnnExplainer } from '../explainability/gnn-explainer.js';
import { pgExplainer } from '../explainability/pg-explainer.js';
import { cfExplainer } from '../explainability/counterfactual.js';
import {
  edgeMaskToExplanation,
  defaultVenueTemplates,
} from '../explainability/template-mapper.js';

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
 * Each node has a 3-dimensional feature vector with class-discriminative signals.
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
  const featureDim = 3;

  const base = buildCSR(edges, numNodes);

  // Assign features that create a clear class separation:
  // Nodes 0,1,2 get features biased toward class 0
  // Nodes 3,4,5 get features biased toward class 1
  const nodeFeatures = new Float64Array(numNodes * featureDim);
  // Class 0 nodes
  nodeFeatures[0 * featureDim + 0] = 1.0; nodeFeatures[0 * featureDim + 1] = 0.2; nodeFeatures[0 * featureDim + 2] = 0.1;
  nodeFeatures[1 * featureDim + 0] = 0.9; nodeFeatures[1 * featureDim + 1] = 0.3; nodeFeatures[1 * featureDim + 2] = 0.1;
  nodeFeatures[2 * featureDim + 0] = 0.8; nodeFeatures[2 * featureDim + 1] = 0.1; nodeFeatures[2 * featureDim + 2] = 0.2;
  // Class 1 nodes
  nodeFeatures[3 * featureDim + 0] = 0.1; nodeFeatures[3 * featureDim + 1] = 0.2; nodeFeatures[3 * featureDim + 2] = 1.0;
  nodeFeatures[4 * featureDim + 0] = 0.2; nodeFeatures[4 * featureDim + 1] = 0.1; nodeFeatures[4 * featureDim + 2] = 0.9;
  nodeFeatures[5 * featureDim + 0] = 0.1; nodeFeatures[5 * featureDim + 1] = 0.3; nodeFeatures[5 * featureDim + 2] = 0.8;

  return {
    ...base,
    nodeFeatures,
    featureDim,
  };
}

/**
 * Create a mock GNN forward function that produces per-node logits.
 * The output is numNodes * numClasses. For node i, the logit for class c
 * is based on the c-th feature dimension summed with neighbor averages.
 * This creates class-discriminative output that the explainer can reason about.
 */
function makeMockGNN(numClasses: number): GNNForwardFn {
  return (graph: Graph, features: Float64Array): Float64Array => {
    const numNodes = graph.numNodes;
    const featureDim = graph.featureDim;
    const output = new Float64Array(numNodes * numClasses);

    for (let i = 0; i < numNodes; i++) {
      // Compute mean neighbor features (1-hop message passing)
      const start = graph.rowPtr[i]!;
      const end = graph.rowPtr[i + 1]!;
      const deg = end - start;

      for (let c = 0; c < numClasses; c++) {
        // Self contribution
        let val = features[i * featureDim + (c % featureDim)]! * 2.0;

        // Neighbor contribution
        if (deg > 0) {
          let neighborSum = 0;
          for (let e = start; e < end; e++) {
            const j = graph.colIdx[e]!;
            const edgeWeight = graph.edgeWeights ? graph.edgeWeights[e]! : 1.0;
            neighborSum += features[j * featureDim + (c % featureDim)]! * edgeWeight;
          }
          val += neighborSum / deg;
        }

        output[i * numClasses + c] = val;
      }
    }

    return output;
  };
}

/**
 * Create a mock GNN whose prediction for a target node flips when key edges
 * are removed. Used to test the counterfactual explainer.
 *
 * The model makes predictions based on the count of edges: if a node has
 * 3 or more edges, predict class 1; otherwise predict class 0.
 */
function makeFlippableGNN(): GNNForwardFn {
  const numClasses = 2;
  return (graph: Graph, features: Float64Array): Float64Array => {
    const numNodes = graph.numNodes;
    const output = new Float64Array(numNodes * numClasses);

    for (let i = 0; i < numNodes; i++) {
      const start = graph.rowPtr[i]!;
      const end = graph.rowPtr[i + 1]!;
      const deg = end - start;

      // If degree >= 3, strongly predict class 1
      // Otherwise, strongly predict class 0
      if (deg >= 3) {
        output[i * numClasses + 0] = -1.0;
        output[i * numClasses + 1] = 2.0;
      } else {
        output[i * numClasses + 0] = 2.0;
        output[i * numClasses + 1] = -1.0;
      }
    }

    return output;
  };
}

// ---------------------------------------------------------------------------
// 1. gnnExplainer returns edgeMask and featureMask with correct dimensions
// ---------------------------------------------------------------------------

describe('gnnExplainer', () => {
  it('returns edgeMask and featureMask with correct dimensions', () => {
    const graph = makeTestGraph();
    const model = makeMockGNN(2);
    const rng = createPRNG(42);

    const config: ExplainerConfig = {
      epochs: 20,
      lr: 0.5,
      sizeReg: 0.01,
      entropyReg: 0.01,
      edgeMaskThreshold: 0.5,
      featureMaskThreshold: 0.5,
    };

    const result = gnnExplainer(model, graph, 0, config, rng);

    // edgeMask should have one entry per edge
    expect(result.edgeMask.length).toBe(graph.numEdges);
    // featureMask should have one entry per feature dimension
    expect(result.featureMask).toBeDefined();
    expect(result.featureMask!.length).toBe(graph.featureDim);
    // importantEdges should be an array of [src, dst] pairs
    expect(Array.isArray(result.importantEdges)).toBe(true);
  });

  // ---------------------------------------------------------------------------
  // 2. gnnExplainer edgeMask values are in [0,1]
  // ---------------------------------------------------------------------------

  it('edgeMask values are in [0,1]', () => {
    const graph = makeTestGraph();
    const model = makeMockGNN(2);
    const rng = createPRNG(101);

    const config: ExplainerConfig = {
      epochs: 10,
      lr: 0.3,
      sizeReg: 0.01,
      entropyReg: 0.01,
      edgeMaskThreshold: 0.5,
      featureMaskThreshold: 0.5,
    };

    const result = gnnExplainer(model, graph, 1, config, rng);

    for (let i = 0; i < result.edgeMask.length; i++) {
      expect(result.edgeMask[i]).toBeGreaterThanOrEqual(0);
      expect(result.edgeMask[i]).toBeLessThanOrEqual(1);
    }

    // Feature mask values should also be in [0,1]
    if (result.featureMask) {
      for (let i = 0; i < result.featureMask.length; i++) {
        expect(result.featureMask[i]).toBeGreaterThanOrEqual(0);
        expect(result.featureMask[i]).toBeLessThanOrEqual(1);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// 3. pgExplainer returns masks for all nodes quickly (no iteration)
// ---------------------------------------------------------------------------

describe('pgExplainer', () => {
  it('returns masks for all nodes quickly (no per-node iteration)', () => {
    const graph = makeTestGraph();
    const model = makeMockGNN(2);
    const rng = createPRNG(202);

    // The model output gives us embeddings of dimension = numClasses per node.
    // PGExplainer MLP takes concatenated endpoint embeddings (2 * embDim)
    // and outputs a scalar log-odds (1-dim).
    const embDim = 2; // numClasses from the mock
    const mlpInputDim = 2 * embDim;

    // Build simple MLP weights: one layer from (2 * embDim) -> 1
    const mlpWeights: MLPWeights = {
      layers: [
        {
          W: xavierInit(mlpInputDim, 1, createPRNG(300)),
          bias: new Float64Array(1),
          inDim: mlpInputDim,
          outDim: 1,
        },
      ],
    };

    const temperature = 0.5;

    const result = pgExplainer(model, graph, mlpWeights, temperature, rng);

    // Edge mask should cover all edges
    expect(result.edgeMask.length).toBe(graph.numEdges);
    // importantEdges is derived from thresholding
    expect(Array.isArray(result.importantEdges)).toBe(true);
  });

  // ---------------------------------------------------------------------------
  // 4. pgExplainer edge masks are in [0,1]
  // ---------------------------------------------------------------------------

  it('edge masks are in [0,1]', () => {
    const graph = makeTestGraph();
    const model = makeMockGNN(2);
    const rng = createPRNG(303);

    const embDim = 2;
    const mlpInputDim = 2 * embDim;

    const mlpWeights: MLPWeights = {
      layers: [
        {
          W: xavierInit(mlpInputDim, 1, createPRNG(400)),
          bias: new Float64Array(1),
          inDim: mlpInputDim,
          outDim: 1,
        },
      ],
    };

    const result = pgExplainer(model, graph, mlpWeights, 0.5, rng);

    for (let i = 0; i < result.edgeMask.length; i++) {
      expect(result.edgeMask[i]).toBeGreaterThanOrEqual(0);
      expect(result.edgeMask[i]).toBeLessThanOrEqual(1);
    }
  });
});

// ---------------------------------------------------------------------------
// 5. cfExplainer returns removedEdges and flippedPrediction
// ---------------------------------------------------------------------------

describe('cfExplainer', () => {
  it('returns removedEdges and prediction info', () => {
    const graph = makeTestGraph();
    const model = makeFlippableGNN();
    const rng = createPRNG(404);

    // Node 1 has degree 3 (edges to 0, 2, 4) => predicted class 1
    // Node 4 has degree 3 (edges to 1, 3, 5) => predicted class 1
    // Removing edges should flip the prediction

    const config: ExplainerConfig = {
      epochs: 10, // max edits
      lr: 0.1,
      sizeReg: 0.01,
      entropyReg: 0.01,
      edgeMaskThreshold: 0.5,
      featureMaskThreshold: 0.5,
    };

    const result = cfExplainer(model, graph, 1, config, rng);

    // Should have removed some edges
    expect(Array.isArray(result.removedEdges)).toBe(true);
    // originalPrediction and newPrediction should be numbers
    expect(typeof result.originalPrediction).toBe('number');
    expect(typeof result.newPrediction).toBe('number');
    // numEditsRequired should be a positive number (or 0 if already flipped)
    expect(result.numEditsRequired).toBeGreaterThanOrEqual(0);

    // The original prediction for node 1 should be class 1 (degree >= 3)
    expect(result.originalPrediction).toBe(1);
    // After edge removal, prediction should have flipped
    expect(result.newPrediction).not.toBe(result.originalPrediction);
  });

  // ---------------------------------------------------------------------------
  // 6. cfExplainer removes minimal edges (< numEdges)
  // ---------------------------------------------------------------------------

  it('removes fewer edges than total', () => {
    const graph = makeTestGraph();
    const model = makeFlippableGNN();
    const rng = createPRNG(505);

    const config: ExplainerConfig = {
      epochs: 10,
      lr: 0.1,
      sizeReg: 0.01,
      entropyReg: 0.01,
      edgeMaskThreshold: 0.5,
      featureMaskThreshold: 0.5,
    };

    const result = cfExplainer(model, graph, 1, config, rng);

    // Should remove fewer edges than total graph edges
    expect(result.removedEdges.length).toBeLessThan(graph.numEdges);
    // And certainly fewer than max edits
    expect(result.numEditsRequired).toBeLessThanOrEqual(config.epochs);
  });
});

// ---------------------------------------------------------------------------
// 7. edgeMaskToExplanation returns string array
// ---------------------------------------------------------------------------

describe('edgeMaskToExplanation', () => {
  it('returns string array for important edges', () => {
    const graph = makeTestGraph();

    // Create an edge mask where some edges are above threshold
    const edgeMask = new Float64Array(graph.numEdges);
    // Mark a few edges as important
    edgeMask[0] = 0.9; // edge 0
    edgeMask[1] = 0.8; // edge 1
    edgeMask[2] = 0.1; // below threshold

    // Use a simple wildcard template
    const templates = [
      {
        edgeType: '*',
        sourceNodeType: '*',
        targetNodeType: '*',
        template: 'Edge from {src} to {dst} is important (weight: {weight}).',
      },
    ];

    const result = edgeMaskToExplanation(edgeMask, graph, templates, 0.5);

    // Should produce explanations for edges above threshold
    expect(result.explanations.length).toBe(2);
    expect(result.importanceScores.length).toBe(2);

    // Explanations should be strings
    for (const explanation of result.explanations) {
      expect(typeof explanation).toBe('string');
      expect(explanation.length).toBeGreaterThan(0);
    }

    // Importance scores should be sorted descending
    for (let i = 1; i < result.importanceScores.length; i++) {
      expect(result.importanceScores[i]).toBeLessThanOrEqual(result.importanceScores[i - 1]!);
    }
  });

  // ---------------------------------------------------------------------------
  // 8. edgeMaskToExplanation with defaultVenueTemplates produces meaningful output
  // ---------------------------------------------------------------------------

  it('with defaultVenueTemplates produces meaningful output', () => {
    const graph = makeTestGraph();

    // Create an edge mask where some edges are above threshold
    const edgeMask = new Float64Array(graph.numEdges);
    edgeMask[0] = 0.95;
    edgeMask[3] = 0.75;
    edgeMask[6] = 0.6;

    const templates = defaultVenueTemplates();

    // defaultVenueTemplates() should return a non-empty array
    expect(templates.length).toBeGreaterThan(0);

    // The last template should be a wildcard fallback
    const lastTemplate = templates[templates.length - 1]!;
    expect(lastTemplate.edgeType).toBe('*');
    expect(lastTemplate.sourceNodeType).toBe('*');
    expect(lastTemplate.targetNodeType).toBe('*');

    const result = edgeMaskToExplanation(edgeMask, graph, templates, 0.5);

    // Should produce 3 explanations (for the 3 edges above 0.5 threshold)
    expect(result.explanations.length).toBe(3);

    // Each explanation should be a meaningful string (not empty)
    for (const explanation of result.explanations) {
      expect(typeof explanation).toBe('string');
      expect(explanation.length).toBeGreaterThan(10);
    }

    // Scores should be sorted descending
    expect(result.importanceScores[0]).toBeCloseTo(0.95, 5);
    expect(result.importanceScores[1]).toBeCloseTo(0.75, 5);
    expect(result.importanceScores[2]).toBeCloseTo(0.6, 5);
  });
});
