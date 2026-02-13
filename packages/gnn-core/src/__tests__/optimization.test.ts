// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-8: Combinatorial Optimization Tests
// Tests for Sinkhorn assignment, Hungarian algorithm, attention model
// encoder/decoder, greedy decode, bipartite GNN assignment, and MIP encoding.
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';

// Types and PRNG
import {
  createPRNG,
  type AttentionModelConfig,
  type AttentionModelWeights,
  type SinkhornConfig,
  type MIPVariable,
  type MIPConstraint,
} from '../types.js';

// Tensor utilities
import { xavierInit } from '../tensor.js';

// Modules under test
import {
  attentionModelEncode,
  greedyDecode,
} from '../optimization/attention-model.js';

import {
  sinkhornAssignment,
  hungarianAlgorithm,
  bipartiteGNNAssignment,
} from '../optimization/event-room-assignment.js';

import {
  encodeMIP,
  mipGNNPredict,
} from '../optimization/mip-gnn.js';

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

/** Build attention model weights for a given config. */
function makeAttentionModelWeights(
  config: AttentionModelConfig,
  seed: number,
): AttentionModelWeights {
  const { dim, numLayers } = config;
  const ffnDim = dim * 2;
  let s = seed;

  const encoderLayers = [];
  for (let l = 0; l < numLayers; l++) {
    encoderLayers.push({
      W_Q: xavierInit(dim, dim, createPRNG(s++)),
      W_K: xavierInit(dim, dim, createPRNG(s++)),
      W_V: xavierInit(dim, dim, createPRNG(s++)),
      W_O: xavierInit(dim, dim, createPRNG(s++)),
      ffnW1: xavierInit(dim, ffnDim, createPRNG(s++)),
      ffnB1: new Float64Array(ffnDim),
      ffnW2: xavierInit(ffnDim, dim, createPRNG(s++)),
      ffnB2: new Float64Array(dim),
    });
  }

  return {
    encoderLayers,
    decoderW_Q: xavierInit(dim, dim, createPRNG(s++)),
    decoderW_K: xavierInit(dim, dim, createPRNG(s++)),
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('GNN-8: Combinatorial Optimization', () => {
  // =========================================================================
  // Sinkhorn Assignment
  // =========================================================================
  describe('sinkhornAssignment', () => {
    it('produces a doubly-stochastic matrix (rows and cols sum to ~1)', () => {
      const numRows = 4;
      const numCols = 4;

      // Random cost matrix
      const rng = createPRNG(1);
      const costMatrix = new Float64Array(numRows * numCols);
      for (let i = 0; i < costMatrix.length; i++) {
        costMatrix[i] = rng() * 10;
      }

      const config: SinkhornConfig = {
        iterations: 50,
        temperature: 1.0,
        epsilon: 1e-12,
      };

      const M = sinkhornAssignment(costMatrix, numRows, numCols, config);

      expect(M.length).toBe(numRows * numCols);
      expect(allFinite(M)).toBe(true);

      // Check row sums are approximately 1
      for (let i = 0; i < numRows; i++) {
        let rowSum = 0;
        for (let j = 0; j < numCols; j++) {
          rowSum += M[i * numCols + j]!;
        }
        expect(rowSum).toBeCloseTo(1.0, 2);
      }

      // Check column sums are approximately 1
      for (let j = 0; j < numCols; j++) {
        let colSum = 0;
        for (let i = 0; i < numRows; i++) {
          colSum += M[i * numCols + j]!;
        }
        expect(colSum).toBeCloseTo(1.0, 2);
      }
    });

    it('all entries are non-negative', () => {
      const numRows = 3;
      const numCols = 3;

      const costMatrix = new Float64Array([
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
      ]);

      const config: SinkhornConfig = {
        iterations: 30,
        temperature: 0.5,
        epsilon: 1e-12,
      };

      const M = sinkhornAssignment(costMatrix, numRows, numCols, config);

      for (let i = 0; i < M.length; i++) {
        expect(M[i]!).toBeGreaterThanOrEqual(0);
      }
    });

    it('lower temperature pushes toward harder assignment', () => {
      const costMatrix = new Float64Array([
        1, 100,
        100, 1,
      ]);

      const softConfig: SinkhornConfig = {
        iterations: 50,
        temperature: 10.0,
        epsilon: 1e-12,
      };

      const hardConfig: SinkhornConfig = {
        iterations: 50,
        temperature: 0.1,
        epsilon: 1e-12,
      };

      const Msoft = sinkhornAssignment(costMatrix, 2, 2, softConfig);
      const Mhard = sinkhornAssignment(costMatrix, 2, 2, hardConfig);

      // With hard temperature, diagonal should be closer to 1
      // (since diagonal has low cost)
      expect(Mhard[0]!).toBeGreaterThan(Msoft[0]!);
      expect(Mhard[3]!).toBeGreaterThan(Msoft[3]!);
    });
  });

  // =========================================================================
  // Hungarian Algorithm
  // =========================================================================
  describe('hungarianAlgorithm', () => {
    it('finds optimal assignment for a known cost matrix', () => {
      // Classic example:
      // Cost matrix:
      //   [1, 2, 3]
      //   [3, 3, 1]
      //   [3, 1, 3]
      //
      // Optimal: row 0->col 0 (cost 1), row 1->col 2 (cost 1), row 2->col 1 (cost 1)
      // Total cost = 3
      const costMatrix = new Float64Array([
        1, 2, 3,
        3, 3, 1,
        3, 1, 3,
      ]);

      const result = hungarianAlgorithm(costMatrix, 3);

      expect(result.feasible).toBe(true);
      expect(result.cost).toBe(3);

      // Verify assignment is a valid permutation
      const assigned = new Set<number>();
      for (let i = 0; i < 3; i++) {
        assigned.add(result.assignment[i]!);
      }
      expect(assigned.size).toBe(3);

      // Verify the specific optimal assignment
      expect(result.assignment[0]).toBe(0); // row 0 -> col 0
      expect(result.assignment[1]).toBe(2); // row 1 -> col 2
      expect(result.assignment[2]).toBe(1); // row 2 -> col 1
    });

    it('handles 1x1 matrix', () => {
      const costMatrix = new Float64Array([42]);
      const result = hungarianAlgorithm(costMatrix, 1);

      expect(result.feasible).toBe(true);
      expect(result.assignment[0]).toBe(0);
      expect(result.cost).toBe(42);
    });

    it('handles 0x0 matrix', () => {
      const result = hungarianAlgorithm(new Float64Array(0), 0);

      expect(result.feasible).toBe(true);
      expect(result.assignment.length).toBe(0);
      expect(result.cost).toBe(0);
    });

    it('finds optimal for diagonal cost matrix', () => {
      // Diagonal matrix: cheapest to assign row i -> col i
      const n = 4;
      const costMatrix = new Float64Array(n * n);
      costMatrix.fill(100);
      for (let i = 0; i < n; i++) {
        costMatrix[i * n + i] = 1;
      }

      const result = hungarianAlgorithm(costMatrix, n);

      expect(result.feasible).toBe(true);
      expect(result.cost).toBe(n); // 4 * 1 = 4
      for (let i = 0; i < n; i++) {
        expect(result.assignment[i]).toBe(i);
      }
    });
  });

  // =========================================================================
  // Attention Model Encode
  // =========================================================================
  describe('attentionModelEncode', () => {
    it('produces output with correct dimensions (numNodes x dim)', () => {
      const numNodes = 6;
      const dim = 8;
      const heads = 2;
      const numLayers = 2;

      const config: AttentionModelConfig = {
        dim,
        heads,
        numLayers,
        clipC: 10,
      };

      const weights = makeAttentionModelWeights(config, 100);

      // Random node features
      const rng = createPRNG(42);
      const nodeFeatures = new Float64Array(numNodes * dim);
      for (let i = 0; i < nodeFeatures.length; i++) {
        nodeFeatures[i] = rng() * 2 - 1;
      }

      const embeddings = attentionModelEncode(nodeFeatures, weights, config);

      expect(embeddings.length).toBe(numNodes * dim);
      expect(allFinite(embeddings)).toBe(true);
    });

    it('single-layer encoder produces non-trivial output', () => {
      const numNodes = 4;
      const dim = 4;

      const config: AttentionModelConfig = {
        dim,
        heads: 1,
        numLayers: 1,
        clipC: 10,
      };

      const weights = makeAttentionModelWeights(config, 200);

      const rng = createPRNG(55);
      const nodeFeatures = new Float64Array(numNodes * dim);
      for (let i = 0; i < nodeFeatures.length; i++) {
        nodeFeatures[i] = rng() * 2 - 1;
      }

      const embeddings = attentionModelEncode(nodeFeatures, weights, config);

      // Output should differ from input (non-identity transformation)
      let different = false;
      for (let i = 0; i < embeddings.length; i++) {
        if (Math.abs(embeddings[i]! - nodeFeatures[i]!) > 1e-10) {
          different = true;
          break;
        }
      }
      expect(different).toBe(true);
    });
  });

  // =========================================================================
  // Greedy Decode
  // =========================================================================
  describe('greedyDecode', () => {
    it('produces a valid permutation of all nodes', () => {
      const numNodes = 5;
      const dim = 8;
      const heads = 2;

      const config: AttentionModelConfig = {
        dim,
        heads,
        numLayers: 1,
        clipC: 10,
      };

      const weights = makeAttentionModelWeights(config, 300);

      // Create embeddings via the encoder
      const rng = createPRNG(42);
      const nodeFeatures = new Float64Array(numNodes * dim);
      for (let i = 0; i < nodeFeatures.length; i++) {
        nodeFeatures[i] = rng() * 2 - 1;
      }
      const embeddings = attentionModelEncode(nodeFeatures, weights, config);

      // Greedy decode
      const tour = greedyDecode(embeddings, weights, config, numNodes);

      expect(tour.length).toBe(numNodes);

      // Check it's a valid permutation: every node appears exactly once
      const visited = new Set<number>();
      for (let i = 0; i < numNodes; i++) {
        visited.add(tour[i]!);
      }
      expect(visited.size).toBe(numNodes);

      // All indices should be in valid range
      for (let i = 0; i < numNodes; i++) {
        expect(tour[i]!).toBeGreaterThanOrEqual(0);
        expect(tour[i]!).toBeLessThan(numNodes);
      }
    });

    it('is deterministic with the same inputs', () => {
      const numNodes = 4;
      const dim = 4;

      const config: AttentionModelConfig = {
        dim,
        heads: 1,
        numLayers: 1,
        clipC: 10,
      };

      const weights = makeAttentionModelWeights(config, 400);

      const rng = createPRNG(99);
      const nodeFeatures = new Float64Array(numNodes * dim);
      for (let i = 0; i < nodeFeatures.length; i++) {
        nodeFeatures[i] = rng() * 2 - 1;
      }
      const embeddings = attentionModelEncode(nodeFeatures, weights, config);

      const tour1 = greedyDecode(embeddings, weights, config, numNodes);
      const tour2 = greedyDecode(embeddings, weights, config, numNodes);

      for (let i = 0; i < numNodes; i++) {
        expect(tour1[i]).toBe(tour2[i]);
      }
    });
  });

  // =========================================================================
  // Bipartite GNN Assignment
  // =========================================================================
  describe('bipartiteGNNAssignment', () => {
    it('produces correct cost matrix dimensions', () => {
      const numEvents = 3;
      const numRooms = 4;
      const featureDim = 5;
      const outDim = 6;
      const rng = createPRNG(500);

      const eventFeatures = new Float64Array(numEvents * featureDim);
      const roomFeatures = new Float64Array(numRooms * featureDim);
      for (let i = 0; i < eventFeatures.length; i++) {
        eventFeatures[i] = rng() * 2 - 1;
      }
      for (let i = 0; i < roomFeatures.length; i++) {
        roomFeatures[i] = rng() * 2 - 1;
      }

      // W = [W_event | W_room], total size = 2 * featureDim * outDim
      const W = xavierInit(2 * featureDim, outDim, createPRNG(501));

      const costMatrix = bipartiteGNNAssignment(
        eventFeatures,
        roomFeatures,
        numEvents,
        numRooms,
        featureDim,
        W,
        outDim,
      );

      expect(costMatrix.length).toBe(numEvents * numRooms);
      expect(allFinite(costMatrix)).toBe(true);
    });
  });

  // =========================================================================
  // MIP Encoding
  // =========================================================================
  describe('encodeMIP', () => {
    it('creates bipartite graph with correct structure', () => {
      const variables: MIPVariable[] = [
        { cost: 1.0, lb: 0, ub: 1, isInteger: true },
        { cost: 2.0, lb: 0, ub: 1, isInteger: true },
        { cost: 3.0, lb: 0, ub: 10, isInteger: false },
      ];

      const constraints: MIPConstraint[] = [
        {
          coeffs: new Float64Array([1, 1]),
          varIndices: new Uint32Array([0, 1]),
          rhs: 1,
          sense: 'le' as const,
        },
        {
          coeffs: new Float64Array([2, 1]),
          varIndices: new Uint32Array([1, 2]),
          rhs: 5,
          sense: 'ge' as const,
        },
      ];

      const encoding = encodeMIP(variables, constraints);

      expect(encoding.numVariables).toBe(3);
      expect(encoding.numConstraints).toBe(2);

      // Total nodes = numVariables + numConstraints = 5
      expect(encoding.graph.numNodes).toBe(5);

      // Feature dim should be 4 for all nodes
      expect(encoding.graph.featureDim).toBe(4);
      expect(encoding.graph.nodeFeatures.length).toBe(5 * 4);

      // Edges: each (variable, constraint) pair with non-zero coefficient
      // gets 2 edges (bidirectional). Constraint 0 has vars [0,1], constraint 1 has vars [1,2]
      // Total: 2 * (2 + 2) = 8 edges
      expect(encoding.graph.numEdges).toBe(8);

      // Check variable node features
      // Variable 0: [cost=1, lb=0, ub=1, isInteger=1]
      expect(encoding.graph.nodeFeatures[0]).toBe(1.0);
      expect(encoding.graph.nodeFeatures[1]).toBe(0);
      expect(encoding.graph.nodeFeatures[2]).toBe(1);
      expect(encoding.graph.nodeFeatures[3]).toBe(1); // isInteger

      // Variable 2: [cost=3, lb=0, ub=10, isInteger=0]
      expect(encoding.graph.nodeFeatures[2 * 4]).toBe(3.0);
      expect(encoding.graph.nodeFeatures[2 * 4 + 3]).toBe(0); // not integer

      // Check constraint node features
      // Constraint 0 (node 3): [rhs=1, sense_le=1, sense_ge=0, sense_eq=0]
      expect(encoding.graph.nodeFeatures[3 * 4]).toBe(1);
      expect(encoding.graph.nodeFeatures[3 * 4 + 1]).toBe(1); // le
      expect(encoding.graph.nodeFeatures[3 * 4 + 2]).toBe(0);
      expect(encoding.graph.nodeFeatures[3 * 4 + 3]).toBe(0);

      // Constraint 1 (node 4): [rhs=5, sense_le=0, sense_ge=1, sense_eq=0]
      expect(encoding.graph.nodeFeatures[4 * 4]).toBe(5);
      expect(encoding.graph.nodeFeatures[4 * 4 + 1]).toBe(0);
      expect(encoding.graph.nodeFeatures[4 * 4 + 2]).toBe(1); // ge
      expect(encoding.graph.nodeFeatures[4 * 4 + 3]).toBe(0);
    });

    it('edge weights correspond to constraint coefficients', () => {
      const variables: MIPVariable[] = [
        { cost: 1.0, lb: 0, ub: 1, isInteger: true },
        { cost: 2.0, lb: 0, ub: 1, isInteger: false },
      ];

      const constraints: MIPConstraint[] = [
        {
          coeffs: new Float64Array([3.5]),
          varIndices: new Uint32Array([0]),
          rhs: 10,
          sense: 'eq' as const,
        },
      ];

      const encoding = encodeMIP(variables, constraints);

      // Should have edge weights
      expect(encoding.graph.edgeWeights).toBeDefined();

      // Find the edges and verify weight = 3.5
      if (encoding.graph.edgeWeights) {
        // All edge weights should be 3.5 (bidirectional edges with same coefficient)
        for (let e = 0; e < encoding.graph.numEdges; e++) {
          expect(encoding.graph.edgeWeights[e]).toBe(3.5);
        }
      }
    });
  });

  // =========================================================================
  // MIP-GNN Predict
  // =========================================================================
  describe('mipGNNPredict', () => {
    it('produces bias predictions in [0, 1] for each variable', () => {
      const variables: MIPVariable[] = [
        { cost: 1.0, lb: 0, ub: 1, isInteger: true },
        { cost: 2.0, lb: 0, ub: 1, isInteger: true },
        { cost: 3.0, lb: 0, ub: 5, isInteger: false },
      ];

      const constraints: MIPConstraint[] = [
        {
          coeffs: new Float64Array([1, 1]),
          varIndices: new Uint32Array([0, 1]),
          rhs: 1,
          sense: 'le' as const,
        },
        {
          coeffs: new Float64Array([1, 2]),
          varIndices: new Uint32Array([0, 2]),
          rhs: 3,
          sense: 'ge' as const,
        },
      ];

      const encoding = encodeMIP(variables, constraints);

      const dim = 8;
      const rounds = 2;

      // W_vc: (dim+1) x dim
      const W_vc = xavierInit(dim + 1, dim, createPRNG(600));
      // W_cv: (dim+1) x dim
      const W_cv = xavierInit(dim + 1, dim, createPRNG(601));
      // W_out: dim x 1
      const W_out = xavierInit(dim, 1, createPRNG(602));

      const biases = mipGNNPredict(encoding, W_vc, W_cv, W_out, dim, rounds);

      expect(biases.length).toBe(variables.length);
      expect(allFinite(biases)).toBe(true);

      // All biases should be in [0, 1] (output of sigmoid)
      for (let i = 0; i < biases.length; i++) {
        expect(biases[i]!).toBeGreaterThanOrEqual(0);
        expect(biases[i]!).toBeLessThanOrEqual(1);
      }
    });
  });
});
