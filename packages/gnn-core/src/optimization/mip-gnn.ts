// ---------------------------------------------------------------------------
// GNN-8: Combinatorial Optimization — MIP-GNN
// GNN-guided branch-and-bound for Mixed Integer Programming.
//
// Encodes MILP instances as bipartite variable-constraint graphs and uses
// half-convolution message passing (Gasse et al. 2019) to predict variable
// branching biases. Pure TypeScript, Float64Array.
// ---------------------------------------------------------------------------

import type { Graph, MIPVariable, MIPConstraint, MIPEncoding } from '../types.js';
import { relu, sigmoid } from '../tensor.js';
import { buildCSR } from '../graph.js';

// ---------------------------------------------------------------------------
// 1. encodeMIP — Encode MILP as bipartite variable-constraint graph
// ---------------------------------------------------------------------------

/**
 * Encode a Mixed Integer Linear Program as a bipartite graph.
 *
 * The bipartite graph has two node types:
 *   - Variable nodes (indices 0..numVariables-1):
 *     Features = [cost, lb, ub, isInteger] (4-dim)
 *   - Constraint nodes (indices numVariables..numVariables+numConstraints-1):
 *     Features = [rhs, sense_le, sense_ge, sense_eq] (4-dim, one-hot sense)
 *
 * Edges connect variable i to constraint j where constraint j involves variable i.
 * Edge weight = coefficient of variable i in constraint j.
 * Edges are bidirectional (variable -> constraint and constraint -> variable).
 *
 * @param variables - Array of MIPVariable definitions.
 * @param constraints - Array of MIPConstraint definitions.
 * @returns MIPEncoding with the bipartite graph, numVariables, numConstraints.
 */
export function encodeMIP(
  variables: MIPVariable[],
  constraints: MIPConstraint[],
): MIPEncoding {
  const numVariables = variables.length;
  const numConstraints = constraints.length;
  const totalNodes = numVariables + numConstraints;
  const featureDim = 4;

  // Build node features
  const nodeFeatures = new Float64Array(totalNodes * featureDim);

  // Variable node features: [cost, lb, ub, isInteger]
  for (let i = 0; i < numVariables; i++) {
    const v = variables[i]!;
    const offset = i * featureDim;
    nodeFeatures[offset] = v.cost;
    nodeFeatures[offset + 1] = v.lb;
    nodeFeatures[offset + 2] = v.ub;
    nodeFeatures[offset + 3] = v.isInteger ? 1 : 0;
  }

  // Constraint node features: [rhs, sense_le, sense_ge, sense_eq]
  for (let j = 0; j < numConstraints; j++) {
    const c = constraints[j]!;
    const offset = (numVariables + j) * featureDim;
    nodeFeatures[offset] = c.rhs;
    nodeFeatures[offset + 1] = c.sense === 'le' ? 1 : 0;
    nodeFeatures[offset + 2] = c.sense === 'ge' ? 1 : 0;
    nodeFeatures[offset + 3] = c.sense === 'eq' ? 1 : 0;
  }

  // Build edges: bidirectional between variables and constraints
  const edges: [number, number][] = [];
  const weights: number[] = [];

  for (let j = 0; j < numConstraints; j++) {
    const c = constraints[j]!;
    const constraintNode = numVariables + j;

    for (let k = 0; k < c.varIndices.length; k++) {
      const varIdx = c.varIndices[k]!;
      const coeff = c.coeffs[k]!;

      // Variable -> Constraint edge
      edges.push([varIdx, constraintNode]);
      weights.push(coeff);

      // Constraint -> Variable edge
      edges.push([constraintNode, varIdx]);
      weights.push(coeff);
    }
  }

  const csrGraph = buildCSR(edges, totalNodes, weights);

  const graph: Graph = {
    ...csrGraph,
    nodeFeatures,
    featureDim,
  };

  return { graph, numVariables, numConstraints };
}

// ---------------------------------------------------------------------------
// 2. mipGNNPredict — Half-convolution message passing for variable biases
// ---------------------------------------------------------------------------

/**
 * Predict variable branching biases using half-convolution message passing
 * on the bipartite variable-constraint graph.
 *
 * Half-convolution (Gasse et al. 2019):
 *   For each round:
 *     1. Variable -> Constraint messages:
 *        c_j = relu( sum_i W_vc * [v_i || edge_weight_ij] )
 *        where [v_i || w] is concatenation of variable embedding and edge weight.
 *     2. Constraint -> Variable messages:
 *        v_i = relu( sum_j W_cv * [c_j || edge_weight_ij] )
 *        where [c_j || w] is concatenation of constraint embedding and edge weight.
 *
 *   After all rounds:
 *     bias_i = sigmoid(W_out * v_i) for each variable node.
 *
 * @param encoding - MIPEncoding from encodeMIP.
 * @param W_vc - Variable-to-constraint weight matrix ((dim+1) x dim), row-major.
 * @param W_cv - Constraint-to-variable weight matrix ((dim+1) x dim), row-major.
 * @param W_out - Output weight matrix (dim x 1), row-major.
 * @param dim - Embedding dimension.
 * @param rounds - Number of message passing rounds.
 * @returns Float64Array of bias predictions for each variable (numVariables).
 */
export function mipGNNPredict(
  encoding: MIPEncoding,
  W_vc: Float64Array,
  W_cv: Float64Array,
  W_out: Float64Array,
  dim: number,
  rounds: number,
): Float64Array {
  const { graph, numVariables, numConstraints } = encoding;
  const totalNodes = numVariables + numConstraints;
  const featureDim = graph.featureDim;

  // Initialize node embeddings from features
  // If featureDim < dim, pad with zeros; if featureDim > dim, truncate
  const varEmb = new Float64Array(numVariables * dim);
  const conEmb = new Float64Array(numConstraints * dim);

  // Initialize variable embeddings from node features
  for (let i = 0; i < numVariables; i++) {
    const copyDim = Math.min(featureDim, dim);
    for (let d = 0; d < copyDim; d++) {
      varEmb[i * dim + d] = graph.nodeFeatures[i * featureDim + d]!;
    }
  }

  // Initialize constraint embeddings from node features
  for (let j = 0; j < numConstraints; j++) {
    const copyDim = Math.min(featureDim, dim);
    for (let d = 0; d < copyDim; d++) {
      conEmb[j * dim + d] = graph.nodeFeatures[(numVariables + j) * featureDim + d]!;
    }
  }

  // Message passing rounds
  for (let round = 0; round < rounds; round++) {
    // ---- Variable -> Constraint messages ----
    // For each constraint node j, aggregate messages from connected variable nodes
    const newConEmb = new Float64Array(numConstraints * dim);

    for (let j = 0; j < numConstraints; j++) {
      const constraintNode = numVariables + j;
      const start = graph.rowPtr[constraintNode]!;
      const end = graph.rowPtr[constraintNode + 1]!;

      // Accumulate messages from variable neighbors
      const aggMsg = new Float64Array(dim);

      for (let e = start; e < end; e++) {
        const neighbor = graph.colIdx[e]!;
        // Only process variable -> constraint edges (neighbor < numVariables)
        if (neighbor >= numVariables) continue;

        const edgeWeight = graph.edgeWeights ? graph.edgeWeights[e]! : 1.0;

        // Construct input: [v_neighbor || edge_weight] of size (dim + 1)
        // Apply W_vc: ((dim+1) x dim)^T * (dim+1) -> dim
        // i.e., out[d] = sum_{k=0}^{dim} W_vc[k * dim + d] * input[k]
        for (let d = 0; d < dim; d++) {
          let val = 0;
          // Variable embedding part
          for (let k = 0; k < dim; k++) {
            val += W_vc[k * dim + d]! * varEmb[neighbor * dim + k]!;
          }
          // Edge weight part (last row of W_vc)
          val += W_vc[dim * dim + d]! * edgeWeight;
          aggMsg[d] = aggMsg[d]! + val;
        }
      }

      // Apply ReLU to aggregated message
      for (let d = 0; d < dim; d++) {
        newConEmb[j * dim + d] = aggMsg[d]! > 0 ? aggMsg[d]! : 0;
      }
    }

    // Update constraint embeddings
    conEmb.set(newConEmb);

    // ---- Constraint -> Variable messages ----
    // For each variable node i, aggregate messages from connected constraint nodes
    const newVarEmb = new Float64Array(numVariables * dim);

    for (let i = 0; i < numVariables; i++) {
      const start = graph.rowPtr[i]!;
      const end = graph.rowPtr[i + 1]!;

      // Accumulate messages from constraint neighbors
      const aggMsg = new Float64Array(dim);

      for (let e = start; e < end; e++) {
        const neighbor = graph.colIdx[e]!;
        // Only process constraint -> variable edges (neighbor >= numVariables)
        if (neighbor < numVariables) continue;

        const constraintIdx = neighbor - numVariables;
        const edgeWeight = graph.edgeWeights ? graph.edgeWeights[e]! : 1.0;

        // Construct input: [c_neighbor || edge_weight] of size (dim + 1)
        // Apply W_cv: ((dim+1) x dim)^T * (dim+1) -> dim
        for (let d = 0; d < dim; d++) {
          let val = 0;
          // Constraint embedding part
          for (let k = 0; k < dim; k++) {
            val += W_cv[k * dim + d]! * conEmb[constraintIdx * dim + k]!;
          }
          // Edge weight part (last row of W_cv)
          val += W_cv[dim * dim + d]! * edgeWeight;
          aggMsg[d] = aggMsg[d]! + val;
        }
      }

      // Apply ReLU to aggregated message
      for (let d = 0; d < dim; d++) {
        newVarEmb[i * dim + d] = aggMsg[d]! > 0 ? aggMsg[d]! : 0;
      }
    }

    // Update variable embeddings
    varEmb.set(newVarEmb);
  }

  // ---- Output layer ----
  // bias_i = sigmoid(W_out * v_i) for each variable
  // W_out: (dim x 1), so output is a scalar per variable
  const biases = new Float64Array(numVariables);

  for (let i = 0; i < numVariables; i++) {
    let val = 0;
    for (let d = 0; d < dim; d++) {
      val += W_out[d]! * varEmb[i * dim + d]!;
    }
    // Numerically stable sigmoid
    if (val >= 0) {
      biases[i] = 1 / (1 + Math.exp(-val));
    } else {
      const ev = Math.exp(val);
      biases[i] = ev / (1 + ev);
    }
  }

  return biases;
}
