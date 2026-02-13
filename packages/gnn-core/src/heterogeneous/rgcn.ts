// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — R-GCN (Relational Graph Convolutional Network)
// Schlichtkrull et al. 2018 — Relation-specific weights with basis decomposition.
//
// For each relation r, W_r = Σ_{b=1}^{numBases} coeffs[r][b] * bases[b].
// Message: h_i += (1/|N_r(i)|) * Σ_{j∈N_r(i)} W_r * h_j
// Output:  ReLU(h_i + bias)
// ---------------------------------------------------------------------------

import type { HeteroGraph, RGCNConfig, RGCNWeights } from '../types.js';
import { matMul, relu, add } from '../tensor.js';

/**
 * R-GCN layer with basis decomposition.
 *
 * Algorithm:
 * 1. For each relation r, reconstruct W_r from basis matrices:
 *    W_r = Σ_{b=0}^{numBases-1} coeffs[r][b] * bases[b]
 *    where each basis is (inDim × outDim).
 * 2. For each edge type, iterate edges in CSR order. For each target node i,
 *    aggregate neighbor features transformed by W_r with mean normalization.
 * 3. Add optional bias and apply ReLU activation.
 *
 * @param heteroGraph - Heterogeneous graph with typed nodes and edges.
 * @param nodeFeatures - Map from node type to row-major feature matrix (count × featureDim).
 * @param weights - R-GCN weights: bases, coefficients, and optional bias.
 * @param config - R-GCN configuration: inDim, outDim, numRelations, numBases, bias.
 * @returns Map from node type to updated feature matrix (count × outDim).
 */
export function rgcnLayer(
  heteroGraph: HeteroGraph,
  nodeFeatures: Map<string, Float64Array>,
  weights: RGCNWeights,
  config: RGCNConfig,
): Map<string, Float64Array> {
  const { inDim, outDim, numBases } = config;

  // --- Step 1: Pre-compute relation-specific weight matrices via basis decomposition ---
  // relationWeights[r] is a Float64Array of size inDim × outDim
  const relationWeights: Float64Array[] = [];

  for (let r = 0; r < config.numRelations; r++) {
    const Wr = new Float64Array(inDim * outDim);
    for (let b = 0; b < numBases; b++) {
      const coeff = weights.coeffs[r * numBases + b]!;
      const basis = weights.bases[b]!;
      for (let idx = 0; idx < inDim * outDim; idx++) {
        Wr[idx] = Wr[idx]! + coeff * basis[idx]!;
      }
    }
    relationWeights.push(Wr);
  }

  // --- Step 2: Initialize output accumulators per node type ---
  // Track aggregated output and neighbor counts per node for normalization
  const outputAccum = new Map<string, Float64Array>();
  const neighborCounts = new Map<string, Float64Array>();

  for (const nodeType of heteroGraph.nodeTypes) {
    const store = heteroGraph.nodes.get(nodeType)!;
    outputAccum.set(nodeType, new Float64Array(store.count * outDim));
    neighborCounts.set(nodeType, new Float64Array(store.count));
  }

  // --- Step 3: For each edge type, aggregate messages ---
  const edgeTypeList = heteroGraph.edgeTypes;

  for (let r = 0; r < edgeTypeList.length; r++) {
    const [srcType, relation, dstType] = edgeTypeList[r]!;
    const edgeKey = `${srcType}/${relation}/${dstType}`;
    const edgeStore = heteroGraph.edges.get(edgeKey);
    if (!edgeStore || edgeStore.numEdges === 0) continue;

    const srcFeatures = nodeFeatures.get(srcType);
    if (!srcFeatures) continue;

    const dstAccum = outputAccum.get(dstType)!;
    const dstCounts = neighborCounts.get(dstType)!;

    // Use the relation index (clamped to available relation weights)
    const Wr = relationWeights[r % config.numRelations]!;

    // Transform all source features: XW_r of shape (srcCount × outDim)
    const srcStore = heteroGraph.nodes.get(srcType)!;
    const transformedSrc = matMul(srcFeatures, Wr, srcStore.count, inDim, outDim);

    // Iterate over destination nodes in CSR format
    // edgeStore.rowPtr is indexed by destination node (target of the message)
    const dstStore = heteroGraph.nodes.get(dstType)!;

    for (let dst = 0; dst < dstStore.count; dst++) {
      const start = edgeStore.rowPtr[dst]!;
      const end = edgeStore.rowPtr[dst + 1]!;
      const degree = end - start;
      if (degree === 0) continue;

      const norm = 1.0 / degree;

      for (let e = start; e < end; e++) {
        const src = edgeStore.colIdx[e]!;
        // Accumulate normalized transformed source feature into dst
        for (let d = 0; d < outDim; d++) {
          dstAccum[dst * outDim + d] =
            dstAccum[dst * outDim + d]! + norm * transformedSrc[src * outDim + d]!;
        }
      }

      dstCounts[dst] = dstCounts[dst]! + 1;
    }
  }

  // --- Step 4: Apply bias and ReLU activation ---
  const result = new Map<string, Float64Array>();

  for (const nodeType of heteroGraph.nodeTypes) {
    let output = outputAccum.get(nodeType)!;
    const store = heteroGraph.nodes.get(nodeType)!;

    // Add bias if configured
    if (config.bias && weights.bias) {
      const biased = new Float64Array(output.length);
      for (let i = 0; i < store.count; i++) {
        for (let d = 0; d < outDim; d++) {
          biased[i * outDim + d] = output[i * outDim + d]! + weights.bias[d]!;
        }
      }
      output = biased;
    }

    // Apply ReLU
    result.set(nodeType, relu(output));
  }

  return result;
}
