// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — HAN (Heterogeneous Attention Network)
// Wang et al. 2019 — Hierarchical Attention: node-level + semantic-level.
//
// Node-level attention within a single meta-path adjacency:
//   α_{ij} = softmax_j(LeakyReLU(a^T [Wh_i ‖ Wh_j]))
//
// Semantic-level attention across meta-paths:
//   β_p = softmax_p(q^T tanh(W * mean_node(z_p)))
//   Final = Σ_p β_p * z_p
// ---------------------------------------------------------------------------

import type {
  Graph,
  HeteroGraph,
  HANConfig,
  HANWeights,
} from '../types.js';
import { matMul, leakyRelu, softmax, dot, scale, add, tanhActivation } from '../tensor.js';

/**
 * Node-level attention within a single meta-path adjacency.
 *
 * Algorithm (GAT-style on a single meta-path subgraph):
 * 1. Transform features: Z = X * W  (numNodes × outDim).
 * 2. For each edge (i, j), compute attention score:
 *    e_{ij} = LeakyReLU(a^T [Wh_i ‖ Wh_j])
 *    where a has length 2*outDim: a_left for source, a_right for target.
 * 3. Softmax normalize attention per node's neighborhood.
 * 4. Aggregate: h'_i = Σ_j α_{ij} * Z_j
 *
 * @param graph - CSR graph for this meta-path adjacency.
 * @param X - Row-major node features (numNodes × inDim).
 * @param W - Weight matrix (inDim × outDim), row-major.
 * @param a - Attention vector of length 2 * outDim ([a_left | a_right]).
 * @param inDim - Input feature dimension.
 * @param outDim - Output feature dimension.
 * @returns Object with `output` (numNodes × outDim) and `weights` (numEdges attention coefficients).
 */
export function hanNodeAttention(
  graph: Graph,
  X: Float64Array,
  W: Float64Array,
  a: Float64Array,
  inDim: number,
  outDim: number,
): { output: Float64Array; weights: Float64Array } {
  const numNodes = graph.numNodes;

  // Step 1: Transform all node features: Z = X * W, shape (numNodes × outDim)
  const Z = matMul(X, W, numNodes, inDim, outDim);

  // Step 2: Precompute a_left^T * Z_i and a_right^T * Z_j for each node
  // a_left = a[0..outDim), a_right = a[outDim..2*outDim)
  const srcScores = new Float64Array(numNodes);
  const dstScores = new Float64Array(numNodes);

  for (let i = 0; i < numNodes; i++) {
    let srcS = 0;
    let dstS = 0;
    for (let d = 0; d < outDim; d++) {
      const zVal = Z[i * outDim + d]!;
      srcS += a[d]! * zVal;
      dstS += a[outDim + d]! * zVal;
    }
    srcScores[i] = srcS;
    dstScores[i] = dstS;
  }

  // Step 3: Compute attention weights per edge and softmax per neighborhood
  const attnWeights = new Float64Array(graph.numEdges);

  for (let i = 0; i < numNodes; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;
    const degree = end - start;
    if (degree === 0) continue;

    // Compute raw attention scores for all neighbors of node i
    // e_{ij} = LeakyReLU(srcScore_i + dstScore_j)
    let maxVal = -Infinity;
    for (let e = start; e < end; e++) {
      const j = graph.colIdx[e]!;
      const raw = srcScores[i]! + dstScores[j]!;
      // LeakyReLU with slope 0.2
      const activated = raw > 0 ? raw : 0.2 * raw;
      attnWeights[e] = activated;
      if (activated > maxVal) maxVal = activated;
    }

    // Softmax normalization over neighbors of node i
    let sumExp = 0;
    for (let e = start; e < end; e++) {
      const expVal = Math.exp(attnWeights[e]! - maxVal);
      attnWeights[e] = expVal;
      sumExp += expVal;
    }
    if (sumExp > 0) {
      for (let e = start; e < end; e++) {
        attnWeights[e] = attnWeights[e]! / sumExp;
      }
    }
  }

  // Step 4: Weighted aggregation: h'_i = Σ_j α_{ij} * Z_j
  const output = new Float64Array(numNodes * outDim);

  for (let i = 0; i < numNodes; i++) {
    const start = graph.rowPtr[i]!;
    const end = graph.rowPtr[i + 1]!;

    for (let e = start; e < end; e++) {
      const j = graph.colIdx[e]!;
      const alpha = attnWeights[e]!;
      for (let d = 0; d < outDim; d++) {
        output[i * outDim + d] = output[i * outDim + d]! + alpha * Z[j * outDim + d]!;
      }
    }
  }

  return { output, weights: attnWeights };
}

/**
 * Semantic-level attention across multiple meta-path outputs.
 *
 * Algorithm:
 * 1. For each meta-path p, compute mean node embedding: mean_p = mean(z_p) over nodes.
 * 2. Transform: t_p = tanh(W * mean_p), where W is (dim × dim).
 * 3. Score: s_p = q^T * t_p.
 * 4. β_p = softmax(s_1, ..., s_P).
 * 5. Final output = Σ_p β_p * z_p  (numNodes × dim).
 *
 * @param metaPathOutputs - Array of meta-path output matrices, each (numNodes × dim), row-major.
 * @param W - Semantic transform matrix (dim × dim), row-major.
 * @param q - Semantic attention query vector of length dim.
 * @param numNodes - Number of nodes.
 * @param dim - Feature dimension.
 * @returns Combined output (numNodes × dim), row-major.
 */
export function hanSemanticAttention(
  metaPathOutputs: Float64Array[],
  W: Float64Array,
  q: Float64Array,
  numNodes: number,
  dim: number,
): Float64Array {
  const numPaths = metaPathOutputs.length;
  if (numPaths === 0) return new Float64Array(numNodes * dim);
  if (numPaths === 1) return new Float64Array(metaPathOutputs[0]!);

  // Step 1-3: Compute attention score for each meta-path
  const scores = new Float64Array(numPaths);

  for (let p = 0; p < numPaths; p++) {
    const zp = metaPathOutputs[p]!;

    // Compute mean node embedding for this meta-path
    const meanEmb = new Float64Array(dim);
    for (let i = 0; i < numNodes; i++) {
      for (let d = 0; d < dim; d++) {
        meanEmb[d] = meanEmb[d]! + zp[i * dim + d]!;
      }
    }
    if (numNodes > 0) {
      for (let d = 0; d < dim; d++) {
        meanEmb[d] = meanEmb[d]! / numNodes;
      }
    }

    // Transform: t_p = tanh(W * meanEmb)
    const transformed = new Float64Array(dim);
    for (let i = 0; i < dim; i++) {
      let val = 0;
      for (let j = 0; j < dim; j++) {
        val += W[i * dim + j]! * meanEmb[j]!;
      }
      transformed[i] = Math.tanh(val);
    }

    // Score: s_p = q^T * t_p
    let score = 0;
    for (let d = 0; d < dim; d++) {
      score += q[d]! * transformed[d]!;
    }
    scores[p] = score;
  }

  // Step 4: Softmax over meta-path scores
  const beta = softmax(scores, numPaths);

  // Step 5: Weighted combination: output = Σ_p β_p * z_p
  const output = new Float64Array(numNodes * dim);

  for (let p = 0; p < numPaths; p++) {
    const bp = beta[p]!;
    const zp = metaPathOutputs[p]!;
    for (let idx = 0; idx < numNodes * dim; idx++) {
      output[idx] = output[idx]! + bp * zp[idx]!;
    }
  }

  return output;
}

/**
 * Full HAN layer: node-level attention per meta-path, then semantic attention.
 *
 * Algorithm:
 * 1. For each meta-path, extract or build the meta-path adjacency subgraph.
 *    (Meta-paths reference edge type indices into heteroGraph.edgeTypes.)
 * 2. Apply hanNodeAttention on each meta-path subgraph.
 * 3. Apply hanSemanticAttention to combine meta-path outputs.
 *
 * Note: This implementation assumes a single target node type. The meta-path
 * adjacency is materialized by composing edge types along the path. For a
 * two-hop meta-path [e0, e1], nodes connected through an intermediate type
 * are found and a direct adjacency is built for the target node type.
 *
 * @param heteroGraph - Heterogeneous graph.
 * @param nodeFeatures - Map from node type to feature matrix.
 * @param weights - HAN weights: nodeAttn (GAT weights), semanticAttnVec, W_semantic.
 * @param config - HAN config: inDim, outDim, heads, metaPaths.
 * @returns Combined output for the target node type (numNodes × outDim), row-major.
 */
export function hanLayer(
  heteroGraph: HeteroGraph,
  nodeFeatures: Map<string, Float64Array>,
  weights: HANWeights,
  config: HANConfig,
): Float64Array {
  const { inDim, outDim, metaPaths } = config;
  const edgeTypeList = heteroGraph.edgeTypes;

  // Determine the target node type from the first meta-path
  // For a meta-path [e0, e1, ...], the target type is the dst of the last edge type
  if (metaPaths.length === 0 || edgeTypeList.length === 0) {
    return new Float64Array(0);
  }

  // Identify target node type from first meta-path's last edge
  const firstPath = metaPaths[0]!;
  const lastEdgeIdx = firstPath[firstPath.length - 1]!;
  const lastEdgeType = edgeTypeList[lastEdgeIdx]!;
  const targetNodeType = lastEdgeType[2];
  const targetStore = heteroGraph.nodes.get(targetNodeType);
  if (!targetStore) return new Float64Array(0);
  const numTargetNodes = targetStore.count;

  // Collect target node features as the input for attention
  const targetFeatures = nodeFeatures.get(targetNodeType);
  if (!targetFeatures) return new Float64Array(numTargetNodes * outDim);

  // For each meta-path, materialize the adjacency and run node attention
  const metaPathOutputs: Float64Array[] = [];

  // W for node-level attention (from nodeAttn GATWeights)
  const W = weights.nodeAttn.W; // inDim × outDim
  // Attention vector a: use a_src and a_dst concatenated as [a_left | a_right]
  const aVec = new Float64Array(outDim * 2);
  aVec.set(weights.nodeAttn.a_src.subarray(0, outDim), 0);
  aVec.set(weights.nodeAttn.a_dst.subarray(0, outDim), outDim);

  for (const metaPath of metaPaths) {
    // Materialize the meta-path adjacency for the target node type.
    // A meta-path of length 1: [edgeIdx] → direct adjacency.
    // A meta-path of length 2: [e0, e1] → compose two adjacencies.

    if (metaPath.length === 1) {
      // Single-hop meta-path: use the edge type's CSR directly
      const edgeIdx = metaPath[0]!;
      const [srcType, relation, dstType] = edgeTypeList[edgeIdx]!;
      const edgeKey = `${srcType}/${relation}/${dstType}`;
      const edgeStore = heteroGraph.edges.get(edgeKey);
      if (!edgeStore) {
        metaPathOutputs.push(new Float64Array(numTargetNodes * outDim));
        continue;
      }

      // Build a Graph for hanNodeAttention
      const graph: import('../types.js').Graph = {
        numNodes: numTargetNodes,
        numEdges: edgeStore.numEdges,
        rowPtr: edgeStore.rowPtr,
        colIdx: edgeStore.colIdx,
        nodeFeatures: targetFeatures,
        featureDim: inDim,
      };

      const { output } = hanNodeAttention(graph, targetFeatures, W, aVec, inDim, outDim);
      metaPathOutputs.push(output);
    } else if (metaPath.length === 2) {
      // Two-hop meta-path: compose edge types
      const e0Idx = metaPath[0]!;
      const e1Idx = metaPath[1]!;
      const [srcType0, rel0, dstType0] = edgeTypeList[e0Idx]!;
      const [srcType1, rel1, dstType1] = edgeTypeList[e1Idx]!;

      const edgeKey0 = `${srcType0}/${rel0}/${dstType0}`;
      const edgeKey1 = `${srcType1}/${rel1}/${dstType1}`;
      const edgeStore0 = heteroGraph.edges.get(edgeKey0);
      const edgeStore1 = heteroGraph.edges.get(edgeKey1);

      if (!edgeStore0 || !edgeStore1) {
        metaPathOutputs.push(new Float64Array(numTargetNodes * outDim));
        continue;
      }

      // Compose: for each target node in dstType1, find intermediate nodes
      // in dstType0 (= srcType1), then find source nodes in srcType0.
      // Build direct edges: srcType0 → dstType1.
      const intermediateStore = heteroGraph.nodes.get(dstType0);
      if (!intermediateStore) {
        metaPathOutputs.push(new Float64Array(numTargetNodes * outDim));
        continue;
      }

      // Build composed adjacency as edge list
      const composedEdges = new Map<number, Set<number>>();
      const dstStore1 = heteroGraph.nodes.get(dstType1);
      if (!dstStore1) {
        metaPathOutputs.push(new Float64Array(numTargetNodes * outDim));
        continue;
      }

      for (let t = 0; t < dstStore1.count; t++) {
        // Neighbors of t in edgeStore1 (intermediate nodes)
        const start1 = edgeStore1.rowPtr[t]!;
        const end1 = edgeStore1.rowPtr[t + 1]!;

        for (let e1 = start1; e1 < end1; e1++) {
          const mid = edgeStore1.colIdx[e1]!;
          // Neighbors of mid in edgeStore0 (source nodes)
          if (mid < edgeStore0.rowPtr.length - 1) {
            const start0 = edgeStore0.rowPtr[mid]!;
            const end0 = edgeStore0.rowPtr[mid + 1]!;
            for (let e0 = start0; e0 < end0; e0++) {
              const src = edgeStore0.colIdx[e0]!;
              if (!composedEdges.has(t)) composedEdges.set(t, new Set());
              composedEdges.get(t)!.add(src);
            }
          }
        }
      }

      // Build CSR for the composed meta-path adjacency
      const rowPtr = new Uint32Array(numTargetNodes + 1);
      const edgeList: number[] = [];

      for (let t = 0; t < numTargetNodes; t++) {
        const neighbors = composedEdges.get(t);
        if (neighbors) {
          for (const n of neighbors) {
            edgeList.push(n);
          }
          rowPtr[t + 1] = rowPtr[t]! + neighbors.size;
        } else {
          rowPtr[t + 1] = rowPtr[t]!;
        }
      }

      const colIdx = new Uint32Array(edgeList);

      const graph: import('../types.js').Graph = {
        numNodes: numTargetNodes,
        numEdges: edgeList.length,
        rowPtr,
        colIdx,
        nodeFeatures: targetFeatures,
        featureDim: inDim,
      };

      const { output } = hanNodeAttention(graph, targetFeatures, W, aVec, inDim, outDim);
      metaPathOutputs.push(output);
    } else {
      // For longer meta-paths, fall back to identity (not commonly used)
      metaPathOutputs.push(new Float64Array(numTargetNodes * outDim));
    }
  }

  // Semantic-level attention to combine meta-path outputs
  return hanSemanticAttention(
    metaPathOutputs,
    weights.W_semantic,
    weights.semanticAttnVec,
    numTargetNodes,
    outDim,
  );
}
