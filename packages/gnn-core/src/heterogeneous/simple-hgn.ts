// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — Simple-HGN (Simple Heterogeneous Graph Network)
// Baseline that outperforms complex HetGNNs on HGB benchmarks.
//
// GAT-style attention + learnable edge-type embeddings:
//   α_{ij} = LeakyReLU(a^T (W*h_i ‖ W*h_j + edgeTypeEmb[r]))
//   Softmax per source node, weighted aggregation.
//   Residual connection + L2 row normalization.
// ---------------------------------------------------------------------------

import type { HeteroGraph, SimpleHGNWeights } from '../types.js';
import { matMul } from '../tensor.js';

/**
 * Simple-HGN layer.
 *
 * Algorithm:
 * 1. Project all node features to outDim: Z = X * W (inDim × outDim).
 *    Since node types may have different input dimensions, W is applied
 *    uniformly (assumed all projected to same inDim first).
 * 2. For each edge type, compute GAT-style attention with edge-type embeddings:
 *    - For edge (i → j) of type r:
 *      e_{ij} = LeakyReLU(a_left^T * Z_i + a_right^T * (Z_j + edgeTypeEmb[r]))
 * 3. Softmax attention per target node (over all incoming edges across all types).
 * 4. Weighted aggregation: h'_i = Σ_j α_{ij} * Z_j
 * 5. Residual: h'_i += residualW * h_i (if residualW provided and dims match,
 *    or h'_i += h_i if inDim === outDim).
 * 6. L2 row normalization: h'_i = h'_i / ‖h'_i‖₂
 *
 * @param heteroGraph - Heterogeneous graph.
 * @param nodeFeatures - Map from node type to feature matrix (count × inDim).
 * @param weights - SimpleHGN weights: W, a, edgeTypeEmb[], optional residualW.
 * @param inDim - Input feature dimension.
 * @param outDim - Output feature dimension.
 * @returns Map from node type to updated feature matrix (count × outDim).
 */
export function simpleHGNLayer(
  heteroGraph: HeteroGraph,
  nodeFeatures: Map<string, Float64Array>,
  weights: SimpleHGNWeights,
  inDim: number,
  outDim: number,
): Map<string, Float64Array> {
  // Attention vector: a has length 2 * outDim, split into a_left and a_right
  const a = weights.a;
  const W = weights.W; // inDim × outDim

  // --- Step 1: Project all node features ---
  const projected = new Map<string, Float64Array>();
  for (const nodeType of heteroGraph.nodeTypes) {
    const store = heteroGraph.nodes.get(nodeType)!;
    const features = nodeFeatures.get(nodeType);
    if (!features) continue;
    projected.set(nodeType, matMul(features, W, store.count, inDim, outDim));
  }

  // --- Step 2: Precompute source scores (a_left^T * Z_i) for each node ---
  const srcScores = new Map<string, Float64Array>();
  for (const nodeType of heteroGraph.nodeTypes) {
    const Z = projected.get(nodeType);
    const store = heteroGraph.nodes.get(nodeType)!;
    if (!Z) continue;

    const scores = new Float64Array(store.count);
    for (let i = 0; i < store.count; i++) {
      let s = 0;
      for (let d = 0; d < outDim; d++) {
        s += a[d]! * Z[i * outDim + d]!;
      }
      scores[i] = s;
    }
    srcScores.set(nodeType, scores);
  }

  // --- Step 3: For each target node, collect all incoming edges with attention ---
  // We need to accumulate attention scores across edge types before softmax.
  // Structure: for each target node type, track raw attention + edge info.

  // First pass: compute raw attention scores for all edges
  interface EdgeInfo {
    srcNode: number;
    srcType: string;
    rawAttn: number;
  }

  // incomingEdges[dstType][dstNode] = list of EdgeInfo
  const incomingEdges = new Map<string, Map<number, EdgeInfo[]>>();
  for (const nodeType of heteroGraph.nodeTypes) {
    incomingEdges.set(nodeType, new Map());
  }

  for (let eIdx = 0; eIdx < heteroGraph.edgeTypes.length; eIdx++) {
    const [srcType, relation, dstType] = heteroGraph.edgeTypes[eIdx]!;
    const edgeKey = `${srcType}/${relation}/${dstType}`;
    const edgeStore = heteroGraph.edges.get(edgeKey);
    if (!edgeStore || edgeStore.numEdges === 0) continue;

    const srcZ = projected.get(srcType);
    const srcSc = srcScores.get(srcType);
    if (!srcZ || !srcSc) continue;

    const dstStore = heteroGraph.nodes.get(dstType)!;
    const edgeTypeEmb = weights.edgeTypeEmb[eIdx % weights.edgeTypeEmb.length]!;
    const dstEdges = incomingEdges.get(dstType)!;

    // Precompute a_right^T * edgeTypeEmb for this edge type
    let edgeTypeBias = 0;
    for (let d = 0; d < outDim; d++) {
      edgeTypeBias += a[outDim + d]! * edgeTypeEmb[d]!;
    }

    for (let t = 0; t < dstStore.count; t++) {
      const start = edgeStore.rowPtr[t]!;
      const end = edgeStore.rowPtr[t + 1]!;

      for (let e = start; e < end; e++) {
        const s = edgeStore.colIdx[e]!;

        // a_right^T * (Z_j + edgeTypeEmb[r])
        // = a_right^T * Z_j + a_right^T * edgeTypeEmb[r]
        let dstScore = 0;
        for (let d = 0; d < outDim; d++) {
          dstScore += a[outDim + d]! * srcZ[s * outDim + d]!;
        }
        dstScore += edgeTypeBias;

        // Note: in Simple-HGN the attention is computed for messages TO node t
        // FROM node s. So the "target" receiving the message is t, "source"
        // sending the message is s.
        // e_{st} = LeakyReLU(a_left^T * Z_t + a_right^T * (Z_s + emb_r))
        // Wait - looking at the specification more carefully:
        // For edge (i→j) of type r: α_{ij} = LeakyReLU(a^T (W*h_i ‖ W*h_j + edgeTypeEmb[r]))
        // The edge goes i→j, so i is the source, j is the target receiving message.
        // In CSR, rowPtr is indexed by "source" of edges stored.
        // But our HeteroGraph CSR indexes by destination node for message passing.
        // So t is the node receiving, s is sending.

        // Raw attention for this edge:
        // We want: LeakyReLU(a_left^T * Z_t + a_right^T * (Z_s + emb_r))
        // But the specification says: a^T (W*h_i ‖ W*h_j + edgeTypeEmb[r])
        // where (i→j), so h_i is the sender, h_j the receiver.
        // Let's use the projected target node score for the first half.

        // Recompute: src of the attention is the "sender" node s,
        // dst is "receiver" node t in the message passing sense.
        // The concat is [Z_s ‖ Z_t + emb] but let's follow spec literally:
        // a^T [W*h_i ‖ W*h_j + edgeTypeEmb[r]] where i→j edge
        // In our CSR, edges point from s to t where t is iterated via rowPtr.
        // Actually, let's interpret the CSR: rowPtr[t] gives edges INTO t.
        // So s is the source, t is the destination.
        // a^T [W*h_s ‖ W*h_t + emb_r]

        // Recompute properly:
        let srcPart = 0;
        for (let d = 0; d < outDim; d++) {
          srcPart += a[d]! * srcZ[s * outDim + d]!;
        }

        // For the destination part we need Z_t
        const dstZ = projected.get(dstType);
        if (!dstZ) continue;

        let dstPart = 0;
        for (let d = 0; d < outDim; d++) {
          dstPart += a[outDim + d]! * (dstZ[t * outDim + d]! + edgeTypeEmb[d]!);
        }

        const rawAttn = srcPart + dstPart;
        // LeakyReLU with slope 0.2
        const activated = rawAttn > 0 ? rawAttn : 0.2 * rawAttn;

        if (!dstEdges.has(t)) dstEdges.set(t, []);
        dstEdges.get(t)!.push({
          srcNode: s,
          srcType,
          rawAttn: activated,
        });
      }
    }
  }

  // --- Step 4: Softmax + aggregation per target node ---
  const result = new Map<string, Float64Array>();

  for (const nodeType of heteroGraph.nodeTypes) {
    const store = heteroGraph.nodes.get(nodeType)!;
    const output = new Float64Array(store.count * outDim);
    const dstEdges = incomingEdges.get(nodeType)!;

    for (let t = 0; t < store.count; t++) {
      const edges = dstEdges.get(t);
      if (!edges || edges.length === 0) continue;

      // Softmax over all incoming edges to node t
      let maxVal = -Infinity;
      for (const edge of edges) {
        if (edge.rawAttn > maxVal) maxVal = edge.rawAttn;
      }

      let sumExp = 0;
      const expVals = new Float64Array(edges.length);
      for (let i = 0; i < edges.length; i++) {
        const expVal = Math.exp(edges[i]!.rawAttn - maxVal);
        expVals[i] = expVal;
        sumExp += expVal;
      }

      // Weighted aggregation
      for (let i = 0; i < edges.length; i++) {
        const alpha = sumExp > 0 ? expVals[i]! / sumExp : 0;
        const edge = edges[i]!;
        const srcZ = projected.get(edge.srcType);
        if (!srcZ) continue;

        for (let d = 0; d < outDim; d++) {
          output[t * outDim + d] = output[t * outDim + d]! +
            alpha * srcZ[edge.srcNode * outDim + d]!;
        }
      }
    }

    // --- Step 5: Residual connection ---
    const origFeatures = nodeFeatures.get(nodeType);
    if (origFeatures) {
      if (weights.residualW && inDim !== outDim) {
        // Project residual: h_res = X * residualW
        const residual = matMul(origFeatures, weights.residualW, store.count, inDim, outDim);
        for (let idx = 0; idx < store.count * outDim; idx++) {
          output[idx] = output[idx]! + residual[idx]!;
        }
      } else if (inDim === outDim) {
        // Direct residual addition
        for (let idx = 0; idx < store.count * outDim; idx++) {
          output[idx] = output[idx]! + origFeatures[idx]!;
        }
      }
    }

    // --- Step 6: L2 row normalization ---
    for (let i = 0; i < store.count; i++) {
      let normSq = 0;
      const offset = i * outDim;
      for (let d = 0; d < outDim; d++) {
        normSq += output[offset + d]! * output[offset + d]!;
      }
      const norm = Math.sqrt(normSq);
      if (norm > 0) {
        for (let d = 0; d < outDim; d++) {
          output[offset + d] = output[offset + d]! / norm;
        }
      }
    }

    result.set(nodeType, output);
  }

  return result;
}
