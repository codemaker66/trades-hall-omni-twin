// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — HGT (Heterogeneous Graph Transformer)
// Hu et al. 2020 — Type-decomposed multi-head attention for heterogeneous graphs.
//
// For each edge type (τ(s), φ(e), τ(t)):
//   Q = W_Q[τ(t)] * h_t,  K = W_K[τ(s)] * h_s,  V = W_V[τ(s)] * h_s
//   ATT = softmax(K * W_ATT[φ(e)] * Q^T * mu[φ(e)] / √d)
//   MSG = W_MSG[φ(e)] * V
//   Output = ATT ⊙ MSG aggregated per target node
// ---------------------------------------------------------------------------

import type { HeteroGraph, HGTConfig, HGTWeights } from '../types.js';
import { matMul } from '../tensor.js';

/**
 * HGT layer — Heterogeneous Graph Transformer.
 *
 * Algorithm:
 * 1. For each node type, project features to Q, K, V spaces using
 *    type-specific weight matrices.
 * 2. For each edge type (srcType, relation, dstType):
 *    a. Compute Q for target nodes: Q = h_t * W_Q[τ(t)]
 *    b. Compute K for source nodes: K = h_s * W_K[τ(s)]
 *    c. Compute V for source nodes: V = h_s * W_V[τ(s)]
 *    d. For each head, compute attention: score = (K * W_ATT[φ(e)] * Q^T) * mu[φ(e)] / √d_head
 *    e. Softmax over source neighbors per target node.
 *    f. Compute messages: MSG = W_MSG[φ(e)] * V
 *    g. Aggregate: output_t += Σ_s ATT(s,t) * MSG(s)
 * 3. Return updated features per node type.
 *
 * @param heteroGraph - Heterogeneous graph.
 * @param nodeFeatures - Map from node type to feature matrix (count × inDim).
 * @param weights - HGT weights: W_Q, W_K, W_V per node type; W_ATT, W_MSG, mu per edge type.
 * @param config - HGT config: inDim, outDim, heads, numNodeTypes, numEdgeTypes.
 * @returns Map from node type to updated feature matrix (count × outDim).
 */
export function hgtLayer(
  heteroGraph: HeteroGraph,
  nodeFeatures: Map<string, Float64Array>,
  weights: HGTWeights,
  config: HGTConfig,
): Map<string, Float64Array> {
  const { inDim, outDim, heads } = config;
  const dHead = Math.floor(outDim / heads);
  const sqrtD = Math.sqrt(dHead);

  const nodeTypeIndex = new Map<string, number>();
  for (let i = 0; i < heteroGraph.nodeTypes.length; i++) {
    nodeTypeIndex.set(heteroGraph.nodeTypes[i]!, i);
  }

  // --- Step 1: Pre-compute K, V per node type ---
  // K[nodeType] = features * W_K[typeIdx], shape (count × outDim)
  // V[nodeType] = features * W_V[typeIdx], shape (count × outDim)
  const projK = new Map<string, Float64Array>();
  const projV = new Map<string, Float64Array>();

  for (const nodeType of heteroGraph.nodeTypes) {
    const store = heteroGraph.nodes.get(nodeType)!;
    const features = nodeFeatures.get(nodeType);
    if (!features) continue;

    const typeIdx = nodeTypeIndex.get(nodeType)!;
    const WK = weights.W_K[typeIdx]!; // inDim × outDim
    const WV = weights.W_V[typeIdx]!; // inDim × outDim

    projK.set(nodeType, matMul(features, WK, store.count, inDim, outDim));
    projV.set(nodeType, matMul(features, WV, store.count, inDim, outDim));
  }

  // --- Step 2: Initialize output accumulators ---
  const outputAccum = new Map<string, Float64Array>();
  const outputNorm = new Map<string, Float64Array>();

  for (const nodeType of heteroGraph.nodeTypes) {
    const store = heteroGraph.nodes.get(nodeType)!;
    outputAccum.set(nodeType, new Float64Array(store.count * outDim));
    outputNorm.set(nodeType, new Float64Array(store.count));
  }

  // --- Step 3: Process each edge type ---
  for (let eIdx = 0; eIdx < heteroGraph.edgeTypes.length; eIdx++) {
    const [srcType, relation, dstType] = heteroGraph.edgeTypes[eIdx]!;
    const edgeKey = `${srcType}/${relation}/${dstType}`;
    const edgeStore = heteroGraph.edges.get(edgeKey);
    if (!edgeStore || edgeStore.numEdges === 0) continue;

    const srcK = projK.get(srcType);
    const srcV = projV.get(srcType);
    const dstFeatures = nodeFeatures.get(dstType);
    if (!srcK || !srcV || !dstFeatures) continue;

    const dstTypeIdx = nodeTypeIndex.get(dstType)!;
    const dstStore = heteroGraph.nodes.get(dstType)!;
    const srcStore = heteroGraph.nodes.get(srcType)!;

    // Q for destination nodes: features * W_Q[τ(t)]
    const WQ = weights.W_Q[dstTypeIdx]!; // inDim × outDim
    const dstQ = matMul(dstFeatures, WQ, dstStore.count, inDim, outDim);

    // Edge-type specific attention and message weights
    const edgeTypeIdx = eIdx % config.numEdgeTypes;
    const WATT = weights.W_ATT[edgeTypeIdx]!; // dHead × dHead (per head, or outDim × outDim)
    const WMSG = weights.W_MSG[edgeTypeIdx]!; // outDim × outDim
    const mu = weights.mu[edgeTypeIdx]!;

    // Transform V by W_MSG: MSG_src = V * W_MSG
    const msgSrc = matMul(srcV, WMSG, srcStore.count, outDim, outDim);

    const dstAccum = outputAccum.get(dstType)!;

    // For each target node, compute attention over its source neighbors
    for (let t = 0; t < dstStore.count; t++) {
      const start = edgeStore.rowPtr[t]!;
      const end = edgeStore.rowPtr[t + 1]!;
      const degree = end - start;
      if (degree === 0) continue;

      // For each head, compute attention scores
      // We process all heads together for efficiency
      const attnScores = new Float64Array(degree);

      for (let eOff = 0; eOff < degree; eOff++) {
        const s = edgeStore.colIdx[start + eOff]!;
        let totalScore = 0;

        // Sum attention across all heads
        for (let h = 0; h < heads; h++) {
          const headOff = h * dHead;
          // Compute K_s * W_ATT * Q_t for this head
          // K_s_head: srcK[s*outDim + headOff .. +dHead]
          // Q_t_head: dstQ[t*outDim + headOff .. +dHead]
          // Attention: K_s_head * W_ATT_head * Q_t_head^T

          // For simplicity, compute as dot product with W_ATT transformation
          // First transform K by W_ATT (per head block)
          let score = 0;
          for (let di = 0; di < dHead; di++) {
            let kTransformed = 0;
            for (let dj = 0; dj < dHead; dj++) {
              kTransformed += srcK[s * outDim + headOff + dj]! * WATT[di * dHead + dj]!;
            }
            score += kTransformed * dstQ[t * outDim + headOff + di]!;
          }
          totalScore += score;
        }

        attnScores[eOff] = (totalScore * mu) / sqrtD;
      }

      // Softmax over source neighbors
      let maxScore = -Infinity;
      for (let eOff = 0; eOff < degree; eOff++) {
        if (attnScores[eOff]! > maxScore) maxScore = attnScores[eOff]!;
      }

      let sumExp = 0;
      for (let eOff = 0; eOff < degree; eOff++) {
        const expVal = Math.exp(attnScores[eOff]! - maxScore);
        attnScores[eOff] = expVal;
        sumExp += expVal;
      }

      if (sumExp > 0) {
        for (let eOff = 0; eOff < degree; eOff++) {
          attnScores[eOff] = attnScores[eOff]! / sumExp;
        }
      }

      // Aggregate: output_t += Σ_s ATT(s,t) * MSG(s)
      for (let eOff = 0; eOff < degree; eOff++) {
        const s = edgeStore.colIdx[start + eOff]!;
        const alpha = attnScores[eOff]!;
        for (let d = 0; d < outDim; d++) {
          dstAccum[t * outDim + d] = dstAccum[t * outDim + d]! + alpha * msgSrc[s * outDim + d]!;
        }
      }
    }
  }

  // --- Step 4: Return updated features ---
  const result = new Map<string, Float64Array>();
  for (const nodeType of heteroGraph.nodeTypes) {
    result.set(nodeType, outputAccum.get(nodeType)!);
  }

  return result;
}
