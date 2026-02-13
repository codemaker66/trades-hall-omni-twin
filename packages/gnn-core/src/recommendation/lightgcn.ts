// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — LightGCN (He et al. 2020)
// Collaborative filtering via pure neighborhood aggregation.
// NO feature transform, NO nonlinearity — only aggregation + final mean.
// ---------------------------------------------------------------------------

import type { PRNG, Graph, LightGCNConfig, LightGCNResult } from '../types.js';
import { bprLoss } from '../tensor.js';
import { degree } from '../graph.js';

// ---------------------------------------------------------------------------
// 1. lightGCNPropagate — Multi-layer propagation with symmetric norm
// ---------------------------------------------------------------------------

/**
 * LightGCN multi-layer propagation.
 *
 * For each layer k:
 *   e_u^{k+1} = SUM_{i in N(u)} (1 / sqrt(|N(u)|) * sqrt(|N(i)|)) * e_i^{k}
 *
 * Final embedding = (1 / (K+1)) * SUM_{k=0}^{K} e^{k}
 *
 * Key insight: NO weight matrices, NO activation functions. Just aggregation.
 *
 * @param graph - Bipartite user-item interaction graph in CSR format.
 * @param embeddings - Initial embeddings, row-major (numNodes x embDim).
 * @param numLayers - Number of propagation layers (K).
 * @param embDim - Embedding dimension per node.
 * @returns Final embeddings after layer-weighted mean (numNodes x embDim).
 */
export function lightGCNPropagate(
  graph: Graph,
  embeddings: Float64Array,
  numLayers: number,
  embDim: number,
): Float64Array {
  const numNodes = graph.numNodes;

  // Precompute inverse sqrt of degree for each node
  const invSqrtDeg = new Float64Array(numNodes);
  for (let u = 0; u < numNodes; u++) {
    const d = degree(graph, u);
    invSqrtDeg[u] = d > 0 ? 1.0 / Math.sqrt(d) : 0.0;
  }

  // Accumulator for the weighted mean across all layers (sum of e^0 .. e^K)
  const accumulated = new Float64Array(numNodes * embDim);

  // Add layer-0 embeddings to accumulator
  for (let i = 0; i < numNodes * embDim; i++) {
    accumulated[i] = embeddings[i]!;
  }

  // Current layer embeddings
  let current = new Float64Array(embeddings);

  for (let layer = 0; layer < numLayers; layer++) {
    const next = new Float64Array(numNodes * embDim);

    // For each node u, aggregate neighbor embeddings with symmetric norm
    for (let u = 0; u < numNodes; u++) {
      const start = graph.rowPtr[u]!;
      const end = graph.rowPtr[u + 1]!;
      const normU = invSqrtDeg[u]!;

      for (let e = start; e < end; e++) {
        const v = graph.colIdx[e]!;
        const normV = invSqrtDeg[v]!;
        const coeff = normU * normV; // 1/sqrt(|N(u)|) * 1/sqrt(|N(v)|)

        const uOff = u * embDim;
        const vOff = v * embDim;
        for (let d = 0; d < embDim; d++) {
          next[uOff + d]! += coeff * current[vOff + d]!;
        }
      }
    }

    // Add this layer's embeddings to accumulator
    for (let i = 0; i < numNodes * embDim; i++) {
      accumulated[i] = accumulated[i]! + next[i]!;
    }

    current = next;
  }

  // Final embedding = (1 / (K+1)) * accumulated
  const scale = 1.0 / (numLayers + 1);
  for (let i = 0; i < numNodes * embDim; i++) {
    accumulated[i] = accumulated[i]! * scale;
  }

  return accumulated;
}

// ---------------------------------------------------------------------------
// 2. bprLossCompute — BPR pairwise ranking loss
// ---------------------------------------------------------------------------

/**
 * BPR loss: -mean(ln(sigmoid(posScore - negScore))).
 *
 * @param posScores - Positive (user, pos-item) interaction dot products.
 * @param negScores - Negative (user, neg-item) interaction dot products.
 * @returns Scalar BPR loss value.
 */
export function bprLossCompute(
  posScores: Float64Array,
  negScores: Float64Array,
): number {
  return bprLoss(posScores, negScores);
}

// ---------------------------------------------------------------------------
// 3. lightGCNTrain — Full training loop
// ---------------------------------------------------------------------------

/**
 * Train LightGCN on a user-item interaction graph.
 *
 * Algorithm:
 * 1. Initialize random embeddings for (numUsers + numItems) nodes.
 * 2. For each epoch:
 *    a. Propagate through K layers (no transforms, no activations).
 *    b. Sample positive edges from graph, sample random negative items.
 *    c. Compute BPR loss on (user, pos-item, neg-item) triplets.
 *    d. SGD update on the raw embedding matrix (NOT the propagated embeddings).
 * 3. Return trained embeddings and loss history.
 *
 * @param graph - Bipartite user-item graph (users: 0..numUsers-1, items: numUsers..numUsers+numItems-1).
 * @param config - LightGCN hyperparameters.
 * @param rng - Deterministic PRNG for initialization and sampling.
 * @returns Trained user/item embeddings and loss history.
 */
export function lightGCNTrain(
  graph: Graph,
  config: LightGCNConfig,
  rng: PRNG,
): LightGCNResult {
  const {
    numUsers,
    numItems,
    embeddingDim,
    numLayers,
    learningRate,
    l2Reg,
    epochs,
  } = config;

  const totalNodes = numUsers + numItems;

  // Initialize embeddings with small random values (Xavier-like scale)
  const embeddings = new Float64Array(totalNodes * embeddingDim);
  const initScale = Math.sqrt(1.0 / embeddingDim);
  for (let i = 0; i < embeddings.length; i++) {
    embeddings[i] = (rng() * 2 - 1) * initScale;
  }

  // Collect positive edges (user -> item interactions) from graph
  const posEdges: { user: number; item: number }[] = [];
  for (let u = 0; u < numUsers; u++) {
    const start = graph.rowPtr[u]!;
    const end = graph.rowPtr[u + 1]!;
    for (let e = start; e < end; e++) {
      const item = graph.colIdx[e]!;
      if (item >= numUsers && item < totalNodes) {
        posEdges.push({ user: u, item });
      }
    }
  }

  const losses: number[] = [];

  for (let epoch = 0; epoch < epochs; epoch++) {
    // Propagate to get multi-layer embeddings
    const propagated = lightGCNPropagate(graph, embeddings, numLayers, embeddingDim);

    // Sample a batch of triplets (all positive edges + random negatives)
    const batchSize = posEdges.length;
    const posScores = new Float64Array(batchSize);
    const negScores = new Float64Array(batchSize);

    // Gradient accumulators for the raw embeddings
    const grad = new Float64Array(totalNodes * embeddingDim);

    for (let b = 0; b < batchSize; b++) {
      const edge = posEdges[b]!;
      const u = edge.user;
      const posItem = edge.item;

      // Sample a random negative item (not in user's neighborhood)
      let negItem = numUsers + Math.floor(rng() * numItems);
      // Simple rejection: just pick a different random item if it's the same as positive
      if (negItem === posItem) {
        negItem = numUsers + ((negItem - numUsers + 1) % numItems);
      }

      // Compute scores as dot product of propagated embeddings
      let posScore = 0;
      let negScore = 0;
      const uOff = u * embeddingDim;
      const posOff = posItem * embeddingDim;
      const negOff = negItem * embeddingDim;

      for (let d = 0; d < embeddingDim; d++) {
        posScore += propagated[uOff + d]! * propagated[posOff + d]!;
        negScore += propagated[uOff + d]! * propagated[negOff + d]!;
      }

      posScores[b] = posScore;
      negScores[b] = negScore;

      // BPR gradient: d/d_theta [-ln(sigmoid(pos - neg))]
      // = -sigmoid(neg - pos) for the positive direction
      const diff = posScore - negScore;
      let sigmoidNegDiff: number;
      if (diff >= 0) {
        sigmoidNegDiff = 1.0 / (1.0 + Math.exp(-diff));
        sigmoidNegDiff = 1.0 - sigmoidNegDiff; // sigmoid(-diff)
      } else {
        const expDiff = Math.exp(diff);
        sigmoidNegDiff = 1.0 / (1.0 + expDiff);
      }
      // Gradient factor: -sigmoid(neg - pos) = -(1 - sigmoid(pos - neg))
      const gFactor = -sigmoidNegDiff;

      // Accumulate gradients on raw embeddings
      // For user u: grad wrt e_u is gFactor * (e_posItem - e_negItem)
      // For posItem: grad wrt e_posItem is gFactor * e_u
      // For negItem: grad wrt e_negItem is -gFactor * e_u
      for (let d = 0; d < embeddingDim; d++) {
        const eu = embeddings[uOff + d]!;
        const ep = embeddings[posOff + d]!;
        const en = embeddings[negOff + d]!;

        grad[uOff + d] = grad[uOff + d]! + gFactor * (ep - en) + l2Reg * eu;
        grad[posOff + d] = grad[posOff + d]! + gFactor * eu + l2Reg * ep;
        grad[negOff + d] = grad[negOff + d]! + (-gFactor) * eu + l2Reg * en;
      }
    }

    // Compute epoch loss
    const epochLoss = bprLossCompute(posScores, negScores);
    losses.push(epochLoss);

    // SGD update on raw embeddings
    const scaledLr = learningRate / Math.max(batchSize, 1);
    for (let i = 0; i < embeddings.length; i++) {
      embeddings[i] = embeddings[i]! - scaledLr * grad[i]!;
    }
  }

  // Final propagation to get output embeddings
  const finalPropagated = lightGCNPropagate(graph, embeddings, numLayers, embeddingDim);

  // Split into user and item embeddings
  const userEmbeddings = new Float64Array(numUsers * embeddingDim);
  const itemEmbeddings = new Float64Array(numItems * embeddingDim);

  for (let u = 0; u < numUsers; u++) {
    const off = u * embeddingDim;
    for (let d = 0; d < embeddingDim; d++) {
      userEmbeddings[off + d] = finalPropagated[off + d]!;
    }
  }

  for (let i = 0; i < numItems; i++) {
    const srcOff = (numUsers + i) * embeddingDim;
    const dstOff = i * embeddingDim;
    for (let d = 0; d < embeddingDim; d++) {
      itemEmbeddings[dstOff + d] = finalPropagated[srcOff + d]!;
    }
  }

  return { userEmbeddings, itemEmbeddings, losses };
}
