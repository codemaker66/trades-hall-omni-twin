// ---------------------------------------------------------------------------
// GNN-8: Combinatorial Optimization — Attention Model
// Kool et al. 2019 "Attention, Learn to Solve Routing Problems!"
//
// Transformer-based encoder + autoregressive decoder for combinatorial
// optimization problems (TSP, VRP, assignment). Pure TypeScript, Float64Array.
// ---------------------------------------------------------------------------

import type { AttentionModelConfig, AttentionModelWeights } from '../types.js';
import { matMul, matVecMul, relu, scale, add, dot, softmax } from '../tensor.js';

// ---------------------------------------------------------------------------
// 1. attentionModelEncode — Multi-layer transformer encoder
// ---------------------------------------------------------------------------

/**
 * Encode node features through a multi-layer transformer encoder.
 *
 * Each layer applies:
 *   1. Multi-head self-attention:
 *      Q = h * W_Q, K = h * W_K, V = h * W_V  (per head)
 *      Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V
 *      Output = concat(heads) * W_O
 *   2. Residual connection + implicit layer norm (skip for simplicity)
 *   3. Position-wise FFN: W2 * relu(W1 * x + B1) + B2
 *   4. Residual connection
 *
 * @param nodeFeatures - Flat row-major (numNodes x dim) input features.
 * @param weights - Encoder layer weights (W_Q, W_K, W_V, W_O, FFN).
 * @param config - Model configuration (dim, heads, numLayers, clipC).
 * @returns Float64Array of shape (numNodes x dim), row-major node embeddings.
 */
export function attentionModelEncode(
  nodeFeatures: Float64Array,
  weights: AttentionModelWeights,
  config: AttentionModelConfig,
): Float64Array {
  const { dim, heads, numLayers } = config;
  const numNodes = nodeFeatures.length / dim;
  const headDim = dim / heads;
  const sqrtDk = Math.sqrt(headDim);

  // h is the current hidden state: (numNodes x dim) row-major
  let h = new Float64Array(nodeFeatures);

  for (let layer = 0; layer < numLayers; layer++) {
    const lw = weights.encoderLayers[layer]!;

    // ---- Multi-Head Self-Attention ----
    // Q, K, V: (numNodes x dim) = h * W  where W is (dim x dim)
    const Q = matMul(h, lw.W_Q, numNodes, dim, dim);
    const K = matMul(h, lw.W_K, numNodes, dim, dim);
    const V = matMul(h, lw.W_V, numNodes, dim, dim);

    // Compute attention per head, then concatenate
    const attnOut = new Float64Array(numNodes * dim);

    for (let hd = 0; hd < heads; hd++) {
      const hdOffset = hd * headDim;

      // Extract per-head Q, K, V slices: each (numNodes x headDim)
      // Compute scaled dot-product attention scores: (numNodes x numNodes)
      const scores = new Float64Array(numNodes * numNodes);

      for (let i = 0; i < numNodes; i++) {
        for (let j = 0; j < numNodes; j++) {
          let s = 0;
          for (let d = 0; d < headDim; d++) {
            s += Q[i * dim + hdOffset + d]! * K[j * dim + hdOffset + d]!;
          }
          scores[i * numNodes + j] = s / sqrtDk;
        }
      }

      // Softmax over each row (over j dimension)
      const attnWeights = softmax(scores, numNodes);

      // Weighted sum of V: out_i = sum_j attn[i,j] * V_j
      for (let i = 0; i < numNodes; i++) {
        for (let j = 0; j < numNodes; j++) {
          const w = attnWeights[i * numNodes + j]!;
          for (let d = 0; d < headDim; d++) {
            attnOut[i * dim + hdOffset + d] = attnOut[i * dim + hdOffset + d]! + w * V[j * dim + hdOffset + d]!;
          }
        }
      }
    }

    // Project concatenated heads: (numNodes x dim) * W_O (dim x dim)
    const projected = matMul(attnOut, lw.W_O, numNodes, dim, dim);

    // Residual connection
    const afterAttn = add(h, projected);

    // ---- Position-wise FFN ----
    // FFN(x) = W2 * relu(W1 * x + B1) + B2
    // W1: (dim x ffnDim), B1: (ffnDim), W2: (ffnDim x dim), B2: (dim)
    // Infer ffnDim from W1 size: W1 is (dim x ffnDim) => ffnDim = W1.length / dim
    const ffnDim = lw.ffnW1.length / dim;

    const ffnHidden = matMul(afterAttn, lw.ffnW1, numNodes, dim, ffnDim);

    // Add bias B1 to each row and apply ReLU
    for (let i = 0; i < numNodes; i++) {
      for (let d = 0; d < ffnDim; d++) {
        ffnHidden[i * ffnDim + d] = ffnHidden[i * ffnDim + d]! + lw.ffnB1[d]!;
      }
    }
    const ffnActivated = relu(ffnHidden);

    const ffnOut = matMul(ffnActivated, lw.ffnW2, numNodes, ffnDim, dim);

    // Add bias B2 to each row
    for (let i = 0; i < numNodes; i++) {
      for (let d = 0; d < dim; d++) {
        ffnOut[i * dim + d] = ffnOut[i * dim + d]! + lw.ffnB2[d]!;
      }
    }

    // Residual connection
    h = add(afterAttn, ffnOut) as Float64Array<ArrayBuffer>;
  }

  return h;
}

// ---------------------------------------------------------------------------
// 2. attentionModelDecode — Autoregressive decoder with clipped log-probs
// ---------------------------------------------------------------------------

/**
 * Compute log probabilities for selecting the next node using the attention
 * model's decoder mechanism.
 *
 * The decoder computes a context query from the mean embedding of all nodes
 * (in practice, the mean of unvisited nodes would be used during sequential
 * decoding; here we compute over all nodes as the initial step).
 *
 * Score_i = clipC * tanh( q^T * k_i / sqrt(dim) )
 *
 * where q = W_query * context, k_i = W_key * embedding_i.
 *
 * @param embeddings - Node embeddings from encoder, (numNodes x dim) row-major.
 * @param W_query - Query projection weight (dim x dim).
 * @param W_key - Key projection weight (dim x dim).
 * @param numNodes - Number of nodes.
 * @param dim - Embedding dimension.
 * @param clipC - Clipping constant for tanh (typically 10).
 * @returns Float64Array of log probabilities (numNodes).
 */
export function attentionModelDecode(
  embeddings: Float64Array,
  W_query: Float64Array,
  W_key: Float64Array,
  numNodes: number,
  dim: number,
  clipC: number,
): Float64Array {
  // Compute context as mean of all node embeddings
  const context = new Float64Array(dim);
  for (let i = 0; i < numNodes; i++) {
    for (let d = 0; d < dim; d++) {
      context[d] = context[d]! + embeddings[i * dim + d]!;
    }
  }
  for (let d = 0; d < dim; d++) {
    context[d] = context[d]! / numNodes;
  }

  // q = W_query * context  (dim x dim) * (dim) -> (dim)
  const q = matVecMul(W_query, context, dim, dim);

  // k_i = W_key * embedding_i for each node
  // Keys: (numNodes x dim) = embeddings * W_key^T, but we need per-node keys
  // Compute keys: each k_i = W_key * emb_i (dim x dim) * (dim) -> (dim)
  const keys = new Float64Array(numNodes * dim);
  for (let i = 0; i < numNodes; i++) {
    const emb_i = embeddings.slice(i * dim, (i + 1) * dim);
    const k_i = matVecMul(W_key, emb_i, dim, dim);
    keys.set(k_i, i * dim);
  }

  const sqrtDim = Math.sqrt(dim);

  // Score_i = clipC * tanh(q^T * k_i / sqrt(dim))
  const scores = new Float64Array(numNodes);
  for (let i = 0; i < numNodes; i++) {
    let s = 0;
    for (let d = 0; d < dim; d++) {
      s += q[d]! * keys[i * dim + d]!;
    }
    scores[i] = clipC * Math.tanh(s / sqrtDim);
  }

  // Convert to log probabilities via log-softmax
  // log_softmax(x_i) = x_i - log(sum_j exp(x_j))
  let maxScore = -Infinity;
  for (let i = 0; i < numNodes; i++) {
    if (scores[i]! > maxScore) maxScore = scores[i]!;
  }

  let sumExp = 0;
  for (let i = 0; i < numNodes; i++) {
    sumExp += Math.exp(scores[i]! - maxScore);
  }
  const logSumExp = maxScore + Math.log(sumExp);

  const logProbs = new Float64Array(numNodes);
  for (let i = 0; i < numNodes; i++) {
    logProbs[i] = scores[i]! - logSumExp;
  }

  return logProbs;
}

// ---------------------------------------------------------------------------
// 3. greedyDecode — Greedy autoregressive decoding
// ---------------------------------------------------------------------------

/**
 * Greedy decoding: repeatedly select the highest-probability unvisited node.
 *
 * At each step:
 *   1. Compute context query from mean of remaining unvisited node embeddings.
 *   2. Compute clipped attention scores for all unvisited nodes.
 *   3. Select the node with highest score.
 *   4. Mark as visited and append to tour.
 *
 * @param embeddings - Node embeddings from encoder, (numNodes x dim) row-major.
 * @param weights - Attention model weights (uses decoderW_Q, decoderW_K).
 * @param config - Model configuration (dim, clipC).
 * @param numNodes - Number of nodes.
 * @returns Uint32Array of tour/assignment order (numNodes).
 */
export function greedyDecode(
  embeddings: Float64Array,
  weights: AttentionModelWeights,
  config: AttentionModelConfig,
  numNodes: number,
): Uint32Array {
  const { dim, clipC } = config;
  const tour = new Uint32Array(numNodes);
  const visited = new Uint8Array(numNodes); // 0 = unvisited, 1 = visited
  const sqrtDim = Math.sqrt(dim);

  // Precompute keys for all nodes: k_i = W_key * emb_i
  const keys = new Float64Array(numNodes * dim);
  for (let i = 0; i < numNodes; i++) {
    const emb_i = embeddings.slice(i * dim, (i + 1) * dim);
    const k_i = matVecMul(weights.decoderW_K, emb_i, dim, dim);
    keys.set(k_i, i * dim);
  }

  for (let step = 0; step < numNodes; step++) {
    // Compute context: mean of unvisited node embeddings
    const context = new Float64Array(dim);
    let unvisitedCount = 0;
    for (let i = 0; i < numNodes; i++) {
      if (!visited[i]) {
        for (let d = 0; d < dim; d++) {
          context[d] = context[d]! + embeddings[i * dim + d]!;
        }
        unvisitedCount++;
      }
    }
    if (unvisitedCount > 0) {
      for (let d = 0; d < dim; d++) {
        context[d] = context[d]! / unvisitedCount;
      }
    }

    // q = W_query * context
    const q = matVecMul(weights.decoderW_Q, context, dim, dim);

    // Find best unvisited node by score = clipC * tanh(q^T k_i / sqrt(dim))
    let bestNode = -1;
    let bestScore = -Infinity;

    for (let i = 0; i < numNodes; i++) {
      if (visited[i]) continue;

      let s = 0;
      for (let d = 0; d < dim; d++) {
        s += q[d]! * keys[i * dim + d]!;
      }
      const score = clipC * Math.tanh(s / sqrtDim);

      if (score > bestScore) {
        bestScore = score;
        bestNode = i;
      }
    }

    tour[step] = bestNode;
    visited[bestNode] = 1;
  }

  return tour;
}
