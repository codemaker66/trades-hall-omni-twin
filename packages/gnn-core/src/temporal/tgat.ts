// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — GNN-6: Temporal Graph Attention (TGAT, Xu et al. 2020)
// Time-aware multi-head attention over temporal neighborhoods.
// ---------------------------------------------------------------------------

import { matVecMul, dot, softmax, add } from '../tensor.js';
import { bochnerTimeEncoding } from './time-encoding.js';

/** A temporal neighbor record with features and interaction timestamp. */
export interface TemporalNeighbor {
  readonly nodeIdx: number;
  readonly features: Float64Array;
  readonly timestamp: number;
}

/**
 * Temporal Graph Attention layer — time-aware multi-head attention.
 *
 * For each target node i:
 *   1. Compute temporal encoding for each neighbor j:
 *      Phi(t_current - t_j) using Bochner time encoding
 *   2. Concatenate neighbor features with time encoding:
 *      h_j' = [h_j || Phi(dt)]
 *   3. Multi-head attention:
 *      Q_h = W_Q * h_i   (per head slice)
 *      K_h = W_K * h_j'  (per head slice)
 *      V_h = W_V * h_j'  (per head slice)
 *      alpha = softmax(Q^T K / sqrt(d_head))
 *      output_h = sum(alpha * V)
 *   4. Average across heads
 *
 * @param nodeFeatures - (numNodes x inDim) feature matrix, row-major
 * @param neighbors - per-node list of temporal neighbors
 * @param timeEncFreqs - (numFreqs) frequency vector for Bochner encoding
 * @param W_Q - (outDim*heads x inDim) query weight matrix
 * @param W_K - (outDim*heads x augDim) key weight matrix, augDim = inDim + 2*numFreqs
 * @param W_V - (outDim*heads x augDim) value weight matrix
 * @param currentTime - current reference timestamp
 * @param heads - number of attention heads
 * @param inDim - input feature dimension per node
 * @param outDim - output dimension per head
 * @returns (numNodes x outDim) updated features, row-major
 */
export function tgatLayer(
  nodeFeatures: Float64Array,
  neighbors: TemporalNeighbor[][],
  timeEncFreqs: Float64Array,
  W_Q: Float64Array,
  W_K: Float64Array,
  W_V: Float64Array,
  currentTime: number,
  heads: number,
  inDim: number,
  outDim: number,
): Float64Array {
  const numNodes = neighbors.length;
  const numFreqs = timeEncFreqs.length;
  const timeDim = 2 * numFreqs;           // Bochner output dimension
  const augDim = inDim + timeDim;          // augmented neighbor dim
  const headDim = outDim;                  // per-head output dimension
  const totalHeadDim = headDim * heads;    // total across all heads

  const output = new Float64Array(numNodes * outDim);

  for (let i = 0; i < numNodes; i++) {
    const nodeNeighbors = neighbors[i]!;
    const numNeigh = nodeNeighbors.length;

    // Target node features
    const hI = nodeFeatures.slice(i * inDim, (i + 1) * inDim);

    if (numNeigh === 0) {
      // No neighbors: self-projection through W_Q, then average heads
      const qFull = matVecMul(W_Q, hI, totalHeadDim, inDim);
      // Average across heads to get outDim
      for (let d = 0; d < outDim; d++) {
        let sum = 0;
        for (let h = 0; h < heads; h++) {
          sum += qFull[h * headDim + d]!;
        }
        output[i * outDim + d] = sum / heads;
      }
      continue;
    }

    // ---- Compute augmented neighbor features ----
    // For each neighbor: [h_j || Phi(currentTime - t_j)]
    const augFeatures = new Float64Array(numNeigh * augDim);

    for (let j = 0; j < numNeigh; j++) {
      const neigh = nodeNeighbors[j]!;
      const dt = currentTime - neigh.timestamp;
      // Single-element time encoding
      const timeEnc = bochnerTimeEncoding(
        new Float64Array([dt]),
        timeEncFreqs,
      );

      // Copy neighbor features
      const rowOffset = j * augDim;
      for (let d = 0; d < inDim; d++) {
        augFeatures[rowOffset + d] = neigh.features[d]!;
      }
      // Append time encoding
      for (let d = 0; d < timeDim; d++) {
        augFeatures[rowOffset + inDim + d] = timeEnc[d]!;
      }
    }

    // ---- Multi-head attention ----
    // Q = W_Q * h_i → (totalHeadDim)
    const qFull = matVecMul(W_Q, hI, totalHeadDim, inDim);

    // For each head, compute attention scores and weighted values
    const headOutputs = new Float64Array(heads * headDim);

    for (let h = 0; h < heads; h++) {
      const headOffset = h * headDim;

      // Extract Q for this head
      const qH = qFull.slice(headOffset, headOffset + headDim);

      // Compute K and V for all neighbors, extract this head's portion
      const scores = new Float64Array(numNeigh);
      const vVectors = new Float64Array(numNeigh * headDim);

      for (let j = 0; j < numNeigh; j++) {
        const augJ = augFeatures.slice(j * augDim, (j + 1) * augDim);

        // K_j = W_K * augJ → (totalHeadDim), take head slice
        const kFull = matVecMul(W_K, augJ, totalHeadDim, augDim);
        const kH = kFull.slice(headOffset, headOffset + headDim);

        // V_j = W_V * augJ → (totalHeadDim), take head slice
        const vFull = matVecMul(W_V, augJ, totalHeadDim, augDim);
        const vH = vFull.slice(headOffset, headOffset + headDim);

        // Store V for this head
        for (let d = 0; d < headDim; d++) {
          vVectors[j * headDim + d] = vH[d]!;
        }

        // Attention score: Q^T K / sqrt(d_head)
        scores[j] = dot(qH, kH) / Math.sqrt(headDim);
      }

      // Softmax over neighbor scores
      const attnWeights = softmax(scores, numNeigh);

      // Weighted sum of values
      for (let j = 0; j < numNeigh; j++) {
        const w = attnWeights[j]!;
        for (let d = 0; d < headDim; d++) {
          headOutputs[headOffset + d] = headOutputs[headOffset + d]! + w * vVectors[j * headDim + d]!;
        }
      }
    }

    // ---- Average across heads ----
    for (let d = 0; d < outDim; d++) {
      let sum = 0;
      for (let h = 0; h < heads; h++) {
        sum += headOutputs[h * headDim + d]!;
      }
      output[i * outDim + d] = sum / heads;
    }
  }

  return output;
}
