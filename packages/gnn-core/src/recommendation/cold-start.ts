// ---------------------------------------------------------------------------
// @omni-twin/gnn-core — Cold-Start Strategies
// Content-based initialization + MAML-style meta-learning adaptation.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 1. contentBasedInit — k-NN initialization for new items
// ---------------------------------------------------------------------------

/**
 * Find k nearest existing items to a new item based on content features.
 *
 * Algorithm (brute-force k-NN by Euclidean distance):
 * 1. For each existing item, compute Euclidean distance to the new item.
 * 2. Maintain a sorted list of the k nearest items.
 * 3. Return the indices of the k nearest items.
 *
 * Use case: When a new item enters the catalog with no interaction history,
 * use its content features (category, price, description embedding, etc.)
 * to find similar existing items. The new item's collaborative embedding
 * can then be initialized as the mean of these neighbors' embeddings.
 *
 * @param newItemFeatures - Feature vector of the new item (length featureDim).
 * @param existingItems - Feature matrix of existing items, row-major (numItems x featureDim).
 * @param numItems - Number of existing items.
 * @param featureDim - Dimension of each item's feature vector.
 * @param k - Number of nearest neighbors to return.
 * @returns Uint32Array of k indices into the existing items array.
 */
export function contentBasedInit(
  newItemFeatures: Float64Array,
  existingItems: Float64Array,
  numItems: number,
  featureDim: number,
  k: number,
): Uint32Array {
  // Compute Euclidean distances from new item to each existing item
  const distances: { idx: number; dist: number }[] = [];

  for (let i = 0; i < numItems; i++) {
    let distSq = 0;
    const off = i * featureDim;
    for (let d = 0; d < featureDim; d++) {
      const diff = newItemFeatures[d]! - existingItems[off + d]!;
      distSq += diff * diff;
    }
    distances.push({ idx: i, dist: Math.sqrt(distSq) });
  }

  // Sort by distance ascending
  distances.sort((a, b) => a.dist - b.dist);

  // Return top-k indices
  const actualK = Math.min(k, numItems);
  const result = new Uint32Array(actualK);
  for (let i = 0; i < actualK; i++) {
    result[i] = distances[i]!.idx;
  }

  return result;
}

// ---------------------------------------------------------------------------
// 2. metaLearnerAdapt — MAML-style inner-loop adaptation
// ---------------------------------------------------------------------------

/**
 * MAML-style inner-loop adaptation for cold-start scenarios.
 *
 * Algorithm (Model-Agnostic Meta-Learning inner loop):
 * Uses a simple linear model: predict = X * W (no bias for simplicity).
 * Loss = MSE(predict, labels).
 *
 * For each step:
 * 1. Forward: predictions = X_support * W  (numSupport x outDim)
 * 2. Loss: MSE = (1/n) * SUM (predict_i - label_i)^2
 * 3. Gradient: dW = (2/n) * X^T * (predictions - labels)
 * 4. Update: W = W - lr * dW
 *
 * After `steps` gradient updates, return the adapted weight matrix.
 *
 * Use case: When a new user/item has very few interactions (the "support set"),
 * perform a few gradient steps starting from the meta-learned base weights
 * to quickly adapt to the new entity.
 *
 * @param baseWeights - Initial weight matrix (featureDim x outDim), row-major.
 *                      These are the meta-learned weights shared across tasks.
 * @param supportFeatures - Support set feature matrix (numSupport x featureDim), row-major.
 * @param supportLabels - Support set label matrix (numSupport x outDim), row-major.
 * @param numSupport - Number of support examples.
 * @param featureDim - Input feature dimension.
 * @param outDim - Output dimension.
 * @param lr - Inner-loop learning rate.
 * @param steps - Number of gradient descent steps.
 * @returns Adapted weight matrix (featureDim x outDim).
 */
export function metaLearnerAdapt(
  baseWeights: Float64Array,
  supportFeatures: Float64Array,
  supportLabels: Float64Array,
  numSupport: number,
  featureDim: number,
  outDim: number,
  lr: number,
  steps: number,
): Float64Array {
  // Clone base weights to avoid mutating the original
  const W = new Float64Array(baseWeights);

  for (let step = 0; step < steps; step++) {
    // Forward pass: predictions = X_support * W
    // X_support: (numSupport x featureDim), W: (featureDim x outDim)
    // predictions: (numSupport x outDim)
    const predictions = new Float64Array(numSupport * outDim);
    for (let i = 0; i < numSupport; i++) {
      for (let j = 0; j < outDim; j++) {
        let val = 0;
        for (let k = 0; k < featureDim; k++) {
          val += supportFeatures[i * featureDim + k]! * W[k * outDim + j]!;
        }
        predictions[i * outDim + j] = val;
      }
    }

    // Compute residuals: R = predictions - labels
    // R: (numSupport x outDim)
    const residuals = new Float64Array(numSupport * outDim);
    for (let i = 0; i < numSupport * outDim; i++) {
      residuals[i] = predictions[i]! - supportLabels[i]!;
    }

    // Gradient: dW = (2/n) * X^T * R
    // X^T: (featureDim x numSupport), R: (numSupport x outDim)
    // dW: (featureDim x outDim)
    const scale = 2.0 / numSupport;
    const grad = new Float64Array(featureDim * outDim);
    for (let k = 0; k < featureDim; k++) {
      for (let j = 0; j < outDim; j++) {
        let val = 0;
        for (let i = 0; i < numSupport; i++) {
          val += supportFeatures[i * featureDim + k]! * residuals[i * outDim + j]!;
        }
        grad[k * outDim + j] = scale * val;
      }
    }

    // SGD update: W = W - lr * dW
    for (let i = 0; i < W.length; i++) {
      W[i] = W[i]! - lr * grad[i]!;
    }
  }

  return W;
}
