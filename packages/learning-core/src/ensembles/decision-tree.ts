// ---------------------------------------------------------------------------
// CART Decision Tree for Regression
// ---------------------------------------------------------------------------

import type { PRNG, TreeNode, TreeLeaf } from '../types.js';

/** Type guard: is this node a leaf? */
export function isLeaf(node: TreeNode | TreeLeaf): node is TreeLeaf {
  return 'values' in node;
}

/**
 * CART decision tree for regression.
 *
 * Builds the tree by exhaustive search over features and thresholds,
 * choosing the split that minimises mean squared error (MSE).
 * Leaves store all training target values that fell into them,
 * which enables quantile predictions downstream.
 */
export class DecisionTree {
  private root: TreeNode | TreeLeaf | null = null;

  // -----------------------------------------------------------------------
  // Public API
  // -----------------------------------------------------------------------

  /** Build the tree from training data. */
  fit(
    X: number[][],
    y: number[],
    maxDepth: number,
    minSamplesLeaf: number,
    rng: PRNG,
    maxFeatures?: number,
  ): void {
    const n = X.length;
    const indices = Array.from({ length: n }, (_, i) => i);
    const nFeatures = (X[0] ?? []).length;

    this.root = this.buildNode(
      X,
      y,
      indices,
      0,
      maxDepth,
      minSamplesLeaf,
      nFeatures,
      rng,
      maxFeatures,
    );
  }

  /** Predict for a batch of samples. */
  predict(X: number[][]): number[] {
    return X.map((x) => this.predictSingle(x));
  }

  /** Predict for a single sample (returns mean of leaf values). */
  predictSingle(x: number[]): number {
    const leaf = this.getLeaf(x);
    return mean(leaf.values);
  }

  /** Return the leaf node that x falls into. */
  getLeaf(x: number[]): TreeLeaf {
    if (this.root === null) {
      return { values: [0] };
    }
    let node: TreeNode | TreeLeaf = this.root;
    while (!isLeaf(node)) {
      const featureVal = x[node.featureIndex] ?? 0;
      node = featureVal <= node.threshold ? node.left : node.right;
    }
    return node;
  }

  /** Expose tree root (e.g. for SHAP). */
  getRoot(): TreeNode | TreeLeaf {
    return this.root ?? { values: [0] };
  }

  // -----------------------------------------------------------------------
  // Private — recursive tree building
  // -----------------------------------------------------------------------

  private buildNode(
    X: number[][],
    y: number[],
    indices: number[],
    depth: number,
    maxDepth: number,
    minSamplesLeaf: number,
    nFeatures: number,
    rng: PRNG,
    maxFeatures?: number,
  ): TreeNode | TreeLeaf {
    // Collect target values for this node
    const values = indices.map((i) => y[i]!);

    // Stop conditions: max depth, too few samples, or pure node
    if (
      depth >= maxDepth ||
      indices.length < 2 * minSamplesLeaf ||
      allEqual(values)
    ) {
      return { values };
    }

    // Search for the best split across the feature subset
    let bestGain = -Infinity;
    let bestFeature = -1;
    let bestThreshold = 0;
    let bestLeftIndices: number[] = [];
    let bestRightIndices: number[] = [];

    const parentMSE = mse(values);
    const featureSubset =
      maxFeatures !== undefined && maxFeatures < nFeatures
        ? sampleFeatureIndices(nFeatures, maxFeatures, rng)
        : Array.from({ length: nFeatures }, (_, i) => i);

    for (const fIdx of featureSubset) {
      // Gather unique sorted values for this feature among current indices
      const featureVals = indices.map((i) => (X[i]![fIdx] ?? 0));
      const sorted = [...new Set(featureVals)].sort((a, b) => a - b);

      // Try midpoints between consecutive unique values
      for (let t = 0; t < sorted.length - 1; t++) {
        const threshold = ((sorted[t] ?? 0) + (sorted[t + 1] ?? 0)) / 2;

        const leftIdx: number[] = [];
        const rightIdx: number[] = [];
        for (const i of indices) {
          if ((X[i]![fIdx] ?? 0) <= threshold) {
            leftIdx.push(i);
          } else {
            rightIdx.push(i);
          }
        }

        if (leftIdx.length < minSamplesLeaf || rightIdx.length < minSamplesLeaf) {
          continue;
        }

        const leftVals = leftIdx.map((i) => y[i]!);
        const rightVals = rightIdx.map((i) => y[i]!);
        const leftMSE = mse(leftVals);
        const rightMSE = mse(rightVals);

        // Weighted reduction in MSE
        const n = indices.length;
        const gain =
          parentMSE -
          (leftIdx.length / n) * leftMSE -
          (rightIdx.length / n) * rightMSE;

        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = fIdx;
          bestThreshold = threshold;
          bestLeftIndices = leftIdx;
          bestRightIndices = rightIdx;
        }
      }
    }

    // If no valid split found, create a leaf
    if (bestFeature === -1 || bestGain <= 0) {
      return { values };
    }

    // Recurse
    const left = this.buildNode(
      X,
      y,
      bestLeftIndices,
      depth + 1,
      maxDepth,
      minSamplesLeaf,
      nFeatures,
      rng,
      maxFeatures,
    );
    const right = this.buildNode(
      X,
      y,
      bestRightIndices,
      depth + 1,
      maxDepth,
      minSamplesLeaf,
      nFeatures,
      rng,
      maxFeatures,
    );

    return {
      featureIndex: bestFeature,
      threshold: bestThreshold,
      left,
      right,
    };
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i] ?? 0;
  }
  return sum / values.length;
}

function mse(values: number[]): number {
  if (values.length === 0) return 0;
  const m = mean(values);
  let sumSq = 0;
  for (let i = 0; i < values.length; i++) {
    const diff = (values[i] ?? 0) - m;
    sumSq += diff * diff;
  }
  return sumSq / values.length;
}

function allEqual(values: number[]): boolean {
  if (values.length <= 1) return true;
  const first = values[0] ?? 0;
  for (let i = 1; i < values.length; i++) {
    if ((values[i] ?? 0) !== first) return false;
  }
  return true;
}

/** Sample `k` unique feature indices from `[0, nFeatures)`. */
function sampleFeatureIndices(nFeatures: number, k: number, rng: PRNG): number[] {
  const all = Array.from({ length: nFeatures }, (_, i) => i);
  // Fisher-Yates partial shuffle
  const count = Math.min(k, nFeatures);
  for (let i = 0; i < count; i++) {
    const j = i + Math.floor(rng() * (nFeatures - i));
    const tmp = all[i]!;
    all[i] = all[j]!;
    all[j] = tmp;
  }
  return all.slice(0, count);
}
