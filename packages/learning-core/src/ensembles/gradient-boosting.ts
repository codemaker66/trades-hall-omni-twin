// ---------------------------------------------------------------------------
// Gradient Boosting for Regression
// ---------------------------------------------------------------------------

import type { GradientBoostConfig, PRNG, TreeNode, TreeLeaf } from '../types.js';
import { createPRNG } from '../types.js';
import { DecisionTree, isLeaf } from './decision-tree.js';

/**
 * Gradient Boosting Machine for regression.
 *
 * Iteratively fits shallow decision trees on the negative gradient
 * (pseudo-residuals) of the chosen loss function. Supports:
 *   - `squared`  (L2): gradient = y - F(x)
 *   - `absolute` (L1): gradient = sign(y - F(x))
 *   - `huber`:         hybrid L1/L2 (delta = 1.0)
 *   - `quantile`:      gradient = I(y > F(x)) - quantile
 *
 * Feature importance is computed by accumulating the weighted MSE
 * reduction at each split across all trees.
 */
export class GradientBoosting {
  private readonly config: GradientBoostConfig;
  private trees: DecisionTree[] = [];
  private treeWeights: number[] = [];
  private initialPrediction = 0;
  private rng: PRNG;
  private splitGains: number[] = []; // per-feature cumulative gain

  constructor(config: GradientBoostConfig) {
    this.config = config;
    this.rng = createPRNG(config.seed ?? 42);
  }

  // -----------------------------------------------------------------------
  // Training
  // -----------------------------------------------------------------------

  /** Fit the gradient boosting model. */
  fit(X: number[][], y: number[]): void {
    const n = X.length;
    const nFeatures = (X[0] ?? []).length;

    this.splitGains = new Array<number>(nFeatures).fill(0);

    // Initial prediction: for L2 it's the mean; for L1/quantile it's the median
    this.initialPrediction = this.computeInitialPrediction(y);

    // Current predictions for each training sample
    const F = new Array<number>(n).fill(this.initialPrediction);

    this.trees = [];
    this.treeWeights = [];

    for (let m = 0; m < this.config.nEstimators; m++) {
      // Compute pseudo-residuals (negative gradient)
      const residuals = new Array<number>(n);
      for (let i = 0; i < n; i++) {
        residuals[i] = this.negativeGradient(y[i]!, F[i]!);
      }

      // Subsample if configured
      let sampleX: number[][];
      let sampleR: number[];
      let sampleIndices: number[];

      if (this.config.subsample < 1.0) {
        const sampleSize = Math.max(1, Math.round(n * this.config.subsample));
        sampleIndices = [];
        sampleX = [];
        sampleR = [];
        for (let i = 0; i < sampleSize; i++) {
          const idx = Math.floor(this.rng() * n);
          sampleIndices.push(idx);
          sampleX.push(X[idx]!);
          sampleR.push(residuals[idx]!);
        }
      } else {
        sampleIndices = Array.from({ length: n }, (_, i) => i);
        sampleX = X;
        sampleR = residuals;
      }

      // Fit a shallow tree on pseudo-residuals
      const tree = new DecisionTree();
      tree.fit(sampleX, sampleR, this.config.maxDepth, 1, this.rng);
      this.trees.push(tree);

      const lr = this.config.learningRate;
      this.treeWeights.push(lr);

      // Update predictions
      for (let i = 0; i < n; i++) {
        const pred = tree.predictSingle(X[i]!);
        F[i] = (F[i] ?? 0) + lr * pred;
      }

      // Accumulate split gains from this tree
      this.accumulateGains(tree.getRoot(), n);
    }
  }

  // -----------------------------------------------------------------------
  // Prediction
  // -----------------------------------------------------------------------

  /** Predict for a batch of samples. */
  predict(X: number[][]): number[] {
    return X.map((x) => this.predictSingle(x));
  }

  /** Predict for a single sample. */
  private predictSingle(x: number[]): number {
    let pred = this.initialPrediction;
    for (let t = 0; t < this.trees.length; t++) {
      pred += (this.treeWeights[t] ?? 0) * this.trees[t]!.predictSingle(x);
    }
    return pred;
  }

  // -----------------------------------------------------------------------
  // Feature importance
  // -----------------------------------------------------------------------

  /**
   * Return normalised feature importance based on split gains.
   * The returned array has one entry per feature (same indexing as columns
   * of the training matrix X).
   */
  featureImportance(): number[] {
    const total = this.splitGains.reduce((a, b) => a + b, 0);
    if (total === 0) return this.splitGains.map(() => 0);
    return this.splitGains.map((g) => g / total);
  }

  // -----------------------------------------------------------------------
  // Private — loss functions
  // -----------------------------------------------------------------------

  /** Compute the negative gradient of the loss at a single point. */
  private negativeGradient(y: number, f: number): number {
    const r = y - f;
    switch (this.config.loss) {
      case 'squared':
        return r;
      case 'absolute':
        return r > 0 ? 1 : r < 0 ? -1 : 0;
      case 'huber': {
        const delta = 1.0;
        return Math.abs(r) <= delta ? r : delta * Math.sign(r);
      }
      case 'quantile': {
        const q = this.config.quantile ?? 0.5;
        return r >= 0 ? q : q - 1;
      }
    }
  }

  /** Compute the initial (constant) prediction for the training targets. */
  private computeInitialPrediction(y: number[]): number {
    switch (this.config.loss) {
      case 'squared':
        return arrayMean(y);
      case 'absolute':
      case 'quantile':
        return arrayMedian(y);
      case 'huber':
        return arrayMedian(y);
    }
  }

  /**
   * Walk the tree and accumulate MSE-reduction-weighted gains
   * for each split feature.
   */
  private accumulateGains(
    node: TreeNode | TreeLeaf,
    totalSamples: number,
  ): number {
    if (isLeaf(node)) {
      return node.values.length;
    }

    const leftCount = this.accumulateGains(node.left, totalSamples);
    const rightCount = this.accumulateGains(node.right, totalSamples);
    const nodeCount = leftCount + rightCount;

    // Compute gain: weighted MSE reduction at this split
    const leftValues = collectLeafValues(node.left);
    const rightValues = collectLeafValues(node.right);
    const allValues = [...leftValues, ...rightValues];

    const parentMSE = computeMSE(allValues);
    const leftMSE = computeMSE(leftValues);
    const rightMSE = computeMSE(rightValues);

    const gain =
      (nodeCount / totalSamples) *
      (parentMSE -
        (leftCount / nodeCount) * leftMSE -
        (rightCount / nodeCount) * rightMSE);

    if (node.featureIndex >= 0 && node.featureIndex < this.splitGains.length) {
      this.splitGains[node.featureIndex] =
        (this.splitGains[node.featureIndex] ?? 0) + Math.max(0, gain);
    }

    return nodeCount;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function arrayMean(arr: number[]): number {
  if (arr.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    sum += arr[i] ?? 0;
  }
  return sum / arr.length;
}

function arrayMedian(arr: number[]): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2;
  }
  return sorted[mid] ?? 0;
}

function computeMSE(values: number[]): number {
  if (values.length === 0) return 0;
  const m = arrayMean(values);
  let sumSq = 0;
  for (let i = 0; i < values.length; i++) {
    const d = (values[i] ?? 0) - m;
    sumSq += d * d;
  }
  return sumSq / values.length;
}

/** Recursively collect all leaf values under a node. */
function collectLeafValues(node: TreeNode | TreeLeaf): number[] {
  if (isLeaf(node)) {
    return node.values;
  }
  return [...collectLeafValues(node.left), ...collectLeafValues(node.right)];
}
