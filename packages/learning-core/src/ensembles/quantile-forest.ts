// ---------------------------------------------------------------------------
// Quantile Regression Forest (Meinshausen 2006)
// ---------------------------------------------------------------------------

import type {
  PRNG,
  RandomForestConfig,
  QuantileForestPrediction,
  PredictionInterval,
} from '../types.js';
import { createPRNG } from '../types.js';
import { DecisionTree } from './decision-tree.js';

/**
 * Quantile Regression Forest.
 *
 * Each tree in the forest is a standard CART tree trained on a bootstrap
 * sample with random feature subsampling. Quantile predictions are
 * formed by collecting all training-target values from the leaves
 * that each test sample falls into across all trees, then computing
 * empirical quantiles of that aggregated set.
 *
 * Reference: Meinshausen, N. (2006). Quantile Regression Forests.
 * Journal of Machine Learning Research, 7, 983-999.
 */
export class QuantileForest {
  private readonly config: RandomForestConfig;
  private trees: DecisionTree[] = [];
  private rng: PRNG;

  constructor(config: RandomForestConfig) {
    this.config = config;
    this.rng = createPRNG(config.seed ?? 42);
  }

  // -----------------------------------------------------------------------
  // Training
  // -----------------------------------------------------------------------

  /** Fit the forest on training data X (n x d) and targets y (n). */
  fit(X: number[][], y: number[]): void {
    const n = X.length;
    const nFeatures = (X[0] ?? []).length;

    // Resolve maxFeatures
    let maxFeatures: number;
    if (this.config.maxFeatures === 'sqrt') {
      maxFeatures = Math.max(1, Math.round(Math.sqrt(nFeatures)));
    } else if (this.config.maxFeatures === 'log2') {
      maxFeatures = Math.max(1, Math.round(Math.log2(nFeatures)));
    } else {
      maxFeatures = this.config.maxFeatures;
    }

    this.trees = [];

    for (let t = 0; t < this.config.nEstimators; t++) {
      // Bootstrap sample (sample n indices with replacement)
      const bootX: number[][] = [];
      const bootY: number[] = [];
      for (let i = 0; i < n; i++) {
        const idx = Math.floor(this.rng() * n);
        bootX.push(X[idx]!);
        bootY.push(y[idx]!);
      }

      const tree = new DecisionTree();
      tree.fit(
        bootX,
        bootY,
        this.config.maxDepth,
        this.config.minSamplesLeaf,
        this.rng,
        maxFeatures,
      );
      this.trees.push(tree);
    }
  }

  // -----------------------------------------------------------------------
  // Prediction
  // -----------------------------------------------------------------------

  /**
   * Predict specified quantiles for each sample in X.
   *
   * For each test point x_i, we collect all training-target values from
   * the leaf node that x_i falls into across every tree, then compute
   * the requested empirical quantiles.
   */
  predict(X: number[][], quantiles: number[]): QuantileForestPrediction {
    const n = X.length;
    const result = new Map<number, number[]>();
    for (const q of quantiles) {
      result.set(q, new Array<number>(n).fill(0));
    }
    const medianArr = new Array<number>(n).fill(0);

    for (let i = 0; i < n; i++) {
      const xi = X[i]!;

      // Aggregate leaf values across all trees
      const allValues: number[] = [];
      for (const tree of this.trees) {
        const leaf = tree.getLeaf(xi);
        for (const v of leaf.values) {
          allValues.push(v);
        }
      }

      // Sort for quantile computation
      allValues.sort((a, b) => a - b);

      for (const q of quantiles) {
        const qArr = result.get(q)!;
        qArr[i] = empiricalQuantile(allValues, q);
      }
      medianArr[i] = empiricalQuantile(allValues, 0.5);
    }

    return { quantiles: result, median: medianArr };
  }

  /**
   * Shorthand for prediction intervals at a given confidence level.
   *
   * @param alpha  Miscoverage rate. E.g. alpha=0.1 gives 90% intervals.
   * @returns Array of `{ lower, upper, confidenceLevel }` for each sample.
   */
  predictIntervals(X: number[][], alpha: number): PredictionInterval[] {
    const qLow = alpha / 2;
    const qHigh = 1 - alpha / 2;
    const pred = this.predict(X, [qLow, 0.5, qHigh]);

    const lowerArr = pred.quantiles.get(qLow)!;
    const upperArr = pred.quantiles.get(qHigh)!;

    const intervals: PredictionInterval[] = [];
    for (let i = 0; i < X.length; i++) {
      intervals.push({
        lower: lowerArr[i] ?? 0,
        upper: upperArr[i] ?? 0,
        confidenceLevel: 1 - alpha,
      });
    }
    return intervals;
  }

  /** Expose individual trees (e.g. for SHAP aggregation). */
  getTrees(): DecisionTree[] {
    return this.trees;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Compute the empirical quantile of a **sorted** array using
 * linear interpolation (same as NumPy's default `method='linear'`).
 */
function empiricalQuantile(sorted: number[], q: number): number {
  const n = sorted.length;
  if (n === 0) return 0;
  if (n === 1) return sorted[0] ?? 0;

  const pos = q * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  const frac = pos - lo;

  if (lo === hi) return sorted[lo] ?? 0;
  return (sorted[lo] ?? 0) * (1 - frac) + (sorted[hi] ?? 0) * frac;
}
