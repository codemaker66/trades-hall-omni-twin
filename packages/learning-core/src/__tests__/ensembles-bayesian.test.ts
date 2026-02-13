// ---------------------------------------------------------------------------
// Tests: Tree Ensembles + Bayesian (DecisionTree, QuantileForest,
// GradientBoosting, SHAP, Gibbs Hierarchical, MC Dropout)
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';
import { createPRNG } from '../types.js';
import type { TreeNode, TreeLeaf } from '../types.js';
import { DecisionTree, isLeaf } from '../ensembles/decision-tree.js';
import { QuantileForest } from '../ensembles/quantile-forest.js';
import { GradientBoosting } from '../ensembles/gradient-boosting.js';
import { computeTreeSHAP, aggregateSHAP } from '../ensembles/shap.js';
import {
  gibbsSampleHierarchical,
  posteriorPredictive,
} from '../bayesian/hierarchical.js';
import { mcDropoutPredict, createDropoutMask } from '../bayesian/mc-dropout.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Generate a simple regression dataset: y = 2*x0 + 3*x1 + noise. */
function makeSimpleDataset(n: number, rng: () => number) {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x0 = rng() * 10 - 5;
    const x1 = rng() * 10 - 5;
    const noise = (rng() - 0.5) * 0.5;
    X.push([x0, x1]);
    y.push(2 * x0 + 3 * x1 + noise);
  }
  return { X, y };
}

/** Mean squared error. */
function mse(predicted: number[], actual: number[]): number {
  let sum = 0;
  for (let i = 0; i < predicted.length; i++) {
    const d = (predicted[i] ?? 0) - (actual[i] ?? 0);
    sum += d * d;
  }
  return sum / predicted.length;
}

// ===========================================================================
// DecisionTree
// ===========================================================================

describe('DecisionTree', () => {
  it('fits and predicts on simple linear data', () => {
    const rng = createPRNG(42);
    const { X, y } = makeSimpleDataset(100, rng);
    const tree = new DecisionTree();
    tree.fit(X, y, 10, 2, rng);

    const preds = tree.predict(X);
    expect(preds).toHaveLength(100);

    // Training MSE should be low (tree can overfit)
    const error = mse(preds, y);
    expect(error).toBeLessThan(5);
  });

  it('handles single-value leaf (all same targets)', () => {
    const rng = createPRNG(99);
    const X = [[1], [2], [3], [4], [5]];
    const y = [7, 7, 7, 7, 7]; // all same

    const tree = new DecisionTree();
    tree.fit(X, y, 5, 1, rng);

    const pred = tree.predictSingle([3]);
    expect(pred).toBe(7);
  });

  it('respects maxDepth = 1 (stump)', () => {
    const rng = createPRNG(123);
    const X = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]];
    const y = [0, 0, 0, 0, 0, 10, 10, 10, 10, 10];

    const tree = new DecisionTree();
    tree.fit(X, y, 1, 1, rng);

    const root = tree.getRoot();
    // Depth-1 means root is an internal node with two leaf children
    expect(isLeaf(root)).toBe(false);
    if (!isLeaf(root)) {
      expect(isLeaf(root.left)).toBe(true);
      expect(isLeaf(root.right)).toBe(true);
    }
  });

  it('samples maxFeatures independently at each split', () => {
    const rngValues = [0.0, 0.9, 0.9, 0.9, 0.9];
    const rng = () => rngValues.shift() ?? 0.9;
    const X = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ];
    const y = [0, 1, 10, 11];

    const tree = new DecisionTree();
    tree.fit(X, y, 2, 1, rng, 1);

    const usedFeatures = new Set<number>();
    const visit = (node: TreeNode | TreeLeaf): void => {
      if (isLeaf(node)) return;
      usedFeatures.add(node.featureIndex);
      visit(node.left);
      visit(node.right);
    };
    visit(tree.getRoot());

    expect(usedFeatures.size).toBeGreaterThan(1);
  });

  it('predicts the mean of leaf values', () => {
    const rng = createPRNG(7);
    const X = [[1], [2], [3], [100], [101], [102]];
    const y = [1, 2, 3, 100, 101, 102];

    const tree = new DecisionTree();
    tree.fit(X, y, 1, 1, rng);

    // The stump should split low vs high; leaves should have mean of their group
    const predLow = tree.predictSingle([2]);
    const predHigh = tree.predictSingle([101]);
    expect(predLow).toBeCloseTo(2, 0);
    expect(predHigh).toBeCloseTo(101, 0);
  });

  it('isLeaf correctly distinguishes nodes and leaves', () => {
    expect(isLeaf({ values: [1, 2, 3] })).toBe(true);
    expect(
      isLeaf({
        featureIndex: 0,
        threshold: 0.5,
        left: { values: [1] },
        right: { values: [2] },
      }),
    ).toBe(false);
  });

  it('returns a default leaf for unfitted tree', () => {
    const tree = new DecisionTree();
    const leaf = tree.getLeaf([1, 2, 3]);
    expect(leaf.values).toEqual([0]);
  });
});

// ===========================================================================
// QuantileForest
// ===========================================================================

describe('QuantileForest', () => {
  it('fits and predict returns correct quantile count', () => {
    const rng = createPRNG(42);
    const { X, y } = makeSimpleDataset(80, rng);

    const forest = new QuantileForest({
      nEstimators: 5,
      maxDepth: 5,
      minSamplesLeaf: 2,
      maxFeatures: 'sqrt',
      seed: 42,
    });
    forest.fit(X, y);

    const quantiles = [0.1, 0.5, 0.9];
    const pred = forest.predict(X.slice(0, 10), quantiles);

    expect(pred.quantiles.size).toBe(3);
    for (const q of quantiles) {
      const vals = pred.quantiles.get(q);
      expect(vals).toBeDefined();
      expect(vals).toHaveLength(10);
    }
  });

  it('median is a reasonable prediction', () => {
    const rng = createPRNG(10);
    const { X, y } = makeSimpleDataset(100, rng);

    const forest = new QuantileForest({
      nEstimators: 10,
      maxDepth: 6,
      minSamplesLeaf: 2,
      maxFeatures: 'sqrt',
      seed: 10,
    });
    forest.fit(X, y);

    const pred = forest.predict(X, [0.5]);
    const medians = pred.median;
    const error = mse(medians, y);
    // Forest median should have reasonable training error
    expect(error).toBeLessThan(20);
  });

  it('predictIntervals produces valid intervals', () => {
    const rng = createPRNG(55);
    const { X, y } = makeSimpleDataset(60, rng);

    const forest = new QuantileForest({
      nEstimators: 8,
      maxDepth: 5,
      minSamplesLeaf: 2,
      maxFeatures: 2,
      seed: 55,
    });
    forest.fit(X, y);

    const intervals = forest.predictIntervals(X.slice(0, 5), 0.1);
    expect(intervals).toHaveLength(5);

    for (const interval of intervals) {
      expect(interval.lower).toBeLessThanOrEqual(interval.upper);
      expect(interval.confidenceLevel).toBeCloseTo(0.9);
    }
  });

  it('getTrees returns the correct number of trees', () => {
    const forest = new QuantileForest({
      nEstimators: 7,
      maxDepth: 3,
      minSamplesLeaf: 1,
      maxFeatures: 1,
      seed: 1,
    });
    const X = [[1], [2], [3], [4], [5]];
    const y = [1, 2, 3, 4, 5];
    forest.fit(X, y);

    expect(forest.getTrees()).toHaveLength(7);
  });
});

// ===========================================================================
// GradientBoosting
// ===========================================================================

describe('GradientBoosting', () => {
  it('fit reduces training error over iterations', () => {
    const rng = createPRNG(42);
    const { X, y } = makeSimpleDataset(80, rng);

    // Fewer trees: higher error
    const gbSmall = new GradientBoosting({
      nEstimators: 5,
      learningRate: 0.1,
      maxDepth: 3,
      subsample: 1.0,
      loss: 'squared',
      seed: 42,
    });
    gbSmall.fit(X, y);
    const predsSmall = gbSmall.predict(X);
    const mseSmall = mse(predsSmall, y);

    // More trees: lower error
    const gbLarge = new GradientBoosting({
      nEstimators: 50,
      learningRate: 0.1,
      maxDepth: 3,
      subsample: 1.0,
      loss: 'squared',
      seed: 42,
    });
    gbLarge.fit(X, y);
    const predsLarge = gbLarge.predict(X);
    const mseLarge = mse(predsLarge, y);

    expect(mseLarge).toBeLessThan(mseSmall);
  });

  it('predict returns reasonable values', () => {
    const rng = createPRNG(77);
    const { X, y } = makeSimpleDataset(100, rng);

    const gb = new GradientBoosting({
      nEstimators: 30,
      learningRate: 0.1,
      maxDepth: 4,
      subsample: 0.8,
      loss: 'squared',
      seed: 77,
    });
    gb.fit(X, y);

    const preds = gb.predict(X);
    expect(preds).toHaveLength(100);

    // Predictions should correlate with actual values
    const error = mse(preds, y);
    expect(error).toBeLessThan(10);
  });

  it('featureImportance sums to approximately 1', () => {
    const rng = createPRNG(42);
    const { X, y } = makeSimpleDataset(100, rng);

    const gb = new GradientBoosting({
      nEstimators: 20,
      learningRate: 0.1,
      maxDepth: 4,
      subsample: 1.0,
      loss: 'squared',
      seed: 42,
    });
    gb.fit(X, y);

    const importance = gb.featureImportance();
    expect(importance).toHaveLength(2);

    const sum = importance.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 1);
  });

  it('supports absolute loss', () => {
    const rng = createPRNG(42);
    const { X, y } = makeSimpleDataset(50, rng);

    const gb = new GradientBoosting({
      nEstimators: 20,
      learningRate: 0.1,
      maxDepth: 3,
      subsample: 1.0,
      loss: 'absolute',
      seed: 42,
    });
    gb.fit(X, y);

    const preds = gb.predict(X);
    expect(preds).toHaveLength(50);
    // Should produce finite values
    for (const p of preds) {
      expect(Number.isFinite(p)).toBe(true);
    }
  });

  it('supports huber loss', () => {
    const rng = createPRNG(42);
    const { X, y } = makeSimpleDataset(50, rng);

    const gb = new GradientBoosting({
      nEstimators: 20,
      learningRate: 0.1,
      maxDepth: 3,
      subsample: 1.0,
      loss: 'huber',
      seed: 42,
    });
    gb.fit(X, y);

    const preds = gb.predict(X);
    for (const p of preds) {
      expect(Number.isFinite(p)).toBe(true);
    }
  });

  it('supports quantile loss', () => {
    const rng = createPRNG(42);
    const { X, y } = makeSimpleDataset(50, rng);

    const gb = new GradientBoosting({
      nEstimators: 20,
      learningRate: 0.1,
      maxDepth: 3,
      subsample: 1.0,
      loss: 'quantile',
      quantile: 0.9,
      seed: 42,
    });
    gb.fit(X, y);

    const preds = gb.predict(X);
    for (const p of preds) {
      expect(Number.isFinite(p)).toBe(true);
    }
  });
});

// ===========================================================================
// SHAP
// ===========================================================================

describe('SHAP', () => {
  it('computeTreeSHAP returns values for all features', () => {
    const rng = createPRNG(42);
    const X = [[0, 0], [1, 0], [0, 1], [1, 1], [2, 2], [3, 3]];
    const y = [0, 2, 3, 5, 8, 12];

    const tree = new DecisionTree();
    tree.fit(X, y, 4, 1, rng);

    const featureNames = ['x0', 'x1'];
    const shapValues = computeTreeSHAP(tree.getRoot(), [1, 1], featureNames);

    expect(shapValues).toHaveLength(2);
    expect(shapValues[0]!.feature).toBe('x0');
    expect(shapValues[1]!.feature).toBe('x1');
    // Each value should be finite
    for (const sv of shapValues) {
      expect(Number.isFinite(sv.value)).toBe(true);
    }
  });

  it('SHAP direction is correct', () => {
    const rng = createPRNG(42);
    const X = [[0, 0], [1, 0], [0, 1], [1, 1]];
    const y = [0, 2, 3, 5];

    const tree = new DecisionTree();
    tree.fit(X, y, 4, 1, rng);

    const shapValues = computeTreeSHAP(tree.getRoot(), [1, 1], ['x0', 'x1']);
    for (const sv of shapValues) {
      if (sv.value >= 0) {
        expect(sv.direction).toBe('increases');
      } else {
        expect(sv.direction).toBe('decreases');
      }
    }
  });

  it('aggregateSHAP averages correctly across trees', () => {
    const featureNames = ['a', 'b'];

    const shapTree1 = [
      { feature: 'a', value: 2.0, direction: 'increases' as const },
      { feature: 'b', value: -1.0, direction: 'decreases' as const },
    ];
    const shapTree2 = [
      { feature: 'a', value: 4.0, direction: 'increases' as const },
      { feature: 'b', value: 1.0, direction: 'increases' as const },
    ];

    const aggregated = aggregateSHAP([shapTree1, shapTree2], featureNames);
    expect(aggregated).toHaveLength(2);

    // Mean of [2.0, 4.0] = 3.0
    expect(aggregated[0]!.value).toBeCloseTo(3.0);
    expect(aggregated[0]!.feature).toBe('a');

    // Mean of [-1.0, 1.0] = 0.0
    expect(aggregated[1]!.value).toBeCloseTo(0.0);
    expect(aggregated[1]!.feature).toBe('b');
  });

  it('aggregateSHAP returns zeros for empty input', () => {
    const result = aggregateSHAP([], ['a', 'b']);
    expect(result).toHaveLength(2);
    expect(result[0]!.value).toBe(0);
    expect(result[1]!.value).toBe(0);
  });
});

// ===========================================================================
// Hierarchical Bayesian (Gibbs sampling)
// ===========================================================================

describe('gibbsSampleHierarchical', () => {
  it('returns a valid posterior object', () => {
    const rng = createPRNG(42);
    const groupIds = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    const X = groupIds.map(() => [1.0]);
    const y = [5, 6, 5, 10, 11, 10, 15, 16, 15];

    const posterior = gibbsSampleHierarchical(
      groupIds, X, y,
      { nGroups: 3, nSamples: 50, nChains: 1, tuneSamples: 20, targetAccept: 0.8 },
      rng,
    );

    expect(Number.isFinite(posterior.globalMean)).toBe(true);
    expect(Number.isFinite(posterior.globalStd)).toBe(true);
    expect(posterior.groupMeans).toHaveLength(3);
    expect(posterior.groupStds).toHaveLength(3);
    expect(Number.isFinite(posterior.observationNoise)).toBe(true);
    expect(posterior.samples.size).toBeGreaterThan(0);
  });

  it('group means lie between global prior and data mean', () => {
    const rng = createPRNG(123);
    const groupIds = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    const X = groupIds.map(() => [0]); // No covariate effect
    const y = [2, 3, 2, 3, 2, 8, 9, 8, 9, 8];

    const posterior = gibbsSampleHierarchical(
      groupIds, X, y,
      { nGroups: 2, nSamples: 200, nChains: 1, tuneSamples: 100, targetAccept: 0.8 },
      rng,
    );

    // Group 0 data mean ~ 2.4, Group 1 data mean ~ 8.4
    // Group means should be shrunk toward global mean but still differentiated
    expect(posterior.groupMeans[0]!).toBeLessThan(posterior.groupMeans[1]!);
    // Group 0 should be roughly near its data
    expect(posterior.groupMeans[0]!).toBeGreaterThan(0);
    expect(posterior.groupMeans[0]!).toBeLessThan(7);
    // Group 1 should be roughly near its data
    expect(posterior.groupMeans[1]!).toBeGreaterThan(4);
    expect(posterior.groupMeans[1]!).toBeLessThan(12);
  });

  it('posterior predictive has reasonable uncertainty', () => {
    const rng = createPRNG(999);
    const groupIds = [0, 0, 0, 0, 1, 1, 1, 1];
    const X = groupIds.map(() => [1]);
    const y = [10, 11, 10, 11, 20, 21, 20, 21];

    const posterior = gibbsSampleHierarchical(
      groupIds, X, y,
      { nGroups: 2, nSamples: 100, nChains: 1, tuneSamples: 50, targetAccept: 0.8 },
      rng,
    );

    const pred = posteriorPredictive(posterior, 0, [1], rng);
    expect(Number.isFinite(pred.mean)).toBe(true);
    expect(pred.std).toBeGreaterThan(0);
  });

  it('stores correct number of posterior samples', () => {
    const rng = createPRNG(42);
    const groupIds = [0, 0, 1, 1];
    const X = [[1], [1], [1], [1]];
    const y = [1, 2, 3, 4];

    const posterior = gibbsSampleHierarchical(
      groupIds, X, y,
      { nGroups: 2, nSamples: 30, nChains: 1, tuneSamples: 10, targetAccept: 0.8 },
      rng,
    );

    const muGlobalSamples = posterior.samples.get('mu_global');
    expect(muGlobalSamples).toBeDefined();
    expect(muGlobalSamples!).toHaveLength(30);
  });
});

// ===========================================================================
// MC Dropout
// ===========================================================================

describe('mcDropoutPredict', () => {
  // Tiny 2-layer network: input=2, hidden=4, output=1
  const W1 = [
    // 4 x 2 = 8 values (hidden x input)
    0.5, -0.3,
    0.2, 0.7,
    -0.4, 0.6,
    0.1, -0.5,
  ];
  const W2 = [
    // 1 x 4 = 4 values (output x hidden)
    0.3, -0.2, 0.5, 0.1,
  ];
  const b1 = [0.1, -0.1, 0.2, 0.0];
  const b2 = [0.05];

  it('epistemic std > 0 with dropout', () => {
    const rng = createPRNG(42);
    const result = mcDropoutPredict(
      [W1, W2], [b1, b2], [1.0, 0.5], 0.5, 50, rng,
    );

    expect(result.mean).toHaveLength(1);
    expect(result.epistemicStd).toHaveLength(1);
    expect(result.epistemicStd[0]!).toBeGreaterThan(0);
  });

  it('multiple passes produce different results (via samples)', () => {
    const rng = createPRNG(42);
    const result = mcDropoutPredict(
      [W1, W2], [b1, b2], [1.0, 0.5], 0.5, 20, rng,
    );

    expect(result.samples).toHaveLength(20);
    // With dropout=0.5, samples should not all be identical
    const uniqueOutputs = new Set(result.samples.map((s) => s[0]));
    expect(uniqueOutputs.size).toBeGreaterThan(1);
  });

  it('mean is average of pass outputs', () => {
    const rng = createPRNG(42);
    const nPasses = 30;
    const result = mcDropoutPredict(
      [W1, W2], [b1, b2], [1.0, 0.5], 0.3, nPasses, rng,
    );

    // Manually compute mean from samples
    let manualSum = 0;
    for (const sample of result.samples) {
      manualSum += sample[0] ?? 0;
    }
    const manualMean = manualSum / nPasses;

    expect(result.mean[0]!).toBeCloseTo(manualMean, 8);
  });

  it('totalStd accounts for both epistemic and aleatoric', () => {
    const rng = createPRNG(42);
    const result = mcDropoutPredict(
      [W1, W2], [b1, b2], [1.0, 0.5], 0.5, 50, rng,
    );

    // total^2 = epistemic^2 + aleatoric^2
    const total2 =
      result.epistemicStd[0]! ** 2 + result.aleatoricStd[0]! ** 2;
    expect(result.totalStd[0]!).toBeCloseTo(Math.sqrt(total2), 6);
  });

  it('zero dropout gives zero epistemic uncertainty', () => {
    const rng = createPRNG(42);
    const result = mcDropoutPredict(
      [W1, W2], [b1, b2], [1.0, 0.5], 0.0, 10, rng,
    );

    // With no dropout, every pass is identical => epistemic std = 0
    expect(result.epistemicStd[0]!).toBeCloseTo(0);
  });

  it('throws when nPasses is not positive', () => {
    const rng = createPRNG(42);
    expect(() =>
      mcDropoutPredict([W1, W2], [b1, b2], [1.0, 0.5], 0.2, 0, rng),
    ).toThrow('nPasses must be a positive integer');
  });
});

describe('createDropoutMask', () => {
  it('returns correct length', () => {
    const rng = createPRNG(42);
    const mask = createDropoutMask(10, 0.5, rng);
    expect(mask).toHaveLength(10);
  });

  it('correct proportion of zeros (approximately)', () => {
    const rng = createPRNG(42);
    const size = 10000;
    const rate = 0.3;
    const mask = createDropoutMask(size, rate, rng);

    const zeroCount = mask.filter((v) => v === 0).length;
    const zeroProportion = zeroCount / size;

    // Should be approximately 0.3 (within tolerance)
    expect(zeroProportion).toBeGreaterThan(0.25);
    expect(zeroProportion).toBeLessThan(0.35);
  });

  it('all ones when rate is 0', () => {
    const rng = createPRNG(42);
    const mask = createDropoutMask(20, 0.0, rng);
    expect(mask.every((v) => v === 1)).toBe(true);
  });

  it('all zeros when rate is 1', () => {
    const rng = createPRNG(42);
    const mask = createDropoutMask(20, 1.0, rng);
    expect(mask.every((v) => v === 0)).toBe(true);
  });
});
