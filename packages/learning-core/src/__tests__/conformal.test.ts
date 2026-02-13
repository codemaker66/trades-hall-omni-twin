// ---------------------------------------------------------------------------
// Tests for Conformal Prediction modules
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';
import {
  computeResiduals,
  conformalQuantile,
  splitConformalPredict,
} from '../conformal/split-conformal.js';
import { cqrScores, cqrPredict } from '../conformal/cqr.js';
import { createACIState, aciUpdate, aciGetAlpha } from '../conformal/aci.js';
import { enbpiPredict } from '../conformal/enbpi.js';
import { weightedQuantile, weightedConformalPredict } from '../conformal/weighted.js';
import { createPRNG } from '../types.js';

// ---------------------------------------------------------------------------
// computeResiduals
// ---------------------------------------------------------------------------

describe('computeResiduals', () => {
  it('returns correct absolute residuals for simple inputs', () => {
    const yTrue = [1, 2, 3, 4, 5];
    const yPred = [1.1, 1.8, 3.2, 3.5, 5.5];
    const residuals = computeResiduals(yTrue, yPred);
    expect(residuals).toHaveLength(5);
    expect(residuals[0]).toBeCloseTo(0.1, 10);
    expect(residuals[1]).toBeCloseTo(0.2, 10);
    expect(residuals[2]).toBeCloseTo(0.2, 10);
    expect(residuals[3]).toBeCloseTo(0.5, 10);
    expect(residuals[4]).toBeCloseTo(0.5, 10);
  });

  it('returns zeros when predictions are exact', () => {
    const y = [10, 20, 30];
    const residuals = computeResiduals(y, y);
    for (const r of residuals) {
      expect(r).toBe(0);
    }
  });

  it('handles mismatched lengths by using the minimum', () => {
    const yTrue = [1, 2, 3, 4];
    const yPred = [1.5, 2.5];
    const residuals = computeResiduals(yTrue, yPred);
    expect(residuals).toHaveLength(2);
  });

  it('handles empty arrays', () => {
    const residuals = computeResiduals([], []);
    expect(residuals).toHaveLength(0);
  });

  it('handles negative values correctly (absolute difference)', () => {
    const yTrue = [-5, -10];
    const yPred = [-3, -12];
    const residuals = computeResiduals(yTrue, yPred);
    expect(residuals[0]).toBeCloseTo(2, 10);
    expect(residuals[1]).toBeCloseTo(2, 10);
  });
});

// ---------------------------------------------------------------------------
// conformalQuantile
// ---------------------------------------------------------------------------

describe('conformalQuantile', () => {
  it('returns the correct quantile value for a simple sorted set', () => {
    // residuals = [1, 2, 3, 4, 5], alpha=0.1
    // level = ceil(6 * 0.9) / 5 = ceil(5.4) / 5 = 6/5 = 1.2
    // level >= 1 so returns the max = 5
    const residuals = [1, 2, 3, 4, 5];
    const q = conformalQuantile(residuals, 0.1);
    expect(q).toBe(5);
  });

  it('returns a value that increases as alpha decreases (wider coverage)', () => {
    const residuals = [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0];
    const qLoose = conformalQuantile(residuals, 0.5); // 50% coverage
    const qTight = conformalQuantile(residuals, 0.1); // 90% coverage
    expect(qTight).toBeGreaterThanOrEqual(qLoose);
  });

  it('returns 0 for empty residuals', () => {
    expect(conformalQuantile([], 0.1)).toBe(0);
  });

  it('handles single-element residuals', () => {
    // n=1, level = ceil(2*(1-0.1))/1 = ceil(1.8)/1 = 2/1 = 2 >= 1 -> returns max
    const q = conformalQuantile([3.5], 0.1);
    expect(q).toBe(3.5);
  });

  it('returns correct quantile for large alpha (narrow coverage)', () => {
    const residuals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const q = conformalQuantile(residuals, 0.9); // 10% coverage
    // level = ceil(11 * 0.1) / 10 = ceil(1.1) / 10 = 2/10 = 0.2
    // idx = min(ceil(0.2*10) - 1, 9) = min(1, 9) = 1
    expect(q).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// splitConformalPredict
// ---------------------------------------------------------------------------

describe('splitConformalPredict', () => {
  it('creates valid intervals containing point estimates', () => {
    const yPred = [10, 20, 30];
    const quantile = 2.5;
    const alpha = 0.1;
    const intervals = splitConformalPredict(yPred, quantile, alpha);

    expect(intervals).toHaveLength(3);
    for (let i = 0; i < intervals.length; i++) {
      const interval = intervals[i]!;
      const pred = yPred[i]!;
      expect(interval.lower).toBeCloseTo(pred - quantile, 10);
      expect(interval.upper).toBeCloseTo(pred + quantile, 10);
      expect(interval.lower).toBeLessThanOrEqual(pred);
      expect(interval.upper).toBeGreaterThanOrEqual(pred);
      expect(interval.confidenceLevel).toBeCloseTo(0.9, 10);
    }
  });

  it('produces symmetric intervals around predictions', () => {
    const yPred = [5.0];
    const quantile = 1.0;
    const intervals = splitConformalPredict(yPred, quantile, 0.1);
    const interval = intervals[0]!;
    expect(interval.upper - 5.0).toBeCloseTo(5.0 - interval.lower, 10);
  });

  it('returns empty array for empty predictions', () => {
    const intervals = splitConformalPredict([], 1.0, 0.1);
    expect(intervals).toHaveLength(0);
  });

  it('wider quantile produces wider intervals', () => {
    const yPred = [100];
    const narrow = splitConformalPredict(yPred, 1.0, 0.1);
    const wide = splitConformalPredict(yPred, 5.0, 0.1);
    const narrowWidth = narrow[0]!.upper - narrow[0]!.lower;
    const wideWidth = wide[0]!.upper - wide[0]!.lower;
    expect(wideWidth).toBeGreaterThan(narrowWidth);
  });
});

// ---------------------------------------------------------------------------
// CQR: cqrScores
// ---------------------------------------------------------------------------

describe('cqrScores', () => {
  it('computes correct conformity scores when y is inside [qLow, qHigh]', () => {
    // y = 5, qLow = 3, qHigh = 7 => max(3-5, 5-7) = max(-2, -2) = -2
    const scores = cqrScores([5], [3], [7]);
    expect(scores[0]).toBeCloseTo(-2, 10);
  });

  it('computes positive score when y is above qHigh', () => {
    // y = 10, qLow = 3, qHigh = 7 => max(3-10, 10-7) = max(-7, 3) = 3
    const scores = cqrScores([10], [3], [7]);
    expect(scores[0]).toBeCloseTo(3, 10);
  });

  it('computes positive score when y is below qLow', () => {
    // y = 1, qLow = 3, qHigh = 7 => max(3-1, 1-7) = max(2, -6) = 2
    const scores = cqrScores([1], [3], [7]);
    expect(scores[0]).toBeCloseTo(2, 10);
  });

  it('handles multiple data points', () => {
    const scores = cqrScores([5, 10, 1], [3, 3, 3], [7, 7, 7]);
    expect(scores).toHaveLength(3);
    expect(scores[0]).toBeCloseTo(-2, 10);
    expect(scores[1]).toBeCloseTo(3, 10);
    expect(scores[2]).toBeCloseTo(2, 10);
  });
});

// ---------------------------------------------------------------------------
// CQR: cqrPredict
// ---------------------------------------------------------------------------

describe('cqrPredict', () => {
  it('creates asymmetric intervals that preserve quantile structure', () => {
    // Different qLow and qHigh -> intervals are not symmetric around midpoint
    const qLow = [2, 5];
    const qHigh = [8, 12];
    const scores = [-1, 0, 1, 2]; // calibration scores
    const intervals = cqrPredict(qLow, qHigh, scores, 0.1);

    expect(intervals).toHaveLength(2);
    for (const interval of intervals) {
      expect(interval.upper).toBeGreaterThan(interval.lower);
      expect(interval.confidenceLevel).toBeCloseTo(0.9, 10);
    }
  });

  it('returns raw intervals when no calibration data is provided', () => {
    const qLow = [3];
    const qHigh = [7];
    const intervals = cqrPredict(qLow, qHigh, [], 0.1);
    expect(intervals[0]!.lower).toBeCloseTo(3, 10);
    expect(intervals[0]!.upper).toBeCloseTo(7, 10);
  });

  it('widens intervals when scores are positive (miscoverage)', () => {
    const qLow = [5];
    const qHigh = [10];
    const scores = [2, 3, 4, 5]; // all positive = miscoverage
    const intervals = cqrPredict(qLow, qHigh, scores, 0.1);
    // qHat should be large -> intervals much wider than [5, 10]
    expect(intervals[0]!.lower).toBeLessThan(5);
    expect(intervals[0]!.upper).toBeGreaterThan(10);
  });
});

// ---------------------------------------------------------------------------
// ACI: createACIState / aciUpdate / aciGetAlpha
// ---------------------------------------------------------------------------

describe('ACI (Adaptive Conformal Inference)', () => {
  it('createACIState initializes with correct values', () => {
    const state = createACIState(0.1, 0.005);
    expect(state.alphaTarget).toBe(0.1);
    expect(state.alphaT).toBe(0.1);
    expect(state.gamma).toBe(0.005);
    expect(state.coverageHistory).toHaveLength(0);
  });

  it('aciGetAlpha returns current alpha', () => {
    const state = createACIState(0.1, 0.005);
    expect(aciGetAlpha(state)).toBe(0.1);
  });

  it('miscoverage decreases alphaT (leading to wider future intervals)', () => {
    const state = createACIState(0.1, 0.01);
    // y=15 is outside [0, 10] -> miscoverage
    const updated = aciUpdate(state, 15, 0, 10);
    // errT = 1, newAlpha = 0.1 + 0.01 * (0.1 - 1) = 0.1 - 0.009 = 0.091
    expect(updated.alphaT).toBeLessThan(state.alphaT);
    expect(updated.alphaT).toBeCloseTo(0.091, 8);
  });

  it('coverage increases alphaT (leading to tighter future intervals)', () => {
    const state = createACIState(0.1, 0.01);
    // y=5 is inside [0, 10] -> coverage
    const updated = aciUpdate(state, 5, 0, 10);
    // errT = 0, newAlpha = 0.1 + 0.01 * (0.1 - 0) = 0.1 + 0.001 = 0.101
    expect(updated.alphaT).toBeGreaterThan(state.alphaT);
    expect(updated.alphaT).toBeCloseTo(0.101, 8);
  });

  it('alpha is clamped to [0.001, 0.999]', () => {
    // Force alpha near zero
    const state: ReturnType<typeof createACIState> = {
      alphaTarget: 0.01,
      alphaT: 0.002,
      gamma: 0.1,
      coverageHistory: [],
    };
    // miscoverage => alpha decreases by gamma*(alphaTarget-1) = 0.1*(0.01-1) = -0.099
    // 0.002 - 0.099 = -0.097 -> clamped to 0.001
    const updated = aciUpdate(state, 15, 0, 10);
    expect(updated.alphaT).toBe(0.001);
  });

  it('updates coverage history correctly', () => {
    let state = createACIState(0.1, 0.01);
    state = aciUpdate(state, 5, 0, 10);   // covered -> 1
    state = aciUpdate(state, 15, 0, 10);  // not covered -> 0
    state = aciUpdate(state, 8, 0, 10);   // covered -> 1
    expect(state.coverageHistory).toEqual([1, 0, 1]);
  });

  it('returns a new object (immutability)', () => {
    const state = createACIState(0.1, 0.01);
    const updated = aciUpdate(state, 5, 0, 10);
    expect(updated).not.toBe(state);
    expect(state.coverageHistory).toHaveLength(0); // original unchanged
  });
});

// ---------------------------------------------------------------------------
// EnbPI: enbpiPredict
// ---------------------------------------------------------------------------

describe('enbpiPredict', () => {
  it('creates valid intervals for simple bootstrap predictions', () => {
    // 3 bootstrap models, 5 training points
    const bootstrapPredictions = [
      [10, 20, 30, 40, 50],
      [11, 19, 31, 39, 51],
      [9, 21, 29, 41, 49],
    ];
    const yTrain = [10, 20, 30, 40, 50];
    const intervals = enbpiPredict(bootstrapPredictions, yTrain, 0.1);

    expect(intervals).toHaveLength(5);
    for (const interval of intervals) {
      expect(interval.upper).toBeGreaterThanOrEqual(interval.lower);
      expect(interval.confidenceLevel).toBeCloseTo(0.9, 10);
    }
  });

  it('returns empty array for empty inputs', () => {
    expect(enbpiPredict([], [1, 2, 3], 0.1)).toHaveLength(0);
    expect(enbpiPredict([[1, 2]], [], 0.1)).toHaveLength(0);
  });

  it('intervals contain the training data when predictions are accurate', () => {
    // Predictions exactly match training data -> residuals are 0
    const yTrain = [5, 10, 15];
    const bootstrapPredictions = [
      [5, 10, 15],
      [5, 10, 15],
    ];
    const intervals = enbpiPredict(bootstrapPredictions, yTrain, 0.1);

    for (let i = 0; i < yTrain.length; i++) {
      const y = yTrain[i]!;
      const interval = intervals[i]!;
      expect(interval.lower).toBeLessThanOrEqual(y);
      expect(interval.upper).toBeGreaterThanOrEqual(y);
    }
  });
});

// ---------------------------------------------------------------------------
// Weighted Conformal: weightedQuantile
// ---------------------------------------------------------------------------

describe('weightedQuantile', () => {
  it('with uniform weights matches approximately the regular quantile', () => {
    const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const uniformWeights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    const alpha = 0.1;

    const wq = weightedQuantile(values, uniformWeights, alpha);
    const regularQ = conformalQuantile(values, alpha);

    // They should be close (not necessarily identical due to the implicit test-point weight)
    // The weighted quantile adds w_{n+1}=1 to the denominator, slightly shifting the result.
    // Both should be in the high range of values for 90% coverage.
    expect(wq).toBeGreaterThanOrEqual(8);
    expect(regularQ).toBeGreaterThanOrEqual(8);
  });

  it('returns 0 for empty values', () => {
    expect(weightedQuantile([], [], 0.1)).toBe(0);
  });

  it('high weight on a single value biases quantile toward it', () => {
    const values = [1, 2, 3, 4, 5];
    // Very high weight on value 3 (index 2)
    const weights = [1, 1, 100, 1, 1];
    const q = weightedQuantile(values, weights, 0.5);
    // With weight 100 on value 3, cumulative weight passes threshold quickly at value 3
    expect(q).toBeLessThanOrEqual(3);
  });

  it('ignores negative weights', () => {
    const values = [1, 2, 3];
    const weights = [-1, 1, 1];
    const q = weightedQuantile(values, weights, 0.1);
    // Only values 2 and 3 are considered (weight of value 1 is negative)
    expect(q).toBeGreaterThanOrEqual(2);
  });
});

// ---------------------------------------------------------------------------
// Weighted Conformal: weightedConformalPredict
// ---------------------------------------------------------------------------

describe('weightedConformalPredict', () => {
  it('produces valid intervals with correct confidence level', () => {
    const yPred = [10, 20, 30];
    const residuals = [0.5, 1.0, 1.5, 2.0, 2.5];
    const weights = [1, 1, 1, 1, 1];
    const intervals = weightedConformalPredict(yPred, residuals, weights, 0.1);

    expect(intervals).toHaveLength(3);
    for (let i = 0; i < intervals.length; i++) {
      const interval = intervals[i]!;
      const pred = yPred[i]!;
      expect(interval.lower).toBeLessThan(pred);
      expect(interval.upper).toBeGreaterThan(pred);
      expect(interval.confidenceLevel).toBeCloseTo(0.9, 10);
    }
  });

  it('produces symmetric intervals around predictions', () => {
    const yPred = [50];
    const residuals = [1, 2, 3, 4, 5];
    const weights = [1, 1, 1, 1, 1];
    const intervals = weightedConformalPredict(yPred, residuals, weights, 0.1);
    const interval = intervals[0]!;
    expect(interval.upper - 50).toBeCloseTo(50 - interval.lower, 10);
  });
});

// ---------------------------------------------------------------------------
// Coverage test: empirical coverage on generated data
// ---------------------------------------------------------------------------

describe('empirical coverage', () => {
  it('split conformal achieves approximately (1 - alpha) coverage on generated data', () => {
    const rng = createPRNG(42);
    const n = 500;
    const nCal = 250;
    const alpha = 0.1;

    // Generate data: y = 2*x + noise
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < n; i++) {
      const x = rng() * 10;
      const noise = (rng() - 0.5) * 2; // Uniform noise in [-1, 1]
      xs.push(x);
      ys.push(2 * x + noise);
    }

    // Simple linear predictions: yPred = 2*x (no noise)
    const yPred = xs.map(x => 2 * x);

    // Split: first nCal for calibration, rest for test
    const calTrue = ys.slice(0, nCal);
    const calPred = yPred.slice(0, nCal);
    const testTrue = ys.slice(nCal);
    const testPred = yPred.slice(nCal);

    // Compute calibration residuals and quantile
    const residuals = computeResiduals(calTrue, calPred);
    const q = conformalQuantile(residuals, alpha);

    // Create prediction intervals for test set
    const intervals = splitConformalPredict(testPred, q, alpha);

    // Count coverage
    let covered = 0;
    for (let i = 0; i < testTrue.length; i++) {
      const y = testTrue[i]!;
      const interval = intervals[i]!;
      if (y >= interval.lower && y <= interval.upper) {
        covered++;
      }
    }

    const empiricalCoverage = covered / testTrue.length;
    // Should be approximately 1 - alpha = 0.9, with some tolerance
    expect(empiricalCoverage).toBeGreaterThanOrEqual(0.85);
    expect(empiricalCoverage).toBeLessThanOrEqual(1.0);
  });
});
