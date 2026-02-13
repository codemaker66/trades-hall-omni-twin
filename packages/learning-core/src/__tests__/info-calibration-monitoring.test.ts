// ---------------------------------------------------------------------------
// Tests for Information Theory, Calibration, Fairness, and Monitoring modules
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';
import {
  entropy,
  mutualInformation,
  mrmrSelect,
} from '../information/mutual-information.js';
import {
  klDivergence,
  jsDivergence,
  jsdMetric,
} from '../information/divergence.js';
import { mdlScore, mdlSelect } from '../information/mdl.js';
import {
  fisherInformationMatrix,
  dOptimality,
} from '../information/fisher-oed.js';
import { plattFit, plattTransform } from '../calibration/platt.js';
import { isotonicFit, isotonicTransform } from '../calibration/isotonic.js';
import {
  temperatureFit,
  temperatureTransform,
} from '../calibration/temperature.js';
import {
  evaluateCalibration,
  multiCalibrate,
} from '../calibration/multi-calibration.js';
import {
  demographicParity,
  equalizedOdds,
  disparateImpact,
} from '../fairness/metrics.js';
import { exponentiatedGradient } from '../fairness/debiasing.js';
import { CoverageTracker } from '../monitoring/coverage-tracker.js';
import {
  computePIT,
  kolmogorovSmirnovTest,
} from '../monitoring/calibration-monitor.js';
import { RetrainingTrigger } from '../monitoring/retraining-trigger.js';
import { createPRNG } from '../types.js';

// ---------------------------------------------------------------------------
// Entropy
// ---------------------------------------------------------------------------

describe('entropy', () => {
  it('returns 0 for a single class', () => {
    expect(entropy([1])).toBe(0);
  });

  it('returns 0 for a single non-zero count among zeros', () => {
    expect(entropy([0, 0, 5, 0])).toBe(0);
  });

  it('returns log(n) for a uniform distribution of n classes', () => {
    const n = 4;
    const counts = Array.from({ length: n }, () => 10);
    const result = entropy(counts);
    expect(result).toBeCloseTo(Math.log(n), 8);
  });

  it('returns log(2) for two equally-sized classes', () => {
    expect(entropy([50, 50])).toBeCloseTo(Math.log(2), 8);
  });

  it('is non-negative', () => {
    const rng = createPRNG(10);
    const counts = Array.from({ length: 5 }, () => Math.floor(rng() * 100));
    expect(entropy(counts)).toBeGreaterThanOrEqual(0);
  });
});

// ---------------------------------------------------------------------------
// Mutual Information
// ---------------------------------------------------------------------------

describe('mutualInformation', () => {
  it('is approximately 0 for independent variables', () => {
    const rng = createPRNG(42);
    const n = 1000;
    const x = Array.from({ length: n }, () => rng());
    const y = Array.from({ length: n }, () => rng());
    const mi = mutualInformation(x, y, 10);
    // MI should be close to 0 for independent data
    expect(mi).toBeLessThan(0.1);
  });

  it('is greater than 0 for correlated data', () => {
    const rng = createPRNG(7);
    const n = 500;
    const x = Array.from({ length: n }, () => rng() * 10);
    // y is a noisy copy of x
    const y = x.map((v) => v + (rng() - 0.5) * 0.1);
    const mi = mutualInformation(x, y, 10);
    expect(mi).toBeGreaterThan(0);
  });

  it('returns 0 for empty arrays', () => {
    expect(mutualInformation([], [], 10)).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// mRMR Feature Selection
// ---------------------------------------------------------------------------

describe('mrmrSelect', () => {
  it('selects exactly k features', () => {
    const rng = createPRNG(33);
    const n = 100;
    const nFeatures = 5;
    const k = 3;
    const X: number[][] = [];
    const y: number[] = [];

    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        row.push(rng());
      }
      X.push(row);
      y.push(rng());
    }

    const featureNames = Array.from({ length: nFeatures }, (_, j) => `f${j}`);
    const result = mrmrSelect(X, y, featureNames, k, 10);

    expect(result.selectedIndices).toHaveLength(k);
    expect(result.features).toHaveLength(k);
    expect(result.scores).toHaveLength(k);
  });

  it('returns empty for k = 0', () => {
    const result = mrmrSelect([[1]], [1], ['f0'], 0, 10);
    expect(result.selectedIndices).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// KL Divergence
// ---------------------------------------------------------------------------

describe('klDivergence', () => {
  it('returns 0 for identical distributions', () => {
    const p = [0.25, 0.25, 0.25, 0.25];
    const result = klDivergence(p, p);
    expect(result).toBeCloseTo(0, 6);
  });

  it('returns > 0 for different distributions', () => {
    const p = [0.9, 0.1];
    const q = [0.1, 0.9];
    const result = klDivergence(p, q);
    expect(result).toBeGreaterThan(0);
  });

  it('is non-negative', () => {
    const p = [0.3, 0.7];
    const q = [0.6, 0.4];
    expect(klDivergence(p, q)).toBeGreaterThanOrEqual(0);
  });
});

// ---------------------------------------------------------------------------
// JS Divergence
// ---------------------------------------------------------------------------

describe('jsDivergence', () => {
  it('is symmetric: JSD(P||Q) = JSD(Q||P)', () => {
    const p = [0.3, 0.7];
    const q = [0.6, 0.4];
    const jsd1 = jsDivergence(p, q);
    const jsd2 = jsDivergence(q, p);
    expect(jsd1).toBeCloseTo(jsd2, 10);
  });

  it('returns 0 for identical distributions', () => {
    const p = [0.5, 0.5];
    expect(jsDivergence(p, p)).toBeCloseTo(0, 6);
  });

  it('is bounded by log(2)', () => {
    const p = [1, 0];
    const q = [0, 1];
    const result = jsDivergence(p, q);
    expect(result).toBeLessThanOrEqual(Math.log(2) + 1e-10);
  });
});

describe('jsdMetric', () => {
  it('is bounded by sqrt(ln(2))', () => {
    const p = [1, 0];
    const q = [0, 1];
    const result = jsdMetric(p, q);
    expect(result).toBeLessThanOrEqual(Math.sqrt(Math.log(2)) + 1e-6);
  });

  it('is 0 for identical distributions', () => {
    const p = [0.25, 0.25, 0.25, 0.25];
    expect(jsdMetric(p, p)).toBeCloseTo(0, 6);
  });
});

// ---------------------------------------------------------------------------
// MDL
// ---------------------------------------------------------------------------

describe('mdlScore', () => {
  it('penalizes more parameters', () => {
    const logLik = -50;
    const n = 100;
    const score1 = mdlScore(logLik, 2, n);
    const score2 = mdlScore(logLik, 10, n);
    // More params => higher MDL score (worse)
    expect(score2).toBeGreaterThan(score1);
  });

  it('returns Infinity for n <= 0', () => {
    expect(mdlScore(-50, 2, 0)).toBe(Infinity);
  });
});

describe('mdlSelect', () => {
  it('picks the simplest adequate model', () => {
    const n = 100;
    const models = [
      { name: 'simple', logLikelihood: -50, nParams: 2 },
      { name: 'complex', logLikelihood: -49, nParams: 20 },
      { name: 'middle', logLikelihood: -48, nParams: 10 },
    ];
    const result = mdlSelect(models, n);
    // The simple model should win because its small nParams more than
    // compensates for the slightly lower logLikelihood
    expect(result.modelIndex).toBe(0);
    expect(result.totalLength).toBeLessThan(Infinity);
    expect(result.modelComplexity).toBeGreaterThan(0);
    expect(result.dataFit).toBeGreaterThan(0);
  });

  it('returns -1 index for empty model list', () => {
    const result = mdlSelect([], 100);
    expect(result.modelIndex).toBe(-1);
    expect(result.totalLength).toBe(Infinity);
  });
});

// ---------------------------------------------------------------------------
// Fisher Information / D-Optimality
// ---------------------------------------------------------------------------

describe('fisherInformationMatrix', () => {
  it('returns a square matrix of correct dimension', () => {
    const jacobian = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 1, 0],
    ];
    const fim = fisherInformationMatrix(jacobian);
    expect(fim).toHaveLength(3);
    for (const row of fim) {
      expect(row).toHaveLength(3);
    }
  });

  it('returns empty for empty jacobian', () => {
    const fim = fisherInformationMatrix([]);
    expect(fim).toHaveLength(0);
  });
});

describe('dOptimality', () => {
  it('is > 0 for a full-rank Fisher matrix', () => {
    // Identity matrix: det = 1
    const jacobian = [
      [1, 0],
      [0, 1],
      [1, 1],
    ];
    const fim = fisherInformationMatrix(jacobian);
    const det = dOptimality(fim);
    expect(det).toBeGreaterThan(0);
  });

  it('is 0 for a rank-deficient matrix', () => {
    // All rows are identical -> rank 1 for a 2x2 FIM
    const jacobian = [
      [1, 2],
      [1, 2],
      [1, 2],
    ];
    const fim = fisherInformationMatrix(jacobian);
    const det = dOptimality(fim);
    expect(det).toBeCloseTo(0, 6);
  });
});

// ---------------------------------------------------------------------------
// Platt Scaling
// ---------------------------------------------------------------------------

describe('plattFit / plattTransform', () => {
  it('plattFit returns params with finite a and b', () => {
    const predictions = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0];
    const labels = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1];
    const params = plattFit(predictions, labels);
    expect(Number.isFinite(params.a)).toBe(true);
    expect(Number.isFinite(params.b)).toBe(true);
  });

  it('plattTransform outputs values in [0, 1]', () => {
    const params = plattFit(
      [0.1, 0.4, 0.6, 0.9],
      [0, 0, 1, 1],
    );
    const scores = [-2, -1, 0, 1, 2, 5, -5];
    const calibrated = plattTransform(scores, params);
    for (const p of calibrated) {
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });
});

// ---------------------------------------------------------------------------
// Isotonic Regression
// ---------------------------------------------------------------------------

describe('isotonicFit / isotonicTransform', () => {
  it('isotonicFit returns monotonic breakpoints', () => {
    const predictions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    const labels = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1];
    const params = isotonicFit(predictions, labels);
    expect(params.xs.length).toBeGreaterThan(0);
    expect(params.ys.length).toBeGreaterThan(0);

    // ys should be monotonically non-decreasing
    for (let i = 1; i < params.ys.length; i++) {
      expect(params.ys[i]!).toBeGreaterThanOrEqual(params.ys[i - 1]! - 1e-10);
    }
  });

  it('isotonicTransform outputs in [0, 1]', () => {
    const params = isotonicFit(
      [0.1, 0.3, 0.5, 0.7, 0.9],
      [0, 0, 1, 1, 1],
    );
    const scores = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    const calibrated = isotonicTransform(scores, params);
    for (const p of calibrated) {
      expect(p).toBeGreaterThanOrEqual(-1e-10);
      expect(p).toBeLessThanOrEqual(1 + 1e-10);
    }
  });
});

// ---------------------------------------------------------------------------
// Temperature Scaling
// ---------------------------------------------------------------------------

describe('temperatureFit / temperatureTransform', () => {
  it('temperatureFit returns T > 0', () => {
    const logits = [-2, -1, 0, 1, 2, 3];
    const labels = [0, 0, 0, 1, 1, 1];
    const params = temperatureFit(logits, labels);
    expect(params.temperature).toBeGreaterThan(0);
  });

  it('temperatureTransform outputs in [0, 1]', () => {
    const params = temperatureFit(
      [-3, -1, 0, 1, 3],
      [0, 0, 0, 1, 1],
    );
    const logits = [-10, -5, -1, 0, 1, 5, 10];
    const calibrated = temperatureTransform(logits, params);
    for (const p of calibrated) {
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });

  it('temperatureFit returns T=1 for empty input', () => {
    const params = temperatureFit([], []);
    expect(params.temperature).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Calibration Evaluation
// ---------------------------------------------------------------------------

describe('evaluateCalibration', () => {
  it('ECE is in [0, 1]', () => {
    const rng = createPRNG(44);
    const n = 100;
    const predictions = Array.from({ length: n }, () => rng());
    const labels = Array.from({ length: n }, () => (rng() > 0.5 ? 1 : 0));
    const result = evaluateCalibration(predictions, labels, 10);
    expect(result.ece).toBeGreaterThanOrEqual(0);
    expect(result.ece).toBeLessThanOrEqual(1);
  });

  it('has correct number of bins', () => {
    const nBins = 5;
    const predictions = [0.1, 0.3, 0.5, 0.7, 0.9];
    const labels = [0, 0, 1, 1, 1];
    const result = evaluateCalibration(predictions, labels, nBins);
    expect(result.bins).toHaveLength(nBins);
  });

  it('Brier score is non-negative', () => {
    const predictions = [0.9, 0.1, 0.8, 0.2];
    const labels = [1, 0, 1, 0];
    const result = evaluateCalibration(predictions, labels, 10);
    expect(result.brier).toBeGreaterThanOrEqual(0);
  });
});

// ---------------------------------------------------------------------------
// Multi-Calibration
// ---------------------------------------------------------------------------

describe('multiCalibrate', () => {
  it('outputs values in [0, 1]', () => {
    const predictions = [0.3, 0.6, 0.2, 0.8, 0.5, 0.7, 0.1, 0.9];
    const labels = [0, 1, 0, 1, 0, 1, 0, 1];
    const subgroupMasks = [
      [true, true, true, true, false, false, false, false],
      [false, false, false, false, true, true, true, true],
    ];

    const adjusted = multiCalibrate(predictions, labels, subgroupMasks, 0.05, 50);

    expect(adjusted).toHaveLength(predictions.length);
    for (const p of adjusted) {
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });
});

// ---------------------------------------------------------------------------
// Fairness: Demographic Parity
// ---------------------------------------------------------------------------

describe('demographicParity', () => {
  it('returns 0 when both groups have the same prediction rate', () => {
    const predictions = [1, 0, 1, 0, 1, 0, 1, 0];
    const sensitiveAttr = [0, 0, 0, 0, 1, 1, 1, 1];
    // Group 0: 2/4 positive, Group 1: 2/4 positive
    const result = demographicParity(predictions, sensitiveAttr);
    expect(result).toBeCloseTo(0, 10);
  });

  it('returns > 0 when prediction rates differ between groups', () => {
    // Group 0: all positive, Group 1: all negative
    const predictions = [1, 1, 1, 1, 0, 0, 0, 0];
    const sensitiveAttr = [0, 0, 0, 0, 1, 1, 1, 1];
    const result = demographicParity(predictions, sensitiveAttr);
    expect(result).toBeGreaterThan(0);
    expect(result).toBeCloseTo(1, 10); // 1.0 - 0.0 = 1.0
  });
});

// ---------------------------------------------------------------------------
// Fairness: Equalized Odds
// ---------------------------------------------------------------------------

describe('equalizedOdds', () => {
  it('returns tprDiff and fprDiff', () => {
    const predictions = [1, 0, 1, 0, 1, 1, 0, 0];
    const labels = [1, 0, 1, 0, 1, 0, 1, 0];
    const sensitiveAttr = [0, 0, 0, 0, 1, 1, 1, 1];
    const result = equalizedOdds(predictions, labels, sensitiveAttr);
    expect(result).toHaveProperty('tprDiff');
    expect(result).toHaveProperty('fprDiff');
    expect(result.tprDiff).toBeGreaterThanOrEqual(0);
    expect(result.fprDiff).toBeGreaterThanOrEqual(0);
  });

  it('returns 0 for both when groups are identical', () => {
    // Same predictions, labels, and rates in both groups
    const predictions = [1, 0, 1, 0];
    const labels = [1, 0, 1, 0];
    const sensitiveAttr = [0, 0, 1, 1];
    const result = equalizedOdds(predictions, labels, sensitiveAttr);
    expect(result.tprDiff).toBeCloseTo(0, 10);
    expect(result.fprDiff).toBeCloseTo(0, 10);
  });
});

// ---------------------------------------------------------------------------
// Fairness: Disparate Impact
// ---------------------------------------------------------------------------

describe('disparateImpact', () => {
  it('returns 1 for equal groups', () => {
    const predictions = [1, 0, 1, 0, 1, 0, 1, 0];
    const sensitiveAttr = [0, 0, 0, 0, 1, 1, 1, 1];
    const result = disparateImpact(predictions, sensitiveAttr);
    expect(result).toBeCloseTo(1, 10);
  });

  it('returns < 1 for unequal rates', () => {
    // Group 0: 3/4 positive, Group 1: 1/4 positive
    const predictions = [1, 1, 1, 0, 1, 0, 0, 0];
    const sensitiveAttr = [0, 0, 0, 0, 1, 1, 1, 1];
    const result = disparateImpact(predictions, sensitiveAttr);
    expect(result).toBeLessThan(1);
    expect(result).toBeGreaterThanOrEqual(0);
  });
});

// ---------------------------------------------------------------------------
// Exponentiated Gradient Debiasing
// ---------------------------------------------------------------------------

describe('exponentiatedGradient', () => {
  it('outputs valid predictions in [0, 1]', () => {
    const rng = createPRNG(88);
    const n = 50;
    const predictions = Array.from({ length: n }, () => rng());
    const labels = Array.from({ length: n }, () => (rng() > 0.5 ? 1 : 0));
    const sensitiveAttr = Array.from({ length: n }, (_, i) => (i < n / 2 ? 0 : 1));

    const adjusted = exponentiatedGradient(
      predictions,
      labels,
      sensitiveAttr,
      'dp',
      0.05,
      50,
      0.5,
    );

    expect(adjusted).toHaveLength(n);
    for (const p of adjusted) {
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });

  it('returns empty for empty input', () => {
    const result = exponentiatedGradient([], [], [], 'dp', 0.05, 10, 0.1);
    expect(result).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// CoverageTracker
// ---------------------------------------------------------------------------

describe('CoverageTracker', () => {
  it('does not alert when coverage is at nominal', () => {
    const tracker = new CoverageTracker(0.9, 100, 0.05);
    // All observations are covered
    for (let i = 0; i < 50; i++) {
      const metrics = tracker.update(5, 0, 10); // yTrue=5 is within [0, 10]
      expect(metrics.alert).toBe(false);
    }
    const finalMetrics = tracker.getMetrics();
    expect(finalMetrics.empirical).toBeCloseTo(1.0, 10);
    expect(finalMetrics.alert).toBe(false);
  });

  it('alerts when coverage drops below nominal - threshold', () => {
    const tracker = new CoverageTracker(0.9, 20, 0.05);
    // None are covered: yTrue = 20 is outside [0, 10]
    for (let i = 0; i < 20; i++) {
      tracker.update(20, 0, 10);
    }
    const metrics = tracker.getMetrics();
    expect(metrics.empirical).toBe(0);
    // 0 < 0.9 - 0.05 = 0.85, so should alert
    expect(metrics.alert).toBe(true);
  });

  it('reports correct nominal value', () => {
    const tracker = new CoverageTracker(0.95, 50, 0.1);
    const metrics = tracker.getMetrics();
    expect(metrics.nominal).toBe(0.95);
    expect(metrics.windowSize).toBe(50);
  });
});

// ---------------------------------------------------------------------------
// PIT
// ---------------------------------------------------------------------------

describe('computePIT', () => {
  it('returns values clamped to [0, 1]', () => {
    const obs = [1, 2, 3, 4, 5];
    const cdfValues = [0.1, 0.3, 0.5, 0.8, 1.2]; // 1.2 should be clamped to 1
    const pit = computePIT(obs, cdfValues);

    expect(pit).toHaveLength(5);
    for (const v of pit) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
    // Last value should be clamped to 1
    expect(pit[4]).toBe(1);
  });

  it('returns empty for empty input', () => {
    const pit = computePIT([], []);
    expect(pit).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Kolmogorov-Smirnov Test
// ---------------------------------------------------------------------------

describe('kolmogorovSmirnovTest', () => {
  it('returns a p-value in [0, 1]', () => {
    const rng = createPRNG(66);
    // Generate approximate uniform(0,1) samples
    const pitValues = Array.from({ length: 100 }, () => rng());
    const pValue = kolmogorovSmirnovTest(pitValues);
    expect(pValue).toBeGreaterThanOrEqual(0);
    expect(pValue).toBeLessThanOrEqual(1);
  });

  it('returns high p-value for uniform data', () => {
    // Perfectly spaced uniform data
    const n = 100;
    const pitValues = Array.from({ length: n }, (_, i) => (i + 0.5) / n);
    const pValue = kolmogorovSmirnovTest(pitValues);
    // Should not reject uniformity
    expect(pValue).toBeGreaterThan(0.05);
  });

  it('returns low p-value for clearly non-uniform data', () => {
    // All values clustered near 0
    const pitValues = Array.from({ length: 100 }, (_, i) => i * 0.001);
    const pValue = kolmogorovSmirnovTest(pitValues);
    expect(pValue).toBeLessThan(0.05);
  });

  it('returns 1 for empty input', () => {
    expect(kolmogorovSmirnovTest([])).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// RetrainingTrigger
// ---------------------------------------------------------------------------

describe('RetrainingTrigger', () => {
  it('does not trigger when metrics are good', () => {
    const trigger = new RetrainingTrigger(0.05, 0.1, 1000);
    const coverage = {
      nominal: 0.9,
      empirical: 0.88, // only 0.02 below nominal, within threshold of 0.05
      rolling: [],
      windowSize: 100,
      alert: false,
    };
    const result = trigger.evaluate(coverage, 0.05, 2000);
    expect(result.shouldRetrain).toBe(false);
    expect(result.reason).toBeNull();
  });

  it('triggers when coverage drops below threshold', () => {
    const trigger = new RetrainingTrigger(0.05, 0.1, 1000);
    const coverage = {
      nominal: 0.9,
      empirical: 0.80, // 0.10 below nominal, exceeds threshold of 0.05
      rolling: [],
      windowSize: 100,
      alert: true,
    };
    const result = trigger.evaluate(coverage, 0.05, 2000);
    expect(result.shouldRetrain).toBe(true);
    expect(result.reason).not.toBeNull();
    expect(result.coverageDrop).toBeCloseTo(0.10, 8);
  });

  it('triggers when drift score exceeds threshold', () => {
    const trigger = new RetrainingTrigger(0.05, 0.1, 1000);
    const coverage = {
      nominal: 0.9,
      empirical: 0.88,
      rolling: [],
      windowSize: 100,
      alert: false,
    };
    const result = trigger.evaluate(coverage, 0.5, 2000); // drift = 0.5 > 0.1
    expect(result.shouldRetrain).toBe(true);
    expect(result.reason).not.toBeNull();
    expect(result.driftScore).toBe(0.5);
  });

  it('does not trigger if min interval has not elapsed', () => {
    const trigger = new RetrainingTrigger(0.05, 0.1, 10000);
    // First trigger to set lastRetrainedAt
    const coverage = {
      nominal: 0.9,
      empirical: 0.80,
      rolling: [],
      windowSize: 100,
      alert: true,
    };
    trigger.evaluate(coverage, 0.5, 1000); // triggers and sets lastRetrainedAt = 1000

    // Immediately try again: only 500ms later (< 10000ms interval)
    const result2 = trigger.evaluate(coverage, 0.5, 1500);
    expect(result2.shouldRetrain).toBe(false);
  });
});
