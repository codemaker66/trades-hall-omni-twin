// ---------------------------------------------------------------------------
// Tests for Drift Detection and Causal Inference modules
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';
import { ADWIN } from '../drift/adwin.js';
import { DDM } from '../drift/ddm.js';
import { PageHinkley } from '../drift/page-hinkley.js';
import { droObjective, droOptimize, wassersteinBall } from '../drift/dro.js';
import { computeFisherDiagonal, ewcLoss, ewcGradient } from '../drift/ewc.js';
import { dmlEstimate } from '../causal/dml.js';
import { ols, twoStageLeastSquares } from '../causal/iv.js';
import { tLearnerEstimate } from '../causal/uplift.js';
import {
  createSequentialTest,
  sequentialTestUpdate,
} from '../causal/sequential-testing.js';
import { syntheticControl } from '../causal/synthetic-control.js';
import { createPRNG } from '../types.js';

// ---------------------------------------------------------------------------
// ADWIN
// ---------------------------------------------------------------------------

describe('ADWIN', () => {
  it('reports no drift on a stationary sequence of the same value', () => {
    const adwin = new ADWIN(0.002);
    let driftSeen = false;
    for (let i = 0; i < 500; i++) {
      if (adwin.update(5.0)) {
        driftSeen = true;
      }
    }
    expect(driftSeen).toBe(false);
    expect(adwin.driftDetected).toBe(false);
  });

  it('detects drift when the mean shifts abruptly', () => {
    const adwin = new ADWIN(0.002);
    // Phase 1: stationary at 0
    for (let i = 0; i < 300; i++) {
      adwin.update(0);
    }
    // Phase 2: abrupt shift to 10
    let driftSeen = false;
    for (let i = 0; i < 300; i++) {
      if (adwin.update(10)) {
        driftSeen = true;
      }
    }
    expect(driftSeen).toBe(true);
  });

  it('getState returns correct structure', () => {
    const adwin = new ADWIN();
    adwin.update(1);
    adwin.update(2);
    const state = adwin.getState();
    expect(state).toHaveProperty('driftDetected');
    expect(state).toHaveProperty('warningDetected');
    expect(state).toHaveProperty('nObservations');
    expect(state.nObservations).toBeGreaterThanOrEqual(1);
  });

  it('reset clears all state', () => {
    const adwin = new ADWIN();
    for (let i = 0; i < 100; i++) adwin.update(i);
    adwin.reset();
    expect(adwin.windowWidth).toBe(0);
    expect(adwin.mean).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// DDM
// ---------------------------------------------------------------------------

describe('DDM', () => {
  it('reports no drift on a stable low error rate', () => {
    const rng = createPRNG(12);
    const ddm = new DDM({ minInstances: 30, warningLevel: 2, driftLevel: 3 });
    let driftSeen = false;
    // Feed a stable ~10% error rate so p+s stays consistent
    for (let i = 0; i < 200; i++) {
      const error = rng() < 0.1 ? 1 : 0;
      const state = ddm.update(error);
      if (state.driftDetected) {
        driftSeen = true;
      }
    }
    expect(driftSeen).toBe(false);
  });

  it('detects drift when error rate increases sharply', () => {
    const ddm = new DDM({ minInstances: 30, warningLevel: 2, driftLevel: 3 });
    // Phase 1: low error rate
    for (let i = 0; i < 100; i++) {
      ddm.update(0);
    }
    // Phase 2: high error rate
    let driftSeen = false;
    for (let i = 0; i < 200; i++) {
      const state = ddm.update(1); // all errors
      if (state.driftDetected) {
        driftSeen = true;
      }
    }
    expect(driftSeen).toBe(true);
  });

  it('getState returns nObservations = 0 after reset', () => {
    const ddm = new DDM();
    ddm.update(0);
    ddm.update(1);
    ddm.reset();
    const state = ddm.getState();
    expect(state.nObservations).toBe(0);
    expect(state.driftDetected).toBe(false);
    expect(state.warningDetected).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// PageHinkley
// ---------------------------------------------------------------------------

describe('PageHinkley', () => {
  it('stays stable on stationary data', () => {
    const ph = new PageHinkley({ delta: 0.005, lambda: 50, alpha: 0.9999 });
    let driftSeen = false;
    for (let i = 0; i < 500; i++) {
      const state = ph.update(5.0);
      if (state.driftDetected) driftSeen = true;
    }
    expect(driftSeen).toBe(false);
  });

  it('detects a downward mean shift', () => {
    // PageHinkley detects drift when M_T - m_T > lambda.
    // A downward shift causes the cumulative sum m_T to decrease while
    // M_T stays at the historical maximum, triggering detection.
    const ph = new PageHinkley({ delta: 0.005, lambda: 50, alpha: 0.9999 });
    // Phase 1: stationary at 10
    for (let i = 0; i < 300; i++) {
      ph.update(10);
    }
    // Phase 2: drop to 0
    let driftSeen = false;
    for (let i = 0; i < 300; i++) {
      const state = ph.update(0);
      if (state.driftDetected) driftSeen = true;
    }
    expect(driftSeen).toBe(true);
  });

  it('testStatistic is non-negative', () => {
    const ph = new PageHinkley();
    for (let i = 0; i < 50; i++) ph.update(i);
    expect(ph.testStatistic).toBeGreaterThanOrEqual(0);
  });
});

// ---------------------------------------------------------------------------
// DRO
// ---------------------------------------------------------------------------

describe('droObjective', () => {
  it('returns a finite value', () => {
    const losses = [0.1, 0.5, 0.3, 0.8, 0.2];
    const weights = [0.2, 0.2, 0.2, 0.2, 0.2];
    const result = droObjective(losses, weights, 0.1);
    expect(Number.isFinite(result)).toBe(true);
  });

  it('returns 0 for empty losses', () => {
    const result = droObjective([], [], 0.1);
    expect(result).toBe(0);
  });

  it('with epsilon=0 returns the weighted average', () => {
    const losses = [1, 2, 3];
    const weights = [0.5, 0.3, 0.2];
    const expected = 0.5 * 1 + 0.3 * 2 + 0.2 * 3;
    const result = droObjective(losses, weights, 0);
    expect(result).toBeCloseTo(expected, 8);
  });
});

describe('droOptimize', () => {
  it('converges: final loss <= initial loss', () => {
    const rng = createPRNG(42);
    // Simple quadratic loss per sample: (params[0] - target)^2
    const targets = [1, 2, 3, 4, 5];
    const lossFn = (params: Float64Array): number[] => {
      return targets.map((t) => {
        const diff = (params[0] ?? 0) - t;
        return diff * diff;
      });
    };

    const initParams = new Float64Array([10]); // Start far from mean
    const epsilon = 0.01;

    // Compute initial DRO loss
    const initLosses = lossFn(initParams);
    const uniformW = initLosses.map(() => 1 / initLosses.length);
    const initialObj = droObjective(initLosses, uniformW, epsilon);

    const optimized = droOptimize(lossFn, initParams, epsilon, 0.1, 100, rng);

    // Compute final DRO loss
    const finalLosses = lossFn(optimized);
    const finalObj = droObjective(finalLosses, uniformW, epsilon);

    expect(finalObj).toBeLessThanOrEqual(initialObj);
  });
});

describe('wassersteinBall', () => {
  it('returns bounds containing the empirical values', () => {
    const empirical = [1, 2, 3, 4, 5];
    const epsilon = 0.5;
    const { lower, upper } = wassersteinBall(empirical, epsilon);

    expect(lower).toHaveLength(5);
    expect(upper).toHaveLength(5);

    for (let i = 0; i < empirical.length; i++) {
      expect(lower[i]).toBeLessThanOrEqual(empirical[i]!);
      expect(upper[i]).toBeGreaterThanOrEqual(empirical[i]!);
    }
  });

  it('returns empty arrays for empty input', () => {
    const { lower, upper } = wassersteinBall([], 0.5);
    expect(lower).toHaveLength(0);
    expect(upper).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// EWC
// ---------------------------------------------------------------------------

describe('computeFisherDiagonal', () => {
  it('returns positive values for non-zero gradients', () => {
    const gradients = [
      [0.5, -0.3, 0.1],
      [0.2, 0.4, -0.6],
      [-0.1, 0.2, 0.3],
    ];
    const fisher = computeFisherDiagonal(gradients);
    expect(fisher).toHaveLength(3);
    for (const f of fisher) {
      expect(f).toBeGreaterThan(0);
    }
  });

  it('returns empty array for empty gradients', () => {
    const fisher = computeFisherDiagonal([]);
    expect(fisher).toHaveLength(0);
  });
});

describe('ewcLoss', () => {
  it('equals task loss alone when params equal old params', () => {
    const params = new Float64Array([1, 2, 3]);
    const oldParams = new Float64Array([1, 2, 3]);
    const fisher = new Float64Array([0.5, 0.5, 0.5]);
    const taskLoss = 2.5;
    const result = ewcLoss(params, oldParams, fisher, 1.0, taskLoss);
    expect(result).toBeCloseTo(taskLoss, 10);
  });

  it('is >= task loss when params differ from old params', () => {
    const params = new Float64Array([1, 2, 3]);
    const oldParams = new Float64Array([0, 1, 2]);
    const fisher = new Float64Array([0.5, 0.5, 0.5]);
    const taskLoss = 2.5;
    const result = ewcLoss(params, oldParams, fisher, 1.0, taskLoss);
    expect(result).toBeGreaterThanOrEqual(taskLoss);
  });
});

describe('ewcGradient', () => {
  it('has correct length matching params', () => {
    const d = 5;
    const params = new Float64Array(d).fill(1);
    const oldParams = new Float64Array(d).fill(0);
    const fisher = new Float64Array(d).fill(0.5);
    const taskGradient = new Float64Array(d).fill(0.1);
    const grad = ewcGradient(params, oldParams, fisher, 1.0, taskGradient);
    expect(grad).toHaveLength(d);
  });

  it('equals task gradient when params equal old params', () => {
    const d = 3;
    const params = new Float64Array([1, 2, 3]);
    const oldParams = new Float64Array([1, 2, 3]);
    const fisher = new Float64Array(d).fill(0.5);
    const taskGradient = new Float64Array([0.1, -0.2, 0.3]);
    const grad = ewcGradient(params, oldParams, fisher, 1.0, taskGradient);
    for (let i = 0; i < d; i++) {
      expect(grad[i]).toBeCloseTo(taskGradient[i]!, 10);
    }
  });
});

// ---------------------------------------------------------------------------
// DML
// ---------------------------------------------------------------------------

describe('dmlEstimate', () => {
  it('returns a finite ATE with valid CI using simple predictors', () => {
    const rng = createPRNG(123);
    const n = 100;
    const Y: number[] = [];
    const T: number[] = [];
    const W: number[][] = [];

    for (let i = 0; i < n; i++) {
      const w = rng() * 2 - 1;
      const t = rng() > 0.5 ? 1 : 0;
      const y = 2 * t + w + (rng() - 0.5) * 0.1;
      Y.push(y);
      T.push(t);
      W.push([w]);
    }

    // Simple mean-based predictors: return the mean of the target variable
    // for the test portion
    const predictY = (wAll: number[][]): number[] => {
      // Return zeros for simplicity (nuisance = mean subtracted)
      return wAll.map(() => 0);
    };
    const predictT = (wAll: number[][]): number[] => {
      return wAll.map(() => 0.5);
    };

    const result = dmlEstimate(Y, T, W, predictY, predictT, 5);

    expect(Number.isFinite(result.ate)).toBe(true);
    expect(Number.isFinite(result.ciLower)).toBe(true);
    expect(Number.isFinite(result.ciUpper)).toBe(true);
    expect(result.ciLower).toBeLessThan(result.ciUpper);
  });

  it('returns degenerate result for too-small data', () => {
    const result = dmlEstimate([1], [0], [[1]], () => [0], () => [0], 5);
    expect(result.ate).toBe(0);
    expect(result.pValue).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// OLS
// ---------------------------------------------------------------------------

describe('ols', () => {
  it('recovers correct coefficients for y = 2x + 1', () => {
    // y = 2*x + 1, design matrix includes intercept column
    const n = 50;
    const y: number[] = [];
    const X: number[][] = [];
    for (let i = 0; i < n; i++) {
      const x = i / 10;
      y.push(2 * x + 1);
      X.push([x, 1]); // [slope column, intercept column]
    }
    const result = ols(y, X);
    expect(result.coefficients).toHaveLength(2);
    // Slope should be ~2
    expect(result.coefficients[0]).toBeCloseTo(2, 4);
    // Intercept should be ~1
    expect(result.coefficients[1]).toBeCloseTo(1, 4);
  });

  it('returns high R-squared for perfect fit', () => {
    const n = 20;
    const y: number[] = [];
    const X: number[][] = [];
    for (let i = 0; i < n; i++) {
      const x = i;
      y.push(3 * x + 5);
      X.push([x, 1]);
    }
    const result = ols(y, X);
    expect(result.rSquared).toBeCloseTo(1, 4);
  });

  it('returns empty for empty input', () => {
    const result = ols([], []);
    expect(result.coefficients).toHaveLength(0);
    expect(result.residuals).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// 2SLS
// ---------------------------------------------------------------------------

describe('twoStageLeastSquares', () => {
  it('returns a result with positive first-stage F', () => {
    const rng = createPRNG(99);
    const n = 200;
    const Y: number[] = [];
    const T: number[] = [];
    const Z: number[][] = [];

    for (let i = 0; i < n; i++) {
      const z = rng() * 2 - 1; // instrument
      const e = (rng() - 0.5) * 0.5; // error
      const t = 0.8 * z + e; // endogenous treatment
      const y = 3 * t + (rng() - 0.5); // outcome
      Y.push(y);
      T.push(t);
      Z.push([z]);
    }

    const result = twoStageLeastSquares(Y, T, Z);

    expect(result.firstStageF).toBeGreaterThan(0);
    expect(Number.isFinite(result.coefficient)).toBe(true);
    expect(Number.isFinite(result.standardError)).toBe(true);
    expect(result.ciLower).toBeLessThan(result.ciUpper);
  });

  it('returns degenerate result for n < 3', () => {
    const result = twoStageLeastSquares([1, 2], [0, 1], [[1], [0]]);
    expect(result.coefficient).toBe(0);
    expect(result.firstStageF).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// T-Learner
// ---------------------------------------------------------------------------

describe('tLearnerEstimate', () => {
  it('returns per-unit uplift values and aggregate ATE', () => {
    const n = 50;
    const rng = createPRNG(77);
    const Y: number[] = [];
    const T: number[] = [];
    const X: number[][] = [];

    for (let i = 0; i < n; i++) {
      const x = rng();
      const t = i % 2; // alternating treatment
      const y = t === 1 ? x + 2 : x; // treatment effect = 2
      Y.push(y);
      T.push(t);
      X.push([x]);
    }

    // Simple predictors: control returns the feature, treatment returns feature + 2
    const predictControl = (xArr: number[][]): number[] =>
      xArr.map((row) => row[0] ?? 0);
    const predictTreatment = (xArr: number[][]): number[] =>
      xArr.map((row) => (row[0] ?? 0) + 2);

    const result = tLearnerEstimate(Y, T, X, predictControl, predictTreatment);

    expect(result.uplift).toHaveLength(n);
    // Each uplift should be ~2
    for (const u of result.uplift) {
      expect(u).toBeCloseTo(2, 4);
    }
    expect(result.ate).toBeCloseTo(2, 4);
  });

  it('returns empty for empty input', () => {
    const result = tLearnerEstimate(
      [],
      [],
      [],
      () => [],
      () => [],
    );
    expect(result.uplift).toHaveLength(0);
    expect(result.ate).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Sequential Testing
// ---------------------------------------------------------------------------

describe('createSequentialTest / sequentialTestUpdate', () => {
  it('initial state has 0 observations and is not rejected', () => {
    const state = createSequentialTest(0.05);
    expect(state.nObservations).toBe(0);
    expect(state.treatmentN).toBe(0);
    expect(state.controlN).toBe(0);
    expect(state.rejected).toBe(false);
    expect(state.confidenceSequence).toHaveLength(0);
  });

  it('n increases after updates', () => {
    let state = createSequentialTest(0.05);
    state = sequentialTestUpdate(state, true, 0.5);
    expect(state.nObservations).toBe(1);
    expect(state.controlN).toBe(1);
    state = sequentialTestUpdate(state, false, 0.6);
    expect(state.nObservations).toBe(2);
    expect(state.treatmentN).toBe(1);
  });

  it('does not mutate the original state', () => {
    const state = createSequentialTest(0.05);
    const updated = sequentialTestUpdate(state, true, 1.0);
    expect(state.nObservations).toBe(0);
    expect(updated.nObservations).toBe(1);
  });

  it('eventually rejects with a large treatment effect', () => {
    const rng = createPRNG(55);
    let state = createSequentialTest(0.05);

    // Feed many observations with a large effect: treatment ~ 10, control ~ 0
    for (let i = 0; i < 500; i++) {
      const isControl = rng() < 0.5;
      const value = isControl ? rng() * 0.1 : 10 + rng() * 0.1;
      state = sequentialTestUpdate(state, isControl, value);
    }

    expect(state.rejected).toBe(true);
    expect(state.nObservations).toBe(500);
  });
});

// ---------------------------------------------------------------------------
// Synthetic Control
// ---------------------------------------------------------------------------

describe('syntheticControl', () => {
  it('weights sum to 1 and are non-negative', () => {
    // Treated unit: 10 pre-periods, 5 post-periods
    const treatedPre = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const treatedPost = [15, 16, 17, 18, 19]; // Post-intervention jump

    // 3 donor units
    const donorsPre = [
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],   // Donor 0: same as treated
      [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],   // Donor 1: shifted by 1
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],     // Donor 2: zero
    ];
    const donorsPost = [
      [11, 12, 13, 14, 15],
      [12, 13, 14, 15, 16],
      [0, 0, 0, 0, 0],
    ];

    const result = syntheticControl(treatedPre, treatedPost, donorsPre, donorsPost);

    // Weights should sum to 1
    const weightSum = result.weights.reduce((a, b) => a + b, 0);
    expect(weightSum).toBeCloseTo(1, 5);

    // All weights should be non-negative
    for (const w of result.weights) {
      expect(w).toBeGreaterThanOrEqual(-1e-10);
    }
  });

  it('computes a finite post effect', () => {
    const treatedPre = [1, 2, 3, 4, 5];
    const treatedPost = [10, 11, 12];
    const donorsPre = [
      [1, 2, 3, 4, 5],
      [2, 3, 4, 5, 6],
    ];
    const donorsPost = [
      [6, 7, 8],
      [7, 8, 9],
    ];

    const result = syntheticControl(treatedPre, treatedPost, donorsPre, donorsPost);
    expect(Number.isFinite(result.postEffect)).toBe(true);
    expect(Number.isFinite(result.preEffect)).toBe(true);
  });

  it('returns empty for no donors', () => {
    const result = syntheticControl([1, 2, 3], [4, 5], [], []);
    expect(result.weights).toHaveLength(0);
    expect(result.postEffect).toBe(0);
  });

  it('includes placebo effects when donors exist', () => {
    const treatedPre = [1, 2, 3, 4, 5];
    const treatedPost = [10, 11, 12];
    const donorsPre = [
      [1, 2, 3, 4, 5],
      [2, 3, 4, 5, 6],
      [0, 1, 2, 3, 4],
    ];
    const donorsPost = [
      [6, 7, 8],
      [7, 8, 9],
      [5, 6, 7],
    ];

    const result = syntheticControl(treatedPre, treatedPost, donorsPre, donorsPost);
    expect(result.placeboEffects.length).toBeGreaterThan(0);
  });
});
