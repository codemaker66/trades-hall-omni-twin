// ---------------------------------------------------------------------------
// Tests for Gaussian Process (Kernels, GP Regression, Bayesian Optimization)
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';
import { createPRNG } from '../types.js';
import type { KernelConfig } from '../types.js';
import {
  rbfKernel,
  matern32Kernel,
  periodicKernel,
  linearKernel,
  computeKernelMatrix,
} from '../gp/kernel.js';
import { GPRegressor } from '../gp/gp-regression.js';
import {
  normalCDF,
  normalPDF,
  expectedImprovement,
  upperConfidenceBound,
  bayesianOptimize,
} from '../gp/bayesian-opt.js';

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

describe('rbfKernel', () => {
  it('returns variance for the same point (distance 0)', () => {
    const variance = 2.0;
    const val = rbfKernel(3.0, 3.0, 1.0, variance);
    expect(val).toBeCloseTo(variance, 10);
  });

  it('decreases with distance', () => {
    const near = rbfKernel(0, 0.1, 1.0, 1.0);
    const far = rbfKernel(0, 5.0, 1.0, 1.0);
    expect(near).toBeGreaterThan(far);
  });

  it('is symmetric', () => {
    const a = rbfKernel(1.0, 3.0, 1.0, 1.0);
    const b = rbfKernel(3.0, 1.0, 1.0, 1.0);
    expect(a).toBeCloseTo(b, 10);
  });

  it('is always positive', () => {
    const val = rbfKernel(-10, 10, 1.0, 1.0);
    expect(val).toBeGreaterThan(0);
  });
});

describe('matern32Kernel', () => {
  it('returns variance at distance 0', () => {
    const variance = 3.0;
    const val = matern32Kernel(5.0, 5.0, 1.0, variance);
    expect(val).toBeCloseTo(variance, 10);
  });

  it('decreases with distance', () => {
    const near = matern32Kernel(0, 0.5, 1.0, 1.0);
    const far = matern32Kernel(0, 5.0, 1.0, 1.0);
    expect(near).toBeGreaterThan(far);
  });
});

describe('periodicKernel', () => {
  it('has correct period: same value at x and x+period', () => {
    const period = 2.0;
    const lengthscale = 1.0;
    const variance = 1.0;
    const x1 = 1.0;
    const x2 = 3.0; // x2 = x1 + period

    // k(0, x1) should equal k(0, x2) since they differ by exactly one period
    const val1 = periodicKernel(0, x1, lengthscale, variance, period);
    const val2 = periodicKernel(0, x2, lengthscale, variance, period);
    expect(val1).toBeCloseTo(val2, 8);
  });

  it('returns variance at distance 0', () => {
    const variance = 2.5;
    const val = periodicKernel(4.0, 4.0, 1.0, variance, 1.0);
    expect(val).toBeCloseTo(variance, 10);
  });
});

describe('linearKernel', () => {
  it('returns variance * x1 * x2', () => {
    const val = linearKernel(3.0, 4.0, 2.0);
    expect(val).toBeCloseTo(24.0, 10);
  });

  it('returns 0 when one input is 0', () => {
    const val = linearKernel(0, 5.0, 1.0);
    expect(val).toBeCloseTo(0, 10);
  });

  it('is negative when inputs have opposite signs', () => {
    const val = linearKernel(-2.0, 3.0, 1.0);
    expect(val).toBeLessThan(0);
  });
});

describe('computeKernelMatrix', () => {
  it('is symmetric', () => {
    const xs = [0, 1, 2, 3];
    const config: KernelConfig = { type: 'rbf', lengthscale: 1.0, variance: 1.0 };
    const K = computeKernelMatrix(xs, config);

    for (let i = 0; i < xs.length; i++) {
      for (let j = 0; j < xs.length; j++) {
        const kij = K.data[i * K.cols + j]!;
        const kji = K.data[j * K.cols + i]!;
        expect(kij).toBeCloseTo(kji, 10);
      }
    }
  });

  it('has positive diagonal', () => {
    const xs = [-2, 0, 1, 5];
    const config: KernelConfig = { type: 'rbf', lengthscale: 1.0, variance: 1.0 };
    const K = computeKernelMatrix(xs, config);

    for (let i = 0; i < xs.length; i++) {
      const kii = K.data[i * K.cols + i]!;
      expect(kii).toBeGreaterThan(0);
    }
  });

  it('has correct dimensions', () => {
    const xs = [0, 1, 2];
    const config: KernelConfig = { type: 'rbf', lengthscale: 1.0, variance: 1.0 };
    const K = computeKernelMatrix(xs, config);
    expect(K.rows).toBe(3);
    expect(K.cols).toBe(3);
    expect(K.data.length).toBe(9);
  });

  it('diagonal equals variance for RBF kernel', () => {
    const xs = [0, 1, 2];
    const variance = 2.5;
    const config: KernelConfig = { type: 'rbf', lengthscale: 1.0, variance };
    const K = computeKernelMatrix(xs, config);

    for (let i = 0; i < xs.length; i++) {
      expect(K.data[i * K.cols + i]!).toBeCloseTo(variance, 10);
    }
  });
});

// ---------------------------------------------------------------------------
// GP Regression
// ---------------------------------------------------------------------------

describe('GPRegressor', () => {
  it('fit on simple linear data, predict returns reasonable mean', () => {
    const gp = new GPRegressor({
      kernel: { type: 'rbf', lengthscale: 1.0, variance: 1.0 },
      noiseVariance: 0.01,
      meanFunction: 'zero',
    });

    // Training data: y = 2x
    const xTrain = [0, 1, 2, 3, 4];
    const yTrain = [0, 2, 4, 6, 8];
    gp.fit(xTrain, yTrain);

    // Predict at training points - should be close to training values
    const pred = gp.predict([1, 2, 3]);
    expect(pred.mean[0]).toBeCloseTo(2, 0);
    expect(pred.mean[1]).toBeCloseTo(4, 0);
    expect(pred.mean[2]).toBeCloseTo(6, 0);
  });

  it('posterior variance is lower near training points', () => {
    const gp = new GPRegressor({
      kernel: { type: 'rbf', lengthscale: 1.0, variance: 1.0 },
      noiseVariance: 0.01,
      meanFunction: 'zero',
    });

    const xTrain = [0, 2, 4];
    const yTrain = [0, 4, 8];
    gp.fit(xTrain, yTrain);

    // Predict near a training point and far from any training point
    const pred = gp.predict([2.0, 10.0]);
    const varNear = pred.variance[0]!;
    const varFar = pred.variance[1]!;
    expect(varNear).toBeLessThan(varFar);
  });

  it('posterior variance is higher far from training points', () => {
    const gp = new GPRegressor({
      kernel: { type: 'rbf', lengthscale: 1.0, variance: 1.0 },
      noiseVariance: 0.01,
      meanFunction: 'zero',
    });

    const xTrain = [0, 1, 2];
    const yTrain = [0, 1, 2];
    gp.fit(xTrain, yTrain);

    const pred = gp.predict([1.0, 100.0]);
    expect(pred.variance[1]).toBeGreaterThan(pred.variance[0]!);
  });

  it('logMarginalLikelihood returns a finite value', () => {
    const gp = new GPRegressor({
      kernel: { type: 'rbf', lengthscale: 1.0, variance: 1.0 },
      noiseVariance: 0.1,
      meanFunction: 'zero',
    });

    const xTrain = [0, 1, 2, 3];
    const yTrain = [0, 1, 2, 3];
    gp.fit(xTrain, yTrain);

    const lml = gp.logMarginalLikelihood();
    expect(Number.isFinite(lml)).toBe(true);
  });

  it('predictions interpolate training data with near-zero residual', () => {
    const gp = new GPRegressor({
      kernel: { type: 'rbf', lengthscale: 1.0, variance: 1.0 },
      noiseVariance: 1e-6,
      meanFunction: 'zero',
    });

    const xTrain = [0, 1, 2, 3, 4];
    const yTrain = [1, 3, 2, 5, 4];
    gp.fit(xTrain, yTrain);

    const pred = gp.predict(xTrain);
    for (let i = 0; i < xTrain.length; i++) {
      expect(pred.mean[i]).toBeCloseTo(yTrain[i]!, 1);
    }
  });

  it('returns prior predictions when not fitted', () => {
    const gp = new GPRegressor({
      kernel: { type: 'rbf', lengthscale: 1.0, variance: 1.0 },
      noiseVariance: 0.1,
      meanFunction: 'zero',
    });

    const pred = gp.predict([0, 1, 2]);
    for (const m of pred.mean) {
      expect(m).toBe(0); // zero mean function
    }
    for (const v of pred.variance) {
      expect(v).toBeCloseTo(1.0, 5); // prior variance = kernel variance
    }
  });

  it('confidence bounds contain the mean', () => {
    const gp = new GPRegressor({
      kernel: { type: 'rbf', lengthscale: 1.0, variance: 1.0 },
      noiseVariance: 0.01,
      meanFunction: 'zero',
    });

    const xTrain = [0, 1, 2, 3];
    const yTrain = [0, 1, 2, 3];
    gp.fit(xTrain, yTrain);

    const pred = gp.predict([0.5, 1.5, 2.5]);
    for (let i = 0; i < 3; i++) {
      expect(pred.lower[i]).toBeLessThanOrEqual(pred.mean[i]!);
      expect(pred.upper[i]).toBeGreaterThanOrEqual(pred.mean[i]!);
    }
  });
});

// ---------------------------------------------------------------------------
// Bayesian Optimization: Helpers
// ---------------------------------------------------------------------------

describe('normalCDF', () => {
  it('normalCDF(0) is approximately 0.5', () => {
    expect(normalCDF(0)).toBeCloseTo(0.5, 4);
  });

  it('normalCDF is monotonically increasing', () => {
    expect(normalCDF(-2)).toBeLessThan(normalCDF(0));
    expect(normalCDF(0)).toBeLessThan(normalCDF(2));
  });

  it('normalCDF(-Infinity-like) is near 0', () => {
    expect(normalCDF(-10)).toBeLessThan(0.001);
  });

  it('normalCDF(+Infinity-like) is near 1', () => {
    expect(normalCDF(10)).toBeGreaterThan(0.999);
  });
});

describe('normalPDF', () => {
  it('normalPDF(0) is approximately 0.3989', () => {
    expect(normalPDF(0)).toBeCloseTo(0.3989, 3);
  });

  it('normalPDF is symmetric', () => {
    expect(normalPDF(-2)).toBeCloseTo(normalPDF(2), 10);
  });

  it('normalPDF is always non-negative', () => {
    expect(normalPDF(-5)).toBeGreaterThanOrEqual(0);
    expect(normalPDF(0)).toBeGreaterThan(0);
    expect(normalPDF(5)).toBeGreaterThanOrEqual(0);
  });
});

// ---------------------------------------------------------------------------
// Bayesian Optimization: Acquisition Functions
// ---------------------------------------------------------------------------

describe('expectedImprovement', () => {
  it('returns > 0 when mean < bestY (for maximization with positive std)', () => {
    // EI should be positive when there is uncertainty
    const ei = expectedImprovement(5.0, 1.0, 4.0, 0.0);
    expect(ei).toBeGreaterThan(0);
  });

  it('returns 0 when std is 0 and mean <= bestY', () => {
    const ei = expectedImprovement(3.0, 0, 5.0, 0.0);
    expect(ei).toBe(0);
  });

  it('increases with higher std (more uncertainty)', () => {
    const eiLow = expectedImprovement(5.0, 0.5, 5.0, 0.0);
    const eiHigh = expectedImprovement(5.0, 2.0, 5.0, 0.0);
    expect(eiHigh).toBeGreaterThan(eiLow);
  });
});

describe('upperConfidenceBound', () => {
  it('increases with kappa', () => {
    const ucb1 = upperConfidenceBound(5.0, 1.0, 1.0);
    const ucb2 = upperConfidenceBound(5.0, 1.0, 3.0);
    expect(ucb2).toBeGreaterThan(ucb1);
  });

  it('equals mean when kappa is 0', () => {
    const ucb = upperConfidenceBound(5.0, 2.0, 0.0);
    expect(ucb).toBeCloseTo(5.0, 10);
  });

  it('equals mean when std is 0', () => {
    const ucb = upperConfidenceBound(5.0, 0, 10.0);
    expect(ucb).toBeCloseTo(5.0, 10);
  });
});

// ---------------------------------------------------------------------------
// Bayesian Optimization: Full Loop
// ---------------------------------------------------------------------------

describe('bayesianOptimize', () => {
  it('on simple quadratic improves over iterations', () => {
    const rng = createPRNG(42);

    // Maximize f(x) = -(x - 3)^2 + 10 => optimum at x=3, f=10
    const objective = (x: number[]): number => {
      const xi = x[0]!;
      return -(xi - 3) * (xi - 3) + 10;
    };

    const result = bayesianOptimize(
      objective,
      {
        bounds: [[0, 6]],
        acquisitionFn: 'ei',
        kappa: 2.0,
        xi: 0.01,
        nInitial: 5,
        maxIterations: 20,
      },
      rng,
    );

    // The best Y should be reasonably close to 10
    expect(result.bestY).toBeGreaterThan(8);
    // The best X should be close to 3
    expect(result.bestX[0]).toBeGreaterThan(1);
    expect(result.bestX[0]).toBeLessThan(5);
  });

  it('returns history with correct length', () => {
    const rng = createPRNG(99);
    const objective = (x: number[]): number => -(x[0]! * x[0]!);

    const result = bayesianOptimize(
      objective,
      {
        bounds: [[-5, 5]],
        acquisitionFn: 'ei',
        kappa: 2.0,
        xi: 0.01,
        nInitial: 3,
        maxIterations: 7,
      },
      rng,
    );

    // nInitial + maxIterations = 10
    expect(result.history).toHaveLength(10);
  });

  it('final best Y is the maximum over all history', () => {
    const rng = createPRNG(55);
    const objective = (x: number[]): number => Math.sin(x[0]!);

    const result = bayesianOptimize(
      objective,
      {
        bounds: [[0, Math.PI]],
        acquisitionFn: 'ucb',
        kappa: 2.0,
        xi: 0.01,
        nInitial: 5,
        maxIterations: 10,
      },
      rng,
    );

    const maxHistory = Math.max(...result.history.map(h => h.y));
    expect(result.bestY).toBeCloseTo(maxHistory, 10);
  });
});
