// ---------------------------------------------------------------------------
// Tests for OC-10: Optimal Experiment Design
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';
import {
  dualControlStep,
  computeInfoGain,
  updateBelief,
  idsSelect,
  idsUpdate,
  thompsonPriceSelect,
  thompsonPriceUpdate,
  safeBONext,
} from '../experiment/index.js';
import { createPRNG } from '../types.js';
import type {
  DualControlConfig,
  IDSConfig,
  ThompsonPricingConfig,
  SafeBOConfig,
} from '../types.js';

// ---------------------------------------------------------------------------
// Tests: Dual Control
// ---------------------------------------------------------------------------

describe('dualControlStep', () => {
  it('explores when uncertainty is high (info gain > 0)', () => {
    const rng = createPRNG(42);
    const nx = 2;

    // High uncertainty prior (large diagonal covariance)
    const config: DualControlConfig = {
      priorMean: new Float64Array([0.5, 0.3]),
      priorCov: new Float64Array([
        4.0, 0.0,
        0.0, 1.0,
      ]),
      nx,
      explorationWeight: 1.0,
    };

    const x = new Float64Array([1, -1]);
    const result = dualControlStep(config, x, rng);

    // Information gain should be positive when there is uncertainty
    expect(result.informationGain).toBeGreaterThan(0);

    // Exploration component should be non-zero
    let exploreNorm = 0;
    for (let i = 0; i < nx; i++) {
      exploreNorm += Math.pow(result.explorationComponent[i]!, 2);
    }
    exploreNorm = Math.sqrt(exploreNorm);
    expect(exploreNorm).toBeGreaterThan(0);

    // Action should be the sum of exploitation and scaled exploration
    for (let i = 0; i < nx; i++) {
      const expected =
        result.exploitationComponent[i]! +
        config.explorationWeight * result.explorationComponent[i]!;
      expect(result.action[i]!).toBeCloseTo(expected, 10);
    }
  });
});

describe('computeInfoGain', () => {
  it('is positive for different covariances (posterior < prior)', () => {
    const nx = 2;

    // Prior: diagonal [4, 2]
    const priorCov = new Float64Array([4, 0, 0, 2]);
    // Posterior: diagonal [2, 1]  (reduced uncertainty)
    const posteriorCov = new Float64Array([2, 0, 0, 1]);

    const ig = computeInfoGain(priorCov, posteriorCov, nx);

    // IG = 0.5 * (ln(4/2) + ln(2/1)) = 0.5 * (ln(2) + ln(2)) = ln(2)
    expect(ig).toBeCloseTo(Math.log(2), 10);
    expect(ig).toBeGreaterThan(0);
  });

  it('is zero when prior and posterior are identical', () => {
    const nx = 2;
    const cov = new Float64Array([1, 0, 0, 1]);
    expect(computeInfoGain(cov, cov, nx)).toBeCloseTo(0, 10);
  });
});

describe('updateBelief', () => {
  it('reduces uncertainty (trace of posterior < trace of prior)', () => {
    const nx = 2;
    const nz = 1;

    const priorMean = new Float64Array([0, 0]);
    const priorCov = new Float64Array([
      2.0, 0.0,
      0.0, 2.0,
    ]);

    // Observe z = H * theta + noise, where H = [1, 0], R = [0.5]
    const H = new Float64Array([1, 0]);
    const R = new Float64Array([0.5]);
    const observation = new Float64Array([1.0]);

    const { mean, cov } = updateBelief(
      priorMean,
      priorCov,
      observation,
      H,
      R,
      nx,
      nz,
    );

    // Posterior mean should shift toward the observation
    expect(mean[0]!).toBeGreaterThan(0);

    // Trace of posterior covariance should be less than trace of prior
    const priorTrace = priorCov[0]! + priorCov[3]!; // 2 + 2 = 4
    const postTrace = cov[0]! + cov[3]!;
    expect(postTrace).toBeLessThan(priorTrace);

    // The observed dimension (dim 0) should have reduced variance
    expect(cov[0]!).toBeLessThan(priorCov[0]!);

    // The unobserved dimension (dim 1) should remain unchanged
    // (since H doesn't involve dim 1)
    expect(cov[3]!).toBeCloseTo(priorCov[3]!, 5);
  });
});

// ---------------------------------------------------------------------------
// Tests: Information-Directed Sampling
// ---------------------------------------------------------------------------

describe('idsSelect', () => {
  it('selects arm with good regret-to-info ratio', () => {
    const rng = createPRNG(42);

    // 3 arms with different prior means
    // Arm 0: best mean (highest alpha/(alpha+beta))
    // Arm 1: moderate
    // Arm 2: worst
    const config: IDSConfig = {
      nArms: 3,
      priorAlpha: new Float64Array([10, 5, 2]),
      priorBeta: new Float64Array([2, 5, 10]),
    };

    const result = idsSelect(config, rng);

    // Arm probabilities should sum to ~1
    let probSum = 0;
    for (let i = 0; i < config.nArms; i++) {
      probSum += result.armProbabilities[i]!;
    }
    expect(probSum).toBeCloseTo(1.0, 5);

    // Expected regret of the best arm should be ~0
    expect(result.expectedRegret[0]!).toBeCloseTo(0, 5);

    // Information gain should be positive for all arms
    for (let i = 0; i < config.nArms; i++) {
      expect(result.informationGain[i]!).toBeGreaterThan(0);
    }

    // IDS ratio for best arm should be very small (near zero regret)
    expect(result.idsRatio[0]!).toBeLessThan(result.idsRatio[2]!);

    // Best arm should have high selection probability
    expect(result.armProbabilities[0]!).toBeGreaterThan(
      result.armProbabilities[2]!,
    );
  });
});

describe('idsUpdate', () => {
  it('updates alpha/beta correctly', () => {
    const config: IDSConfig = {
      nArms: 2,
      priorAlpha: new Float64Array([1, 1]),
      priorBeta: new Float64Array([1, 1]),
    };

    // Success on arm 0
    const updated1 = idsUpdate(config, 0, 1);
    expect(updated1.priorAlpha[0]!).toBe(2);
    expect(updated1.priorBeta[0]!).toBe(1);
    // Arm 1 unchanged
    expect(updated1.priorAlpha[1]!).toBe(1);
    expect(updated1.priorBeta[1]!).toBe(1);

    // Failure on arm 1
    const updated2 = idsUpdate(config, 1, 0);
    expect(updated2.priorAlpha[1]!).toBe(1);
    expect(updated2.priorBeta[1]!).toBe(2);
    // Arm 0 unchanged
    expect(updated2.priorAlpha[0]!).toBe(1);
    expect(updated2.priorBeta[0]!).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Tests: Thompson Pricing
// ---------------------------------------------------------------------------

describe('thompsonPriceSelect', () => {
  it('selects price within bounds', () => {
    const rng = createPRNG(42);

    const config: ThompsonPricingConfig = {
      priceGrid: new Float64Array([5, 10, 15, 20, 25]),
      priorAlpha: new Float64Array([2, 3, 5, 3, 2]),
      priorBeta: new Float64Array([3, 2, 1, 2, 3]),
      priceBounds: { min: 5, max: 25 },
      rateLimit: 100, // no rate limit restriction
    };

    const lastPrice = 15;
    const { price, expectedRevenue } = thompsonPriceSelect(config, lastPrice, rng);

    // Price should be in the grid
    const gridValues = Array.from(config.priceGrid);
    expect(gridValues).toContain(price);

    // Price should be within bounds
    expect(price).toBeGreaterThanOrEqual(config.priceBounds.min);
    expect(price).toBeLessThanOrEqual(config.priceBounds.max);

    // Expected revenue should be non-negative
    expect(expectedRevenue).toBeGreaterThanOrEqual(0);
  });

  it('respects rate-limit constraint', () => {
    const rng = createPRNG(42);

    const config: ThompsonPricingConfig = {
      priceGrid: new Float64Array([5, 10, 15, 20, 25]),
      priorAlpha: new Float64Array([2, 3, 5, 3, 2]),
      priorBeta: new Float64Array([3, 2, 1, 2, 3]),
      priceBounds: { min: 5, max: 25 },
      rateLimit: 5, // max change of 5
    };

    const lastPrice = 15;
    const { price } = thompsonPriceSelect(config, lastPrice, rng);

    // Price should be within rate-limit of lastPrice
    expect(Math.abs(price - lastPrice)).toBeLessThanOrEqual(5 + 0.01);
  });
});

describe('thompsonPriceUpdate', () => {
  it('alpha increases on purchase', () => {
    const config: ThompsonPricingConfig = {
      priceGrid: new Float64Array([10, 20, 30]),
      priorAlpha: new Float64Array([1, 1, 1]),
      priorBeta: new Float64Array([1, 1, 1]),
      priceBounds: { min: 10, max: 30 },
      rateLimit: 20,
    };

    const updated = thompsonPriceUpdate(config, 1, true);

    // Alpha for price index 1 should have increased by 1
    expect(updated.priorAlpha[1]!).toBe(2);
    // Beta for price index 1 should be unchanged
    expect(updated.priorBeta[1]!).toBe(1);
    // Other prices unchanged
    expect(updated.priorAlpha[0]!).toBe(1);
    expect(updated.priorAlpha[2]!).toBe(1);
  });

  it('beta increases on no-purchase', () => {
    const config: ThompsonPricingConfig = {
      priceGrid: new Float64Array([10, 20, 30]),
      priorAlpha: new Float64Array([1, 1, 1]),
      priorBeta: new Float64Array([1, 1, 1]),
      priceBounds: { min: 10, max: 30 },
      rateLimit: 20,
    };

    const updated = thompsonPriceUpdate(config, 2, false);

    // Beta for price index 2 should have increased by 1
    expect(updated.priorBeta[2]!).toBe(2);
    // Alpha for price index 2 should be unchanged
    expect(updated.priorAlpha[2]!).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Tests: Safe Bayesian Optimization
// ---------------------------------------------------------------------------

describe('safeBONext', () => {
  it('respects safety threshold with observed data', () => {
    const config: SafeBOConfig = {
      bounds: [{ min: 0, max: 1 }],
      safetyThreshold: 0.0,
      kernelLengthscale: 0.3,
      kernelVariance: 1.0,
      noiseVariance: 0.01,
      beta: 2.0,
    };

    // Provide observations: x=0.5 is safe and good, x=0.9 is unsafe
    const observations = [
      { x: new Float64Array([0.2]), y: 0.5, safe: true },
      { x: new Float64Array([0.5]), y: 1.0, safe: true },
      { x: new Float64Array([0.8]), y: -0.5, safe: false },
    ];

    const result = safeBONext(config, observations);

    // The next point should be within bounds
    expect(result.nextPoint[0]!).toBeGreaterThanOrEqual(0);
    expect(result.nextPoint[0]!).toBeLessThanOrEqual(1);

    // If the algorithm found a safe point, it should be marked as such
    if (result.isSafe) {
      // Safety probability should be reasonable for a safe point
      expect(result.safetyProbability).toBeGreaterThan(0);
      // Expected improvement should be non-negative
      expect(result.expectedImprovement).toBeGreaterThanOrEqual(0);
    }
  });

  it('returns center on cold start (no observations)', () => {
    const config: SafeBOConfig = {
      bounds: [
        { min: 0, max: 2 },
        { min: -1, max: 1 },
      ],
      safetyThreshold: 0.0,
      kernelLengthscale: 0.5,
      kernelVariance: 1.0,
      noiseVariance: 0.01,
      beta: 2.0,
    };

    const result = safeBONext(config, []);

    // Should return center of bounds
    expect(result.nextPoint[0]!).toBeCloseTo(1.0, 10); // (0+2)/2
    expect(result.nextPoint[1]!).toBeCloseTo(0.0, 10); // (-1+1)/2
    expect(result.expectedImprovement).toBe(0);
    expect(result.isSafe).toBe(false);
  });
});
