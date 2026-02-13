// ---------------------------------------------------------------------------
// Tests for PAC-Bayes Bounds & Model Selection
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';
import {
  pacBayesKLBound,
  mcAllesterBound,
  catoniBound,
} from '../pac-bayes/bounds.js';
import {
  selectModelComplexity,
  computeSampleComplexity,
  rademacherBound,
  recommendModel,
} from '../pac-bayes/model-selection.js';

// ---------------------------------------------------------------------------
// pacBayesKLBound
// ---------------------------------------------------------------------------

describe('pacBayesKLBound', () => {
  it('returns a value >= empRisk', () => {
    const bound = pacBayesKLBound(0.1, 50, 10_000, 0.05);
    expect(bound).toBeGreaterThanOrEqual(0.1);
  });

  it('tightens (decreases) with more data (larger n)', () => {
    const boundSmall = pacBayesKLBound(0.1, 50, 1_000, 0.05);
    const boundLarge = pacBayesKLBound(0.1, 50, 50_000, 0.05);
    expect(boundLarge).toBeLessThan(boundSmall);
  });

  it('returns a value <= 1', () => {
    const bound = pacBayesKLBound(0.1, 50, 1_000, 0.05);
    expect(bound).toBeLessThanOrEqual(1);
  });

  it('returns empRisk when klDivergence is 0 and n is very large', () => {
    // With zero KL and huge n, the bound should be very close to empRisk
    const bound = pacBayesKLBound(0.2, 0, 1_000_000, 0.05);
    expect(bound).toBeCloseTo(0.2, 2);
  });

  it('handles empRisk of 0 correctly', () => {
    const bound = pacBayesKLBound(0, 50, 10_000, 0.05);
    expect(bound).toBeGreaterThanOrEqual(0);
    expect(bound).toBeLessThanOrEqual(1);
  });

  it('handles empRisk of 1 correctly', () => {
    const bound = pacBayesKLBound(1, 50, 10_000, 0.05);
    expect(bound).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// mcAllesterBound
// ---------------------------------------------------------------------------

describe('mcAllesterBound', () => {
  it('returns a value >= empRisk', () => {
    const bound = mcAllesterBound(0.1, 50, 10_000, 0.05);
    expect(bound).toBeGreaterThanOrEqual(0.1);
  });

  it('is >= pacBayesKLBound for the same inputs (KL bound is tighter)', () => {
    const empRisk = 0.1;
    const kl = 50;
    const n = 10_000;
    const delta = 0.05;
    const klBound = pacBayesKLBound(empRisk, kl, n, delta);
    const mcBound = mcAllesterBound(empRisk, kl, n, delta);
    expect(mcBound).toBeGreaterThanOrEqual(klBound);
  });

  it('returns a value <= 1', () => {
    const bound = mcAllesterBound(0.1, 50, 1_000, 0.05);
    expect(bound).toBeLessThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// catoniBound
// ---------------------------------------------------------------------------

describe('catoniBound', () => {
  it('returns a value >= empRisk', () => {
    const bound = catoniBound(0.1, 50, 10_000, 1.0, 0.05);
    expect(bound).toBeGreaterThanOrEqual(0.1);
  });

  it('with lossBound=1 returns a reasonable value in (0, 1]', () => {
    const bound = catoniBound(0.1, 50, 10_000, 1.0, 0.05);
    expect(bound).toBeGreaterThan(0);
    expect(bound).toBeLessThanOrEqual(1);
  });

  it('returns empRisk when complexity term is zero', () => {
    // klDivergence = 0, delta very close to 1 => complexity ~ 0
    // Actually complexity = kl + ln(1/delta). If kl=0 and delta=1, ln(1)=0
    const bound = catoniBound(0.3, 0, 10_000, 1.0, 1.0);
    // complexity = 0 + ln(1) = 0 => returns empRisk
    expect(bound).toBe(0.3);
  });

  it('bound is clamped to lossBound', () => {
    // Large KL + small n should produce a bound at or below lossBound
    const bound = catoniBound(0.5, 1000, 10, 2.0, 0.05);
    expect(bound).toBeLessThanOrEqual(2.0);
  });
});

// ---------------------------------------------------------------------------
// selectModelComplexity
// ---------------------------------------------------------------------------

describe('selectModelComplexity', () => {
  it('picks the model with the lowest PAC-Bayes bound', () => {
    const candidates = [
      { name: 'simple', empRisk: 0.15, klDiv: 10 },
      { name: 'complex', empRisk: 0.10, klDiv: 500 },
    ];
    // At n=5000, the complex model's high KL (500) produces a worse bound
    // than the simple model's low KL (10) despite its lower empRisk.
    const selected = selectModelComplexity(candidates, 5_000, 0.05);
    expect(selected).toBe('simple');
  });

  it('prefers simpler model when data is scarce', () => {
    const candidates = [
      { name: 'linear', empRisk: 0.12, klDiv: 5 },
      { name: 'deep-nn', empRisk: 0.02, klDiv: 500 },
    ];
    // With only 200 samples, the deep NN's high KL gives a vacuous bound
    const selected = selectModelComplexity(candidates, 200, 0.05);
    expect(selected).toBe('linear');
  });

  it('returns empty string for empty candidates', () => {
    const selected = selectModelComplexity([], 1000, 0.05);
    expect(selected).toBe('');
  });
});

// ---------------------------------------------------------------------------
// computeSampleComplexity
// ---------------------------------------------------------------------------

describe('computeSampleComplexity', () => {
  it('for VC=21, epsilon=0.05, delta=0.05 returns approximately 1320', () => {
    const m = computeSampleComplexity(21, 0.05, 0.05);
    // m = (1/0.05) * (21 * ln(20) + ln(20))
    //   = 20 * (21 * 2.9957 + 2.9957)
    //   = 20 * (62.91 + 3.00) = 20 * 65.91 = 1318.2 => ceil = 1319
    expect(m).toBeGreaterThanOrEqual(1300);
    expect(m).toBeLessThanOrEqual(1350);
  });

  it('increases with VC dimension', () => {
    const m10 = computeSampleComplexity(10, 0.05, 0.05);
    const m50 = computeSampleComplexity(50, 0.05, 0.05);
    expect(m50).toBeGreaterThan(m10);
  });

  it('returns Infinity for epsilon=0', () => {
    const m = computeSampleComplexity(10, 0, 0.05);
    expect(m).toBe(Infinity);
  });

  it('returns Infinity for delta=0', () => {
    const m = computeSampleComplexity(10, 0.05, 0);
    expect(m).toBe(Infinity);
  });

  it('returns a positive integer', () => {
    const m = computeSampleComplexity(21, 0.05, 0.05);
    expect(m).toBeGreaterThan(0);
    expect(Number.isInteger(m)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// rademacherBound
// ---------------------------------------------------------------------------

describe('rademacherBound', () => {
  it('tightens with more data', () => {
    const boundSmall = rademacherBound(0.1, 1.0, 1.0, 100, 0.05);
    const boundLarge = rademacherBound(0.1, 1.0, 1.0, 10_000, 0.05);
    expect(boundLarge).toBeLessThan(boundSmall);
  });

  it('returns Infinity for n=0', () => {
    const bound = rademacherBound(0.1, 1.0, 1.0, 0, 0.05);
    expect(bound).toBe(Infinity);
  });

  it('returns value >= empRisk', () => {
    const bound = rademacherBound(0.1, 1.0, 1.0, 1_000, 0.05);
    expect(bound).toBeGreaterThanOrEqual(0.1);
  });
});

// ---------------------------------------------------------------------------
// recommendModel
// ---------------------------------------------------------------------------

describe('recommendModel', () => {
  it('returns "linear" recommendation for n=1000', () => {
    const rec = recommendModel(1000);
    expect(rec).toMatch(/^tree-ensemble/);
  });

  it('returns "linear" recommendation for n < 1000', () => {
    const rec = recommendModel(500);
    expect(rec).toMatch(/^linear/);
  });

  it('returns appropriate model for n=10000', () => {
    const rec = recommendModel(10_000);
    expect(rec).toMatch(/^small-nn/);
  });

  it('returns appropriate model for n=100000', () => {
    const rec = recommendModel(100_000);
    expect(rec).toMatch(/^deep/);
  });
});
