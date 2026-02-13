// ---------------------------------------------------------------------------
// Tests for Bandit Algorithms (Hedge, UCB, EXP3, LinUCB, Thompson)
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';
import { createPRNG } from '../types.js';
import {
  createHedgeState,
  hedgeDistribution,
  hedgeUpdate,
  hedgeSelect,
} from '../bandits/hedge.js';
import {
  createUCBState,
  ucbSelect,
  ucbUpdate,
} from '../bandits/ucb.js';
import {
  createEXP3State,
  exp3Select,
  exp3Update,
} from '../bandits/exp3.js';
import { LinUCB } from '../bandits/lin-ucb.js';
import {
  createThompsonState,
  thompsonSelect,
  thompsonUpdate,
} from '../bandits/thompson.js';

// ---------------------------------------------------------------------------
// Hedge
// ---------------------------------------------------------------------------

describe('Hedge', () => {
  it('createHedgeState returns uniform weights', () => {
    const weights = createHedgeState(5, 0.1);
    expect(weights).toHaveLength(5);
    for (const w of weights) {
      expect(w).toBe(1);
    }
  });

  it('hedgeDistribution sums to 1', () => {
    const weights = createHedgeState(4, 0.1);
    const dist = hedgeDistribution(weights);
    const sum = dist.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 10);
  });

  it('hedgeDistribution returns uniform for equal weights', () => {
    const weights = createHedgeState(3, 0.1);
    const dist = hedgeDistribution(weights);
    for (const p of dist) {
      expect(p).toBeCloseTo(1 / 3, 10);
    }
  });

  it('hedgeUpdate increases weight of low-loss expert', () => {
    const weights = createHedgeState(3, 0.1);
    const losses = [1.0, 0.0, 0.5]; // expert 1 has zero loss
    const eta = 0.5;
    const updated = hedgeUpdate(weights, losses, eta);
    // w_0 *= exp(-0.5 * 1.0) = exp(-0.5) ~ 0.607
    // w_1 *= exp(-0.5 * 0.0) = 1.0
    // w_2 *= exp(-0.5 * 0.5) = exp(-0.25) ~ 0.779
    expect(updated[1]).toBeGreaterThan(updated[0]!);
    expect(updated[1]).toBeGreaterThan(updated[2]!);
  });

  it('hedgeSelect returns a valid index', () => {
    const rng = createPRNG(42);
    const weights = createHedgeState(5, 0.1);
    const selected = hedgeSelect(weights, rng);
    expect(selected).toBeGreaterThanOrEqual(0);
    expect(selected).toBeLessThan(5);
  });

  it('hedgeSelect distributes selections roughly uniformly for equal weights', () => {
    const rng = createPRNG(123);
    const weights = createHedgeState(3, 0.1);
    const counts = [0, 0, 0];
    for (let i = 0; i < 3000; i++) {
      const idx = hedgeSelect(weights, rng);
      counts[idx]!++;
    }
    // Each should be around 1000 +/- 100
    for (const c of counts) {
      expect(c).toBeGreaterThan(800);
      expect(c).toBeLessThan(1200);
    }
  });

  it('hedgeDistribution falls back to uniform for zero weights', () => {
    const dist = hedgeDistribution([0, 0, 0]);
    expect(dist).toHaveLength(3);
    for (const p of dist) {
      expect(p).toBeCloseTo(1 / 3, 10);
    }
  });
});

// ---------------------------------------------------------------------------
// UCB
// ---------------------------------------------------------------------------

describe('UCB', () => {
  it('createUCBState initializes arms with 0 pulls', () => {
    const state = createUCBState(4);
    expect(state.arms).toHaveLength(4);
    for (const arm of state.arms) {
      expect(arm.pulls).toBe(0);
      expect(arm.totalReward).toBe(0);
      expect(arm.meanReward).toBe(0);
    }
    expect(state.totalPulls).toBe(0);
  });

  it('ucbSelect selects each arm once initially (exploration phase)', () => {
    let state = createUCBState(3);
    const selected: number[] = [];

    // First 3 selections should be arms 0, 1, 2 (unpulled arms)
    for (let i = 0; i < 3; i++) {
      const arm = ucbSelect(state);
      selected.push(arm);
      state = ucbUpdate(state, arm, 0.5);
    }

    // All 3 arms should have been pulled exactly once
    const uniqueArms = new Set(selected);
    expect(uniqueArms.size).toBe(3);
  });

  it('ucbUpdate updates the correct arm stats', () => {
    let state = createUCBState(3);
    state = ucbUpdate(state, 1, 0.8);
    expect(state.arms[1]!.pulls).toBe(1);
    expect(state.arms[1]!.totalReward).toBeCloseTo(0.8, 10);
    expect(state.arms[1]!.meanReward).toBeCloseTo(0.8, 10);
    expect(state.arms[0]!.pulls).toBe(0);
    expect(state.arms[2]!.pulls).toBe(0);
    expect(state.totalPulls).toBe(1);
  });

  it('ucbUpdate preserves immutability', () => {
    const state = createUCBState(3);
    const newState = ucbUpdate(state, 0, 1.0);
    expect(newState).not.toBe(state);
    expect(state.arms[0]!.pulls).toBe(0);
    expect(newState.arms[0]!.pulls).toBe(1);
  });

  it('after many pulls, exploits best arm most often', () => {
    const rng = createPRNG(7);
    let state = createUCBState(3);
    // True means: arm0=0.3, arm1=0.9, arm2=0.5
    const trueMeans = [0.3, 0.9, 0.5];
    const counts = [0, 0, 0];

    for (let t = 0; t < 1000; t++) {
      const arm = ucbSelect(state);
      counts[arm]!++;
      // Bernoulli reward with true mean
      const reward = rng() < trueMeans[arm]! ? 1 : 0;
      state = ucbUpdate(state, arm, reward);
    }

    // Arm 1 (best arm) should have the most pulls
    expect(counts[1]).toBeGreaterThan(counts[0]!);
    expect(counts[1]).toBeGreaterThan(counts[2]!);
  });
});

// ---------------------------------------------------------------------------
// EXP3
// ---------------------------------------------------------------------------

describe('EXP3', () => {
  it('createEXP3State returns uniform weights', () => {
    const weights = createEXP3State(4, 0.1);
    expect(weights).toHaveLength(4);
    for (const w of weights) {
      expect(w).toBe(1);
    }
  });

  it('exp3Select returns a valid arm index', () => {
    const rng = createPRNG(99);
    const weights = createEXP3State(5, 0.1);
    const arm = exp3Select(weights, 0.1, rng);
    expect(arm).toBeGreaterThanOrEqual(0);
    expect(arm).toBeLessThan(5);
  });

  it('exp3Update returns updated weights with correct length', () => {
    const weights = createEXP3State(3, 0.1);
    const updated = exp3Update(weights, 1, 0.8, 0.5, 0.1);
    expect(updated).toHaveLength(3);
  });

  it('exp3Update only modifies the played arm weight', () => {
    const weights = [1, 1, 1];
    const updated = exp3Update(weights, 1, 0.8, 0.5, 0.1);
    // Arm 0 and 2 should remain unchanged
    expect(updated[0]).toBe(1);
    expect(updated[2]).toBe(1);
    // Arm 1 should be increased (positive reward)
    expect(updated[1]).toBeGreaterThan(1);
  });

  it('probability distribution sums to 1 after updates', () => {
    const rng = createPRNG(55);
    let weights = createEXP3State(4, 0.2);
    const gamma = 0.2;

    for (let t = 0; t < 20; t++) {
      const arm = exp3Select(weights, gamma, rng);
      // Compute the probability that was used for selection
      let totalW = 0;
      for (const w of weights) totalW += w;
      const prob = (1 - gamma) * (weights[arm]! / totalW) + gamma / 4;
      weights = exp3Update(weights, arm, rng(), prob, gamma);
    }

    // Check that the implicit distribution still sums to 1
    let totalW = 0;
    for (const w of weights) totalW += w;
    const dist = weights.map(w => (1 - gamma) * (w / totalW) + gamma / 4);
    const sum = dist.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 6);
  });
});

// ---------------------------------------------------------------------------
// LinUCB
// ---------------------------------------------------------------------------

describe('LinUCB', () => {
  it('constructor creates valid state', () => {
    const linucb = new LinUCB(3, 2, 1.0);
    expect(linucb.nArms).toBe(3);
    expect(linucb.d).toBe(2);
    expect(linucb.alpha).toBe(1.0);
  });

  it('select returns a valid arm index', () => {
    const linucb = new LinUCB(4, 3, 1.0);
    const context = new Float64Array([1.0, 0.5, -0.3]);
    const arm = linucb.select(context);
    expect(arm).toBeGreaterThanOrEqual(0);
    expect(arm).toBeLessThan(4);
  });

  it('update modifies internal state (subsequent selects may differ)', () => {
    const linucb = new LinUCB(2, 2, 1.0);
    const ctx = new Float64Array([1.0, 0.0]);

    // Before any updates, both arms are symmetric
    const arm1 = linucb.select(ctx);

    // Update arm 0 with high reward
    linucb.update(0, ctx, 10.0);

    // After update, arm 0 should be preferred for the same context
    const arm2 = linucb.select(ctx);
    expect(arm2).toBe(0);
  });

  it('converges to best arm on linear reward', () => {
    const rng = createPRNG(42);
    const d = 3;
    const nArms = 3;
    // True weight vectors for each arm
    const trueTheta = [
      new Float64Array([0.1, 0.1, 0.1]),  // arm 0: low reward
      new Float64Array([1.0, 1.0, 1.0]),  // arm 1: high reward
      new Float64Array([0.5, 0.5, 0.5]),  // arm 2: medium reward
    ];

    const linucb = new LinUCB(nArms, d, 1.0);
    const counts = [0, 0, 0];

    for (let t = 0; t < 500; t++) {
      // Random context with positive components
      const ctx = new Float64Array([rng(), rng(), rng()]);
      const arm = linucb.select(ctx);
      counts[arm]!++;

      // Reward is dot(trueTheta[arm], ctx) + noise
      let reward = 0;
      for (let j = 0; j < d; j++) {
        reward += (trueTheta[arm]![j] ?? 0) * (ctx[j] ?? 0);
      }
      reward += (rng() - 0.5) * 0.1; // small noise
      linucb.update(arm, ctx, reward);
    }

    // Arm 1 should have the most pulls
    expect(counts[1]).toBeGreaterThan(counts[0]!);
    expect(counts[1]).toBeGreaterThan(counts[2]!);
  });
});

// ---------------------------------------------------------------------------
// Thompson Sampling
// ---------------------------------------------------------------------------

describe('Thompson Sampling', () => {
  it('createThompsonState has uniform (Beta(1,1)) priors', () => {
    const state = createThompsonState(4);
    expect(state.alpha).toHaveLength(4);
    expect(state.beta).toHaveLength(4);
    for (let i = 0; i < 4; i++) {
      expect(state.alpha[i]).toBe(1);
      expect(state.beta[i]).toBe(1);
    }
  });

  it('thompsonSelect returns a valid arm index', () => {
    const rng = createPRNG(77);
    const state = createThompsonState(5);
    const arm = thompsonSelect(state, rng);
    expect(arm).toBeGreaterThanOrEqual(0);
    expect(arm).toBeLessThan(5);
  });

  it('thompsonUpdate increments alpha on success (reward=1)', () => {
    const state = createThompsonState(3);
    const updated = thompsonUpdate(state, 1, 1.0);
    // alpha[1] should increase by reward=1.0
    expect(updated.alpha[1]).toBe(2);
    // beta[1] should increase by (1 - 1.0) = 0
    expect(updated.beta[1]).toBe(1);
  });

  it('thompsonUpdate increments beta on failure (reward=0)', () => {
    const state = createThompsonState(3);
    const updated = thompsonUpdate(state, 1, 0.0);
    // alpha[1] should increase by reward=0
    expect(updated.alpha[1]).toBe(1);
    // beta[1] should increase by (1 - 0) = 1
    expect(updated.beta[1]).toBe(2);
  });

  it('thompsonUpdate preserves other arms', () => {
    const state = createThompsonState(3);
    const updated = thompsonUpdate(state, 1, 0.7);
    expect(updated.alpha[0]).toBe(1);
    expect(updated.beta[0]).toBe(1);
    expect(updated.alpha[2]).toBe(1);
    expect(updated.beta[2]).toBe(1);
  });

  it('thompsonUpdate is immutable', () => {
    const state = createThompsonState(3);
    const updated = thompsonUpdate(state, 0, 1.0);
    expect(updated).not.toBe(state);
    expect(state.alpha[0]).toBe(1);
    expect(updated.alpha[0]).toBe(2);
  });

  it('after many updates, selects best arm consistently', () => {
    const rng = createPRNG(42);
    let state = createThompsonState(3);
    // True success probabilities: arm0=0.2, arm1=0.8, arm2=0.5
    const trueProbs = [0.2, 0.8, 0.5];

    // Simulate many rounds
    for (let t = 0; t < 500; t++) {
      const arm = thompsonSelect(state, rng);
      const reward = rng() < trueProbs[arm]! ? 1.0 : 0.0;
      state = thompsonUpdate(state, arm, reward);
    }

    // Now sample selections many times and check arm 1 is most popular
    const counts = [0, 0, 0];
    for (let i = 0; i < 1000; i++) {
      const arm = thompsonSelect(state, rng);
      counts[arm]!++;
    }

    expect(counts[1]).toBeGreaterThan(counts[0]!);
    expect(counts[1]).toBeGreaterThan(counts[2]!);
  });

  it('thompsonUpdate clamps reward to [0, 1]', () => {
    const state = createThompsonState(2);
    // Reward > 1 should be clamped to 1
    const updated = thompsonUpdate(state, 0, 5.0);
    expect(updated.alpha[0]).toBe(2); // 1 + 1.0 (clamped)
    expect(updated.beta[0]).toBe(1);  // 1 + 0.0

    // Reward < 0 should be clamped to 0
    const updated2 = thompsonUpdate(state, 0, -3.0);
    expect(updated2.alpha[0]).toBe(1); // 1 + 0.0 (clamped)
    expect(updated2.beta[0]).toBe(2);  // 1 + 1.0
  });
});
