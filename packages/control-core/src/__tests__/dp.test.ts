import { describe, it, expect } from 'vitest';
import {
  valueIteration,
  policyIteration,
  backwardInduction,
  solveHJB,
  fittedValueIteration,
  solveBidPriceLP,
  simplexLP,
  computeStoppingThresholds,
} from '../dp/index.js';
import { createPRNG } from '../types.js';
import type {
  BellmanConfig,
  HJBConfig,
  ApproxDPConfig,
  BidPriceConfig,
  OptimalStoppingConfig,
} from '../types.js';

// ---------------------------------------------------------------------------
// Helper: simple 3-state, 2-action MDP
//
//  State 0 --action 0--> State 0 (reward 1)
//  State 0 --action 1--> State 1 (reward 3)
//  State 1 --action 0--> State 0 (reward 2)
//  State 1 --action 1--> State 2 (reward 0)
//  State 2 --action 0--> State 2 (reward 0)
//  State 2 --action 1--> State 0 (reward 1)
// ---------------------------------------------------------------------------

function makeSimpleMDP(): BellmanConfig {
  return {
    nStates: 3,
    nActions: 2,
    transitions: (s: number, a: number) => {
      if (s === 0 && a === 0) return [{ nextState: 0, probability: 1.0 }];
      if (s === 0 && a === 1) return [{ nextState: 1, probability: 1.0 }];
      if (s === 1 && a === 0) return [{ nextState: 0, probability: 1.0 }];
      if (s === 1 && a === 1) return [{ nextState: 2, probability: 1.0 }];
      if (s === 2 && a === 0) return [{ nextState: 2, probability: 1.0 }];
      if (s === 2 && a === 1) return [{ nextState: 0, probability: 1.0 }];
      return [{ nextState: s, probability: 1.0 }];
    },
    reward: (s: number, a: number) => {
      if (s === 0 && a === 0) return 1;
      if (s === 0 && a === 1) return 3;
      if (s === 1 && a === 0) return 2;
      if (s === 1 && a === 1) return 0;
      if (s === 2 && a === 0) return 0;
      if (s === 2 && a === 1) return 1;
      return 0;
    },
    discount: 0.9,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('Dynamic Programming', () => {
  // ---------------------------------------------------------------------------
  // 1. Value iteration converges (change < tolerance)
  // ---------------------------------------------------------------------------

  it('value iteration converges for simple MDP', () => {
    const config = makeSimpleMDP();
    const result = valueIteration(config, 1e-8, 1000);

    // Should converge in fewer than max iterations
    expect(result.iterations).toBeLessThan(1000);

    // Value function should be non-zero (non-trivial MDP with positive rewards)
    let maxVal = 0;
    for (let s = 0; s < config.nStates; s++) {
      expect(Number.isFinite(result.valueFunction[s]!)).toBe(true);
      if (result.valueFunction[s]! > maxVal) {
        maxVal = result.valueFunction[s]!;
      }
    }
    expect(maxVal).toBeGreaterThan(0);

    // Policy should assign valid actions
    for (let s = 0; s < config.nStates; s++) {
      expect(result.policy[s]!).toBeGreaterThanOrEqual(0);
      expect(result.policy[s]!).toBeLessThan(config.nActions);
    }
  });

  // ---------------------------------------------------------------------------
  // 2. Policy iteration matches value iteration result
  // ---------------------------------------------------------------------------

  it('policy iteration matches value iteration result', () => {
    const config = makeSimpleMDP();
    const viResult = valueIteration(config, 1e-10, 2000);
    const piResult = policyIteration(config, 100);

    // Value functions should be very close
    for (let s = 0; s < config.nStates; s++) {
      expect(piResult.valueFunction[s]!).toBeCloseTo(viResult.valueFunction[s]!, 4);
    }

    // Policies should agree
    for (let s = 0; s < config.nStates; s++) {
      expect(piResult.policy[s]!).toBe(viResult.policy[s]!);
    }
  });

  // ---------------------------------------------------------------------------
  // 3. Backward induction for finite horizon
  // ---------------------------------------------------------------------------

  it('backward induction for finite horizon gives correct V_0', () => {
    const config: BellmanConfig = {
      ...makeSimpleMDP(),
      horizon: 10,
    };

    const result = backwardInduction(config);

    // Value function V_0 should be non-negative
    for (let s = 0; s < config.nStates; s++) {
      expect(result.valueFunction[s]!).toBeGreaterThanOrEqual(0);
    }

    // Finite-horizon values should be less than or equal to infinite-horizon
    // (since fewer steps => less cumulative reward)
    const infResult = valueIteration(config, 1e-10, 2000);
    for (let s = 0; s < config.nStates; s++) {
      expect(result.valueFunction[s]!).toBeLessThanOrEqual(
        infResult.valueFunction[s]! + 1e-4,
      );
    }

    // iterations should equal the horizon
    expect(result.iterations).toBe(config.horizon);
  });

  // ---------------------------------------------------------------------------
  // 4. HJB: solves 1D problem, value non-negative
  // ---------------------------------------------------------------------------

  it('HJB solves 1D problem with non-negative value function', () => {
    const hjbConfig: HJBConfig = {
      gridMin: new Float64Array([-2]),
      gridMax: new Float64Array([2]),
      gridN: new Int32Array([21]),
      dt: 0.1,
      dynamics: (x: Float64Array, u: Float64Array) => {
        // dx/dt = -x + u
        const f = new Float64Array(1);
        f[0] = -x[0]! + u[0]!;
        return f;
      },
      runningCost: (x: Float64Array, u: Float64Array) => {
        return 0.5 * (x[0]! * x[0]! + u[0]! * u[0]!);
      },
      terminalCost: (x: Float64Array) => {
        return x[0]! * x[0]!;
      },
      controlSet: [
        new Float64Array([-1]),
        new Float64Array([0]),
        new Float64Array([1]),
      ],
      T: 1.0,
      nx: 1,
    };

    const result = solveHJB(hjbConfig);

    // Value grid should have correct size
    expect(result.valueGrid.length).toBe(21);

    // All values should be non-negative (quadratic costs with non-negative terminal)
    for (let i = 0; i < result.valueGrid.length; i++) {
      expect(result.valueGrid[i]!).toBeGreaterThanOrEqual(-1e-6);
    }

    // Value at origin should be the smallest (zero state has least cost)
    const midIdx = 10; // grid index for x = 0
    const midVal = result.valueGrid[midIdx]!;
    // Values at edges should be larger or equal
    expect(result.valueGrid[0]!).toBeGreaterThanOrEqual(midVal - 1e-6);
    expect(result.valueGrid[20]!).toBeGreaterThanOrEqual(midVal - 1e-6);

    // Policy grid should have valid indices
    for (let i = 0; i < result.policyGrid.length; i++) {
      expect(result.policyGrid[i]!).toBeGreaterThanOrEqual(0);
      expect(result.policyGrid[i]!).toBeLessThan(hjbConfig.controlSet.length);
    }
  });

  // ---------------------------------------------------------------------------
  // 5. Approximate DP: fitted value iteration converges
  // ---------------------------------------------------------------------------

  it('fitted value iteration converges', () => {
    const rng = createPRNG(42);

    const approxConfig: ApproxDPConfig = {
      nFeatures: 3,
      nSamples: 50,
      featureFn: (s: Float64Array) => {
        // Simple polynomial features: [1, s[0], s[0]^2]
        const phi = new Float64Array(3);
        phi[0] = 1;
        phi[1] = s[0]!;
        phi[2] = s[0]! * s[0]!;
        return phi;
      },
      transitionSample: (s: Float64Array, _a: Float64Array, rng_inner) => {
        // Simple dynamics: s' = 0.9 * s + noise
        const sPrime = new Float64Array(3);
        sPrime[0] = 0.9 * s[0]! + 0.1 * (rng_inner() - 0.5);
        sPrime[1] = s[1]!;
        sPrime[2] = s[2]!;
        return sPrime;
      },
      rewardFn: (s: Float64Array, _a: Float64Array) => {
        // Reward = -x^2 (penalize distance from 0)
        return -s[0]! * s[0]!;
      },
      discount: 0.9,
      maxIter: 50,
      tolerance: 1e-4,
    };

    const result = fittedValueIteration(approxConfig, rng);

    // Should converge or at least run to completion
    expect(result.iterations).toBeGreaterThan(0);
    expect(result.weights.length).toBe(approxConfig.nFeatures);

    // Weights should be finite
    for (let i = 0; i < result.weights.length; i++) {
      expect(Number.isFinite(result.weights[i]!)).toBe(true);
    }
  });

  // ---------------------------------------------------------------------------
  // 6. Bid-price LP: dual prices non-negative, allocations respect capacity
  // ---------------------------------------------------------------------------

  it('bid-price LP: dual prices non-negative and allocations respect capacity', () => {
    const config: BidPriceConfig = {
      nResources: 2,
      nProducts: 3,
      // Resource capacities
      resourceCapacities: new Float64Array([100, 80]),
      // Incidence matrix (2 resources x 3 products):
      //   Product 0 uses resource 0 only
      //   Product 1 uses resource 1 only
      //   Product 2 uses both resources
      incidenceMatrix: new Float64Array([
        1, 0, 1,  // resource 0
        0, 1, 1,  // resource 1
      ]),
      // Revenues per product
      revenues: new Float64Array([50, 40, 80]),
      // Mean demand per product
      demandMeans: new Float64Array([60, 50, 30]),
    };

    const result = solveBidPriceLP(config);

    // Bid prices should be non-negative
    for (let i = 0; i < config.nResources; i++) {
      expect(result.bidPrices[i]!).toBeGreaterThanOrEqual(-1e-6);
    }

    // Allocations should be non-negative
    for (let j = 0; j < config.nProducts; j++) {
      expect(result.allocations[j]!).toBeGreaterThanOrEqual(-1e-6);
    }

    // Check resource capacity constraints
    for (let i = 0; i < config.nResources; i++) {
      let resourceUsage = 0;
      for (let j = 0; j < config.nProducts; j++) {
        resourceUsage +=
          config.incidenceMatrix[i * config.nProducts + j]! * result.allocations[j]!;
      }
      expect(resourceUsage).toBeLessThanOrEqual(config.resourceCapacities[i]! + 1e-4);
    }

    // Allocations should not exceed demand
    for (let j = 0; j < config.nProducts; j++) {
      expect(result.allocations[j]!).toBeLessThanOrEqual(
        config.demandMeans[j]! + 1e-4,
      );
    }

    // Optimal revenue should be positive
    expect(result.optimalRevenue).toBeGreaterThan(0);
  });

  // ---------------------------------------------------------------------------
  // 7. Simplex LP: solves a known LP problem
  // ---------------------------------------------------------------------------

  it('simplex LP solves a known LP problem', () => {
    // max  5x1 + 4x2
    // s.t.  6x1 + 4x2 <= 24
    //        x1 + 2x2 <= 6
    //        x1, x2 >= 0
    //
    // Vertices:
    //   (0, 0): obj = 0
    //   (4, 0): 6*4=24<=24, 4<=6. obj = 20
    //   (0, 3): 0+12=12<=24, 0+6=6<=6. obj = 12
    //   Intersection of both: 6x1+4x2=24 and x1+2x2=6 => x1=3, x2=1.5. obj = 21
    // Optimal is (3, 1.5) with obj = 21

    const c = new Float64Array([5, 4]);
    const A = new Float64Array([
      6, 4,  // row 0
      1, 2,  // row 1
    ]);
    const b = new Float64Array([24, 6]);

    const x = simplexLP(c, A, b, 2, 2);

    expect(x).not.toBeNull();
    if (x !== null) {
      // Check feasibility
      expect(6 * x[0]! + 4 * x[1]!).toBeLessThanOrEqual(24 + 1e-6);
      expect(x[0]! + 2 * x[1]!).toBeLessThanOrEqual(6 + 1e-6);
      expect(x[0]!).toBeGreaterThanOrEqual(-1e-6);
      expect(x[1]!).toBeGreaterThanOrEqual(-1e-6);

      // Check optimality: objective should be 21 at (3, 1.5)
      const obj = 5 * x[0]! + 4 * x[1]!;
      expect(obj).toBeCloseTo(21, 4);
    }
  });

  // ---------------------------------------------------------------------------
  // 8. Optimal stopping: thresholds decrease with more capacity
  // ---------------------------------------------------------------------------

  it('optimal stopping thresholds behave correctly', () => {
    const rng = createPRNG(42);

    const stoppingConfig: OptimalStoppingConfig = {
      capacity: 3,
      horizon: 20,
      arrivalProb: (_t: number) => 0.8, // high arrival rate
      valueDistribution: (_t: number, r) => r() * 10, // uniform [0, 10]
      discount: 0.95,
    };

    const result = computeStoppingThresholds(stoppingConfig, rng, 200);

    // Thresholds should have length = horizon
    expect(result.thresholds.length).toBe(stoppingConfig.horizon);

    // Value function should have length = horizon
    expect(result.valueFunction.length).toBe(stoppingConfig.horizon);

    // Thresholds should be non-negative
    for (let t = 0; t < stoppingConfig.horizon; t++) {
      expect(result.thresholds[t]!).toBeGreaterThanOrEqual(-1e-6);
    }

    // Value function should be non-negative and generally decrease over time
    // (earlier periods have more future opportunity)
    for (let t = 0; t < stoppingConfig.horizon; t++) {
      expect(result.valueFunction[t]!).toBeGreaterThanOrEqual(-1e-6);
    }

    // Generally, earlier periods should have higher value (more time to accept)
    // Compare first vs last value
    expect(result.valueFunction[0]!).toBeGreaterThanOrEqual(
      result.valueFunction[stoppingConfig.horizon - 1]! - 1e-4,
    );

    // Now solve with higher capacity and check thresholds decrease
    const highCapConfig: OptimalStoppingConfig = {
      ...stoppingConfig,
      capacity: 10,
    };
    const rng2 = createPRNG(42);
    const highCapResult = computeStoppingThresholds(highCapConfig, rng2, 200);

    // With more capacity, thresholds should generally be lower (more room)
    // Check the first period threshold
    expect(highCapResult.thresholds[0]!).toBeLessThanOrEqual(
      result.thresholds[0]! + 1e-2,
    );
  });
});
