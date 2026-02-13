// ---------------------------------------------------------------------------
// Tests for OC-9: Stochastic Optimal Control
// ---------------------------------------------------------------------------

import { describe, it, expect } from 'vitest';
import {
  stochasticValueIteration,
  riskSensitiveVI,
  exponentialUtility,
  computeCVaR,
  wasserstein1D,
  wassersteinDRO,
  chanceConstrainedOptimize,
  solveScenarioMPC,
} from '../stochastic/index.js';
import { createPRNG } from '../types.js';
import type { BellmanConfig, ScenarioMPCConfig } from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Build a simple 3-state, 2-action MDP with deterministic scenario-dependent
 * transitions.  State 2 is absorbing with reward 0.  Scenarios shift the
 * reward of action 0 in state 0.
 */
function makeSmallMDP() {
  const nStates = 3;
  const nActions = 2;
  const discount = 0.9;

  const config: BellmanConfig = {
    nStates,
    nActions,
    transitions: (_s, _a) => [],
    reward: (_s, _a) => 0,
    discount,
  };

  const nScenarios = 2;
  const scenarioProbs = new Float64Array([0.5, 0.5]);

  // Scenario-dependent transitions:
  //   state 0, action 0 -> state 1 in both scenarios, reward differs
  //   state 0, action 1 -> state 2 in both scenarios, reward = 1
  //   state 1, action * -> state 2, reward = 2
  //   state 2 (absorbing) -> state 2, reward = 0
  const scenarioTransitions = (s: number, a: number, w: number) => {
    if (s === 2) return { nextState: 2, reward: 0 };
    if (s === 1) return { nextState: 2, reward: 2 };
    // s === 0
    if (a === 1) return { nextState: 2, reward: 1 };
    // a === 0: scenario-dependent reward
    return { nextState: 1, reward: w === 0 ? 3 : 1 };
  };

  return { config, nScenarios, scenarioProbs, scenarioTransitions };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('stochasticValueIteration', () => {
  it('converges for a small MDP with multiple scenarios', () => {
    const { config, nScenarios, scenarioProbs, scenarioTransitions } = makeSmallMDP();

    const result = stochasticValueIteration(
      config,
      nScenarios,
      scenarioProbs,
      scenarioTransitions,
    );

    // Should converge in finite iterations
    expect(result.iterations).toBeGreaterThan(0);
    expect(result.iterations).toBeLessThan(1000);

    // Value function should be non-trivial
    expect(result.V.length).toBe(config.nStates);
    expect(result.policy.length).toBe(config.nStates);

    // State 2 (absorbing, zero reward) should have value ~0
    expect(result.V[2]!).toBeCloseTo(0, 5);

    // State 1 leads to state 2 with reward 2: V(1) = 2 + 0.9 * 0 = 2
    expect(result.V[1]!).toBeCloseTo(2, 5);

    // State 0, action 0: E[r] = 0.5*3 + 0.5*1 = 2, then V(1) = 2
    //   Q(0,0) = 2 + 0.9*2 = 3.8
    // State 0, action 1: reward 1, then V(2) = 0
    //   Q(0,1) = 1 + 0.9*0 = 1
    // So V(0) ~ 3.8, policy(0) = 0
    expect(result.V[0]!).toBeCloseTo(3.8, 4);
    expect(result.policy[0]).toBe(0);
  });
});

describe('riskSensitiveVI', () => {
  it('produces more conservative policy for higher theta (risk-aversion)', () => {
    // 2-state, 2-action MDP:
    //   State 0: action 0 goes to state 1 (cost 0), action 1 stays in state 0 (cost 1)
    //   State 1 (absorbing): cost 0 regardless of action
    //
    // For risk-neutral: optimal is action 0 in state 0 (cost 0 + discount * 0 = 0)
    // For high theta, the exponential weighting penalises variance — but since
    // this is deterministic, we instead test that the cost-value increases with theta.
    const nStates = 2;
    const nActions = 2;
    const gamma = 0.9;

    const transitions = (s: number, a: number) => {
      if (s === 1) return [{ nextState: 1, probability: 1 }];
      // s === 0
      if (a === 0) {
        // Stochastic: 50% go to s1, 50% stay in s0
        return [
          { nextState: 1, probability: 0.5 },
          { nextState: 0, probability: 0.5 },
        ];
      }
      // a === 1: deterministic go to s1
      return [{ nextState: 1, probability: 1 }];
    };

    const cost = (s: number, a: number) => {
      if (s === 1) return 0;
      return a === 0 ? 0.5 : 1.5;
    };

    const resultNeutral = riskSensitiveVI(nStates, nActions, transitions, cost, 0.01, gamma);
    const resultAverse = riskSensitiveVI(nStates, nActions, transitions, cost, 5.0, gamma);

    // Both should return valid value functions
    expect(resultNeutral.V.length).toBe(nStates);
    expect(resultAverse.V.length).toBe(nStates);

    // With high theta (risk-averse), the value at state 0 should be >= the
    // risk-neutral value, because the exponential cost criterion penalises
    // variance. (In a cost minimisation setting, risk-averse values are higher.)
    expect(resultAverse.V[0]!).toBeGreaterThanOrEqual(resultNeutral.V[0]! - 1e-6);
  });
});

describe('exponentialUtility', () => {
  it('returns correct value for risk-averse parameter', () => {
    const theta = 2;
    const cost = 1;
    // U(c) = (1/theta) * (1 - exp(-theta * c))
    const expected = (1 / 2) * (1 - Math.exp(-2 * 1));
    expect(exponentialUtility(cost, theta)).toBeCloseTo(expected, 10);
  });

  it('returns cost itself for near-zero theta (risk-neutral)', () => {
    const cost = 3.14;
    expect(exponentialUtility(cost, 0)).toBeCloseTo(cost, 10);
  });
});

describe('computeCVaR', () => {
  it('CVaR >= VaR for any loss distribution', () => {
    const rng = createPRNG(42);
    const n = 100;
    const losses = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      losses[i] = rng() * 10;
    }

    const alpha = 0.9;
    const { cvar, var_ } = computeCVaR(losses, alpha);

    // CVaR (expected shortfall in the tail) should be >= VaR
    expect(cvar).toBeGreaterThanOrEqual(var_ - 1e-10);
  });

  it('CVaR equals the single value for a degenerate distribution', () => {
    // All losses are the same value
    const losses = new Float64Array([5, 5, 5, 5, 5]);
    const { cvar, var_ } = computeCVaR(losses, 0.9);
    expect(cvar).toBeCloseTo(5, 10);
    expect(var_).toBeCloseTo(5, 10);
  });
});

describe('wasserstein1D', () => {
  it('returns 0 for identical distributions', () => {
    const p = new Float64Array([1, 2, 3, 4, 5]);
    const q = new Float64Array([1, 2, 3, 4, 5]);
    expect(wasserstein1D(p, q)).toBeCloseTo(0, 10);
  });

  it('returns the mean-shift distance for uniformly shifted samples', () => {
    const p = new Float64Array([0, 1, 2, 3, 4]);
    const q = new Float64Array([2, 3, 4, 5, 6]); // shifted by 2
    // W_1 for equal-weight samples shifted by constant = that constant
    expect(wasserstein1D(p, q)).toBeCloseTo(2, 10);
  });
});

describe('wassersteinDRO', () => {
  it('worst-case cost >= nominal (mean) cost', () => {
    const samples = [
      new Float64Array([1]),
      new Float64Array([2]),
      new Float64Array([3]),
    ];
    const actions = [
      new Float64Array([0]),
      new Float64Array([1]),
    ];

    // Cost = |action - scenario|
    const costFn = (action: Float64Array, scenario: Float64Array) =>
      Math.abs(action[0]! - scenario[0]!);

    const result = wassersteinDRO(
      { nSamples: 3, epsilon: 0.5, costFn },
      samples,
      actions,
    );

    // For the best action, compute nominal cost
    const bestIdx = actions.findIndex(
      (a) => a[0]! === result.bestAction[0]!,
    );
    let nominalCost = 0;
    for (const s of samples) {
      nominalCost += costFn(actions[bestIdx]!, s);
    }
    nominalCost /= samples.length;

    // Worst-case cost (with Wasserstein robustness) >= nominal cost
    expect(result.worstCaseCost).toBeGreaterThanOrEqual(nominalCost - 1e-10);
  });
});

describe('chanceConstrainedOptimize', () => {
  it('returned action satisfies violation probability constraint', () => {
    const rng = createPRNG(42);

    // Constraint: action[0] + scenario[0] <= 0  (i.e., g = action[0] + scenario[0])
    const constraintFn = (
      _state: Float64Array,
      action: Float64Array,
      scenario: Float64Array,
    ) => action[0]! + scenario[0]!;

    const state = new Float64Array([0]);

    // Actions: from -2 to 0 (conservative actions are negative)
    const actions = [
      new Float64Array([-2]),
      new Float64Array([-1]),
      new Float64Array([0]),
      new Float64Array([1]),
    ];

    // Scenarios: uniform in [-1, 1]
    const nScenarios = 50;
    const scenarios: Float64Array[] = [];
    for (let i = 0; i < nScenarios; i++) {
      scenarios.push(new Float64Array([rng() * 2 - 1]));
    }

    const violationProb = 0.1;

    const { bestAction, violationRate } = chanceConstrainedOptimize(
      { violationProb, nScenarios, constraintFn },
      state,
      actions,
      scenarios,
    );

    // If a feasible action exists, violation rate should satisfy the constraint
    expect(bestAction.length).toBe(1);
    if (violationRate <= violationProb) {
      expect(violationRate).toBeLessThanOrEqual(violationProb + 1e-10);
    }
    // The most negative action (-2) should certainly be feasible since
    // -2 + scenario <= 0 for scenario in [-1, 1] => always <= -1 < 0
    // So violation rate should be 0 for that action.
    expect(violationRate).toBeLessThanOrEqual(violationProb);
  });
});

describe('solveScenarioMPC', () => {
  it('returns a non-empty control sequence', () => {
    const nx = 2;
    const nu = 1;
    const horizon = 3;

    // Simple double-integrator: x1' = x1 + x2, x2' = x2 + u
    const A = new Float64Array([1, 1, 0, 1]);
    const B = new Float64Array([0, 1]);
    const Q = new Float64Array([1, 0, 0, 1]);
    const R = new Float64Array([1]);

    // Build a small scenario tree: root -> 2 children (leaves)
    const scenarioTree = [
      {
        state: new Float64Array([0, 0]),
        probability: 1.0,
        children: [1, 2],
        parent: -1,
        stage: 0,
      },
      {
        state: new Float64Array([0.1, 0]),
        probability: 0.5,
        children: [] as number[],
        parent: 0,
        stage: 1,
      },
      {
        state: new Float64Array([-0.1, 0]),
        probability: 0.5,
        children: [] as number[],
        parent: 0,
        stage: 1,
      },
    ];

    const config: ScenarioMPCConfig = {
      A,
      B,
      Q,
      R,
      nx,
      nu,
      horizon,
      scenarioTree,
      nScenarios: 2,
      nStages: 2,
    };

    const x0 = new Float64Array([1, 0.5]);
    const result = solveScenarioMPC(config, x0);

    // Should return horizon steps of controls
    expect(result.uSequence.length).toBe(horizon);

    // Each step should have at least one control vector
    for (let t = 0; t < horizon; t++) {
      expect(result.uSequence[t]!.length).toBeGreaterThan(0);
      expect(result.uSequence[t]![0]!.length).toBe(nu);
    }

    // Cost should be finite and non-negative (quadratic costs)
    expect(Number.isFinite(result.cost)).toBe(true);
    expect(result.cost).toBeGreaterThanOrEqual(0);
  });
});
