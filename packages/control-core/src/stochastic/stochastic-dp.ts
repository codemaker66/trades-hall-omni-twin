// ---------------------------------------------------------------------------
// OC-9  Stochastic Optimal Control -- Stochastic Dynamic Programming
// ---------------------------------------------------------------------------

import type { BellmanConfig } from '../types.js';

// ---------------------------------------------------------------------------
// Stochastic Value Iteration
// ---------------------------------------------------------------------------

/**
 * Stochastic value iteration with explicit scenario-weighted Bellman backup.
 *
 * Extends standard VI by incorporating exogenous uncertainty through
 * scenario-dependent transitions and rewards:
 *
 *   V(s) = max_a  sum_w  p(w) [ r(s,a,w) + gamma * V(s') ]
 *
 * where w indexes scenarios drawn from a finite probability distribution,
 * and s' = scenarioTransitions(s, a, w).nextState.
 *
 * @param config        - Base Bellman configuration (nStates, nActions, discount, etc.)
 * @param nScenarios    - Number of exogenous scenarios
 * @param scenarioProbs - Probability of each scenario (must sum to 1)
 * @param scenarioTransitions - (s, a, w) => { nextState, reward } for scenario w
 * @param tol           - Convergence tolerance (default 1e-8)
 * @param maxIter       - Maximum iterations (default 1000)
 * @returns Value function V, greedy policy, and iteration count
 */
export function stochasticValueIteration(
  config: BellmanConfig,
  nScenarios: number,
  scenarioProbs: Float64Array,
  scenarioTransitions: (
    s: number,
    a: number,
    w: number,
  ) => { nextState: number; reward: number },
  tol = 1e-8,
  maxIter = 1000,
): { V: Float64Array; policy: Int32Array; iterations: number } {
  const { nStates, nActions, discount } = config;

  let V = new Float64Array(nStates);
  const policy = new Int32Array(nStates);
  let iterations = 0;

  for (let iter = 0; iter < maxIter; iter++) {
    const Vnext = new Float64Array(nStates);
    let maxDelta = 0;

    for (let s = 0; s < nStates; s++) {
      let bestValue = -Infinity;
      let bestAction = 0;

      for (let a = 0; a < nActions; a++) {
        // Scenario-weighted Bellman backup
        let qValue = 0;

        for (let w = 0; w < nScenarios; w++) {
          const pw = scenarioProbs[w]!;
          const { nextState, reward } = scenarioTransitions(s, a, w);
          qValue += pw * (reward + discount * V[nextState]!);
        }

        if (qValue > bestValue) {
          bestValue = qValue;
          bestAction = a;
        }
      }

      Vnext[s] = bestValue;
      policy[s] = bestAction;

      const delta = Math.abs(bestValue - V[s]!);
      if (delta > maxDelta) maxDelta = delta;
    }

    V = Vnext;
    iterations = iter + 1;

    if (maxDelta < tol) break;
  }

  return { V, policy, iterations };
}
