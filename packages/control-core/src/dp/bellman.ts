// ---------------------------------------------------------------------------
// OC-4  Dynamic Programming -- Bellman Equation Solvers
// ---------------------------------------------------------------------------

import type { BellmanConfig, BellmanResult } from '../types.js';

// ---------------------------------------------------------------------------
// Value Iteration
// ---------------------------------------------------------------------------

/**
 * Standard value iteration for discrete MDPs.
 *
 * V_{k+1}(s) = max_a { r(s,a) + gamma * sum_s' P(s'|s,a) V_k(s') }
 *
 * Iterates until the max absolute change in V falls below `tolerance`
 * or `maxIter` iterations are reached.
 */
export function valueIteration(
  config: BellmanConfig,
  tolerance = 1e-8,
  maxIter = 1000,
): BellmanResult {
  const { nStates, nActions, transitions, reward, discount } = config;

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
        const r = reward(s, a);
        const trans = transitions(s, a);

        let futureValue = 0;
        for (const { nextState, probability } of trans) {
          futureValue += probability * V[nextState]!;
        }

        const qValue = r + discount * futureValue;
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

    if (maxDelta < tolerance) break;
  }

  return { valueFunction: V, policy, iterations };
}

// ---------------------------------------------------------------------------
// Policy Iteration
// ---------------------------------------------------------------------------

/**
 * Policy iteration: alternate between policy evaluation (solve linear system)
 * and policy improvement until the policy stabilises.
 */
export function policyIteration(
  config: BellmanConfig,
  maxIter = 100,
): BellmanResult {
  const { nStates, nActions, transitions, reward, discount } = config;

  // Initialise with a greedy policy using zero value
  const policy = new Int32Array(nStates); // action 0 everywhere
  let V: Float64Array = new Float64Array(nStates);
  let iterations = 0;

  for (let iter = 0; iter < maxIter; iter++) {
    // ---- Policy Evaluation (iterative, since we avoid building a full matrix) ----
    V = evaluatePolicy(nStates, policy, transitions, reward, discount);

    // ---- Policy Improvement ----
    let stable = true;
    for (let s = 0; s < nStates; s++) {
      const oldAction = policy[s]!;
      let bestValue = -Infinity;
      let bestAction = 0;

      for (let a = 0; a < nActions; a++) {
        const r = reward(s, a);
        const trans = transitions(s, a);
        let futureValue = 0;
        for (const { nextState, probability } of trans) {
          futureValue += probability * V[nextState]!;
        }
        const qValue = r + discount * futureValue;
        if (qValue > bestValue) {
          bestValue = qValue;
          bestAction = a;
        }
      }

      policy[s] = bestAction;
      if (bestAction !== oldAction) stable = false;
    }

    iterations = iter + 1;
    if (stable) break;
  }

  return { valueFunction: V, policy, iterations };
}

/**
 * Iterative policy evaluation: compute V^pi by repeatedly applying the
 * Bellman expectation equation until convergence.
 */
function evaluatePolicy(
  nStates: number,
  policy: Int32Array,
  transitions: BellmanConfig['transitions'],
  reward: BellmanConfig['reward'],
  discount: number,
  tolerance = 1e-10,
  maxEvalIter = 500,
) {
  let V = new Float64Array(nStates);

  for (let iter = 0; iter < maxEvalIter; iter++) {
    const Vnext = new Float64Array(nStates);
    let maxDelta = 0;

    for (let s = 0; s < nStates; s++) {
      const a = policy[s]!;
      const r = reward(s, a);
      const trans = transitions(s, a);

      let futureValue = 0;
      for (const { nextState, probability } of trans) {
        futureValue += probability * V[nextState]!;
      }

      Vnext[s] = r + discount * futureValue;
      const delta = Math.abs(Vnext[s]! - V[s]!);
      if (delta > maxDelta) maxDelta = delta;
    }

    V = Vnext;
    if (maxDelta < tolerance) break;
  }

  return V;
}

// ---------------------------------------------------------------------------
// Backward Induction (Finite Horizon)
// ---------------------------------------------------------------------------

/**
 * Finite-horizon backward induction.
 *
 * Computes the optimal value function and policy for the initial time step
 * by working backward from the terminal time t = H to t = 0.
 *
 * Requires `config.horizon` to be set to a positive integer.
 * Returns V_0(s) and the optimal action at t = 0.
 */
export function backwardInduction(config: BellmanConfig): BellmanResult {
  const { nStates, nActions, transitions, reward, discount, horizon } = config;

  if (horizon === undefined || horizon <= 0) {
    throw new Error('backwardInduction requires a finite positive horizon');
  }

  const H = horizon;

  // Terminal value is zero
  let Vnext = new Float64Array(nStates);

  // Track the t = 0 policy
  let policy = new Int32Array(nStates);
  let V = new Float64Array(nStates);

  // Work backward from t = H-1 to t = 0
  for (let t = H - 1; t >= 0; t--) {
    V = new Float64Array(nStates);
    const pi = new Int32Array(nStates);

    for (let s = 0; s < nStates; s++) {
      let bestValue = -Infinity;
      let bestAction = 0;

      for (let a = 0; a < nActions; a++) {
        const r = reward(s, a);
        const trans = transitions(s, a);

        let futureValue = 0;
        for (const { nextState, probability } of trans) {
          futureValue += probability * Vnext[nextState]!;
        }

        const qValue = r + discount * futureValue;
        if (qValue > bestValue) {
          bestValue = qValue;
          bestAction = a;
        }
      }

      V[s] = bestValue;
      pi[s] = bestAction;
    }

    Vnext = V;
    policy = pi;
  }

  return { valueFunction: V, policy, iterations: H };
}
