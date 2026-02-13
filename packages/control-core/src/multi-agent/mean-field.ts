// ---------------------------------------------------------------------------
// OC-7: Mean-Field Game Approximation
// ---------------------------------------------------------------------------

import type { MeanFieldConfig } from '../types.js';
import { vecNorm, vecSub } from '../types.js';

// ---------------------------------------------------------------------------
// meanFieldStep
// ---------------------------------------------------------------------------

/**
 * Advance one time step in a mean-field game.
 *
 * Computes the population distribution as the element-wise mean of all agent
 * states, then advances each agent's state according to the dynamics function
 * f(x, u, distribution).
 *
 * @param config   Mean-field configuration with agent dynamics and state/control dims.
 * @param states   Current state vectors for all agents.
 * @param actions  Current control inputs for all agents.
 * @param dt       Time step size.
 * @returns        Updated state vectors after one step.
 */
export function meanFieldStep(
  config: MeanFieldConfig,
  states: Float64Array[],
  actions: Float64Array[],
  dt: number,
): Float64Array[] {
  const { agentDynamics, nx } = config;
  const nAgents = states.length;

  // Compute population distribution: element-wise mean of all states
  const distribution = new Float64Array(nx);
  for (let a = 0; a < nAgents; a++) {
    const s = states[a]!;
    for (let i = 0; i < nx; i++) {
      distribution[i] = distribution[i]! + s[i]!;
    }
  }
  for (let i = 0; i < nx; i++) {
    distribution[i] = distribution[i]! / nAgents;
  }

  // Advance each agent using Euler integration
  const nextStates: Float64Array[] = [];
  for (let a = 0; a < nAgents; a++) {
    const x = states[a]!;
    const u = actions[a]!;
    const dxdt = agentDynamics(x, u, distribution);
    const xNext = new Float64Array(nx);
    for (let i = 0; i < nx; i++) {
      xNext[i] = x[i]! + dt * dxdt[i]!;
    }
    nextStates.push(xNext);
  }

  return nextStates;
}

// ---------------------------------------------------------------------------
// meanFieldEquilibrium
// ---------------------------------------------------------------------------

/**
 * Find a mean-field equilibrium via fixed-point iteration on the population
 * distribution.
 *
 * Starting from the initial states, the algorithm repeatedly:
 *   1. Computes the population distribution (mean of states).
 *   2. Advances each agent forward using zero control and the current
 *      distribution.
 *   3. Checks whether the distribution has converged (L2 norm of change
 *      below tolerance).
 *
 * The zero-control assumption is a simplification; in a full implementation,
 * agents would optimize their controls given the distribution (forward-backward
 * sweep). Here we iterate the forward dynamics to find a steady-state
 * distribution.
 *
 * @param config         Mean-field configuration.
 * @param initialStates  Starting state vectors for all agents.
 * @param dt             Time step for dynamics integration.
 * @param maxIter        Maximum fixed-point iterations (default 100).
 * @param tol            Convergence tolerance on distribution change (default 1e-6).
 * @returns              Converged states and population distribution.
 */
export function meanFieldEquilibrium(
  config: MeanFieldConfig,
  initialStates: Float64Array[],
  dt: number,
  maxIter: number = 100,
  tol: number = 1e-6,
): { states: Float64Array[]; distribution: Float64Array } {
  const { nx, nu } = config;
  const nAgents = initialStates.length;

  let states: Float64Array[] = [];
  for (let a = 0; a < nAgents; a++) {
    states.push(new Float64Array(initialStates[a]!));
  }

  // Zero control for the simplified fixed-point iteration
  const zeroActions: Float64Array[] = [];
  for (let a = 0; a < nAgents; a++) {
    zeroActions.push(new Float64Array(nu));
  }

  // Compute initial distribution
  let distribution = computeDistribution(states, nx);

  for (let iter = 0; iter < maxIter; iter++) {
    // Advance one step
    const nextStates = meanFieldStep(config, states, zeroActions, dt);

    // Compute new distribution
    const newDistribution = computeDistribution(nextStates, nx);

    // Check convergence
    const diff = vecSub(newDistribution, distribution);
    const change = vecNorm(diff);

    states = nextStates;
    distribution = newDistribution;

    if (change < tol) {
      break;
    }
  }

  return { states, distribution };
}

// ---------------------------------------------------------------------------
// computeDistribution (internal helper)
// ---------------------------------------------------------------------------

/**
 * Compute the population distribution as the element-wise mean of all agent
 * state vectors.
 */
function computeDistribution(states: Float64Array[], nx: number): Float64Array {
  const nAgents = states.length;
  const dist = new Float64Array(nx);

  for (let a = 0; a < nAgents; a++) {
    const s = states[a]!;
    for (let i = 0; i < nx; i++) {
      dist[i] = dist[i]! + s[i]!;
    }
  }
  for (let i = 0; i < nx; i++) {
    dist[i] = dist[i]! / nAgents;
  }

  return dist;
}
