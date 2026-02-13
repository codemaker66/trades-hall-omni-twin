// ---------------------------------------------------------------------------
// EXP3 — Adversarial Multi-Armed Bandits
// ---------------------------------------------------------------------------

import type { PRNG } from '../types.js';

/**
 * Create initial EXP3 state with uniform weights.
 * @param nArms Number of arms
 * @param _gamma Exploration parameter (stored externally)
 * @returns Uniform weight vector of length nArms
 */
export function createEXP3State(nArms: number, _gamma: number): number[] {
  const weights = new Array<number>(nArms);
  for (let i = 0; i < nArms; i++) {
    weights[i] = 1;
  }
  return weights;
}

/**
 * Compute the EXP3 mixed strategy probability distribution.
 * p_i = (1 - gamma) * (w_i / sum(w)) + gamma / K
 *
 * @param weights Current weight vector
 * @param gamma Exploration parameter in [0, 1]
 * @returns Probability distribution over arms
 */
function exp3Distribution(weights: number[], gamma: number): number[] {
  const K = weights.length;
  let totalWeight = 0;
  for (let i = 0; i < K; i++) {
    totalWeight += (weights[i] ?? 0);
  }

  if (totalWeight <= 0) {
    // Fallback to uniform
    const uniform = new Array<number>(K);
    for (let i = 0; i < K; i++) {
      uniform[i] = 1 / K;
    }
    return uniform;
  }

  const dist = new Array<number>(K);
  for (let i = 0; i < K; i++) {
    dist[i] = (1 - gamma) * ((weights[i] ?? 0) / totalWeight) + gamma / K;
  }
  return dist;
}

/**
 * Select an arm using EXP3's mixed strategy.
 * @param weights Current weight vector
 * @param gamma Exploration parameter
 * @param rng Seedable PRNG
 * @returns Index of selected arm
 */
export function exp3Select(weights: number[], gamma: number, rng: PRNG): number {
  const dist = exp3Distribution(weights, gamma);
  const u = rng();
  let cumulative = 0;
  for (let i = 0; i < dist.length; i++) {
    cumulative += (dist[i] ?? 0);
    if (u <= cumulative) {
      return i;
    }
  }
  // Fallback for floating point rounding
  return dist.length - 1;
}

/**
 * Update EXP3 weights using importance-weighted feedback.
 *
 * The estimated reward is: reward_hat = reward / p_i
 * Weight update: w_i *= exp(gamma * reward_hat / K)
 *
 * @param weights Current weight vector
 * @param arm Index of the arm that was played
 * @param reward Observed reward for the played arm (in [0, 1])
 * @param prob Probability with which the arm was selected
 * @param gamma Exploration parameter
 * @returns Updated weight vector
 */
export function exp3Update(
  weights: number[],
  arm: number,
  reward: number,
  prob: number,
  gamma: number,
): number[] {
  const K = weights.length;
  const updated = new Array<number>(K);

  for (let i = 0; i < K; i++) {
    const w = weights[i] ?? 0;
    if (i === arm) {
      // Importance-weighted reward estimate
      const estimatedReward = reward / prob;
      updated[i] = w * Math.exp((gamma * estimatedReward) / K);
    } else {
      updated[i] = w;
    }
  }

  return updated;
}
