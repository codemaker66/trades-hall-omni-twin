// ---------------------------------------------------------------------------
// Hedge (Multiplicative Weights Update) — Full Information Setting
// ---------------------------------------------------------------------------

import type { PRNG } from '../types.js';

/**
 * Create initial Hedge state with uniform weights.
 * @param nExperts Number of experts
 * @param _eta Learning rate (stored externally, not in weights)
 * @returns Uniform weight vector of length nExperts
 */
export function createHedgeState(nExperts: number, _eta: number): number[] {
  const weights = new Array<number>(nExperts);
  for (let i = 0; i < nExperts; i++) {
    weights[i] = 1;
  }
  return weights;
}

/**
 * Compute the Hedge probability distribution from weights.
 * Normalizes the weight vector to sum to 1.
 * @param weights Current weight vector
 * @returns Probability distribution over experts
 */
export function hedgeDistribution(weights: number[]): number[] {
  let total = 0;
  for (let i = 0; i < weights.length; i++) {
    total += (weights[i] ?? 0);
  }
  if (total <= 0) {
    // Fallback to uniform if weights collapsed
    const uniform = new Array<number>(weights.length);
    for (let i = 0; i < weights.length; i++) {
      uniform[i] = 1 / weights.length;
    }
    return uniform;
  }
  const dist = new Array<number>(weights.length);
  for (let i = 0; i < weights.length; i++) {
    dist[i] = (weights[i] ?? 0) / total;
  }
  return dist;
}

/**
 * Update Hedge weights using multiplicative weights rule.
 * w_i *= exp(-eta * loss_i)
 * @param weights Current weight vector
 * @param losses Loss vector for each expert (same length as weights)
 * @param eta Learning rate
 * @returns Updated weight vector
 */
export function hedgeUpdate(weights: number[], losses: number[], eta: number): number[] {
  const updated = new Array<number>(weights.length);
  for (let i = 0; i < weights.length; i++) {
    const w = weights[i] ?? 0;
    const loss = losses[i] ?? 0;
    updated[i] = w * Math.exp(-eta * loss);
  }
  return updated;
}

/**
 * Select an expert by sampling from the Hedge distribution.
 * @param weights Current weight vector
 * @param rng Seedable PRNG
 * @returns Index of selected expert
 */
export function hedgeSelect(weights: number[], rng: PRNG): number {
  const dist = hedgeDistribution(weights);
  const u = rng();
  let cumulative = 0;
  for (let i = 0; i < dist.length; i++) {
    cumulative += (dist[i] ?? 0);
    if (u <= cumulative) {
      return i;
    }
  }
  // Fallback: return last expert (handles floating point rounding)
  return dist.length - 1;
}
