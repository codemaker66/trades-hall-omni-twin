// ---------------------------------------------------------------------------
// Thompson Sampling with Beta-Bernoulli Model
// ---------------------------------------------------------------------------

import type { PRNG, ThompsonState } from '../types.js';
import { betaSample } from '../types.js';

/**
 * Create initial Thompson Sampling state.
 * Each arm starts with Beta(1, 1) = Uniform(0, 1) prior.
 *
 * @param nArms Number of arms
 * @returns Initial Thompson state
 */
export function createThompsonState(nArms: number): ThompsonState {
  const alpha = new Array<number>(nArms);
  const beta = new Array<number>(nArms);
  for (let i = 0; i < nArms; i++) {
    alpha[i] = 1; // Beta(1, 1) = Uniform prior
    beta[i] = 1;
  }
  return { alpha, beta };
}

/**
 * Select an arm using Thompson Sampling.
 * For each arm, sample theta_i ~ Beta(alpha_i, beta_i), then pick argmax.
 *
 * @param state Current Thompson state
 * @param rng Seedable PRNG
 * @returns Index of the selected arm
 */
export function thompsonSelect(state: ThompsonState, rng: PRNG): number {
  let bestArm = 0;
  let bestSample = -Infinity;

  for (let i = 0; i < state.alpha.length; i++) {
    const a = state.alpha[i] ?? 1;
    const b = state.beta[i] ?? 1;
    const sample = betaSample(a, b, rng);

    if (sample > bestSample) {
      bestSample = sample;
      bestArm = i;
    }
  }

  return bestArm;
}

/**
 * Update Thompson state after observing a reward.
 *
 * For Bernoulli rewards:
 *   If reward >= 0.5 (success): alpha_arm += 1
 *   If reward < 0.5 (failure): beta_arm += 1
 *
 * For continuous rewards in [0, 1], we interpret reward as the probability
 * of success and update proportionally:
 *   alpha_arm += reward
 *   beta_arm += (1 - reward)
 *
 * This implements the continuous fractional update for flexibility.
 *
 * @param state Current Thompson state
 * @param arm Index of the arm that was played
 * @param reward Observed reward in [0, 1]
 * @returns Updated Thompson state (immutable)
 */
export function thompsonUpdate(state: ThompsonState, arm: number, reward: number): ThompsonState {
  const newAlpha = new Array<number>(state.alpha.length);
  const newBeta = new Array<number>(state.beta.length);

  for (let i = 0; i < state.alpha.length; i++) {
    newAlpha[i] = state.alpha[i] ?? 1;
    newBeta[i] = state.beta[i] ?? 1;
  }

  // Clamp reward to [0, 1]
  const r = Math.max(0, Math.min(1, reward));
  newAlpha[arm] = (newAlpha[arm] ?? 1) + r;
  newBeta[arm] = (newBeta[arm] ?? 1) + (1 - r);

  return { alpha: newAlpha, beta: newBeta };
}
