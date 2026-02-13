// ---------------------------------------------------------------------------
// OC-5: Reward Shaping for Venue Operations
// ---------------------------------------------------------------------------

import type { RewardConfig } from '../types.js';

// ---------------------------------------------------------------------------
// Venue Reward Function
// ---------------------------------------------------------------------------

/**
 * Compute the composite reward for venue operations.
 *
 * The reward is a weighted sum of multiple objectives:
 *
 *   R = revenueWeight * revenue
 *     - overcrowdingPenalty * max(0, occupancy - overcrowdingSafe)^2
 *     - priceStabilityPenalty * |priceChange|
 *     - queuePenalty * max(0, queueLength - queueMax)
 *     - churnPenalty * churnRate
 *
 * - Revenue bonus: linear reward proportional to revenue.
 * - Overcrowding penalty: quadratic penalty when occupancy exceeds the
 *   safe density threshold, encouraging safe crowd levels.
 * - Price stability penalty: penalises large absolute price changes
 *   to avoid jarring price swings for customers.
 * - Queue penalty: linear penalty for queues exceeding the acceptable max.
 * - Churn penalty: linear penalty for customer churn rate.
 *
 * @param config       Reward shaping weights and thresholds.
 * @param revenue      Current period revenue.
 * @param occupancy    Current occupancy density (e.g., people/m^2).
 * @param priceChange  Absolute price change from previous period.
 * @param queueLength  Current queue length.
 * @param churnRate    Customer churn rate in [0, 1].
 * @returns            Scalar composite reward.
 */
export function computeVenueReward(
  config: RewardConfig,
  revenue: number,
  occupancy: number,
  priceChange: number,
  queueLength: number,
  churnRate: number,
): number {
  // Revenue bonus
  let reward = config.revenueWeight * revenue;

  // Overcrowding penalty (quadratic beyond safe threshold)
  const overcrowdingExcess = Math.max(0, occupancy - config.overcrowdingSafe);
  reward -= config.overcrowdingPenalty * overcrowdingExcess * overcrowdingExcess;

  // Price stability penalty (absolute change)
  reward -= config.priceStabilityPenalty * Math.abs(priceChange);

  // Queue penalty (linear beyond max acceptable queue)
  const queueExcess = Math.max(0, queueLength - config.queueMax);
  reward -= config.queuePenalty * queueExcess;

  // Churn penalty
  reward -= config.churnPenalty * churnRate;

  return reward;
}

// ---------------------------------------------------------------------------
// Potential-Based Reward Shaping
// ---------------------------------------------------------------------------

/**
 * Compute potential-based reward shaping that preserves the optimal policy.
 *
 * The shaping reward is:
 *
 *   F(s, s') = gamma * Phi(s') - Phi(s)
 *
 * where Phi is an arbitrary potential function over states. This form is
 * guaranteed to preserve the optimal policy (Ng, Harada & Russell, 1999).
 *
 * @param prevPotential  Potential Phi(s) at the previous state.
 * @param currPotential  Potential Phi(s') at the current state.
 * @param gamma          Discount factor.
 * @returns              Scalar shaping bonus to add to the environment reward.
 */
export function potentialShaping(
  prevPotential: number,
  currPotential: number,
  gamma: number,
): number {
  return gamma * currPotential - prevPotential;
}
