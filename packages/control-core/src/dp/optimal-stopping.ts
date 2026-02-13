// ---------------------------------------------------------------------------
// OC-4  Dynamic Programming -- Optimal Stopping
// ---------------------------------------------------------------------------

import type { PRNG, OptimalStoppingConfig, OptimalStoppingResult } from '../types.js';

// ---------------------------------------------------------------------------
// computeStoppingThresholds
// ---------------------------------------------------------------------------

/**
 * Compute optimal acceptance thresholds via backward induction with
 * Monte Carlo sampling of offer values.
 *
 * The state is (remaining capacity, time period).  At each period the
 * decision-maker observes a random offer value and must either accept
 * (consuming one unit of capacity and receiving the value) or reject.
 *
 * V(c, t) = E[ max{ v + gamma * V(c-1, t+1),  gamma * V(c, t+1) } ]
 *
 * The threshold at time t is the minimum offer value that should be
 * accepted (with full capacity), i.e. the breakeven value where accepting
 * and continuing have equal expected payoff.
 *
 * Thresholds decrease as remaining capacity increases (more room to accept).
 *
 * @param config    Optimal stopping problem specification.
 * @param rng       Seeded PRNG for Monte Carlo value sampling.
 * @param nSamples  Number of MC samples per (c, t) pair (default 500).
 */
export function computeStoppingThresholds(
  config: OptimalStoppingConfig,
  rng: PRNG,
  nSamples = 500,
): OptimalStoppingResult {
  const { capacity, horizon, arrivalProb, valueDistribution, discount } = config;

  // V[c][t] -- value function.
  // c in [0, capacity], t in [0, horizon].
  // Flatten as V[(capacity+1) * t + c].
  const cap1 = capacity + 1;
  const V = new Float64Array(cap1 * (horizon + 1));

  // Terminal values: V(c, horizon) = 0 for all c  (already zero-initialised)

  // Backward induction from t = horizon-1 down to t = 0
  for (let t = horizon - 1; t >= 0; t--) {
    for (let c = 0; c <= capacity; c++) {
      if (c === 0) {
        // No capacity left -- can only reject
        V[cap1 * t + c] = discount * V[cap1 * (t + 1) + c]!;
        continue;
      }

      const pArrive = arrivalProb(t);
      let totalValue = 0;

      for (let s = 0; s < nSamples; s++) {
        const arrives = rng() < pArrive;

        if (!arrives) {
          // No arrival this period
          totalValue += discount * V[cap1 * (t + 1) + c]!;
        } else {
          const v = valueDistribution(t, rng);
          const acceptValue = v + discount * V[cap1 * (t + 1) + (c - 1)]!;
          const rejectValue = discount * V[cap1 * (t + 1) + c]!;
          totalValue += Math.max(acceptValue, rejectValue);
        }
      }

      V[cap1 * t + c] = totalValue / nSamples;
    }
  }

  // Extract thresholds for full capacity, per period.
  // threshold(t) = opportunity cost of consuming one unit at time t
  //              = gamma * V(cap, t+1) - gamma * V(cap-1, t+1)
  const thresholds = new Float64Array(horizon);
  const valueFunction = new Float64Array(horizon);

  for (let t = 0; t < horizon; t++) {
    valueFunction[t] = V[cap1 * t + capacity]!;

    if (t < horizon - 1) {
      const rejectFuture = discount * V[cap1 * (t + 1) + capacity]!;
      const acceptFuture = discount * V[cap1 * (t + 1) + (capacity - 1)]!;
      thresholds[t] = rejectFuture - acceptFuture;
    } else {
      // Last period: accept anything with non-negative value
      thresholds[t] = 0;
    }

    // Clamp to non-negative
    if (thresholds[t]! < 0) thresholds[t] = 0;
  }

  return { thresholds, valueFunction };
}

// ---------------------------------------------------------------------------
// shouldAccept
// ---------------------------------------------------------------------------

/**
 * Real-time acceptance decision using pre-computed thresholds.
 *
 * The base threshold (computed at full capacity) is scaled down linearly
 * with remaining capacity: more capacity => lower threshold (more room to
 * accept), less capacity => higher threshold (must be more selective).
 *
 * @param thresholds        Pre-computed acceptance thresholds per period (horizon).
 * @param period            Current time period index (0-based).
 * @param remainingCapacity Current remaining capacity (>= 1 to be able to accept).
 * @param value             Value of the current offer.
 * @returns `true` if the offer should be accepted.
 */
export function shouldAccept(
  thresholds: Float64Array,
  period: number,
  remainingCapacity: number,
  value: number,
): boolean {
  if (remainingCapacity <= 0) return false;
  if (period < 0 || period >= thresholds.length) return false;

  const baseThreshold = thresholds[period]!;

  // Scale threshold inversely with capacity: scarce capacity raises the bar.
  // With capacity == 1 the full threshold applies; with higher capacity the
  // threshold is reduced proportionally.
  const adjustedThreshold = baseThreshold / remainingCapacity;

  return value >= adjustedThreshold;
}
