/**
 * Inverse Optimal Transport (OT-7).
 *
 * Learn the cost function weights from observed successful matchings
 * (Li et al. 2018, arXiv:1802.03644).
 *
 * Given observed venue-event matchings, find cost weights w* that minimize:
 *   ||T_OT(C(w)) - T_obs||²
 *
 * This creates a data network effect: the more bookings the platform processes,
 * the better the matching becomes.
 */

import type {
  CostWeights,
  EventRequirements,
  VenueFeatures,
  ObservedMatching,
  InverseOTConfig,
} from './types'
import { DEFAULT_COST_WEIGHTS, DEFAULT_INVERSE_OT_CONFIG } from './types'
import { buildCostMatrix } from './cost-matrix'
import { sinkhorn } from './sinkhorn'
import { normalizeDistribution } from './utils'

/**
 * Build the "observed" transport plan from historical matchings.
 * Each observed matching (event i, venue j) contributes a mass to T_obs[i,j].
 * The plan is normalized so row sums ≈ source distribution.
 */
export function buildObservedPlan(
  matchings: ObservedMatching[],
  N: number,
  M: number,
  successWeight: number = 1.0,
  failureWeight: number = 0.1,
): Float64Array {
  const T = new Float64Array(N * M)

  for (const match of matchings) {
    const weight = match.success ? successWeight : failureWeight
    T[match.eventIndex * M + match.venueIndex]! += weight
  }

  // Normalize rows to sum to 1
  for (let i = 0; i < N; i++) {
    let rowSum = 0
    for (let j = 0; j < M; j++) {
      rowSum += T[i * M + j]!
    }
    if (rowSum > 0) {
      for (let j = 0; j < M; j++) {
        T[i * M + j] = T[i * M + j]! / rowSum
      }
    } else {
      // Uniform for unmatched events
      for (let j = 0; j < M; j++) {
        T[i * M + j] = 1 / M
      }
    }
  }

  return T
}

/**
 * Compute squared Frobenius loss between predicted and observed plans.
 */
function planLoss(predicted: Float64Array, observed: Float64Array): number {
  let loss = 0
  for (let k = 0; k < predicted.length; k++) {
    const diff = predicted[k]! - observed[k]!
    loss += diff * diff
  }
  return loss
}

/**
 * Clamp cost weights to valid range [0.01, 1] and normalize to sum to 1.
 */
function clampAndNormalizeWeights(w: CostWeights): CostWeights {
  const clamp = (v: number) => Math.max(0.01, Math.min(1, v))
  const c = clamp(w.capacity)
  const p = clamp(w.price)
  const a = clamp(w.amenity)
  const l = clamp(w.location)
  const total = c + p + a + l
  return {
    capacity: c / total,
    price: p / total,
    amenity: a / total,
    location: l / total,
  }
}

/**
 * Learn optimal cost weights from observed matchings via gradient descent.
 *
 * Uses finite differences to compute gradients (simpler than backprop through
 * Sinkhorn, and adequate for 4 parameters).
 *
 * @param matchings - Historical event-venue matchings with success flags
 * @param events - Event feature vectors
 * @param venues - Venue feature vectors
 * @param initialWeights - Starting weights (default: equal)
 * @param config - Learning configuration
 */
export function learnCostWeights(
  matchings: ObservedMatching[],
  events: EventRequirements[],
  venues: VenueFeatures[],
  initialWeights: CostWeights = DEFAULT_COST_WEIGHTS,
  config: Partial<InverseOTConfig> = {},
): CostWeights {
  const cfg = { ...DEFAULT_INVERSE_OT_CONFIG, ...config }
  const N = events.length
  const M = venues.length

  if (N === 0 || M === 0 || matchings.length === 0) {
    return initialWeights
  }

  // Build observed plan
  const observedPlan = buildObservedPlan(matchings, N, M)

  // Build uniform marginals
  const a = new Float64Array(N).fill(1 / N)
  const b = new Float64Array(M).fill(1 / M)

  let weights = { ...initialWeights }
  const h = cfg.finiteDiffStep

  for (let iter = 0; iter < cfg.iterations; iter++) {
    // Current loss
    const C = buildCostMatrix(events, venues, weights)
    const result = sinkhorn(a, b, C, { epsilon: cfg.epsilon })
    const currentLoss = planLoss(result.plan, observedPlan)

    // Compute gradient via finite differences for each weight dimension
    const grad: CostWeights = { capacity: 0, price: 0, amenity: 0, location: 0 }

    for (const dim of ['capacity', 'price', 'amenity', 'location'] as const) {
      const wPlus = { ...weights, [dim]: weights[dim] + h }
      const CPlus = buildCostMatrix(events, venues, wPlus)
      const resultPlus = sinkhorn(a, b, CPlus, { epsilon: cfg.epsilon })
      const lossPlus = planLoss(resultPlus.plan, observedPlan)

      grad[dim] = (lossPlus - currentLoss) / h
    }

    // Gradient descent step
    weights = {
      capacity: weights.capacity - cfg.learningRate * grad.capacity,
      price: weights.price - cfg.learningRate * grad.price,
      amenity: weights.amenity - cfg.learningRate * grad.amenity,
      location: weights.location - cfg.learningRate * grad.location,
    }

    // Project back to valid range
    weights = clampAndNormalizeWeights(weights)
  }

  return weights
}

/**
 * Evaluate how well learned weights predict observed matchings.
 * Returns a score in [0, 1] where 1 = perfect prediction.
 *
 * @param matchings - Test set of observed matchings
 * @param events - Event features
 * @param venues - Venue features
 * @param weights - Learned weights to evaluate
 * @param epsilon - Sinkhorn regularization
 */
export function evaluateWeights(
  matchings: ObservedMatching[],
  events: EventRequirements[],
  venues: VenueFeatures[],
  weights: CostWeights,
  epsilon: number = 0.05,
): number {
  const N = events.length
  const M = venues.length

  if (N === 0 || M === 0 || matchings.length === 0) return 0

  const observedPlan = buildObservedPlan(matchings, N, M)
  const a = new Float64Array(N).fill(1 / N)
  const b = new Float64Array(M).fill(1 / M)

  const C = buildCostMatrix(events, venues, weights)
  const result = sinkhorn(a, b, C, { epsilon })
  const loss = planLoss(result.plan, observedPlan)

  // Convert loss to [0,1] score (exponential decay)
  return Math.exp(-loss * N * M)
}
