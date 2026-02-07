/**
 * Heterogeneous cost matrix builder (OT-2).
 *
 * Handles mixed feature types with asymmetric penalties:
 * - Capacity: severe penalty for venue too small, mild for too large
 * - Price: painful for over budget, mild for under budget
 * - Amenities: missing required = penalty, extra = no penalty
 * - Location: Haversine (great-circle distance)
 *
 * Also: Sinkhorn Divergence (Feydy et al. 2019, arXiv:1810.08278)
 */

import type { VenueFeatures, EventRequirements, CostWeights } from './types'
import { DEFAULT_COST_WEIGHTS } from './types'
import { normalize01, toRad } from './utils'
import { sinkhornCost } from './sinkhorn'

// ─── Feature Distance Functions ────────────────────────────────────────────

/**
 * Capacity distance — ASYMMETRIC:
 * Venue too small = severe penalty (2.0× normalized shortfall)
 * Venue too large = mild penalty (0.3× normalized excess)
 */
export function capacityDistance(eventGuests: number, venueCapacity: number): number {
  if (eventGuests <= 0) return 0
  if (venueCapacity < eventGuests) {
    return 2.0 * (eventGuests - venueCapacity) / eventGuests
  }
  return 0.3 * (venueCapacity - eventGuests) / eventGuests
}

/**
 * Amenity distance — ASYMMETRIC:
 * Missing REQUIRED amenities = penalty proportional to fraction missing
 * Extra amenities = no penalty
 */
export function amenityDistance(required: boolean[], available: boolean[]): number {
  let missing = 0
  let totalRequired = 0
  const len = Math.min(required.length, available.length)
  for (let i = 0; i < len; i++) {
    if (required[i]) {
      totalRequired++
      if (!available[i]) missing++
    }
  }
  // Also count required beyond available length
  for (let i = len; i < required.length; i++) {
    if (required[i]) {
      totalRequired++
      missing++
    }
  }
  return totalRequired > 0 ? missing / totalRequired : 0
}

/**
 * Location distance — Haversine (great-circle distance in km).
 */
export function locationDistance(
  a: { lat: number; lng: number },
  b: { lat: number; lng: number },
): number {
  const R = 6371.0 // Earth radius in km
  const dLat = toRad(b.lat - a.lat)
  const dLon = toRad(b.lng - a.lng)
  const sinLat = Math.sin(dLat / 2)
  const sinLon = Math.sin(dLon / 2)
  const h = sinLat * sinLat
    + Math.cos(toRad(a.lat)) * Math.cos(toRad(b.lat)) * sinLon * sinLon
  return 2 * R * Math.asin(Math.sqrt(h))
}

/**
 * Price distance — ASYMMETRIC:
 * Over budget = 1.5× normalized overage (painful), capped at 3.0
 * Under budget = 0.1× normalized savings (mild)
 */
export function priceDistance(budget: number, venuePrice: number): number {
  if (budget <= 0) return 0
  const diff = venuePrice - budget
  if (diff > 0) return Math.min(1.5 * diff / budget, 3.0)
  return 0.1 * Math.abs(diff) / budget
}

// ─── Cost Matrix Builder ───────────────────────────────────────────────────

/**
 * Build the full N×M cost matrix from events × venues.
 *
 * Each per-feature distance matrix is normalized to [0,1], then combined
 * with configurable weights.
 */
export function buildCostMatrix(
  events: EventRequirements[],
  venues: VenueFeatures[],
  weights: CostWeights = DEFAULT_COST_WEIGHTS,
): Float64Array {
  const N = events.length
  const M = venues.length

  // Compute per-feature distance matrices
  const capDist = new Float64Array(N * M)
  const priceDist = new Float64Array(N * M)
  const amenDist = new Float64Array(N * M)
  const locDist = new Float64Array(N * M)

  for (let i = 0; i < N; i++) {
    const ev = events[i]!
    for (let j = 0; j < M; j++) {
      const ve = venues[j]!
      const idx = i * M + j
      capDist[idx] = capacityDistance(ev.guestCount, ve.capacity)
      priceDist[idx] = priceDistance(ev.budget, ve.pricePerEvent)
      amenDist[idx] = amenityDistance(ev.requiredAmenities, ve.amenities)
      locDist[idx] = locationDistance(ev.preferredLocation, ve.location)
    }
  }

  // Normalize each to [0, 1]
  normalize01(capDist)
  normalize01(priceDist)
  normalize01(amenDist)
  normalize01(locDist)

  // Weighted combination
  const C = new Float64Array(N * M)
  for (let k = 0; k < N * M; k++) {
    C[k] = weights.capacity * capDist[k]!
      + weights.price * priceDist[k]!
      + weights.amenity * amenDist[k]!
      + weights.location * locDist[k]!
  }

  return C
}

/**
 * Build a self-cost matrix (n×n) for a set of distributions on the same support.
 * Used for Sinkhorn divergence computation.
 * Here we use squared Euclidean distance on the support indices.
 */
export function buildSelfCostMatrix(n: number): Float64Array {
  const C = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const diff = i - j
      C[i * n + j] = diff * diff
    }
  }
  // Normalize
  normalize01(C)
  return C
}

// ─── Sinkhorn Divergence (Debiased) ────────────────────────────────────────

/**
 * Sinkhorn Divergence: S_ε(a,b) = OT_ε(a,b) - ½·OT_ε(a,a) - ½·OT_ε(b,b)
 *
 * Properties (Feydy et al. 2019, arXiv:1810.08278):
 * - S_ε(a,a) = 0 (identity of indiscernibles)
 * - S_ε(a,b) ≥ 0 (positive definite)
 * - Converges to W(a,b) as ε → 0
 * - Better behaved as a loss function than raw Sinkhorn distance
 *
 * ALWAYS use this instead of raw sinkhornCost for scoring/ranking.
 */
export function sinkhornDivergence(
  a: Float64Array,
  b: Float64Array,
  C_ab: Float64Array,
  C_aa: Float64Array,
  C_bb: Float64Array,
  epsilon: number,
): number {
  const OT_ab = sinkhornCost(a, b, C_ab, epsilon)
  const OT_aa = sinkhornCost(a, a, C_aa, epsilon)
  const OT_bb = sinkhornCost(b, b, C_bb, epsilon)
  return OT_ab - 0.5 * OT_aa - 0.5 * OT_bb
}

/**
 * Simplified Sinkhorn divergence when a and b share the same support
 * and cost matrix (common case).
 */
export function sinkhornDivergenceSymmetric(
  a: Float64Array,
  b: Float64Array,
  C: Float64Array,
  epsilon: number,
): number {
  return sinkhornDivergence(a, b, C, C, C, epsilon)
}
