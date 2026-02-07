/**
 * CT-5: Venue Matching via Kan Extensions
 *
 * Concrete implementation of Kan extensions for venue-event matching.
 * Given partial client preferences, find the optimal venue match.
 *
 * The left Kan extension computes the best approximation:
 *   the venue that satisfies the most specified preferences
 *   while making optimal assumptions about unspecified ones.
 */

import type { VenueSpec, EventSpec, Amenity, CompatibilityScore, RankedVenueMatches, RankedVenueMatch } from './objects'
import { compatibilityScore } from './objects'
import { buildKanExtension, coendFormula } from './kan-extension'
import type { KanConfig, KanDimension, RankedResult } from './kan-extension'

// ─── Partial Event Preferences ──────────────────────────────────────────────

/** Partial event preferences for matching. */
export interface VenuePreferences {
  readonly guestCount?: number
  readonly eventType?: string
  readonly requiredAmenities?: readonly Amenity[]
  readonly minArea?: number
  readonly maxBudget?: number
  readonly needsFocalPoint?: boolean
  readonly outdoorPreferred?: boolean
}

// ─── Venue Quality Metric ───────────────────────────────────────────────────

/**
 * Intrinsic quality of a venue (independent of preferences).
 * This is G(a) in the coend formula.
 */
export function venueQuality(venue: VenueSpec): number {
  let quality = 0.5  // base quality

  // More amenities = higher quality
  quality += Math.min(0.3, venue.amenities.length * 0.03)

  // More exits = safer
  quality += Math.min(0.1, venue.exits.length * 0.025)

  // Larger capacity = more versatile
  quality += Math.min(0.1, venue.maxCapacity / 1000)

  return Math.min(1, quality)
}

// ─── Kan Extension Configuration ────────────────────────────────────────────

const venueDimensions: readonly KanDimension<VenuePreferences, VenueSpec>[] = [
  {
    name: 'capacity',
    weight: 3,
    extractPref: (p) => p.guestCount,
    score: (guestCount, venue) => {
      const count = guestCount as number
      if (count > venue.maxCapacity) return 0
      const utilization = count / venue.maxCapacity
      // Ideal utilization: 60-85%
      if (utilization >= 0.6 && utilization <= 0.85) return 1
      if (utilization > 0.85) return 1 - (utilization - 0.85) / 0.15
      return utilization / 0.6
    },
    optimistic: 0.8,
    conservative: 0.3,
  },
  {
    name: 'area',
    weight: 2,
    extractPref: (p) => p.minArea,
    score: (minArea, venue) => {
      const area = venue.width * venue.depth
      const required = minArea as number
      if (area < required) return area / required
      return Math.min(1, 1 - (area - required) / (required * 2))
    },
    optimistic: 0.9,
    conservative: 0.4,
  },
  {
    name: 'amenities',
    weight: 2,
    extractPref: (p) => p.requiredAmenities,
    score: (required, venue) => {
      const amenities = required as readonly Amenity[]
      if (amenities.length === 0) return 1
      const matched = amenities.filter(a => venue.amenities.includes(a))
      return matched.length / amenities.length
    },
    optimistic: 1.0,
    conservative: 0.2,
  },
  {
    name: 'focalPoint',
    weight: 1.5,
    extractPref: (p) => p.needsFocalPoint,
    score: (needs, venue) => {
      if (!(needs as boolean)) return 1
      return venue.focalPoint ? 1 : 0
    },
    optimistic: 1.0,
    conservative: 0.5,
  },
  {
    name: 'outdoor',
    weight: 1,
    extractPref: (p) => p.outdoorPreferred,
    score: (preferred, venue) => {
      if (!(preferred as boolean)) return 1
      return venue.amenities.includes('outdoor-space') ? 1 : 0.3
    },
    optimistic: 1.0,
    conservative: 0.5,
  },
]

const venueKanConfig: KanConfig<VenuePreferences, VenueSpec> = {
  dimensions: venueDimensions,
  defaultWeight: 1,
}

// ─── Venue Matching Functions ───────────────────────────────────────────────

const kanExtension = buildKanExtension(venueKanConfig)

/**
 * Optimistic venue matching using left Kan extension.
 * Best for exploratory searches — shows the full range of possibilities.
 */
export function matchVenuesOptimistic(
  preferences: VenuePreferences,
  venues: readonly VenueSpec[],
): RankedVenueMatches {
  const results = kanExtension.leftKan(preferences, venues)
  return results.map(toRankedMatch)
}

/**
 * Conservative venue matching using right Kan extension.
 * Best for final selection — only highly confident matches.
 */
export function matchVenuesConservative(
  preferences: VenuePreferences,
  venues: readonly VenueSpec[],
): RankedVenueMatches {
  const results = kanExtension.rightKan(preferences, venues)
  return results.map(toRankedMatch)
}

/**
 * Direct coend formula matching.
 * Uses compatibility weights and venue quality together.
 */
export function matchVenuesCoend(
  preferences: VenuePreferences,
  venues: readonly VenueSpec[],
): RankedVenueMatches {
  // Compute Hom weights (compatibility of each venue with preferences)
  const optimistic = kanExtension.leftKan(preferences, venues)
  const homWeights = optimistic.map(r => r.score)
  const qualities = venues.map(venueQuality)

  // The coend gives us the overall score
  const coendScore = coendFormula(homWeights, qualities)

  // Rank by compatibility × quality
  return optimistic.map((r, i) => ({
    venue: r.item,
    score: compatibilityScore(r.score * qualities[i]!),
    reasons: r.factors
      .filter(f => f.specified && f.score > 0.5)
      .map(f => `${f.name}: ${(f.score * 100).toFixed(0)}%`),
  }))
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function toRankedMatch(result: RankedResult<VenueSpec, number>): RankedVenueMatch {
  return {
    venue: result.item,
    score: compatibilityScore(result.score),
    reasons: result.factors
      .filter(f => f.specified)
      .map(f => `${f.name}: ${(f.score * 100).toFixed(0)}%`),
  }
}
