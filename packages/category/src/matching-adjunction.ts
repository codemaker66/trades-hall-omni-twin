/**
 * CT-7: Matching Adjunction
 *
 * F: Preferences → VenueSelection  (free: find best venue from preferences)
 * G: VenueSelection → Preferences  (forgetful: extract what preferences a selection satisfies)
 *
 * The adjunction captures the duality between:
 *   "What the client wants" ↔ "What the venue provides"
 */

import { createAdjunction } from './adjunction'
import type { Adjunction } from './adjunction'
import type { Amenity } from './objects'

// ─── Types ──────────────────────────────────────────────────────────────────

/** Client preferences (what they want). */
export interface MatchPreferences {
  readonly guestCount: number
  readonly requiredAmenities: readonly Amenity[]
  readonly minArea: number        // m²
  readonly maxBudgetPerHour: number  // cents
  readonly needsFocalPoint: boolean
}

/** A venue selection (what is offered). */
export interface VenueSelection {
  readonly venueId: string
  readonly capacity: number
  readonly area: number           // m²
  readonly costPerHour: number    // cents
  readonly amenities: readonly Amenity[]
  readonly hasFocalPoint: boolean
  readonly matchScore: number     // 0-1
}

/** A pool of available venues to match against. */
export interface VenuePool {
  readonly venues: readonly VenueSelection[]
}

// ─── Free Functor: Preferences → Selection ──────────────────────────────────

/**
 * F: Given preferences, find the best venue.
 * This is the "free" construction — generates the optimal selection.
 */
function findBestVenue(preferences: MatchPreferences): VenueSelection {
  // This is a simplified matcher — in production, uses the Kan extension.
  // For the adjunction, we demonstrate the algebraic structure.
  return {
    venueId: `venue-for-${preferences.guestCount}-guests`,
    capacity: Math.ceil(preferences.guestCount * 1.2),  // 20% buffer
    area: Math.max(preferences.minArea, preferences.guestCount * 2),
    costPerHour: preferences.maxBudgetPerHour,
    amenities: [...preferences.requiredAmenities],
    hasFocalPoint: preferences.needsFocalPoint,
    matchScore: 1.0,  // The free functor always produces a "perfect" match
  }
}

// ─── Forgetful Functor: Selection → Preferences ─────────────────────────────

/**
 * G: Extract what preferences a venue selection satisfies.
 * This is the "forgetful" functor — forgets the venue-specific details.
 */
function extractPreferences(selection: VenueSelection): MatchPreferences {
  return {
    guestCount: selection.capacity,
    requiredAmenities: [...selection.amenities],
    minArea: selection.area,
    maxBudgetPerHour: selection.costPerHour,
    needsFocalPoint: selection.hasFocalPoint,
  }
}

// ─── Matching Adjunction ────────────────────────────────────────────────────

/**
 * The Matching Adjunction: F ⊣ G
 *
 * F: MatchPreferences → VenueSelection  (find best venue)
 * G: VenueSelection → MatchPreferences  (extract preferences)
 *
 * The adjunction says:
 *   Hom(F(prefs), selection) ≅ Hom(prefs, G(selection))
 *   "How to get from the optimal venue to a given selection"
 *   ≡ "How to get from the preferences to what the selection offers"
 */
export const matchingAdjunction: Adjunction<MatchPreferences, VenueSelection> =
  createAdjunction(
    'Preference ⊣ Venue',
    findBestVenue,
    extractPreferences,
  )

/**
 * Find the optimal venue for given preferences.
 */
export function findOptimalVenue(preferences: MatchPreferences): VenueSelection {
  return matchingAdjunction.leftAdjoint(preferences)
}

/**
 * Determine what preferences a venue selection satisfies.
 */
export function venueCapabilities(selection: VenueSelection): MatchPreferences {
  return matchingAdjunction.rightAdjoint(selection)
}

/**
 * The unit: preferences round-trip.
 * η(prefs) = G(F(prefs)) — find best venue, then extract its preferences.
 * The result should be "at least as good" as the original preferences.
 */
export function preferencesRoundTrip(preferences: MatchPreferences): MatchPreferences {
  return matchingAdjunction.unit(preferences)
}

/**
 * Check if a venue selection satisfies given preferences.
 */
export function selectionSatisfies(
  selection: VenueSelection,
  preferences: MatchPreferences,
): { satisfied: boolean; reasons: string[] } {
  const reasons: string[] = []

  if (selection.capacity < preferences.guestCount) {
    reasons.push(`Capacity ${selection.capacity} < required ${preferences.guestCount}`)
  }

  if (selection.area < preferences.minArea) {
    reasons.push(`Area ${selection.area}m² < required ${preferences.minArea}m²`)
  }

  if (selection.costPerHour > preferences.maxBudgetPerHour) {
    reasons.push(`Cost ${selection.costPerHour} > budget ${preferences.maxBudgetPerHour}`)
  }

  const missingAmenities = preferences.requiredAmenities.filter(
    a => !selection.amenities.includes(a),
  )
  if (missingAmenities.length > 0) {
    reasons.push(`Missing amenities: ${missingAmenities.join(', ')}`)
  }

  if (preferences.needsFocalPoint && !selection.hasFocalPoint) {
    reasons.push('Missing focal point')
  }

  return { satisfied: reasons.length === 0, reasons }
}
