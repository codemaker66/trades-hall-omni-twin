/**
 * Kan extension matching tests (CT-5) and adjunction tests (CT-7).
 */

import { describe, test, expect } from 'vitest'
import fc from 'fast-check'
import type { VenueSpec } from '../objects'
import { isoDateTime, minutes, cents } from '../objects'
import {
  buildKanExtension, coendFormula, endFormula,
} from '../kan-extension'
import type { KanConfig, KanDimension } from '../kan-extension'
import {
  matchVenuesOptimistic, matchVenuesConservative, matchVenuesCoend,
  venueQuality,
} from '../venue-matching'
import type { VenuePreferences } from '../venue-matching'
import {
  createAdjunction, verifyLeftTriangle, verifyRightTriangle,
  monadFromAdjunction,
} from '../adjunction'
import {
  layoutAdjunction, optimizeLayout, satisfiedConstraints, constraintRoundTrip,
} from '../layout-adjunction'
import type { LayoutConstraints, OptimizedLayout } from '../layout-adjunction'
import {
  matchingAdjunction, findOptimalVenue, venueCapabilities,
  preferencesRoundTrip, selectionSatisfies,
} from '../matching-adjunction'
import type { MatchPreferences } from '../matching-adjunction'

// ─── Test Data ──────────────────────────────────────────────────────────────

const venues: VenueSpec[] = [
  {
    id: 'v1', name: 'Small Hall', width: 10, depth: 8, height: 3,
    maxCapacity: 50, exits: [{ x: 0, z: 4, width: 1, facing: Math.PI }],
    obstacles: [], amenities: ['wifi', 'lighting'],
  },
  {
    id: 'v2', name: 'Grand Ballroom', width: 30, depth: 25, height: 5,
    maxCapacity: 500, exits: [
      { x: 0, z: 12, width: 2, facing: Math.PI },
      { x: 30, z: 12, width: 2, facing: 0 },
    ],
    obstacles: [], amenities: ['stage', 'av-system', 'lighting', 'wifi', 'bar', 'kitchen', 'accessible'],
    focalPoint: { x: 15, z: 0 },
  },
  {
    id: 'v3', name: 'Garden Pavilion', width: 20, depth: 15, height: 4,
    maxCapacity: 150, exits: [{ x: 10, z: 15, width: 3, facing: Math.PI / 2 }],
    obstacles: [], amenities: ['outdoor-space', 'lighting', 'wifi', 'parking'],
  },
]

// ─── Kan Extension Tests ────────────────────────────────────────────────────

describe('CT-5: Kan Extensions', () => {
  test('coend formula computes weighted average', () => {
    const weights = [0.8, 0.6, 0.4]
    const values = [0.9, 0.7, 0.5]
    const result = coendFormula(weights, values)
    // (0.8*0.9 + 0.6*0.7 + 0.4*0.5) / (0.8 + 0.6 + 0.4) = (0.72+0.42+0.2)/1.8
    expect(result).toBeCloseTo((0.72 + 0.42 + 0.2) / 1.8, 5)
  })

  test('coend formula handles empty arrays', () => {
    expect(coendFormula([], [])).toBe(0)
  })

  test('end formula computes conservative estimate', () => {
    const weights = [0.8, 0.6, 0.4]
    const values = [0.9, 0.7, 0.5]
    const result = endFormula(weights, values)
    // min(0.9/0.8, 0.7/0.6, 0.5/0.4) = min(1.125, 1.167, 1.25) → clamped to 1
    expect(result).toBeLessThanOrEqual(1)
    expect(result).toBeGreaterThan(0)
  })

  test('end formula handles empty arrays', () => {
    expect(endFormula([], [])).toBe(0)
  })

  test('left Kan (optimistic) always scores >= right Kan (conservative)', () => {
    const config: KanConfig<{ value: number }, number> = {
      dimensions: [{
        name: 'test',
        weight: 1,
        extractPref: (p) => p.value,
        score: (pref, item) => 1 - Math.abs((pref as number) - item) / 10,
        optimistic: 1.0,
        conservative: 0.0,
      }],
      defaultWeight: 1,
    }

    const kan = buildKanExtension(config)
    const items = [1, 3, 5, 7, 9]

    // Unspecified preferences: left should score higher
    const left = kan.leftKan({}, items)
    const right = kan.rightKan({}, items)

    for (let i = 0; i < items.length; i++) {
      expect(left[i]!.score).toBeGreaterThanOrEqual(right[i]!.score)
    }
  })
})

// ─── Venue Matching Tests ───────────────────────────────────────────────────

describe('CT-5: Venue Matching', () => {
  test('optimistic matching ranks venues', () => {
    const prefs: VenuePreferences = {
      guestCount: 200,
      requiredAmenities: ['stage', 'av-system'],
      needsFocalPoint: true,
    }

    const results = matchVenuesOptimistic(prefs, venues)
    expect(results).toHaveLength(3)
    // Grand Ballroom should rank highest (has stage, av, focal point, capacity)
    expect(results[0]!.venue.id).toBe('v2')
  })

  test('conservative matching is more restrictive', () => {
    const prefs: VenuePreferences = { guestCount: 100 }
    const optimistic = matchVenuesOptimistic(prefs, venues)
    const conservative = matchVenuesConservative(prefs, venues)

    // Scores should be >= in optimistic vs conservative
    for (let i = 0; i < venues.length; i++) {
      const optScore = optimistic.find(r => r.venue.id === venues[i]!.id)!.score
      const conScore = conservative.find(r => r.venue.id === venues[i]!.id)!.score
      expect(optScore).toBeGreaterThanOrEqual(conScore as number)
    }
  })

  test('coend matching returns results', () => {
    const prefs: VenuePreferences = {
      guestCount: 100,
      requiredAmenities: ['wifi'],
    }
    const results = matchVenuesCoend(prefs, venues)
    expect(results).toHaveLength(3)
    results.forEach(r => {
      expect(r.score).toBeGreaterThanOrEqual(0)
      expect(r.score).toBeLessThanOrEqual(1)
    })
  })

  test('venue quality is bounded 0-1', () => {
    venues.forEach(v => {
      const q = venueQuality(v)
      expect(q).toBeGreaterThanOrEqual(0)
      expect(q).toBeLessThanOrEqual(1)
    })
  })

  test('empty preferences returns all venues', () => {
    const results = matchVenuesOptimistic({}, venues)
    expect(results).toHaveLength(3)
  })
})

// ─── Adjunction Tests ───────────────────────────────────────────────────────

describe('CT-7: Adjunctions', () => {
  test('simple number adjunction', () => {
    // F: number → number (double)
    // G: number → number (halve)
    const adj = createAdjunction(
      'double ⊣ halve',
      (x: number) => x * 2,
      (x: number) => x / 2,
    )

    // Unit: x → G(F(x)) = x/2 * 2 = x (should be identity-like)
    expect(adj.unit(5)).toBe(5)

    // Counit: F(G(x)) = 2 * x/2 = x (should be identity-like)
    expect(adj.counit(10)).toBe(10)
  })

  test('monad from adjunction', () => {
    const adj = createAdjunction(
      'test',
      (x: number) => x * 2,
      (x: number) => x / 2,
    )
    const monad = monadFromAdjunction(adj)

    // Pure should be the unit
    expect(monad.pure(5)).toBe(5)

    // Apply is G ∘ F
    expect(monad.apply(5)).toBe(5)
  })
})

describe('CT-7: Layout Adjunction', () => {
  const constraints: LayoutConstraints = {
    maxItems: 20,
    minSpacing: 0.5,
    roomWidth: 10,
    roomDepth: 8,
    exitClearance: 1.0,
    aisleWidth: 1.5,
  }

  test('left adjoint generates a layout', () => {
    const layout = optimizeLayout(constraints)
    expect(layout.items.length).toBeGreaterThan(0)
    expect(layout.items.length).toBeLessThanOrEqual(constraints.maxItems)
    expect(layout.score).toBeGreaterThan(0)
    expect(layout.score).toBeLessThanOrEqual(1)
  })

  test('right adjoint extracts constraints', () => {
    const layout = optimizeLayout(constraints)
    const extracted = satisfiedConstraints(layout)
    expect(extracted.maxItems).toBe(layout.items.length)
    expect(extracted.roomWidth).toBe(constraints.roomWidth)
    expect(extracted.roomDepth).toBe(constraints.roomDepth)
  })

  test('unit (round-trip) preserves room dimensions', () => {
    const roundTripped = constraintRoundTrip(constraints)
    expect(roundTripped.roomWidth).toBe(constraints.roomWidth)
    expect(roundTripped.roomDepth).toBe(constraints.roomDepth)
  })

  test('generated layout has items within bounds', () => {
    const layout = optimizeLayout(constraints)
    for (const item of layout.items) {
      expect(item.x).toBeGreaterThan(0)
      expect(item.x).toBeLessThan(constraints.roomWidth)
      expect(item.z).toBeGreaterThan(0)
      expect(item.z).toBeLessThan(constraints.roomDepth)
    }
  })
})

describe('CT-7: Matching Adjunction', () => {
  const prefs: MatchPreferences = {
    guestCount: 100,
    requiredAmenities: ['wifi', 'stage'],
    minArea: 150,
    maxBudgetPerHour: 50000,
    needsFocalPoint: true,
  }

  test('left adjoint finds optimal venue', () => {
    const venue = findOptimalVenue(prefs)
    expect(venue.capacity).toBeGreaterThanOrEqual(prefs.guestCount)
    expect(venue.area).toBeGreaterThanOrEqual(prefs.minArea)
    expect(venue.amenities).toContain('wifi')
    expect(venue.amenities).toContain('stage')
    expect(venue.hasFocalPoint).toBe(true)
  })

  test('right adjoint extracts capabilities', () => {
    const venue = findOptimalVenue(prefs)
    const caps = venueCapabilities(venue)
    expect(caps.guestCount).toBe(venue.capacity)
    expect(caps.minArea).toBe(venue.area)
  })

  test('round-trip preserves amenity requirements', () => {
    const roundTripped = preferencesRoundTrip(prefs)
    expect(roundTripped.requiredAmenities).toEqual(prefs.requiredAmenities)
    expect(roundTripped.needsFocalPoint).toBe(prefs.needsFocalPoint)
  })

  test('selectionSatisfies detects failures', () => {
    const venue = findOptimalVenue(prefs)
    const bigPrefs: MatchPreferences = {
      ...prefs,
      guestCount: 10000,
    }
    const result = selectionSatisfies(venue, bigPrefs)
    expect(result.satisfied).toBe(false)
    expect(result.reasons.length).toBeGreaterThan(0)
  })

  test('selectionSatisfies passes for matched venue', () => {
    const venue = findOptimalVenue(prefs)
    const result = selectionSatisfies(venue, prefs)
    expect(result.satisfied).toBe(true)
  })
})
