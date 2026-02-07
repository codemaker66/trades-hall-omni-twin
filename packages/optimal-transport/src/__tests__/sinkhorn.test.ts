/**
 * Sinkhorn solver correctness tests against known solutions.
 */

import { describe, test, expect } from 'vitest'
import { sinkhorn, sinkhornCost } from '../sinkhorn'
import { sinkhornLog, sinkhornLogCost } from '../sinkhorn-log'
import { computeRowSums, computeColSums } from '../utils'
import { sinkhornDivergence, sinkhornDivergenceSymmetric } from '../cost-matrix'
import { partialSinkhorn, unbalancedSinkhorn } from '../partial'
import {
  fixedSupportBarycenter,
  scoreAgainstBarycenter,
  featuresToDistribution,
} from '../barycenter'
import {
  displacementInterpolation,
  generateTransitionKeyframes,
  extractAssignment,
  buildPositionCostMatrix,
} from '../interpolation'
import {
  learnCostWeights,
  evaluateWeights,
  buildObservedPlan,
} from '../inverse-ot'
import type { FurniturePosition } from '../types'

// ─── Helpers ───────────────────────────────────────────────────────────────

function expectClose(a: Float64Array, b: Float64Array, tol: number) {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) {
    expect(Math.abs(a[i]! - b[i]!)).toBeLessThan(tol)
  }
}

function uniform(n: number): Float64Array {
  return new Float64Array(n).fill(1 / n)
}

// ─── OT-1: Standard Sinkhorn ──────────────────────────────────────────────

describe('OT-1: Standard Sinkhorn', () => {
  test('2×2 known solution: uniform distributions', () => {
    const a = new Float64Array([0.5, 0.5])
    const b = new Float64Array([0.5, 0.5])
    // Cost: identity-like → diagonal is cheap, off-diagonal expensive
    const C = new Float64Array([0, 1, 1, 0])

    const result = sinkhorn(a, b, C, { epsilon: 0.01, maxIterations: 200 })

    // Should transport mostly along diagonal
    expect(result.plan[0]).toBeGreaterThan(0.4)  // (0,0)
    expect(result.plan[3]).toBeGreaterThan(0.4)  // (1,1)
    expect(result.cost).toBeLessThan(0.1)
  })

  test('3×3 uniform with identity cost', () => {
    const a = uniform(3)
    const b = uniform(3)
    const C = new Float64Array([
      0, 1, 2,
      1, 0, 1,
      2, 1, 0,
    ])

    const result = sinkhorn(a, b, C, { epsilon: 0.05 })

    // Row sums should match source
    const rowSums = computeRowSums(result.plan, 3, 3)
    expectClose(rowSums, a, 1e-3)

    // Column sums should match target
    const colSums = computeColSums(result.plan, 3, 3)
    expectClose(colSums, b, 1e-3)
  })

  test('asymmetric distributions', () => {
    const a = new Float64Array([0.7, 0.3])
    const b = new Float64Array([0.4, 0.6])
    const C = new Float64Array([1, 2, 3, 1])

    const result = sinkhorn(a, b, C, { epsilon: 0.05 })

    const rowSums = computeRowSums(result.plan, 2, 2)
    expectClose(rowSums, a, 1e-3)

    const colSums = computeColSums(result.plan, 2, 2)
    expectClose(colSums, b, 1e-3)
  })

  test('transport plan is non-negative', () => {
    const a = new Float64Array([0.25, 0.25, 0.25, 0.25])
    const b = new Float64Array([0.1, 0.2, 0.3, 0.4])
    const C = new Float64Array(16)
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        C[i * 4 + j] = Math.abs(i - j)
      }
    }

    const result = sinkhorn(a, b, C, { epsilon: 0.05 })

    for (let k = 0; k < result.plan.length; k++) {
      expect(result.plan[k]).toBeGreaterThanOrEqual(0)
    }
  })

  test('zero cost → plan equals outer product', () => {
    const a = new Float64Array([0.5, 0.5])
    const b = new Float64Array([0.5, 0.5])
    const C = new Float64Array(4).fill(0)

    const result = sinkhorn(a, b, C, { epsilon: 0.1 })

    // With zero cost, plan should be close to a⊗b
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        expect(result.plan[i * 2 + j]).toBeCloseTo(a[i]! * b[j]!, 2)
      }
    }
  })

  test('convergence flag is set', () => {
    const a = uniform(3)
    const b = uniform(3)
    const C = new Float64Array(9).fill(1)

    const result = sinkhorn(a, b, C, { epsilon: 0.1, maxIterations: 500 })
    expect(result.converged).toBe(true)
  })

  test('cost is correct: <T, C>', () => {
    const a = new Float64Array([0.5, 0.5])
    const b = new Float64Array([0.5, 0.5])
    const C = new Float64Array([0, 1, 1, 0])

    const result = sinkhorn(a, b, C, { epsilon: 0.01, maxIterations: 200 })

    // Manually compute cost
    let manualCost = 0
    for (let k = 0; k < 4; k++) {
      manualCost += result.plan[k]! * C[k]!
    }
    expect(result.cost).toBeCloseTo(manualCost, 10)
  })
})

// ─── OT-1: Log-Domain Sinkhorn ────────────────────────────────────────────

describe('OT-1: Log-Domain Sinkhorn', () => {
  test('matches standard Sinkhorn for large epsilon', () => {
    const a = new Float64Array([0.3, 0.3, 0.4])
    const b = new Float64Array([0.2, 0.5, 0.3])
    const C = new Float64Array([
      1, 2, 3,
      4, 1, 2,
      3, 4, 1,
    ])

    const standard = sinkhorn(a, b, C, { epsilon: 0.1 })
    const logDomain = sinkhornLog(a, b, C, { epsilon: 0.1 })

    // Costs should be close
    expect(Math.abs(standard.cost - logDomain.cost)).toBeLessThan(0.05)
  })

  test('works for small epsilon where standard may struggle', () => {
    const a = uniform(3)
    const b = uniform(3)
    const C = new Float64Array([
      0, 5, 10,
      5, 0, 5,
      10, 5, 0,
    ])

    // Small epsilon (where standard Sinkhorn may lose precision)
    const result = sinkhornLog(a, b, C, { epsilon: 0.005, maxIterations: 500 })

    // Should converge and produce valid plan
    expect(result.cost).toBeGreaterThanOrEqual(0)
    expect(result.cost).toBeLessThan(10)

    // Plan should be nearly diagonal (cheapest assignment is identity)
    const diag = result.plan[0]! + result.plan[4]! + result.plan[8]!
    expect(diag).toBeGreaterThan(0.5)

    const rowSums = computeRowSums(result.plan, 3, 3)
    expectClose(rowSums, a, 0.1)
  })

  test('marginals satisfied', () => {
    const a = new Float64Array([0.6, 0.4])
    const b = new Float64Array([0.3, 0.7])
    const C = new Float64Array([1, 3, 2, 1])

    const result = sinkhornLog(a, b, C, { epsilon: 0.05 })

    const rowSums = computeRowSums(result.plan, 2, 2)
    const colSums = computeColSums(result.plan, 2, 2)
    expectClose(rowSums, a, 1e-3)
    expectClose(colSums, b, 1e-3)
  })

  test('transport plan is non-negative', () => {
    const a = uniform(4)
    const b = uniform(4)
    const C = new Float64Array(16)
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        C[i * 4 + j] = (i - j) * (i - j)
      }
    }

    const result = sinkhornLog(a, b, C, { epsilon: 0.05 })

    for (let k = 0; k < result.plan.length; k++) {
      expect(result.plan[k]).toBeGreaterThanOrEqual(-1e-10)
    }
  })
})

// ─── OT-2: Sinkhorn Divergence ────────────────────────────────────────────

describe('OT-2: Sinkhorn Divergence', () => {
  test('S(a, a) = 0 (identity of indiscernibles)', () => {
    const a = new Float64Array([0.25, 0.25, 0.25, 0.25])
    const C = new Float64Array(16)
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        C[i * 4 + j] = (i - j) * (i - j)
      }
    }

    const div = sinkhornDivergenceSymmetric(a, a, C, 0.05)
    expect(Math.abs(div)).toBeLessThan(1e-4)
  })

  test('S(a, b) >= 0 (positive definite)', () => {
    const a = new Float64Array([0.5, 0.3, 0.2])
    const b = new Float64Array([0.1, 0.4, 0.5])
    const C = new Float64Array([
      0, 1, 4,
      1, 0, 1,
      4, 1, 0,
    ])

    const div = sinkhornDivergenceSymmetric(a, b, C, 0.05)
    expect(div).toBeGreaterThanOrEqual(-1e-6)
  })

  test('S(a, b) > 0 when a ≠ b', () => {
    const a = new Float64Array([1, 0, 0])
    const b = new Float64Array([0, 0, 1])
    const C = new Float64Array([
      0, 1, 4,
      1, 0, 1,
      4, 1, 0,
    ])

    const div = sinkhornDivergenceSymmetric(a, b, C, 0.05)
    expect(div).toBeGreaterThan(0.01)
  })
})

// ─── OT-3: Wasserstein Barycenters ────────────────────────────────────────

describe('OT-3: Wasserstein Barycenters', () => {
  test('barycenter of identical distributions is the same distribution', () => {
    const n = 5
    const dist = new Float64Array([0.1, 0.2, 0.4, 0.2, 0.1])
    const C = new Float64Array(n * n)
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        C[i * n + j] = (i - j) * (i - j)
      }
    }

    const bary = fixedSupportBarycenter(
      [dist, dist, dist],
      C,
      new Float64Array([1 / 3, 1 / 3, 1 / 3]),
      { epsilon: 0.05, maxIterations: 50 },
    )

    // Should be close to the input distribution
    expect(bary.length).toBe(n)
    for (let i = 0; i < n; i++) {
      expect(Math.abs(bary[i]! - dist[i]!)).toBeLessThan(0.1)
    }
  })

  test('barycenter is a distribution (sums to 1)', () => {
    const n = 4
    const d1 = new Float64Array([0.5, 0.3, 0.1, 0.1])
    const d2 = new Float64Array([0.1, 0.1, 0.3, 0.5])
    const C = new Float64Array(n * n)
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        C[i * n + j] = (i - j) * (i - j)
      }
    }

    const bary = fixedSupportBarycenter(
      [d1, d2],
      C,
      new Float64Array([0.5, 0.5]),
      { epsilon: 0.1 },
    )

    let sum = 0
    for (let i = 0; i < n; i++) {
      sum += bary[i]!
      expect(bary[i]).toBeGreaterThanOrEqual(0)
    }
    expect(sum).toBeCloseTo(1, 2)
  })

  test('scoreAgainstBarycenter returns 0 for identical', () => {
    const n = 3
    const dist = new Float64Array([0.33, 0.34, 0.33])
    const C = new Float64Array(n * n)
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        C[i * n + j] = (i - j) * (i - j)
      }
    }

    const score = scoreAgainstBarycenter(dist, dist, C, 0.1)
    expect(Math.abs(score)).toBeLessThan(0.01)
  })

  test('featuresToDistribution produces valid distribution', () => {
    const features = [50, 100, 150, 200, 250, 300]
    const binEdges = [0, 100, 200, 300, 400]
    const dist = featuresToDistribution(features, binEdges)

    expect(dist.length).toBe(4) // 4 bins
    let sum = 0
    for (let i = 0; i < dist.length; i++) {
      expect(dist[i]).toBeGreaterThan(0)
      sum += dist[i]!
    }
    expect(sum).toBeCloseTo(1, 5)
  })
})

// ─── OT-4: Partial Transport ──────────────────────────────────────────────

describe('OT-4: Partial Sinkhorn', () => {
  test('partial transport moves less mass than full', () => {
    const a = uniform(3)
    const b = uniform(3)
    const C = new Float64Array([
      1, 2, 3,
      2, 1, 2,
      3, 2, 1,
    ])

    const full = sinkhorn(a, b, C, { epsilon: 0.05 })
    const partial = partialSinkhorn(a, b, C, 0.5, 0.05)

    // Partial should transport less total mass
    let fullMass = 0
    let partialMass = 0
    for (let k = 0; k < 9; k++) {
      fullMass += full.plan[k]!
      partialMass += partial.plan[k]!
    }
    expect(partialMass).toBeLessThan(fullMass + 0.01)
  })

  test('partial plan is non-negative', () => {
    const a = uniform(3)
    const b = uniform(3)
    const C = new Float64Array(9).fill(1)

    const result = partialSinkhorn(a, b, C, 0.7, 0.1)

    for (let k = 0; k < 9; k++) {
      expect(result.plan[k]).toBeGreaterThanOrEqual(-1e-10)
    }
  })

  test('mass=1 approaches full transport', () => {
    const a = uniform(2)
    const b = uniform(2)
    const C = new Float64Array([0, 1, 1, 0])

    const full = sinkhorn(a, b, C, { epsilon: 0.05 })
    const almost = partialSinkhorn(a, b, C, 1.0, 0.05)

    expect(Math.abs(full.cost - almost.cost)).toBeLessThan(0.2)
  })
})

describe('OT-4: Unbalanced Sinkhorn', () => {
  test('large ρ approaches balanced OT', () => {
    const a = uniform(3)
    const b = uniform(3)
    const C = new Float64Array([
      1, 2, 3,
      2, 1, 2,
      3, 2, 1,
    ])

    const balanced = sinkhorn(a, b, C, { epsilon: 0.05 })
    const unbalanced = unbalancedSinkhorn(a, b, C, 0.05, 100)

    // With very large ρ, should be close to balanced
    expect(Math.abs(balanced.cost - unbalanced.cost)).toBeLessThan(0.5)
  })

  test('unbalanced plan is non-negative', () => {
    const a = new Float64Array([0.8, 0.2])
    const b = new Float64Array([0.3, 0.7])
    const C = new Float64Array([1, 2, 3, 1])

    const result = unbalancedSinkhorn(a, b, C, 0.05, 0.1)

    for (let k = 0; k < 4; k++) {
      expect(result.plan[k]).toBeGreaterThanOrEqual(-1e-10)
    }
  })

  test('small ρ allows mass destruction', () => {
    const a = new Float64Array([0.9, 0.1])
    const b = new Float64Array([0.1, 0.9])
    const C = new Float64Array([0, 10, 10, 0])

    const strict = unbalancedSinkhorn(a, b, C, 0.05, 10)
    const relaxed = unbalancedSinkhorn(a, b, C, 0.05, 0.01)

    // Relaxed should have lower cost (can destroy mass)
    expect(relaxed.cost).toBeLessThanOrEqual(strict.cost + 0.1)
  })
})

// ─── OT-5: Displacement Interpolation ─────────────────────────────────────

describe('OT-5: Displacement Interpolation', () => {
  const layoutA: FurniturePosition[] = [
    { id: 'c1', x: 0, z: 0, rotation: 0, type: 'chair' },
    { id: 'c2', x: 1, z: 0, rotation: 0, type: 'chair' },
  ]
  const layoutB: FurniturePosition[] = [
    { id: 'c3', x: 5, z: 5, rotation: Math.PI, type: 'chair' },
    { id: 'c4', x: 6, z: 5, rotation: Math.PI, type: 'chair' },
  ]

  // Squared Euclidean costs range 25-61, so epsilon must be large enough
  // to keep Gibbs kernel K=exp(-C/ε) from underflowing to zero.
  const INTERP_EPSILON = 10

  test('t=0 returns source layout positions', () => {
    const C = buildPositionCostMatrix(layoutA, layoutB)
    const a = uniform(2)
    const b = uniform(2)
    const result = sinkhorn(a, b, C, { epsilon: INTERP_EPSILON })

    const interp = displacementInterpolation(layoutA, layoutB, result.plan, 0)
    expect(interp[0]!.x).toBeCloseTo(0, 5)
    expect(interp[0]!.z).toBeCloseTo(0, 5)
  })

  test('t=1 returns target layout positions', () => {
    const C = buildPositionCostMatrix(layoutA, layoutB)
    const a = uniform(2)
    const b = uniform(2)
    const result = sinkhorn(a, b, C, { epsilon: INTERP_EPSILON })

    const interp = displacementInterpolation(layoutA, layoutB, result.plan, 1)
    // At t=1, all visible items should be at layoutB positions
    const visible = interp.filter(p => (p.opacity ?? 1) > 0.5)
    const targetXs = layoutB.map(p => p.x).sort((a, b) => a - b)
    const interpXs = visible.map(p => p.x).sort((a, b) => a - b)
    expect(interpXs.length).toBeGreaterThanOrEqual(targetXs.length)
    for (let i = 0; i < targetXs.length; i++) {
      expect(interpXs[i]).toBeCloseTo(targetXs[i]!, 0)
    }
  })

  test('t=0.5 returns midpoint positions', () => {
    const C = buildPositionCostMatrix(layoutA, layoutB)
    const a = uniform(2)
    const b = uniform(2)
    const result = sinkhorn(a, b, C, { epsilon: INTERP_EPSILON })

    const interp = displacementInterpolation(layoutA, layoutB, result.plan, 0.5)
    // Midpoint should be roughly between source and target
    for (const pos of interp) {
      if (pos.opacity === 1) {
        expect(pos.x).toBeGreaterThan(-1)
        expect(pos.x).toBeLessThan(7)
      }
    }
  })

  test('extractAssignment returns valid indices', () => {
    const plan = new Float64Array([0.4, 0.1, 0.1, 0.4])
    const assignments = extractAssignment(plan, 2, 2)
    expect(assignments).toHaveLength(2)
    expect(assignments[0]![0]).toBe(0) // source 0
    expect(assignments[0]![1]).toBe(0) // maps to target 0
    expect(assignments[1]![0]).toBe(1) // source 1
    expect(assignments[1]![1]).toBe(1) // maps to target 1
  })

  test('generateTransitionKeyframes produces correct number of frames', () => {
    const frames = generateTransitionKeyframes(layoutA, layoutB, 10)
    expect(frames).toHaveLength(11) // 0 through 10 inclusive
  })

  test('position cost matrix is symmetric for same layouts', () => {
    const C = buildPositionCostMatrix(layoutA, layoutA)
    // Diagonal should be 0
    for (let i = 0; i < 2; i++) {
      expect(C[i * 2 + i]).toBeCloseTo(0, 10)
    }
  })
})

// ─── OT-7: Inverse OT ────────────────────────────────────────────────────

describe('OT-7: Inverse OT', () => {
  test('buildObservedPlan creates valid plan', () => {
    const matchings = [
      { eventIndex: 0, venueIndex: 1, success: true },
      { eventIndex: 1, venueIndex: 0, success: true },
    ]
    const plan = buildObservedPlan(matchings, 2, 2)

    // Row sums should be 1
    expect(plan[0]! + plan[1]!).toBeCloseTo(1, 5) // row 0
    expect(plan[2]! + plan[3]!).toBeCloseTo(1, 5) // row 1

    // Primary matches should dominate
    expect(plan[1]).toBeGreaterThan(0.5) // event 0 → venue 1
    expect(plan[2]).toBeGreaterThan(0.5) // event 1 → venue 0
  })

  test('learnCostWeights returns valid weights', () => {
    const events = [
      {
        guestCount: 100,
        requiredAmenities: [true, false],
        preferredLocation: { lat: 0, lng: 0 },
        budget: 5000,
        minSqFootage: 200,
        eventType: 'conference',
      },
      {
        guestCount: 50,
        requiredAmenities: [false, true],
        preferredLocation: { lat: 1, lng: 1 },
        budget: 3000,
        minSqFootage: 100,
        eventType: 'party',
      },
    ]
    const venues = [
      {
        capacity: 120,
        amenities: [true, true],
        location: { lat: 0, lng: 0 },
        pricePerEvent: 4000,
        sqFootage: 250,
        venueType: 'conference',
      },
      {
        capacity: 60,
        amenities: [false, true],
        location: { lat: 1, lng: 1 },
        pricePerEvent: 2500,
        sqFootage: 120,
        venueType: 'bar',
      },
    ]
    const matchings = [
      { eventIndex: 0, venueIndex: 0, success: true },
      { eventIndex: 1, venueIndex: 1, success: true },
    ]

    const weights = learnCostWeights(matchings, events, venues, undefined, {
      iterations: 10,
      epsilon: 0.1,
    })

    // Weights should be positive and sum to ~1
    expect(weights.capacity).toBeGreaterThan(0)
    expect(weights.price).toBeGreaterThan(0)
    expect(weights.amenity).toBeGreaterThan(0)
    expect(weights.location).toBeGreaterThan(0)
    const total = weights.capacity + weights.price + weights.amenity + weights.location
    expect(total).toBeCloseTo(1, 2)
  })

  test('evaluateWeights returns score in [0, 1]', () => {
    const events = [
      {
        guestCount: 100,
        requiredAmenities: [true],
        preferredLocation: { lat: 0, lng: 0 },
        budget: 5000,
        minSqFootage: 200,
        eventType: 'conference',
      },
    ]
    const venues = [
      {
        capacity: 120,
        amenities: [true],
        location: { lat: 0, lng: 0 },
        pricePerEvent: 4000,
        sqFootage: 250,
        venueType: 'conference',
      },
    ]
    const matchings = [{ eventIndex: 0, venueIndex: 0, success: true }]
    const weights = { capacity: 0.3, price: 0.3, amenity: 0.2, location: 0.2 }

    const score = evaluateWeights(matchings, events, venues, weights, 0.1)
    expect(score).toBeGreaterThanOrEqual(0)
    expect(score).toBeLessThanOrEqual(1)
  })
})
