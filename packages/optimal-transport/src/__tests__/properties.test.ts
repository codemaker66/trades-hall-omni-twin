/**
 * Property-based tests for OT correctness guarantees.
 */

import { describe, test, expect } from 'vitest'
import fc from 'fast-check'
import { sinkhorn } from '../sinkhorn'
import { sinkhornLog } from '../sinkhorn-log'
import { computeRowSums, computeColSums, normalizeDistribution } from '../utils'
import {
  capacityDistance,
  amenityDistance,
  locationDistance,
  priceDistance,
  buildCostMatrix,
  sinkhornDivergenceSymmetric,
} from '../cost-matrix'

// ─── Arbitrary Generators ──────────────────────────────────────────────────

/**
 * Generate a valid probability distribution of length n.
 */
function arbitraryDistribution(n: number) {
  return fc.array(fc.float({ min: Math.fround(0.01), max: 1, noNaN: true, noDefaultInfinity: true }), {
    minLength: n,
    maxLength: n,
  }).map(arr => {
    const f = new Float64Array(arr)
    normalizeDistribution(f)
    return f
  })
}

/**
 * Generate a valid N×M cost matrix with non-negative entries.
 */
function arbitraryCostMatrix(N: number, M: number) {
  return fc.array(
    fc.float({ min: 0, max: 10, noNaN: true, noDefaultInfinity: true }),
    { minLength: N * M, maxLength: N * M },
  ).map(arr => new Float64Array(arr))
}

// ─── Property Tests ────────────────────────────────────────────────────────

describe('Property: Marginal Constraints', () => {
  test('transport plan row sums match source distribution', () => {
    fc.assert(
      fc.property(
        arbitraryDistribution(5),
        arbitraryDistribution(5),
        arbitraryCostMatrix(5, 5),
        (a, b, C) => {
          // Use more iterations + larger epsilon for robustness with extreme distributions
          const result = sinkhorn(a, b, C, { epsilon: 0.5, maxIterations: 500 })
          const rowSums = computeRowSums(result.plan, 5, 5)
          for (let i = 0; i < 5; i++) {
            if (Math.abs(rowSums[i]! - a[i]!) > 0.1) return false
          }
          return true
        },
      ),
      { numRuns: 20 },
    )
  })

  test('transport plan column sums match target distribution', () => {
    fc.assert(
      fc.property(
        arbitraryDistribution(4),
        arbitraryDistribution(4),
        arbitraryCostMatrix(4, 4),
        (a, b, C) => {
          const result = sinkhorn(a, b, C, { epsilon: 0.5, maxIterations: 500 })
          const colSums = computeColSums(result.plan, 4, 4)
          for (let j = 0; j < 4; j++) {
            if (Math.abs(colSums[j]! - b[j]!) > 0.1) return false
          }
          return true
        },
      ),
      { numRuns: 20 },
    )
  })
})

describe('Property: Non-negativity', () => {
  test('transport plan entries are non-negative', () => {
    fc.assert(
      fc.property(
        arbitraryDistribution(6),
        arbitraryDistribution(6),
        arbitraryCostMatrix(6, 6),
        (a, b, C) => {
          const result = sinkhorn(a, b, C, { epsilon: 0.1 })
          for (let k = 0; k < result.plan.length; k++) {
            if (result.plan[k]! < -1e-10) return false
          }
          return true
        },
      ),
      { numRuns: 20 },
    )
  })
})

describe('Property: Sinkhorn Divergence', () => {
  test('S(a, a) ≈ 0 for any distribution', () => {
    fc.assert(
      fc.property(
        arbitraryDistribution(4),
        (a) => {
          const C = new Float64Array(16)
          for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
              C[i * 4 + j] = (i - j) * (i - j)
            }
          }
          const div = sinkhornDivergenceSymmetric(a, a, C, 0.1)
          return Math.abs(div) < 0.05
        },
      ),
      { numRuns: 15 },
    )
  })

  test('S(a, b) >= -epsilon for any a, b (approximately non-negative)', () => {
    fc.assert(
      fc.property(
        arbitraryDistribution(3),
        arbitraryDistribution(3),
        (a, b) => {
          const C = new Float64Array(9)
          for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
              C[i * 3 + j] = (i - j) * (i - j)
            }
          }
          const div = sinkhornDivergenceSymmetric(a, b, C, 0.1)
          return div >= -0.05
        },
      ),
      { numRuns: 15 },
    )
  })
})

describe('Property: Cost Matrix Features', () => {
  test('capacity distance is non-negative', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 1000 }),
        fc.integer({ min: 1, max: 1000 }),
        (guests, capacity) => capacityDistance(guests, capacity) >= 0,
      ),
    )
  })

  test('capacity distance: venue too small is worse than too large', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 50, max: 500 }),
        (guests) => {
          const tooSmall = capacityDistance(guests, Math.floor(guests * 0.5))
          const tooLarge = capacityDistance(guests, Math.floor(guests * 1.5))
          return tooSmall > tooLarge
        },
      ),
    )
  })

  test('amenity distance is in [0, 1]', () => {
    fc.assert(
      fc.property(
        fc.array(fc.boolean(), { minLength: 5, maxLength: 5 }),
        fc.array(fc.boolean(), { minLength: 5, maxLength: 5 }),
        (req, avail) => {
          const d = amenityDistance(req, avail)
          return d >= 0 && d <= 1
        },
      ),
    )
  })

  test('location distance is non-negative', () => {
    fc.assert(
      fc.property(
        fc.float({ min: -90, max: 90, noNaN: true, noDefaultInfinity: true }),
        fc.float({ min: -180, max: 180, noNaN: true, noDefaultInfinity: true }),
        fc.float({ min: -90, max: 90, noNaN: true, noDefaultInfinity: true }),
        fc.float({ min: -180, max: 180, noNaN: true, noDefaultInfinity: true }),
        (lat1, lng1, lat2, lng2) => {
          const d = locationDistance({ lat: lat1, lng: lng1 }, { lat: lat2, lng: lng2 })
          return d >= 0
        },
      ),
    )
  })

  test('location distance: same point = 0', () => {
    const d = locationDistance({ lat: 37.7749, lng: -122.4194 }, { lat: 37.7749, lng: -122.4194 })
    expect(d).toBeCloseTo(0, 5)
  })

  test('price distance is non-negative', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 100, max: 100000 }),
        fc.integer({ min: 100, max: 100000 }),
        (budget, price) => priceDistance(budget, price) >= 0,
      ),
    )
  })

  test('price distance: over budget is worse than under budget', () => {
    const budget = 5000
    const overBudget = priceDistance(budget, 7000)
    const underBudget = priceDistance(budget, 3000)
    expect(overBudget).toBeGreaterThan(underBudget)
  })
})

describe('Property: Log vs Standard Equivalence', () => {
  test('log-domain cost is close to standard for moderate epsilon', () => {
    // Use a smaller cost range to keep the Gibbs kernel from underflowing
    // in standard Sinkhorn (standard can't handle exp(-10/0.5) = exp(-20) ≈ 2e-9)
    const smallCostMatrix = (N: number, M: number) =>
      fc.array(
        fc.float({ min: 0, max: 2, noNaN: true, noDefaultInfinity: true }),
        { minLength: N * M, maxLength: N * M },
      ).map(arr => new Float64Array(arr))

    fc.assert(
      fc.property(
        arbitraryDistribution(3),
        arbitraryDistribution(3),
        smallCostMatrix(3, 3),
        (a, b, C) => {
          const standard = sinkhorn(a, b, C, { epsilon: 0.5, maxIterations: 300 })
          const logDom = sinkhornLog(a, b, C, { epsilon: 0.5, maxIterations: 300 })
          return Math.abs(standard.cost - logDom.cost) < 0.5
        },
      ),
      { numRuns: 15 },
    )
  })
})
