import { describe, it, expect } from 'vitest'
import {
  emsrb,
  bidPriceControl,
  choiceBasedRM,
  gallegoVanRyzinPrice,
} from '../revenue-management'

describe('SP-8: Revenue Management', () => {
  describe('EMSRb', () => {
    it('computes protection levels for 3 fare classes', () => {
      const result = emsrb(
        [
          { name: 'Wedding', revenue: 15000, meanDemand: 3, stdDemand: 1.5 },
          { name: 'Corporate', revenue: 8000, meanDemand: 5, stdDemand: 2 },
          { name: 'Party', revenue: 3000, meanDemand: 8, stdDemand: 3 },
        ],
        20, // Total capacity
      )

      expect(result.protectionLevels.length).toBe(3)
      expect(result.bookingLimits.length).toBe(3)
      expect(result.expectedRevenue).toBeGreaterThan(0)

      // Wedding (highest revenue) should have protection
      expect(result.protectionLevels[0]).toBeGreaterThanOrEqual(0)
      // Lowest class has no protection
      expect(result.protectionLevels[2]).toBe(0)

      // Booking limits should be non-negative and ≤ capacity
      for (const bl of result.bookingLimits) {
        expect(bl).toBeGreaterThanOrEqual(0)
        expect(bl).toBeLessThanOrEqual(20)
      }
    })

    it('single fare class gets full capacity', () => {
      const result = emsrb(
        [{ name: 'Standard', revenue: 5000, meanDemand: 10, stdDemand: 3 }],
        20,
      )

      expect(result.bookingLimits[0]).toBe(20)
    })

    it('higher capacity yields higher expected revenue', () => {
      const classes = [
        { name: 'Wedding', revenue: 15000, meanDemand: 5, stdDemand: 2 },
        { name: 'Corporate', revenue: 8000, meanDemand: 8, stdDemand: 3 },
      ]

      const small = emsrb(classes, 5)
      const large = emsrb(classes, 20)
      expect(large.expectedRevenue).toBeGreaterThan(small.expectedRevenue)
    })
  })

  describe('Bid-Price Control', () => {
    it('accepts profitable requests', () => {
      const result = bidPriceControl(
        [
          { name: 'Ballroom', capacity: 5 },
          { name: 'Garden', capacity: 3 },
        ],
        [
          { revenue: 10000, resourceUsage: { Ballroom: 1 } },
          { revenue: 15000, resourceUsage: { Ballroom: 1, Garden: 1 } },
          { revenue: 3000, resourceUsage: { Ballroom: 1 } },
          { revenue: 20000, resourceUsage: { Ballroom: 2, Garden: 1 } },
        ],
      )

      expect(result.acceptedRequests.length).toBeGreaterThan(0)
      expect(result.optimalRevenue).toBeGreaterThan(0)
      expect(result.bidPrices.size).toBe(2)
    })

    it('respects capacity constraints', () => {
      const result = bidPriceControl(
        [{ name: 'Room', capacity: 2 }],
        [
          { revenue: 1000, resourceUsage: { Room: 1 } },
          { revenue: 2000, resourceUsage: { Room: 1 } },
          { revenue: 500, resourceUsage: { Room: 1 } },
        ],
      )

      // Should accept at most 2 requests (capacity = 2)
      const totalUsage = result.acceptedRequests.reduce(
        (s, idx) => s + ([{ revenue: 1000, resourceUsage: { Room: 1 } },
          { revenue: 2000, resourceUsage: { Room: 1 } },
          { revenue: 500, resourceUsage: { Room: 1 } }][idx]!.resourceUsage['Room'] ?? 0),
        0,
      )
      expect(totalUsage).toBeLessThanOrEqual(2)
    })

    it('prefers higher-revenue requests', () => {
      const result = bidPriceControl(
        [{ name: 'Room', capacity: 1 }],
        [
          { revenue: 500, resourceUsage: { Room: 1 } },
          { revenue: 5000, resourceUsage: { Room: 1 } },
        ],
      )

      // Should accept the $5000 request
      expect(result.acceptedRequests).toContain(1)
      expect(result.optimalRevenue).toBe(5000)
    })
  })

  describe('Choice-Based RM', () => {
    it('selects an offer set', () => {
      const result = choiceBasedRM(
        ['Wedding Saturday', 'Wedding Sunday', 'Corporate Monday', 'Party Friday'],
        [3.0, 2.5, 2.0, 1.5], // utilities
        0.5, // no-choice utility
        [{ name: 'Weekend', capacity: 3 }, { name: 'Weekday', capacity: 5 }],
        [
          { Weekend: 1 },
          { Weekend: 1 },
          { Weekday: 1 },
          { Weekend: 1 },
        ],
        [15000, 12000, 8000, 3000], // revenues
        2.0, // arrival rate
        30, // time horizon
      )

      expect(result.offerSet.length).toBe(4)
      expect(result.expectedRevenue).toBeGreaterThan(0)
      expect(result.choiceProbabilities.length).toBe(4)

      // All probabilities should sum to ≤ 1
      const totalProb = result.choiceProbabilities.reduce((s, p) => s + p, 0)
      expect(totalProb).toBeLessThanOrEqual(1.01)
    })
  })

  describe('Gallego-van Ryzin Pricing', () => {
    it('markup is 1/α when capacity is abundant', () => {
      const price = gallegoVanRyzinPrice(1.0, 0.01, 100, 10)
      expect(price).toBeCloseTo(100, -1) // 1/0.01 = 100
    })

    it('price increases with scarcity', () => {
      const priceAbundant = gallegoVanRyzinPrice(1.0, 0.01, 100, 10)
      const priceScarce = gallegoVanRyzinPrice(1.0, 0.01, 2, 10)
      expect(priceScarce).toBeGreaterThan(priceAbundant)
    })

    it('returns Infinity when sold out', () => {
      const price = gallegoVanRyzinPrice(1.0, 0.01, 0, 10)
      expect(price).toBe(Infinity)
    })

    it('returns 0 when no time remaining', () => {
      const price = gallegoVanRyzinPrice(1.0, 0.01, 5, 0)
      expect(price).toBe(0)
    })
  })
})
