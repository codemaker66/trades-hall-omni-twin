import { describe, it, expect } from 'vitest'
import {
  solveHJBPricing,
  getOptimalPrice,
  getShadowPrice,
  getValueFunction,
  shouldAcceptBooking,
  generatePricingSchedule,
  solveHJBMultiSegment,
} from '../hjb-solver'

describe('SP-3: HJB Dynamic Pricing', () => {
  const baseConfig = {
    maxCapacity: 10,
    timeHorizonDays: 30,
    dt: 1,
    baseDemandRate: 0.5,
    priceSensitivity: 0.01,
  }

  it('solves HJB and returns correct dimensions', () => {
    const result = solveHJBPricing(baseConfig)

    expect(result.nCapacity).toBe(10)
    expect(result.nTimeSteps).toBe(30)
    expect(result.optimalPrices.length).toBe(11 * 31)
    expect(result.valueFunction.length).toBe(11 * 31)
    expect(result.shadowPrices.length).toBe(11 * 31)
    expect(result.iterations).toBeGreaterThan(0)
    expect(result.iterations).toBeLessThanOrEqual(20)
  })

  it('value function is zero at terminal time', () => {
    const result = solveHJBPricing(baseConfig)
    for (let n = 0; n <= 10; n++) {
      expect(getValueFunction(result, n, 30)).toBe(0)
    }
  })

  it('value function is zero when capacity is zero', () => {
    const result = solveHJBPricing(baseConfig)
    for (let t = 0; t <= 30; t++) {
      expect(getValueFunction(result, 0, t)).toBe(0)
    }
  })

  it('value function increases with capacity', () => {
    const result = solveHJBPricing(baseConfig)
    for (let t = 0; t < 30; t++) {
      for (let n = 1; n <= 10; n++) {
        expect(getValueFunction(result, n, t)).toBeGreaterThanOrEqual(
          getValueFunction(result, n - 1, t) - 0.001,
        )
      }
    }
  })

  it('optimal prices are non-negative', () => {
    const result = solveHJBPricing(baseConfig)
    for (let n = 1; n <= 10; n++) {
      for (let t = 0; t < 30; t++) {
        expect(getOptimalPrice(result, n, t)).toBeGreaterThanOrEqual(0)
      }
    }
  })

  it('prices increase as capacity decreases (scarcity effect)', () => {
    const result = solveHJBPricing(baseConfig)
    const t = 0 // Start of horizon

    // Price at capacity 2 should be >= price at capacity 10
    const priceScarcity = getOptimalPrice(result, 2, t)
    const priceAbundance = getOptimalPrice(result, 10, t)
    expect(priceScarcity).toBeGreaterThanOrEqual(priceAbundance - 0.01)
  })

  it('shadow price is non-negative', () => {
    const result = solveHJBPricing(baseConfig)
    for (let n = 1; n <= 10; n++) {
      for (let t = 0; t <= 30; t++) {
        expect(getShadowPrice(result, n, t)).toBeGreaterThanOrEqual(-0.001)
      }
    }
  })

  it('shouldAcceptBooking correctly compares to shadow price', () => {
    const result = solveHJBPricing(baseConfig)
    const shadow = getShadowPrice(result, 5, 10)

    const accept = shouldAcceptBooking(result, shadow + 10, 5, 10)
    expect(accept.accept).toBe(true)
    expect(accept.surplus).toBeCloseTo(10, 0)

    const reject = shouldAcceptBooking(result, shadow - 10, 5, 10)
    expect(reject.accept).toBe(false)
  })

  it('generates pricing schedule', () => {
    const result = solveHJBPricing(baseConfig)
    const schedule = generatePricingSchedule(result, 1)

    expect(schedule.length).toBeGreaterThan(0)
    expect(schedule[0]!.daysBeforeEvent).toBeGreaterThan(0)
    expect(schedule[0]!.capacity).toBeGreaterThanOrEqual(1)
  })

  it('seasonal factors affect pricing', () => {
    // High season: 2x demand
    const highSeason = solveHJBPricing({
      ...baseConfig,
      seasonalFactors: new Array(30).fill(2.0),
    })

    const normalResult = solveHJBPricing(baseConfig)

    // High season should have higher value function
    const highV = getValueFunction(highSeason, 5, 0)
    const normalV = getValueFunction(normalResult, 5, 0)
    expect(highV).toBeGreaterThan(normalV)
  })

  it('multi-segment solver handles multiple customer types', () => {
    const result = solveHJBMultiSegment(
      {
        maxCapacity: 10,
        timeHorizonDays: 30,
        dt: 1,
      },
      [
        { demandRate: 0.3, priceSensitivity: 0.005, name: 'wedding' },
        { demandRate: 0.5, priceSensitivity: 0.015, name: 'corporate' },
        { demandRate: 0.8, priceSensitivity: 0.025, name: 'party' },
      ],
    )

    expect(result.segmentPrices.length).toBe(3)
    // Weddings (low price sensitivity) should have higher optimal prices
    const weddingPrice = result.segmentPrices[0]![5 * 31 + 15]!
    const partyPrice = result.segmentPrices[2]![5 * 31 + 15]!
    expect(weddingPrice).toBeGreaterThan(partyPrice)
  })
})
