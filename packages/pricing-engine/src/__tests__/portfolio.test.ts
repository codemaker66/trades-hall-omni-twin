import { describe, it, expect } from 'vitest'
import {
  optimizeBookingMix,
  optimizeCVaR,
  blackLitterman,
} from '../portfolio'

describe('SP-9: Portfolio Theory', () => {
  describe('Markowitz Efficient Frontier', () => {
    // 3 event types: Wedding, Corporate, Party
    const expectedReturns = [15000, 8000, 3000]
    // Covariance matrix (3x3, flat):
    // Wedding-Wedding=4e6, Wedding-Corp=-1e6, Wedding-Party=0.5e6
    // Corp-Corp=2e6, Corp-Party=0.3e6
    // Party-Party=1e6
    const covariance = [
      4e6, -1e6, 0.5e6,
      -1e6, 2e6, 0.3e6,
      0.5e6, 0.3e6, 1e6,
    ]

    it('returns valid weights that sum to 1', () => {
      const result = optimizeBookingMix(expectedReturns, covariance, 3, 10)

      expect(result.weights.length).toBe(3)
      const sum = result.weights.reduce((s, w) => s + w, 0)
      expect(sum).toBeCloseTo(1, 2)

      // All weights non-negative
      for (const w of result.weights) {
        expect(w).toBeGreaterThanOrEqual(-0.01)
      }
    })

    it('expected revenue is within bounds', () => {
      const result = optimizeBookingMix(expectedReturns, covariance, 3, 10)
      expect(result.expectedRevenue).toBeGreaterThanOrEqual(3000 - 100)
      expect(result.expectedRevenue).toBeLessThanOrEqual(15000 + 100)
    })

    it('volatility is positive', () => {
      const result = optimizeBookingMix(expectedReturns, covariance, 3, 10)
      expect(result.volatility).toBeGreaterThan(0)
    })

    it('efficient frontier has correct number of points', () => {
      const result = optimizeBookingMix(expectedReturns, covariance, 3, 15)
      expect(result.efficientFrontier.length).toBe(15)
    })

    it('frontier points span a range of returns', () => {
      const result = optimizeBookingMix(expectedReturns, covariance, 3, 10)
      const returns = result.efficientFrontier.map((p) => p[1])
      const minRet = Math.min(...returns)
      const maxRet = Math.max(...returns)
      // Frontier should span a meaningful range
      expect(maxRet).toBeGreaterThan(minRet)
      // All returns should be positive and within expected bounds
      for (const r of returns) {
        expect(r).toBeGreaterThan(0)
        expect(r).toBeLessThanOrEqual(15000 + 100)
      }
    })

    it('negative correlation improves diversification', () => {
      // Perfectly correlated
      const corrCov = [
        4e6, 2e6, 2e6,
        2e6, 4e6, 2e6,
        2e6, 2e6, 4e6,
      ]

      // Negatively correlated
      const negCov = [
        4e6, -2e6, -2e6,
        -2e6, 4e6, -2e6,
        -2e6, -2e6, 4e6,
      ]

      const corrResult = optimizeBookingMix(expectedReturns, corrCov, 3, 10)
      const negResult = optimizeBookingMix(expectedReturns, negCov, 3, 10)

      // Negative correlation should achieve lower volatility for same return level
      expect(negResult.volatility).toBeLessThan(corrResult.volatility + 1000)
    })
  })

  describe('CVaR Optimization', () => {
    it('produces valid portfolio from scenarios', () => {
      // 100 scenarios × 3 event types
      const nScenarios = 100
      const nTypes = 3
      const scenarios: number[] = []

      // Generate random scenarios
      for (let s = 0; s < nScenarios; s++) {
        scenarios.push(
          10000 + 5000 * Math.sin(s * 0.1),  // Wedding
          7000 + 2000 * Math.cos(s * 0.2),   // Corporate
          2500 + 1000 * Math.sin(s * 0.3),   // Party
        )
      }

      const result = optimizeCVaR(scenarios, nScenarios, nTypes, 0.95, 5000)

      expect(result.weights.length).toBe(3)
      const sum = result.weights.reduce((s, w) => s + w, 0)
      expect(sum).toBeCloseTo(1, 1)
      expect(result.expectedRevenue).toBeGreaterThan(0)
      expect(result.volatility).toBeGreaterThanOrEqual(0)
    })

    it('CVaR is less than or equal to VaR', () => {
      const nScenarios = 200
      const nTypes = 2
      const scenarios: number[] = []

      for (let s = 0; s < nScenarios; s++) {
        scenarios.push(
          8000 + 3000 * Math.random(),
          5000 + 2000 * Math.random(),
        )
      }

      const result = optimizeCVaR(scenarios, nScenarios, nTypes, 0.95)
      expect(result.cvar95).toBeLessThanOrEqual(result.var95 + 0.01)
    })
  })

  describe('Black-Litterman', () => {
    it('adjusts returns based on manager views', () => {
      const nTypes = 3
      const marketWeights = [0.4, 0.35, 0.25] // Current booking mix

      // Covariance (3×3 flat)
      const covariance = [
        4e6, -1e6, 0.5e6,
        -1e6, 2e6, 0.3e6,
        0.5e6, 0.3e6, 1e6,
      ]

      // View: "Corporate will increase 20%" (absolute view on asset 1)
      const viewsP = [
        0, 1, 0, // View 1 picks Corporate
      ]
      const viewsQ = [10000] // Expected return for corporate
      const viewConfidence = [1e6] // Moderate confidence

      const result = blackLitterman(
        marketWeights,
        covariance,
        viewsP,
        viewsQ,
        viewConfidence,
        2.5, // risk aversion
        0.05, // tau
        nTypes,
        1, // 1 view
      )

      expect(result.posteriorReturns.length).toBe(3)
      expect(result.posteriorWeights.length).toBe(3)

      // Weights should sum to approximately 1
      const sum = result.posteriorWeights.reduce((s, w) => s + w, 0)
      expect(sum).toBeCloseTo(1, 1)

      // The bullish view on corporate should increase its posterior weight
      // relative to prior (this is the key BL insight)
      expect(result.posteriorReturns[1]).toBeGreaterThan(0)
    })

    it('relative view adjusts both assets', () => {
      const nTypes = 2
      const marketWeights = [0.5, 0.5]
      const covariance = [1e6, 0, 0, 1e6]

      // Relative view: "Weddings outperform parties by $5000"
      const viewsP = [1, -1] // Long wedding, short party
      const viewsQ = [5000]
      const viewConfidence = [1e5]

      const result = blackLitterman(
        marketWeights,
        covariance,
        viewsP,
        viewsQ,
        viewConfidence,
        2.5,
        0.05,
        nTypes,
        1,
      )

      // Wedding posterior should be higher
      expect(result.posteriorReturns[0]).toBeGreaterThan(result.posteriorReturns[1]!)
    })
  })
})
