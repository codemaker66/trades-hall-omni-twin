import { describe, it, expect } from 'vitest'
import {
  blackScholesCall,
  blackScholesPut,
  blackScholes,
  impliedVolatility,
  americanOptionBinomial,
  americanOptionWithBoundary,
  computeFloorPrice,
  computeHoldFee,
  normCDF,
} from '../black-scholes'

describe('SP-2: Black-Scholes Booking Options', () => {
  // -----------------------------------------------------------------------
  // Normal Distribution
  // -----------------------------------------------------------------------
  describe('Normal CDF', () => {
    it('N(0) = 0.5', () => {
      expect(normCDF(0)).toBeCloseTo(0.5, 5)
    })

    it('N(-∞) ≈ 0', () => {
      expect(normCDF(-10)).toBeCloseTo(0, 5)
    })

    it('N(+∞) ≈ 1', () => {
      expect(normCDF(10)).toBeCloseTo(1, 5)
    })

    it('N(1.96) ≈ 0.975', () => {
      expect(normCDF(1.96)).toBeCloseTo(0.975, 2)
    })

    it('symmetry: N(-x) = 1 - N(x)', () => {
      expect(normCDF(-1.5) + normCDF(1.5)).toBeCloseTo(1, 5)
    })
  })

  // -----------------------------------------------------------------------
  // European Options
  // -----------------------------------------------------------------------
  describe('European Call', () => {
    it('ATM call has price > 0', () => {
      const result = blackScholesCall(100, 100, 1, 0.05, 0.2)
      expect(result.price).toBeGreaterThan(0)
    })

    it('deep ITM call ≈ S - K·e^{-rT}', () => {
      const result = blackScholesCall(200, 100, 1, 0.05, 0.2)
      const intrinsic = 200 - 100 * Math.exp(-0.05)
      expect(result.price).toBeCloseTo(intrinsic, 0)
    })

    it('deep OTM call ≈ 0', () => {
      const result = blackScholesCall(50, 200, 0.1, 0.05, 0.2)
      expect(result.price).toBeCloseTo(0, 1)
    })

    it('delta is between 0 and 1 for calls', () => {
      const result = blackScholesCall(100, 100, 1, 0.05, 0.2)
      expect(result.delta).toBeGreaterThan(0)
      expect(result.delta).toBeLessThan(1)
    })

    it('ATM delta ≈ 0.5', () => {
      const result = blackScholesCall(100, 100, 1, 0.0, 0.2)
      expect(result.delta).toBeCloseTo(0.5, 1)
    })

    it('gamma is positive', () => {
      const result = blackScholesCall(100, 100, 1, 0.05, 0.2)
      expect(result.gamma).toBeGreaterThan(0)
    })

    it('theta is negative (time decay)', () => {
      const result = blackScholesCall(100, 100, 1, 0.05, 0.2)
      expect(result.theta).toBeLessThan(0)
    })

    it('vega is positive', () => {
      const result = blackScholesCall(100, 100, 1, 0.05, 0.2)
      expect(result.vega).toBeGreaterThan(0)
    })
  })

  describe('European Put', () => {
    it('ATM put has price > 0', () => {
      const result = blackScholesPut(100, 100, 1, 0.05, 0.2)
      expect(result.price).toBeGreaterThan(0)
    })

    it('put delta is between -1 and 0', () => {
      const result = blackScholesPut(100, 100, 1, 0.05, 0.2)
      expect(result.delta).toBeGreaterThan(-1)
      expect(result.delta).toBeLessThan(0)
    })

    it('put-call parity: C - P = S - K·e^{-rT}', () => {
      const s = 100, k = 100, t = 1, r = 0.05, sigma = 0.2
      const call = blackScholesCall(s, k, t, r, sigma)
      const put = blackScholesPut(s, k, t, r, sigma)
      const parity = s - k * Math.exp(-r * t)
      expect(call.price - put.price).toBeCloseTo(parity, 4)
    })
  })

  describe('blackScholes dispatcher', () => {
    it('call type matches blackScholesCall', () => {
      const a = blackScholes(100, 100, 1, 0.05, 0.2, 'call')
      const b = blackScholesCall(100, 100, 1, 0.05, 0.2)
      expect(a.price).toBe(b.price)
    })

    it('put type matches blackScholesPut', () => {
      const a = blackScholes(100, 100, 1, 0.05, 0.2, 'put')
      const b = blackScholesPut(100, 100, 1, 0.05, 0.2)
      expect(a.price).toBe(b.price)
    })
  })

  // -----------------------------------------------------------------------
  // Implied Volatility
  // -----------------------------------------------------------------------
  describe('Implied Volatility', () => {
    it('recovers original volatility from BS price', () => {
      const sigma = 0.25
      const price = blackScholesCall(100, 100, 1, 0.05, sigma).price
      const iv = impliedVolatility(price, 100, 100, 1, 0.05, 'call')
      expect(iv).toBeCloseTo(sigma, 3)
    })

    it('works for OTM call', () => {
      const sigma = 0.30
      const price = blackScholesCall(100, 120, 0.5, 0.03, sigma).price
      const iv = impliedVolatility(price, 100, 120, 0.5, 0.03, 'call')
      expect(iv).toBeCloseTo(sigma, 2)
    })

    it('works for ITM put', () => {
      const sigma = 0.35
      const price = blackScholesPut(100, 110, 1, 0.05, sigma).price
      const iv = impliedVolatility(price, 100, 110, 1, 0.05, 'put')
      expect(iv).toBeCloseTo(sigma, 2)
    })
  })

  // -----------------------------------------------------------------------
  // American Options (Binomial Tree)
  // -----------------------------------------------------------------------
  describe('American Option (CRR Binomial)', () => {
    it('American call on non-dividend = European call', () => {
      const european = blackScholesCall(100, 100, 1, 0.05, 0.2).price
      const american = americanOptionBinomial(100, 100, 1, 0.05, 0.2, 200, 'call')
      // American call = European call when no dividends
      expect(american).toBeCloseTo(european, 1)
    })

    it('American put ≥ European put', () => {
      const european = blackScholesPut(100, 100, 1, 0.05, 0.2).price
      const american = americanOptionBinomial(100, 100, 1, 0.05, 0.2, 200, 'put')
      expect(american).toBeGreaterThanOrEqual(european - 0.01)
    })

    it('American put ≥ intrinsic value', () => {
      const k = 110
      const s = 100
      const intrinsic = Math.max(k - s, 0)
      const american = americanOptionBinomial(s, k, 1, 0.05, 0.2, 200, 'put')
      expect(american).toBeGreaterThanOrEqual(intrinsic - 0.01)
    })

    it('converges with more steps', () => {
      const p50 = americanOptionBinomial(100, 100, 1, 0.05, 0.3, 50, 'put')
      const p200 = americanOptionBinomial(100, 100, 1, 0.05, 0.3, 200, 'put')
      const p500 = americanOptionBinomial(100, 100, 1, 0.05, 0.3, 500, 'put')
      // Should converge: difference decreases
      expect(Math.abs(p500 - p200)).toBeLessThan(Math.abs(p200 - p50))
    })

    it('americanOptionWithBoundary returns exercise boundary', () => {
      const result = americanOptionWithBoundary(100, 100, 1, 0.05, 0.2, 50, 'put')
      expect(result.price).toBeGreaterThan(0)
      expect(result.exerciseBoundary.length).toBe(51) // nSteps + 1
    })
  })

  // -----------------------------------------------------------------------
  // Floor Price & Hold Fee
  // -----------------------------------------------------------------------
  describe('Floor Price', () => {
    it('floor price < current demand value', () => {
      const floor = computeFloorPrice(1000, 30, 0.3, 0.05)
      expect(floor).toBeLessThan(1000)
      expect(floor).toBeGreaterThan(0)
    })

    it('floor price increases as event approaches (less time value)', () => {
      const floor30 = computeFloorPrice(1000, 30, 0.3, 0.05)
      const floor7 = computeFloorPrice(1000, 7, 0.3, 0.05)
      const floor1 = computeFloorPrice(1000, 1, 0.3, 0.05)
      expect(floor7).toBeGreaterThan(floor30)
      expect(floor1).toBeGreaterThan(floor7)
    })

    it('floor price decreases with higher volatility', () => {
      const floorLow = computeFloorPrice(1000, 30, 0.1, 0.05)
      const floorHigh = computeFloorPrice(1000, 30, 0.5, 0.05)
      expect(floorHigh).toBeLessThan(floorLow)
    })
  })

  describe('Hold Fee', () => {
    it('hold fee is positive', () => {
      const result = computeHoldFee(1000, 1000, 30, 0.3, 0.05)
      expect(result.price).toBeGreaterThan(0)
    })

    it('hold fee increases with longer hold period', () => {
      const fee7 = computeHoldFee(1000, 1000, 7, 0.3, 0.05)
      const fee30 = computeHoldFee(1000, 1000, 30, 0.3, 0.05)
      expect(fee30.price).toBeGreaterThan(fee7.price)
    })

    it('hold fee increases with higher volatility', () => {
      const feeLow = computeHoldFee(1000, 1000, 14, 0.1, 0.05)
      const feeHigh = computeHoldFee(1000, 1000, 14, 0.5, 0.05)
      expect(feeHigh.price).toBeGreaterThan(feeLow.price)
    })
  })
})
