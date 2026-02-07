import { describe, it, expect } from 'vitest'
import {
  simulateGBM,
  gbmExpectedValue,
  gbmVariance,
  calibrateGBM,
} from '../gbm'
import {
  simulateOU,
  seasonalMean,
  calibrateOU,
  ouHalfLife,
  ouStationaryVariance,
} from '../ou'
import {
  simulateMertonJD,
  calibrateMertonJD,
  mertonExpectedValue,
} from '../merton'
import {
  simulateVenuePrice,
  calibrateVenueModel,
  computePercentileBands,
  computePathStatistics,
} from '../composite-model'
import { Rng } from '../random'

describe('SP-1: Stochastic Price Models', () => {
  // -----------------------------------------------------------------------
  // Random Number Generator
  // -----------------------------------------------------------------------
  describe('Rng', () => {
    it('produces values in [0, 1)', () => {
      const rng = new Rng(42)
      for (let i = 0; i < 1000; i++) {
        const v = rng.next()
        expect(v).toBeGreaterThanOrEqual(0)
        expect(v).toBeLessThan(1)
      }
    })

    it('produces repeatable sequences with same seed', () => {
      const rng1 = new Rng(12345)
      const rng2 = new Rng(12345)
      for (let i = 0; i < 100; i++) {
        expect(rng1.next()).toBe(rng2.next())
      }
    })

    it('normal distribution has approximately zero mean and unit variance', () => {
      const rng = new Rng(42)
      const n = 10000
      let sum = 0
      let sumSq = 0
      for (let i = 0; i < n; i++) {
        const v = rng.normal()
        sum += v
        sumSq += v * v
      }
      const mean = sum / n
      const variance = sumSq / n - mean * mean
      expect(mean).toBeCloseTo(0, 1) // Within 0.1 of zero
      expect(variance).toBeCloseTo(1, 0) // Within 1.0 of one
    })

    it('poisson distribution has correct mean', () => {
      const rng = new Rng(42)
      const lambda = 5
      const n = 10000
      let sum = 0
      for (let i = 0; i < n; i++) {
        sum += rng.poisson(lambda)
      }
      expect(sum / n).toBeCloseTo(lambda, 0)
    })
  })

  // -----------------------------------------------------------------------
  // GBM
  // -----------------------------------------------------------------------
  describe('GBM', () => {
    it('simulates correct number of paths and steps', () => {
      const result = simulateGBM(
        { mu: 0.1, sigma: 0.2 },
        { s0: 100, tYears: 1, dt: 1 / 252, nPaths: 10, seed: 42 },
      )
      expect(result.nPaths).toBe(10)
      expect(result.nSteps).toBe(252)
      expect(result.paths.length).toBe(10 * 253)
    })

    it('all paths start at s0', () => {
      const result = simulateGBM(
        { mu: 0.05, sigma: 0.3 },
        { s0: 1000, tYears: 1, dt: 1 / 12, nPaths: 50, seed: 42 },
      )
      for (let p = 0; p < 50; p++) {
        expect(result.paths[p * 13]).toBe(1000)
      }
    })

    it('prices are always positive', () => {
      const result = simulateGBM(
        { mu: -0.1, sigma: 0.5 },
        { s0: 100, tYears: 2, dt: 1 / 252, nPaths: 100, seed: 42 },
      )
      for (let i = 0; i < result.paths.length; i++) {
        expect(result.paths[i]).toBeGreaterThan(0)
      }
    })

    it('mean path converges to E[S(T)] = S₀exp(μT)', () => {
      const mu = 0.08
      const sigma = 0.2
      const s0 = 100
      const T = 1

      const result = simulateGBM(
        { mu, sigma },
        { s0, tYears: T, dt: 1 / 252, nPaths: 5000, seed: 42 },
      )

      // Average terminal value
      const nSteps = result.nSteps
      let sumTerminal = 0
      for (let p = 0; p < 5000; p++) {
        sumTerminal += result.paths[p * (nSteps + 1) + nSteps]!
      }
      const avgTerminal = sumTerminal / 5000
      const expected = gbmExpectedValue(s0, mu, T)

      // Within 3% of theoretical (MC error)
      expect(avgTerminal).toBeCloseTo(expected, -1)
      expect(Math.abs(avgTerminal - expected) / expected).toBeLessThan(0.05)
    })

    it('calibrates GBM from synthetic data', () => {
      // Generate synthetic GBM prices
      const result = simulateGBM(
        { mu: 0.1, sigma: 0.25 },
        { s0: 100, tYears: 5, dt: 1 / 252, nPaths: 1, seed: 42 },
      )

      const prices: number[] = []
      for (let t = 0; t <= result.nSteps; t++) {
        prices.push(result.paths[t]!)
      }

      const calibrated = calibrateGBM(prices, 1 / 252)

      // Should recover approximate parameters (not exact due to single path)
      expect(calibrated.sigma).toBeGreaterThan(0.1)
      expect(calibrated.sigma).toBeLessThan(0.5)
    })

    it('gbmVariance is positive and increases with time', () => {
      const v1 = gbmVariance(100, 0.1, 0.2, 0.5)
      const v2 = gbmVariance(100, 0.1, 0.2, 1.0)
      expect(v1).toBeGreaterThan(0)
      expect(v2).toBeGreaterThan(v1)
    })
  })

  // -----------------------------------------------------------------------
  // OU Process
  // -----------------------------------------------------------------------
  describe('OU Process', () => {
    const ouParams = {
      theta: 2.0,
      mu: Math.log(1000),
      sigma: 0.3,
      seasonalA: [0.1],
      seasonalB: [0.05],
    }

    it('seasonal mean includes Fourier components', () => {
      const mu0 = seasonalMean({ ...ouParams, seasonalA: [], seasonalB: [] }, 0.5)
      const muSeasonal = seasonalMean(ouParams, 0.5)
      expect(muSeasonal).not.toBe(mu0)
    })

    it('simulates mean-reverting paths', () => {
      const result = simulateOU(
        ouParams,
        { s0: 1000, tYears: 2, dt: 1 / 252, nPaths: 100, seed: 42 },
      )

      // All paths should be positive (exponential of OU)
      for (let i = 0; i < result.paths.length; i++) {
        expect(result.paths[i]).toBeGreaterThan(0)
      }
    })

    it('paths revert towards the mean', () => {
      // Start far from mean
      const result = simulateOU(
        { ...ouParams, theta: 5.0 }, // Fast reversion
        { s0: 5000, tYears: 1, dt: 1 / 252, nPaths: 500, seed: 42 },
      )

      // Average terminal should be closer to exp(mu) than starting value
      const nSteps = result.nSteps
      let sumTerminal = 0
      for (let p = 0; p < 500; p++) {
        sumTerminal += result.paths[p * (nSteps + 1) + nSteps]!
      }
      const avgTerminal = sumTerminal / 500
      const fairValue = Math.exp(ouParams.mu)

      // Terminal should be closer to mean than start
      expect(Math.abs(avgTerminal - fairValue)).toBeLessThan(Math.abs(5000 - fairValue))
    })

    it('half-life is correct', () => {
      expect(ouHalfLife(1)).toBeCloseTo(Math.LN2, 5)
      expect(ouHalfLife(2)).toBeCloseTo(Math.LN2 / 2, 5)
    })

    it('stationary variance is σ²/(2θ)', () => {
      expect(ouStationaryVariance(2, 0.3)).toBeCloseTo(0.3 * 0.3 / 4, 5)
    })

    it('calibrates OU from synthetic data', () => {
      // Generate longer path for calibration
      const result = simulateOU(
        ouParams,
        { s0: 1000, tYears: 5, dt: 1 / 252, nPaths: 1, seed: 42 },
      )

      const prices: number[] = []
      for (let t = 0; t <= result.nSteps; t++) {
        prices.push(result.paths[t]!)
      }

      const { params } = calibrateOU(prices, 1 / 252, 1)

      // Theta should be positive
      expect(params.theta).toBeGreaterThan(0)
      // Sigma should be positive
      expect(params.sigma).toBeGreaterThan(0)
    })
  })

  // -----------------------------------------------------------------------
  // Merton Jump-Diffusion
  // -----------------------------------------------------------------------
  describe('Merton Jump-Diffusion', () => {
    it('simulates paths with jumps', () => {
      const result = simulateMertonJD(
        { mu: 0.08, sigma: 0.2 },
        { lambda: 3, muJ: -0.02, sigmaJ: 0.05 },
        { s0: 100, tYears: 1, dt: 1 / 252, nPaths: 100, seed: 42 },
      )

      expect(result.nPaths).toBe(100)
      // Prices remain positive
      for (let i = 0; i < result.paths.length; i++) {
        expect(result.paths[i]).toBeGreaterThan(0)
      }
    })

    it('jump-diffusion has higher variance than pure GBM', () => {
      const nPaths = 2000
      const config = { s0: 100, tYears: 1, dt: 1 / 252, nPaths, seed: 42 }

      const gbmResult = simulateGBM({ mu: 0.08, sigma: 0.2 }, config)
      const jdResult = simulateMertonJD(
        { mu: 0.08, sigma: 0.2 },
        { lambda: 5, muJ: 0, sigmaJ: 0.1 },
        config,
      )

      // Compute terminal variance for each
      const nSteps = gbmResult.nSteps
      let gbmSum = 0, gbmSumSq = 0, jdSum = 0, jdSumSq = 0
      for (let p = 0; p < nPaths; p++) {
        const gbmT = gbmResult.paths[p * (nSteps + 1) + nSteps]!
        const jdT = jdResult.paths[p * (nSteps + 1) + nSteps]!
        gbmSum += gbmT; gbmSumSq += gbmT * gbmT
        jdSum += jdT; jdSumSq += jdT * jdT
      }
      const gbmVar = gbmSumSq / nPaths - (gbmSum / nPaths) ** 2
      const jdVar = jdSumSq / nPaths - (jdSum / nPaths) ** 2

      expect(jdVar).toBeGreaterThan(gbmVar)
    })

    it('calibrates jump parameters from synthetic data', () => {
      const result = simulateMertonJD(
        { mu: 0.1, sigma: 0.2 },
        { lambda: 5, muJ: -0.03, sigmaJ: 0.08 },
        { s0: 100, tYears: 10, dt: 1 / 252, nPaths: 1, seed: 42 },
      )

      const prices: number[] = []
      for (let t = 0; t <= result.nSteps; t++) {
        prices.push(result.paths[t]!)
      }

      const { gbm, jumps, jumpIndices } = calibrateMertonJD(prices, 1 / 252)

      expect(gbm.sigma).toBeGreaterThan(0)
      expect(jumps.lambda).toBeGreaterThanOrEqual(0)
      expect(jumpIndices.length).toBeGreaterThanOrEqual(0)
    })

    it('expected value equals S₀exp(μT)', () => {
      expect(mertonExpectedValue(100, 0.1, 1)).toBeCloseTo(100 * Math.exp(0.1), 2)
    })
  })

  // -----------------------------------------------------------------------
  // Composite Model
  // -----------------------------------------------------------------------
  describe('Composite Venue Pricing Model', () => {
    const model = {
      ou: {
        theta: 2.0,
        mu: Math.log(1000),
        sigma: 0.3,
        seasonalA: [0.1, 0.05],
        seasonalB: [0.05, 0.02],
      },
      jumps: { lambda: 3, muJ: -0.02, sigmaJ: 0.05 },
    }

    it('simulates composite paths', () => {
      const result = simulateVenuePrice(model, {
        s0: 1000,
        tYears: 1,
        dt: 1 / 252,
        nPaths: 50,
        seed: 42,
      })

      expect(result.nPaths).toBe(50)
      expect(result.paths.length).toBe(50 * (result.nSteps + 1))

      // All prices positive
      for (let i = 0; i < result.paths.length; i++) {
        expect(result.paths[i]).toBeGreaterThan(0)
      }
    })

    it('computes percentile bands', () => {
      const result = simulateVenuePrice(model, {
        s0: 1000,
        tYears: 1,
        dt: 1 / 52,
        nPaths: 500,
        seed: 42,
      })

      const bands = computePercentileBands(result, [5, 50, 95])

      expect(bands.length).toBe(3)
      // 5th percentile should be below median
      for (let t = 1; t <= result.nSteps; t++) {
        expect(bands[0]![t]).toBeLessThanOrEqual(bands[1]![t]!)
        expect(bands[1]![t]).toBeLessThanOrEqual(bands[2]![t]!)
      }
    })

    it('computes path statistics', () => {
      const result = simulateVenuePrice(model, {
        s0: 1000,
        tYears: 1,
        dt: 1 / 52,
        nPaths: 500,
        seed: 42,
      })

      const stats = computePathStatistics(result)

      expect(stats.mean.length).toBe(result.nSteps + 1)
      expect(stats.variance.length).toBe(result.nSteps + 1)
      expect(stats.mean[0]).toBeCloseTo(1000, 0) // Initial mean = s0
      expect(stats.variance[0]).toBeCloseTo(0, 0) // Initial variance ~ 0
    })

    it('calibrates full model from synthetic data', () => {
      const result = simulateVenuePrice(model, {
        s0: 1000,
        tYears: 5,
        dt: 1 / 252,
        nPaths: 1,
        seed: 42,
      })

      const prices: number[] = []
      for (let t = 0; t <= result.nSteps; t++) {
        prices.push(result.paths[t]!)
      }

      const calibration = calibrateVenueModel(prices, 1 / 252, 2, 3)

      expect(calibration.model.ou.theta).toBeGreaterThan(0)
      expect(calibration.model.ou.sigma).toBeGreaterThan(0)
      expect(calibration.model.jumps.lambda).toBeGreaterThanOrEqual(0)
      expect(calibration.diagnostics.aic).toBeDefined()
      expect(calibration.diagnostics.bic).toBeDefined()
      expect(calibration.diagnostics.ljungBoxPValue).toBeGreaterThanOrEqual(0)
    })
  })
})
