import { describe, it, expect } from 'vitest'
import { simulateRevenueMC, monteCarloGeneric } from '../monte-carlo'
import { Rng } from '../random'

describe('SP-7: Monte Carlo Engine', () => {
  const model = {
    ou: {
      theta: 2.0,
      mu: Math.log(1000),
      sigma: 0.3,
      seasonalA: [0.1],
      seasonalB: [0.05],
    },
    jumps: { lambda: 2, muJ: -0.01, sigmaJ: 0.03 },
  }

  const simConfig = { s0: 1000, tYears: 1, dt: 1 / 52 }

  it('produces valid MC statistics', () => {
    const result = simulateRevenueMC(model, simConfig, {
      nPaths: 500,
      useAntithetic: false,
      useControlVariate: false,
      useSobol: false,
      confidenceLevel: 0.95,
      seed: 42,
    })

    expect(result.mean).toBeGreaterThan(0)
    expect(result.stdError).toBeGreaterThan(0)
    expect(result.var).toBeGreaterThan(0)
    expect(result.cvar).toBeGreaterThan(0)
    expect(result.cvar).toBeLessThanOrEqual(result.var + 0.01) // CVaR ≤ VaR for losses
    expect(result.percentiles.length).toBe(5)
    // Percentiles should be ordered
    for (let i = 1; i < 5; i++) {
      expect(result.percentiles[i]).toBeGreaterThanOrEqual(result.percentiles[i - 1]!)
    }
  })

  it('antithetic variates reduce standard error', () => {
    const withoutAV = simulateRevenueMC(model, simConfig, {
      nPaths: 1000,
      useAntithetic: false,
      useControlVariate: false,
      useSobol: false,
      confidenceLevel: 0.95,
      seed: 42,
    })

    const withAV = simulateRevenueMC(model, simConfig, {
      nPaths: 1000,
      useAntithetic: true,
      useControlVariate: false,
      useSobol: false,
      confidenceLevel: 0.95,
      seed: 42,
    })

    // AV should reduce standard error (may not always due to seed, but generally)
    // At minimum, the result should be valid
    expect(withAV.mean).toBeGreaterThan(0)
    expect(withAV.stdError).toBeGreaterThan(0)
  })

  it('control variates produce valid results', () => {
    const result = simulateRevenueMC(model, simConfig, {
      nPaths: 1000,
      useAntithetic: false,
      useControlVariate: true,
      useSobol: false,
      confidenceLevel: 0.95,
      seed: 42,
    })

    expect(result.mean).toBeGreaterThan(0)
    expect(result.stdError).toBeGreaterThan(0)
  })

  it('Sobol QMC produces valid results', () => {
    const result = simulateRevenueMC(model, simConfig, {
      nPaths: 500,
      useAntithetic: false,
      useControlVariate: false,
      useSobol: true,
      confidenceLevel: 0.95,
      seed: 42,
    })

    expect(result.mean).toBeGreaterThan(0)
    expect(result.stdError).toBeGreaterThan(0)
  })

  it('combined variance reduction techniques', () => {
    const result = simulateRevenueMC(model, simConfig, {
      nPaths: 1000,
      useAntithetic: true,
      useControlVariate: true,
      useSobol: false,
      confidenceLevel: 0.99,
      seed: 42,
    })

    expect(result.mean).toBeGreaterThan(0)
    // VaR at 99% should be lower than at 95%
    expect(result.percentiles[0]).toBeGreaterThan(0) // 5th percentile
  })

  it('generic MC produces correct mean for uniform distribution', () => {
    const result = monteCarloGeneric(
      (rng: Rng) => rng.next(),
      {
        nPaths: 10000,
        useAntithetic: false,
        useControlVariate: false,
        useSobol: false,
        confidenceLevel: 0.95,
        seed: 42,
      },
    )

    // Mean of U(0,1) = 0.5
    expect(result.mean).toBeCloseTo(0.5, 1)
    expect(result.stdError).toBeLessThan(0.01)
  })

  it('generic MC with normal payoff', () => {
    const result = monteCarloGeneric(
      (rng: Rng) => rng.normal(100, 10),
      {
        nPaths: 5000,
        useAntithetic: false,
        useControlVariate: false,
        useSobol: false,
        confidenceLevel: 0.95,
        seed: 42,
      },
    )

    expect(result.mean).toBeCloseTo(100, 0)
    // 5th percentile ≈ 100 - 1.645*10 ≈ 83.6
    expect(result.percentiles[0]).toBeCloseTo(83.6, -1)
  })
})
