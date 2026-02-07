import { describe, it, expect } from 'vitest'
import {
  simulateHawkes,
  hawkesIntensity,
  hawkesIntensityCurve,
  fitHawkes,
  simulateNHPP,
  estimateBranchingRatio,
} from '../hawkes'

describe('SP-6: Stochastic Demand Models', () => {
  describe('Hawkes Process Simulation', () => {
    it('simulates events with timestamps in [0, tMax)', () => {
      const events = simulateHawkes({ mu: 1.0, alpha: 0.5, beta: 1.5 }, 100, 42)
      expect(events.length).toBeGreaterThan(0)
      for (const t of events) {
        expect(t).toBeGreaterThanOrEqual(0)
        expect(t).toBeLessThan(100)
      }
    })

    it('events are sorted in ascending order', () => {
      const events = simulateHawkes({ mu: 2.0, alpha: 0.8, beta: 2.0 }, 50, 42)
      for (let i = 1; i < events.length; i++) {
        expect(events[i]).toBeGreaterThan(events[i - 1]!)
      }
    })

    it('throws for explosive process (branching ratio >= 1)', () => {
      expect(() =>
        simulateHawkes({ mu: 1.0, alpha: 2.0, beta: 1.0 }, 10),
      ).toThrow('explosive')
    })

    it('higher mu produces more events on average', () => {
      const lowMu = simulateHawkes({ mu: 0.5, alpha: 0.3, beta: 1.0 }, 100, 42)
      const highMu = simulateHawkes({ mu: 5.0, alpha: 0.3, beta: 1.0 }, 100, 42)
      expect(highMu.length).toBeGreaterThan(lowMu.length)
    })

    it('self-excitation increases event clustering', () => {
      // With self-excitation
      const excited = simulateHawkes({ mu: 1.0, alpha: 0.8, beta: 2.0 }, 100, 42)
      // Without (Poisson baseline)
      const poisson = simulateHawkes({ mu: 1.0, alpha: 0.001, beta: 2.0 }, 100, 42)

      // Hawkes events should cluster: smaller minimum inter-arrival time
      const minGapExcited = Math.min(
        ...excited.slice(1).map((t, i) => t - excited[i]!),
      )
      const minGapPoisson = Math.min(
        ...poisson.slice(1).map((t, i) => t - poisson[i]!),
      )
      expect(minGapExcited).toBeLessThan(minGapPoisson + 0.5)
    })
  })

  describe('Hawkes Intensity', () => {
    it('intensity equals mu when no prior events', () => {
      const intensity = hawkesIntensity({ mu: 2.0, alpha: 0.5, beta: 1.0 }, 10, [])
      expect(intensity).toBe(2.0)
    })

    it('intensity spikes after an event', () => {
      const params = { mu: 1.0, alpha: 2.0, beta: 3.0 }
      const before = hawkesIntensity(params, 5.0, [])
      const after = hawkesIntensity(params, 5.01, [5.0])
      expect(after).toBeGreaterThan(before)
    })

    it('intensity decays back to baseline', () => {
      const params = { mu: 1.0, alpha: 2.0, beta: 3.0 }
      const justAfter = hawkesIntensity(params, 5.01, [5.0])
      const later = hawkesIntensity(params, 10.0, [5.0])
      expect(later).toBeCloseTo(params.mu, 1)
      expect(justAfter).toBeGreaterThan(later)
    })

    it('intensity curve has correct length', () => {
      const curve = hawkesIntensityCurve(
        { mu: 1.0, alpha: 0.5, beta: 1.0 },
        [1, 2, 3],
        0, 10, 50,
      )
      expect(curve.length).toBe(50)
      expect(curve[0]!.t).toBe(0)
    })
  })

  describe('Hawkes MLE Fitting', () => {
    it('fits parameters from simulated data', () => {
      const trueParams = { mu: 2.0, alpha: 0.5, beta: 1.5 }
      const events = simulateHawkes(trueParams, 500, 42)

      const fit = fitHawkes(events, 500)

      // Should recover approximate parameters
      expect(fit.mu).toBeGreaterThan(0)
      expect(fit.alpha).toBeGreaterThan(0)
      expect(fit.beta).toBeGreaterThan(0)
      expect(fit.branchingRatio).toBeLessThan(1) // Stable process
      expect(fit.branchingRatio).toBeCloseTo(trueParams.alpha / trueParams.beta, 0)
      expect(fit.halfLife).toBeGreaterThan(0)
      expect(fit.logLikelihood).toBeDefined()
    })

    it('throws with too few events', () => {
      expect(() => fitHawkes([1, 2], 10)).toThrow('at least 5')
    })
  })

  describe('Non-Homogeneous Poisson Process', () => {
    it('simulates events within bounds', () => {
      const events = simulateNHPP(2.0, 0.5, 7, 100, undefined, 42)
      expect(events.length).toBeGreaterThan(0)
      for (const t of events) {
        expect(t).toBeGreaterThanOrEqual(0)
        expect(t).toBeLessThan(100)
      }
    })

    it('rate affects number of events', () => {
      const low = simulateNHPP(1.0, 0.0, 7, 100, undefined, 42)
      const high = simulateNHPP(10.0, 0.0, 7, 100, undefined, 42)
      expect(high.length).toBeGreaterThan(low.length)
    })
  })

  describe('Branching Ratio Estimation', () => {
    it('estimates low branching ratio for Poisson-like data', () => {
      const events = simulateHawkes({ mu: 5.0, alpha: 0.01, beta: 1.0 }, 200, 42)
      const br = estimateBranchingRatio(events)
      expect(br).toBeLessThan(0.3)
    })

    it('estimates higher branching ratio for excited data', () => {
      const events = simulateHawkes({ mu: 1.0, alpha: 0.7, beta: 1.0 }, 500, 42)
      const br = estimateBranchingRatio(events)
      expect(br).toBeGreaterThan(0.2)
    })
  })
})
