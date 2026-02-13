/**
 * Comprehensive tests for the optimization solvers:
 *   PS-8: NSGA-II multi-objective optimization
 *   PS-9: CMA-ES covariance matrix adaptation
 *   PS-7: MCMC layout sampling (MH + HMC + diagnostics)
 */

import { describe, it, expect } from 'vitest'
import { nsga2 } from '../nsga2.js'
import { cmaes } from '../cmaes.js'
import {
  sampleLayoutsMH,
  sampleLayoutsHMC,
  layoutDiversity,
  effectiveSampleSize,
} from '../mcmc.js'
import type { HMCConfig } from '../mcmc.js'
import { CrossoverType, createPRNG } from '../types.js'
import type { NSGA2Config, CMAESConfig, MCMCConfig } from '../types.js'

// ---------------------------------------------------------------------------
// Shared test helpers
// ---------------------------------------------------------------------------

/** Simple quadratic energy: f(x) = sum(x_i^2) */
function sphereEnergy(state: Float64Array): number {
  let sum = 0
  for (let i = 0; i < state.length; i++) {
    sum += state[i]! * state[i]!
  }
  return sum
}

/** Rosenbrock 2D: f(x,y) = (1-x)^2 + 100*(y-x^2)^2.  Minimum at (1,1) = 0 */
function rosenbrockEnergy(state: Float64Array): number {
  const x = state[0]!
  const y = state[1]!
  return (1 - x) ** 2 + 100 * (y - x * x) ** 2
}

/** Neighbor function: perturb each dimension by small Gaussian noise */
function gaussianNeighbor(state: Float64Array, rng: { random(): number }): Float64Array {
  const result = new Float64Array(state.length)
  for (let i = 0; i < state.length; i++) {
    // Box-Muller for a quick normal variate
    const u1 = rng.random() || 1e-10
    const u2 = rng.random()
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
    result[i] = state[i]! + 0.1 * z
  }
  return result
}

// ---------------------------------------------------------------------------
// PS-8: NSGA-II Tests
// ---------------------------------------------------------------------------

describe('PS-8: NSGA-II', () => {
  const baseConfig: NSGA2Config = {
    populationSize: 20,
    generations: 10,
    crossoverRate: 0.9,
    mutationRate: 0.1,
    crossoverType: CrossoverType.SBX,
    seed: 123,
  }

  /** Two-objective function: minimize [x^2, (x-2)^2] */
  function biObjective(state: Float64Array): Float64Array {
    const x = state[0]!
    return new Float64Array([x * x, (x - 2) * (x - 2)])
  }

  /** Simple 2D multi-objective: minimize [sum(x_i^2), sum((x_i - 1)^2)] */
  function twoObjSphere(state: Float64Array): Float64Array {
    let obj1 = 0
    let obj2 = 0
    for (let i = 0; i < state.length; i++) {
      obj1 += state[i]! * state[i]!
      obj2 += (state[i]! - 1) * (state[i]! - 1)
    }
    return new Float64Array([obj1, obj2])
  }

  it('returns ParetoSolution[] with correct shape', () => {
    const initial = [new Float64Array([1.0])]
    const result = nsga2(initial, biObjective, baseConfig)

    expect(Array.isArray(result)).toBe(true)
    expect(result.length).toBeGreaterThan(0)

    for (const sol of result) {
      expect(sol).toHaveProperty('state')
      expect(sol).toHaveProperty('objectives')
      expect(sol).toHaveProperty('frontRank')
      expect(sol).toHaveProperty('crowdingDistance')
      expect(sol.state).toBeInstanceOf(Float64Array)
      expect(sol.objectives).toBeInstanceOf(Float64Array)
      expect(sol.objectives.length).toBe(2)
      expect(typeof sol.frontRank).toBe('number')
      expect(typeof sol.crowdingDistance).toBe('number')
    }
  })

  it('front rank 0 solutions are non-dominated', () => {
    const initial = [new Float64Array([0.5])]
    const result = nsga2(initial, biObjective, baseConfig)

    const rank0 = result.filter(s => s.frontRank === 0)
    expect(rank0.length).toBeGreaterThan(0)

    // Verify no rank-0 solution is dominated by another rank-0 solution
    for (let i = 0; i < rank0.length; i++) {
      for (let j = 0; j < rank0.length; j++) {
        if (i === j) continue
        const a = rank0[i]!.objectives
        const b = rank0[j]!.objectives

        // Check that b does NOT dominate a
        // (b dominates a: b <= a on all objectives AND b < a on at least one)
        let bNoWorse = true
        let bStrictlyBetter = false
        for (let m = 0; m < a.length; m++) {
          if (b[m]! > a[m]!) bNoWorse = false
          if (b[m]! < a[m]!) bStrictlyBetter = true
        }
        const bDominatesA = bNoWorse && bStrictlyBetter
        expect(bDominatesA).toBe(false)
      }
    }
  })

  it('crowding distances are positive for front-0 solutions', () => {
    const initial = [new Float64Array([0.5])]
    const result = nsga2(initial, biObjective, baseConfig)

    const rank0 = result.filter(s => s.frontRank === 0)
    for (const sol of rank0) {
      // Crowding distance should be > 0 (boundary solutions get Infinity)
      expect(sol.crowdingDistance).toBeGreaterThan(0)
    }
  })

  it('multi-objective [x^2, (x-2)^2] — Pareto front spans [0, 2]', () => {
    const config: NSGA2Config = {
      ...baseConfig,
      populationSize: 60,
      generations: 50,
      seed: 42,
    }
    // Seed the initial population with points near both extremes
    const initial = [
      new Float64Array([0.1]),
      new Float64Array([1.0]),
      new Float64Array([1.9]),
    ]
    const result = nsga2(initial, biObjective, config)

    const rank0 = result.filter(s => s.frontRank === 0)
    expect(rank0.length).toBeGreaterThan(1)

    // Extract x values from Pareto front
    const xValues = rank0.map(s => s.state[0]!)
    const minX = Math.min(...xValues)
    const maxX = Math.max(...xValues)

    // Pareto front should span a range within [0, 2] since:
    //   - x=0 minimizes x^2 (obj1=0, obj2=4)
    //   - x=2 minimizes (x-2)^2 (obj1=4, obj2=0)
    // With sufficient generations the front should cover a decent part of [0,2]
    expect(minX).toBeLessThan(0.5)
    expect(maxX).toBeGreaterThan(1.5)
  })

  it('population size matches config', () => {
    const initial = [new Float64Array([1.0])]
    const result = nsga2(initial, biObjective, baseConfig)

    // nsga2 returns the full final population
    expect(result.length).toBe(baseConfig.populationSize)
  })

  it('works with Uniform crossover type', () => {
    const config: NSGA2Config = {
      ...baseConfig,
      crossoverType: CrossoverType.Uniform,
    }
    const initial = [new Float64Array([1.0])]
    const result = nsga2(initial, biObjective, config)

    expect(result.length).toBe(config.populationSize)
    expect(result[0]!.frontRank).toBe(0)
  })

  it('works with higher-dimensional state vectors', () => {
    const initial = [new Float64Array([0.5, 0.5])]
    const result = nsga2(initial, twoObjSphere, baseConfig)

    expect(result.length).toBe(baseConfig.populationSize)
    for (const sol of result) {
      expect(sol.state.length).toBe(2)
      expect(sol.objectives.length).toBe(2)
    }
  })

  it('is deterministic with the same seed', () => {
    const initial = [new Float64Array([1.0])]
    const result1 = nsga2(initial, biObjective, baseConfig)
    const result2 = nsga2(initial, biObjective, baseConfig)

    expect(result1.length).toBe(result2.length)
    for (let i = 0; i < result1.length; i++) {
      expect(result1[i]!.state[0]).toBe(result2[i]!.state[0])
      expect(result1[i]!.objectives[0]).toBe(result2[i]!.objectives[0])
      expect(result1[i]!.objectives[1]).toBe(result2[i]!.objectives[1])
    }
  })
})

// ---------------------------------------------------------------------------
// PS-9: CMA-ES Tests
// ---------------------------------------------------------------------------

describe('PS-9: CMA-ES', () => {
  const baseConfig: CMAESConfig = {
    initialSigma: 0.5,
    maxEvaluations: 2000,
    seed: 42,
  }

  it('minimizes sphere function f(x) = sum(x_i^2)', () => {
    const initial = new Float64Array([2.0, -1.5])
    const result = cmaes(initial, baseConfig, sphereEnergy)

    expect(result.bestEnergy).toBeLessThan(0.01)
    // Best state should be near the origin
    for (let i = 0; i < result.bestState.length; i++) {
      expect(Math.abs(result.bestState[i]!)).toBeLessThan(0.2)
    }
  })

  it('minimizes Rosenbrock (2D) to near-optimum', () => {
    const config: CMAESConfig = {
      initialSigma: 0.5,
      maxEvaluations: 5000,
      seed: 42,
    }
    const initial = new Float64Array([0.0, 0.0])
    const result = cmaes(initial, config, rosenbrockEnergy)

    // Rosenbrock is harder; we just want substantial progress toward (1,1)
    expect(result.bestEnergy).toBeLessThan(1.0)
  })

  it('result has lower energy than initial', () => {
    const initial = new Float64Array([3.0, -2.0, 1.5])
    const initialEnergy = sphereEnergy(initial)
    const result = cmaes(initial, baseConfig, sphereEnergy)

    expect(result.bestEnergy).toBeLessThan(initialEnergy)
  })

  it('is deterministic with the same seed', () => {
    const initial = new Float64Array([1.0, -1.0])
    const result1 = cmaes(initial, baseConfig, sphereEnergy)
    const result2 = cmaes(initial, baseConfig, sphereEnergy)

    expect(result1.bestEnergy).toBe(result2.bestEnergy)
    expect(result1.evaluations).toBe(result2.evaluations)
    for (let i = 0; i < result1.bestState.length; i++) {
      expect(result1.bestState[i]).toBe(result2.bestState[i])
    }
  })

  it('returns correct result shape', () => {
    const initial = new Float64Array([1.0])
    const result = cmaes(initial, baseConfig, sphereEnergy)

    expect(result).toHaveProperty('bestState')
    expect(result).toHaveProperty('bestEnergy')
    expect(result).toHaveProperty('evaluations')
    expect(result.bestState).toBeInstanceOf(Float64Array)
    expect(typeof result.bestEnergy).toBe('number')
    expect(typeof result.evaluations).toBe('number')
    expect(result.evaluations).toBeGreaterThan(0)
    expect(result.evaluations).toBeLessThanOrEqual(baseConfig.maxEvaluations)
  })

  it('respects maxEvaluations limit', () => {
    const config: CMAESConfig = {
      initialSigma: 0.5,
      maxEvaluations: 50,
      seed: 42,
    }
    const initial = new Float64Array([5.0, 5.0])
    const result = cmaes(initial, config, sphereEnergy)

    expect(result.evaluations).toBeLessThanOrEqual(config.maxEvaluations + 10)
  })

  it('handles bounds constraints', () => {
    const initial = new Float64Array([0.5, 0.5])
    const bounds = {
      lower: new Float64Array([-1, -1]),
      upper: new Float64Array([1, 1]),
    }
    const result = cmaes(initial, baseConfig, sphereEnergy, bounds)

    // Best state should remain within or very close to bounds
    for (let i = 0; i < result.bestState.length; i++) {
      expect(result.bestState[i]!).toBeGreaterThanOrEqual(-1.01)
      expect(result.bestState[i]!).toBeLessThanOrEqual(1.01)
    }
  })

  it('produces different results with different seeds', () => {
    const initial = new Float64Array([2.0, -1.0])
    const config1: CMAESConfig = { ...baseConfig, seed: 1 }
    const config2: CMAESConfig = { ...baseConfig, seed: 9999 }

    const result1 = cmaes(initial, config1, sphereEnergy)
    const result2 = cmaes(initial, config2, sphereEnergy)

    // With different seeds, at least the evaluation trajectories differ
    // (both should converge to near-zero, but paths differ)
    const statesDiffer = result1.bestState[0] !== result2.bestState[0] ||
      result1.bestState[1] !== result2.bestState[1]
    const evalsDiffer = result1.evaluations !== result2.evaluations
    expect(statesDiffer || evalsDiffer).toBe(true)
  })
})

// ---------------------------------------------------------------------------
// PS-7: MCMC Tests
// ---------------------------------------------------------------------------

describe('PS-7: MCMC', () => {
  const baseMCMCConfig: MCMCConfig = {
    temperature: 1.0,
    nSamples: 20,
    thin: 1,
    burnIn: 10,
    seed: 42,
  }

  describe('sampleLayoutsMH', () => {
    it('returns the correct number of samples', () => {
      const initial = new Float64Array([1.0, -0.5])
      const result = sampleLayoutsMH(initial, baseMCMCConfig, sphereEnergy, gaussianNeighbor)

      expect(result.samples.length).toBe(baseMCMCConfig.nSamples)
      expect(result.energies.length).toBe(baseMCMCConfig.nSamples)
    })

    it('acceptance rate is in (0, 1)', () => {
      const initial = new Float64Array([0.5, 0.3])
      const result = sampleLayoutsMH(initial, baseMCMCConfig, sphereEnergy, gaussianNeighbor)

      expect(result.acceptanceRate).toBeGreaterThan(0)
      expect(result.acceptanceRate).toBeLessThanOrEqual(1)
    })

    it('samples have correct dimensionality', () => {
      const dim = 4
      const initial = new Float64Array(dim).fill(1.0)
      const result = sampleLayoutsMH(initial, baseMCMCConfig, sphereEnergy, gaussianNeighbor)

      for (const sample of result.samples) {
        expect(sample).toBeInstanceOf(Float64Array)
        expect(sample.length).toBe(dim)
      }
    })

    it('energies correspond to sample states', () => {
      const initial = new Float64Array([1.0, 0.5])
      const result = sampleLayoutsMH(initial, baseMCMCConfig, sphereEnergy, gaussianNeighbor)

      // Each reported energy should match the energy of the corresponding sample
      for (let i = 0; i < result.samples.length; i++) {
        const computedEnergy = sphereEnergy(result.samples[i]!)
        expect(result.energies[i]).toBeCloseTo(computedEnergy, 10)
      }
    })

    it('higher temperature increases acceptance rate', () => {
      const initial = new Float64Array([5.0, 5.0])

      const coldConfig: MCMCConfig = { ...baseMCMCConfig, temperature: 0.01, nSamples: 50, burnIn: 0 }
      const hotConfig: MCMCConfig = { ...baseMCMCConfig, temperature: 100, nSamples: 50, burnIn: 0 }

      const coldResult = sampleLayoutsMH(initial, coldConfig, sphereEnergy, gaussianNeighbor)
      const hotResult = sampleLayoutsMH(initial, hotConfig, sphereEnergy, gaussianNeighbor)

      // Higher temperature should generally produce a higher acceptance rate
      expect(hotResult.acceptanceRate).toBeGreaterThanOrEqual(coldResult.acceptanceRate)
    })

    it('thinning works correctly', () => {
      const config: MCMCConfig = {
        ...baseMCMCConfig,
        nSamples: 10,
        thin: 3,
        burnIn: 5,
      }
      const initial = new Float64Array([1.0])
      const result = sampleLayoutsMH(initial, config, sphereEnergy, gaussianNeighbor)

      expect(result.samples.length).toBe(10)
    })

    it('is deterministic with the same seed', () => {
      const initial = new Float64Array([1.0, -0.5])
      const result1 = sampleLayoutsMH(initial, baseMCMCConfig, sphereEnergy, gaussianNeighbor)
      const result2 = sampleLayoutsMH(initial, baseMCMCConfig, sphereEnergy, gaussianNeighbor)

      expect(result1.samples.length).toBe(result2.samples.length)
      for (let i = 0; i < result1.samples.length; i++) {
        for (let j = 0; j < result1.samples[i]!.length; j++) {
          expect(result1.samples[i]![j]).toBe(result2.samples[i]![j])
        }
      }
      expect(result1.acceptanceRate).toBe(result2.acceptanceRate)
    })
  })

  describe('sampleLayoutsHMC', () => {
    const hmcConfig: HMCConfig = {
      temperature: 1.0,
      nSamples: 10,
      thin: 1,
      burnIn: 5,
      leapfrogSteps: 5,
      stepSize: 0.05,
      seed: 42,
    }

    it('returns samples with correct count', () => {
      const initial = new Float64Array([0.5, -0.3])
      const result = sampleLayoutsHMC(initial, hmcConfig, sphereEnergy)

      expect(result.samples.length).toBe(hmcConfig.nSamples)
      expect(result.energies.length).toBe(hmcConfig.nSamples)
    })

    it('acceptance rate is in (0, 1)', () => {
      const initial = new Float64Array([0.5, -0.3])
      const result = sampleLayoutsHMC(initial, hmcConfig, sphereEnergy)

      expect(result.acceptanceRate).toBeGreaterThan(0)
      expect(result.acceptanceRate).toBeLessThanOrEqual(1)
    })

    it('samples have correct dimensionality', () => {
      const dim = 3
      const initial = new Float64Array(dim).fill(0.5)
      const result = sampleLayoutsHMC(initial, hmcConfig, sphereEnergy)

      for (const sample of result.samples) {
        expect(sample).toBeInstanceOf(Float64Array)
        expect(sample.length).toBe(dim)
      }
    })

    it('is deterministic with the same seed', () => {
      const initial = new Float64Array([0.5, -0.3])
      const result1 = sampleLayoutsHMC(initial, hmcConfig, sphereEnergy)
      const result2 = sampleLayoutsHMC(initial, hmcConfig, sphereEnergy)

      expect(result1.acceptanceRate).toBe(result2.acceptanceRate)
      for (let i = 0; i < result1.samples.length; i++) {
        for (let j = 0; j < result1.samples[i]!.length; j++) {
          expect(result1.samples[i]![j]).toBe(result2.samples[i]![j])
        }
      }
    })

    it('produces finite energy values', () => {
      const initial = new Float64Array([0.5, -0.3])
      const result = sampleLayoutsHMC(initial, hmcConfig, sphereEnergy)

      for (let i = 0; i < result.energies.length; i++) {
        expect(Number.isFinite(result.energies[i])).toBe(true)
      }
    })
  })

  describe('layoutDiversity', () => {
    it('returns positive number for diverse samples', () => {
      const samples = [
        new Float64Array([0, 0]),
        new Float64Array([1, 0]),
        new Float64Array([0, 1]),
        new Float64Array([1, 1]),
      ]
      const div = layoutDiversity(samples)
      expect(div).toBeGreaterThan(0)
    })

    it('returns 0 for a single sample', () => {
      const samples = [new Float64Array([1, 2, 3])]
      expect(layoutDiversity(samples)).toBe(0)
    })

    it('returns 0 for empty array', () => {
      expect(layoutDiversity([])).toBe(0)
    })

    it('returns 0 for identical samples', () => {
      const samples = [
        new Float64Array([1, 1]),
        new Float64Array([1, 1]),
        new Float64Array([1, 1]),
      ]
      expect(layoutDiversity(samples)).toBe(0)
    })

    it('more spread-out samples have higher diversity', () => {
      const tight = [
        new Float64Array([0, 0]),
        new Float64Array([0.1, 0]),
        new Float64Array([0, 0.1]),
      ]
      const spread = [
        new Float64Array([0, 0]),
        new Float64Array([10, 0]),
        new Float64Array([0, 10]),
      ]
      expect(layoutDiversity(spread)).toBeGreaterThan(layoutDiversity(tight))
    })

    it('computes correct average pairwise L2 distance', () => {
      // Two points at distance sqrt(2)
      const samples = [
        new Float64Array([0, 0]),
        new Float64Array([1, 1]),
      ]
      const div = layoutDiversity(samples)
      expect(div).toBeCloseTo(Math.sqrt(2), 10)
    })
  })

  describe('effectiveSampleSize', () => {
    it('returns reasonable ESS for uncorrelated data', () => {
      // Generate pseudo-independent samples using PRNG
      const rng = createPRNG(42)
      const n = 100
      const energies = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        energies[i] = rng.random() * 10
      }
      const ess = effectiveSampleSize(energies)

      // ESS should be close to n for truly independent samples
      // Using a loose bound since the estimator can overshoot
      expect(ess).toBeGreaterThan(n * 0.3)
      expect(ess).toBeLessThanOrEqual(n * 2) // Generous upper bound
    })

    it('returns n for constant energies', () => {
      const energies = new Float64Array(50).fill(5.0)
      const ess = effectiveSampleSize(energies)
      // Constant data has zero variance, function returns n
      expect(ess).toBe(50)
    })

    it('returns n for very small sample sizes (< 4)', () => {
      const energies = new Float64Array([1.0, 2.0, 3.0])
      expect(effectiveSampleSize(energies)).toBe(3)
    })

    it('correlated data has lower ESS than independent data', () => {
      const n = 100

      // Independent data
      const rng = createPRNG(99)
      const independent = new Float64Array(n)
      for (let i = 0; i < n; i++) independent[i] = rng.random() * 10

      // Highly correlated: random walk
      const rng2 = createPRNG(99)
      const correlated = new Float64Array(n)
      correlated[0] = 5.0
      for (let i = 1; i < n; i++) {
        correlated[i] = correlated[i - 1]! + (rng2.random() - 0.5) * 0.1
      }

      const essIndependent = effectiveSampleSize(independent)
      const essCorrelated = effectiveSampleSize(correlated)

      expect(essCorrelated).toBeLessThan(essIndependent)
    })

    it('returns a positive number', () => {
      const rng = createPRNG(7)
      const energies = new Float64Array(30)
      for (let i = 0; i < 30; i++) energies[i] = rng.random()

      const ess = effectiveSampleSize(energies)
      expect(ess).toBeGreaterThan(0)
    })
  })
})

// ---------------------------------------------------------------------------
// createPRNG sanity checks
// ---------------------------------------------------------------------------

describe('createPRNG', () => {
  it('produces values in [0, 1)', () => {
    const rng = createPRNG(42)
    for (let i = 0; i < 1000; i++) {
      const v = rng.random()
      expect(v).toBeGreaterThanOrEqual(0)
      expect(v).toBeLessThan(1)
    }
  })

  it('is deterministic with the same seed', () => {
    const rng1 = createPRNG(123)
    const rng2 = createPRNG(123)
    for (let i = 0; i < 100; i++) {
      expect(rng1.random()).toBe(rng2.random())
    }
  })

  it('produces different sequences for different seeds', () => {
    const rng1 = createPRNG(1)
    const rng2 = createPRNG(2)
    let allSame = true
    for (let i = 0; i < 10; i++) {
      if (rng1.random() !== rng2.random()) allSame = false
    }
    expect(allSame).toBe(false)
  })
})
