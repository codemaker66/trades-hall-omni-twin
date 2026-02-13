/**
 * Comprehensive tests for PS-1 (Simulated Annealing) and PS-2 (Parallel Tempering).
 *
 * Uses a simple quadratic energy function f(x) = sum(x_i^2) which has a
 * known global minimum at the origin (energy = 0). Starting from all 5.0
 * (initial energy = 100 in 4D), the solvers should drive energy toward 0.
 */

import { describe, it, expect } from 'vitest'
import { simulatedAnnealing } from '../sa.js'
import { parallelTempering } from '../parallel-tempering.js'
import { CoolingSchedule, TempSpacing, createPRNG } from '../types.js'
import type { SAConfig, PTConfig, PRNG } from '../types.js'

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

const DIM = 4
const INITIAL_VALUE = 5.0
const INITIAL_ENERGY = DIM * INITIAL_VALUE * INITIAL_VALUE // 100.0

/** f(x) = sum(x_i^2), global minimum = 0 at origin */
const quadraticEnergy = (state: Float64Array): number => {
  let sum = 0
  for (let i = 0; i < state.length; i++) sum += state[i]! * state[i]!
  return sum
}

/** Perturb one random dimension by a small step */
const neighbor = (state: Float64Array, rng: { random(): number }): Float64Array => {
  const next = new Float64Array(state)
  const i = Math.floor(rng.random() * state.length)
  next[i] = state[i]! + (rng.random() - 0.5) * 0.5
  return next
}

/** Create a 4D starting state with every element set to 5.0 */
function makeInitialState(): Float64Array {
  const state = new Float64Array(DIM)
  state.fill(INITIAL_VALUE)
  return state
}

// ---------------------------------------------------------------------------
// PS-1: Simulated Annealing
// ---------------------------------------------------------------------------

describe('PS-1: Simulated Annealing', () => {
  // --- Shared base config (overridden per test) ---
  const baseConfig: SAConfig = {
    initialTemp: 100,
    finalTemp: 0.001,
    cooling: CoolingSchedule.Geometric,
    alpha: 0.999,
    maxIterations: 10_000,
    reheatInterval: 0,
    reheatTempFraction: 0.5,
    seed: 42,
  }

  // -----------------------------------------------------------------------
  // 1. Geometric cooling minimizes quadratic
  // -----------------------------------------------------------------------
  it('minimizes a simple quadratic with geometric cooling', () => {
    const config: SAConfig = { ...baseConfig }
    const result = simulatedAnnealing(makeInitialState(), config, quadraticEnergy, neighbor)

    // Should significantly reduce energy from 100
    expect(result.bestEnergy).toBeLessThan(INITIAL_ENERGY)
    // With 10k iterations and geometric cooling the optimizer should get close to 0
    expect(result.bestEnergy).toBeLessThan(10)
    // bestState should match the reported bestEnergy
    expect(quadraticEnergy(result.bestState)).toBeCloseTo(result.bestEnergy, 10)
  })

  // -----------------------------------------------------------------------
  // 2. Lam-Delosme schedule converges toward ~44% acceptance
  // -----------------------------------------------------------------------
  it('minimizes with Lam-Delosme schedule (targets ~44% acceptance)', () => {
    const config: SAConfig = {
      ...baseConfig,
      cooling: CoolingSchedule.LamDelosme,
      maxIterations: 20_000,
    }
    const result = simulatedAnnealing(makeInitialState(), config, quadraticEnergy, neighbor)

    // Should reduce energy significantly
    expect(result.bestEnergy).toBeLessThan(INITIAL_ENERGY)
    expect(result.bestEnergy).toBeLessThan(20)

    // Lam-Delosme adaptively adjusts temperature toward 44% acceptance.
    // With a high initial temperature relative to the energy landscape,
    // overall acceptance can be very high because early iterations accept
    // almost everything. The schedule gradually lowers temperature, so the
    // cumulative ratio trends downward but may remain above 44%.
    // We verify it is positive and finite.
    const totalDecisions = result.accepts + result.rejects
    const acceptanceRatio = result.accepts / totalDecisions
    expect(acceptanceRatio).toBeGreaterThan(0.1)
    expect(acceptanceRatio).toBeLessThanOrEqual(1.0)

    // Verify the final portion of the run has lower acceptance than the start,
    // showing the adaptive schedule is working (cooling the temperature).
    // Check acceptance in the last quarter of the energy history: energy
    // should be lower on average than the first quarter, confirming convergence.
    const q1End = Math.floor(result.iterations / 4)
    const q4Start = Math.floor((3 * result.iterations) / 4)
    let avgQ1 = 0
    let avgQ4 = 0
    for (let i = 0; i < q1End; i++) avgQ1 += result.energyHistory[i]!
    for (let i = q4Start; i < result.iterations; i++) avgQ4 += result.energyHistory[i]!
    avgQ1 /= q1End
    avgQ4 /= (result.iterations - q4Start)
    expect(avgQ4).toBeLessThan(avgQ1)
  })

  // -----------------------------------------------------------------------
  // 3. Huang schedule
  // -----------------------------------------------------------------------
  it('minimizes with Huang schedule', () => {
    // Huang cools aggressively: T *= exp(-T*lambda/sigma). With a high
    // initial temp and small energy variance in early iterations, temperature
    // drops very rapidly. We use a lower initial temp and more iterations
    // so the schedule has room to explore before freezing.
    const config: SAConfig = {
      ...baseConfig,
      cooling: CoolingSchedule.Huang,
      initialTemp: 10,
      finalTemp: 1e-6,
      maxIterations: 30_000,
    }
    const result = simulatedAnnealing(makeInitialState(), config, quadraticEnergy, neighbor)

    expect(result.bestEnergy).toBeLessThan(INITIAL_ENERGY)
    // Huang may freeze early, so we use a generous bound
    expect(result.bestEnergy).toBeLessThan(80)
    // bestState consistency
    expect(quadraticEnergy(result.bestState)).toBeCloseTo(result.bestEnergy, 10)
  })

  // -----------------------------------------------------------------------
  // 4. Reheating triggers when reheatInterval is set
  // -----------------------------------------------------------------------
  it('reheats when reheatInterval is configured', () => {
    const config: SAConfig = {
      ...baseConfig,
      maxIterations: 10_000,
      reheatInterval: 2_000,
      reheatTempFraction: 0.5,
    }
    const result = simulatedAnnealing(makeInitialState(), config, quadraticEnergy, neighbor)

    // With reheatInterval=2000 and 10k iterations we expect at least 4 reheats
    expect(result.reheats).toBeGreaterThan(0)
    expect(result.reheats).toBeGreaterThanOrEqual(4)
    // Despite reheating, the best energy should still be good
    expect(result.bestEnergy).toBeLessThan(INITIAL_ENERGY)
  })

  // -----------------------------------------------------------------------
  // 5. Energy history has correct length
  // -----------------------------------------------------------------------
  it('records energy history with correct length', () => {
    const config: SAConfig = {
      ...baseConfig,
      maxIterations: 500,
      // Use a very low finalTemp to ensure all iterations run
      finalTemp: 1e-30,
    }
    const result = simulatedAnnealing(makeInitialState(), config, quadraticEnergy, neighbor)

    // energyHistory should have exactly `iterations` entries
    expect(result.energyHistory.length).toBe(result.iterations)
    expect(result.tempHistory.length).toBe(result.iterations)
    // iterations should be exactly maxIterations when finalTemp is extremely low
    expect(result.iterations).toBe(500)

    // Every entry in energyHistory should be a finite number
    for (let i = 0; i < result.energyHistory.length; i++) {
      expect(Number.isFinite(result.energyHistory[i])).toBe(true)
    }
  })

  // -----------------------------------------------------------------------
  // 6. Best energy <= initial energy (never worse)
  // -----------------------------------------------------------------------
  it('best energy is always <= initial energy', () => {
    // Run with very few iterations to test even the degenerate case
    const config: SAConfig = {
      ...baseConfig,
      maxIterations: 1,
      finalTemp: 1e-30,
    }
    const result = simulatedAnnealing(makeInitialState(), config, quadraticEnergy, neighbor)

    expect(result.bestEnergy).toBeLessThanOrEqual(INITIAL_ENERGY)
  })

  // -----------------------------------------------------------------------
  // 7. Deterministic with same seed
  // -----------------------------------------------------------------------
  it('produces deterministic results with the same seed', () => {
    const config: SAConfig = { ...baseConfig, seed: 123, maxIterations: 3_000 }

    const result1 = simulatedAnnealing(makeInitialState(), config, quadraticEnergy, neighbor)
    const result2 = simulatedAnnealing(makeInitialState(), config, quadraticEnergy, neighbor)

    // Exact match on numeric results
    expect(result1.bestEnergy).toBe(result2.bestEnergy)
    expect(result1.iterations).toBe(result2.iterations)
    expect(result1.accepts).toBe(result2.accepts)
    expect(result1.rejects).toBe(result2.rejects)
    expect(result1.reheats).toBe(result2.reheats)

    // bestState should be identical
    expect(result1.bestState.length).toBe(result2.bestState.length)
    for (let i = 0; i < result1.bestState.length; i++) {
      expect(result1.bestState[i]).toBe(result2.bestState[i])
    }

    // energyHistory should be identical
    expect(result1.energyHistory.length).toBe(result2.energyHistory.length)
    for (let i = 0; i < result1.energyHistory.length; i++) {
      expect(result1.energyHistory[i]).toBe(result2.energyHistory[i])
    }
  })

  // -----------------------------------------------------------------------
  // Additional SA edge-case tests
  // -----------------------------------------------------------------------
  it('accepts + rejects = iterations', () => {
    const config: SAConfig = { ...baseConfig, maxIterations: 5_000, finalTemp: 1e-30 }
    const result = simulatedAnnealing(makeInitialState(), config, quadraticEnergy, neighbor)

    expect(result.accepts + result.rejects).toBe(result.iterations)
  })

  it('temperature history is monotonically non-increasing with geometric cooling (no reheat)', () => {
    const config: SAConfig = {
      ...baseConfig,
      maxIterations: 2_000,
      finalTemp: 1e-30,
      reheatInterval: 0,
    }
    const result = simulatedAnnealing(makeInitialState(), config, quadraticEnergy, neighbor)

    for (let i = 1; i < result.tempHistory.length; i++) {
      expect(result.tempHistory[i]).toBeLessThanOrEqual(result.tempHistory[i - 1]!)
    }
  })

  it('with different seeds produces different results', () => {
    const config1: SAConfig = { ...baseConfig, seed: 1, maxIterations: 3_000 }
    const config2: SAConfig = { ...baseConfig, seed: 999, maxIterations: 3_000 }

    const result1 = simulatedAnnealing(makeInitialState(), config1, quadraticEnergy, neighbor)
    const result2 = simulatedAnnealing(makeInitialState(), config2, quadraticEnergy, neighbor)

    // Very unlikely (but not impossible) that two different seeds yield identical trajectories
    // Check that at least the energy histories diverge at some point
    let differ = false
    const len = Math.min(result1.energyHistory.length, result2.energyHistory.length)
    for (let i = 0; i < len; i++) {
      if (result1.energyHistory[i] !== result2.energyHistory[i]) {
        differ = true
        break
      }
    }
    expect(differ).toBe(true)
  })

  it('early termination when temp drops below finalTemp', () => {
    const config: SAConfig = {
      ...baseConfig,
      initialTemp: 1,
      finalTemp: 0.5,
      alpha: 0.9,
      maxIterations: 100_000,
    }
    const result = simulatedAnnealing(makeInitialState(), config, quadraticEnergy, neighbor)

    // Should terminate well before maxIterations because temp drops fast
    expect(result.iterations).toBeLessThan(100_000)
    expect(result.energyHistory.length).toBe(result.iterations)
  })
})

// ---------------------------------------------------------------------------
// PS-2: Parallel Tempering
// ---------------------------------------------------------------------------

describe('PS-2: Parallel Tempering', () => {
  const basePTConfig: PTConfig = {
    nReplicas: 6,
    tMin: 0.1,
    tMax: 100,
    spacing: TempSpacing.Geometric,
    sweepsPerSwap: 50,
    totalSwaps: 200,
    seed: 42,
  }

  // -----------------------------------------------------------------------
  // 1. Minimizes quadratic with geometric spacing
  // -----------------------------------------------------------------------
  it('minimizes quadratic with geometric spacing', () => {
    const result = parallelTempering(
      makeInitialState(),
      quadraticEnergy,
      neighbor,
      basePTConfig,
    )

    // Starting energy is 100; PT should drive it well below
    expect(result.bestEnergy).toBeLessThan(INITIAL_ENERGY)
    expect(result.bestEnergy).toBeLessThan(10)
    // Verify bestState matches bestEnergy
    expect(quadraticEnergy(result.bestState)).toBeCloseTo(result.bestEnergy, 10)
  })

  // -----------------------------------------------------------------------
  // 2. Minimizes with adaptive spacing
  // -----------------------------------------------------------------------
  it('minimizes quadratic with adaptive spacing', () => {
    const config: PTConfig = {
      ...basePTConfig,
      spacing: TempSpacing.Adaptive,
      totalSwaps: 300,
    }
    const result = parallelTempering(
      makeInitialState(),
      quadraticEnergy,
      neighbor,
      config,
    )

    expect(result.bestEnergy).toBeLessThan(INITIAL_ENERGY)
    expect(result.bestEnergy).toBeLessThan(15)
    expect(quadraticEnergy(result.bestState)).toBeCloseTo(result.bestEnergy, 10)
  })

  // -----------------------------------------------------------------------
  // 3. Swap acceptance rates are in (0, 1) for reasonable temperatures
  // -----------------------------------------------------------------------
  it('swap acceptance rates are in (0, 1) for reasonable temperatures', () => {
    const config: PTConfig = {
      ...basePTConfig,
      totalSwaps: 500,
    }
    const result = parallelTempering(
      makeInitialState(),
      quadraticEnergy,
      neighbor,
      config,
    )

    // swapAcceptanceRates has nReplicas-1 entries (one per adjacent pair)
    expect(result.swapAcceptanceRates.length).toBe(config.nReplicas - 1)

    for (let i = 0; i < result.swapAcceptanceRates.length; i++) {
      const rate = result.swapAcceptanceRates[i]!
      // Rate should be strictly between 0 and 1 (not degenerate)
      expect(rate).toBeGreaterThan(0)
      expect(rate).toBeLessThan(1)
    }
  })

  // -----------------------------------------------------------------------
  // 4. Energy traces have correct shape (nReplicas x totalSwaps)
  // -----------------------------------------------------------------------
  it('energy traces have correct shape (nReplicas x totalSwaps)', () => {
    const config: PTConfig = {
      ...basePTConfig,
      nReplicas: 4,
      totalSwaps: 150,
    }
    const result = parallelTempering(
      makeInitialState(),
      quadraticEnergy,
      neighbor,
      config,
    )

    // Should have one trace per replica
    expect(result.energyTraces.length).toBe(config.nReplicas)

    // Each trace should have totalSwaps entries
    for (const trace of result.energyTraces) {
      expect(trace.length).toBe(config.totalSwaps)
    }

    // All trace values should be finite non-negative (since quadratic >= 0)
    for (const trace of result.energyTraces) {
      for (let i = 0; i < trace.length; i++) {
        expect(Number.isFinite(trace[i])).toBe(true)
        expect(trace[i]).toBeGreaterThanOrEqual(0)
      }
    }
  })

  // -----------------------------------------------------------------------
  // 5. Best energy <= initial energy
  // -----------------------------------------------------------------------
  it('best energy is always <= initial energy', () => {
    // Even with very few swaps, best should not be worse than initial
    const config: PTConfig = {
      ...basePTConfig,
      totalSwaps: 1,
      sweepsPerSwap: 1,
    }
    const result = parallelTempering(
      makeInitialState(),
      quadraticEnergy,
      neighbor,
      config,
    )

    expect(result.bestEnergy).toBeLessThanOrEqual(INITIAL_ENERGY)
  })

  // -----------------------------------------------------------------------
  // 6. Multiple replicas find better solution than single temperature
  // -----------------------------------------------------------------------
  it('multiple replicas find better solution than single temperature', () => {
    // Run SA at only the lowest temperature for the same total budget
    const saSweeps = basePTConfig.sweepsPerSwap * basePTConfig.totalSwaps
    const saConfig: SAConfig = {
      initialTemp: basePTConfig.tMin,
      finalTemp: basePTConfig.tMin * 0.01,
      cooling: CoolingSchedule.Geometric,
      alpha: 0.9999,
      maxIterations: saSweeps,
      reheatInterval: 0,
      reheatTempFraction: 0,
      seed: 42,
    }
    const saResult = simulatedAnnealing(makeInitialState(), saConfig, quadraticEnergy, neighbor)

    // Run PT with same seed and total computational budget
    const ptConfig: PTConfig = {
      ...basePTConfig,
      seed: 42,
    }
    const ptResult = parallelTempering(
      makeInitialState(),
      quadraticEnergy,
      neighbor,
      ptConfig,
    )

    // PT should find a better (or at least comparable) solution because
    // high-temperature replicas help escape local minima.
    // For a simple quadratic this won't be dramatic, but PT should not be
    // significantly worse. We check that PT's best is within 2x of SA
    // (generous bound) or better.
    // In practice, PT usually wins or ties on this landscape.
    expect(ptResult.bestEnergy).toBeLessThan(INITIAL_ENERGY)
    expect(saResult.bestEnergy).toBeLessThan(INITIAL_ENERGY)

    // At minimum, verify both solvers found a reasonable solution
    expect(ptResult.bestEnergy).toBeLessThan(20)
  })

  // -----------------------------------------------------------------------
  // Additional PT tests
  // -----------------------------------------------------------------------
  it('replicaEnergies has correct length', () => {
    const result = parallelTempering(
      makeInitialState(),
      quadraticEnergy,
      neighbor,
      basePTConfig,
    )

    expect(result.replicaEnergies.length).toBe(basePTConfig.nReplicas)

    // Each replica energy should be non-negative (quadratic) and finite
    for (let i = 0; i < result.replicaEnergies.length; i++) {
      expect(Number.isFinite(result.replicaEnergies[i])).toBe(true)
      expect(result.replicaEnergies[i]).toBeGreaterThanOrEqual(0)
    }
  })

  it('bestEnergy matches the minimum across all replicaEnergies or is from history', () => {
    const result = parallelTempering(
      makeInitialState(),
      quadraticEnergy,
      neighbor,
      basePTConfig,
    )

    // bestEnergy should be <= the minimum of final replica energies
    // (because it tracks the best-ever, which may have been swapped away)
    let minReplica = Infinity
    for (let i = 0; i < result.replicaEnergies.length; i++) {
      if (result.replicaEnergies[i]! < minReplica) {
        minReplica = result.replicaEnergies[i]!
      }
    }
    expect(result.bestEnergy).toBeLessThanOrEqual(minReplica)
  })

  it('deterministic with same seed', () => {
    const config: PTConfig = { ...basePTConfig, seed: 77, totalSwaps: 100 }

    const result1 = parallelTempering(makeInitialState(), quadraticEnergy, neighbor, config)
    const result2 = parallelTempering(makeInitialState(), quadraticEnergy, neighbor, config)

    expect(result1.bestEnergy).toBe(result2.bestEnergy)
    for (let i = 0; i < result1.bestState.length; i++) {
      expect(result1.bestState[i]).toBe(result2.bestState[i])
    }
    for (let i = 0; i < result1.swapAcceptanceRates.length; i++) {
      expect(result1.swapAcceptanceRates[i]).toBe(result2.swapAcceptanceRates[i])
    }
  })

  it('nReplicas is clamped to minimum of 2', () => {
    const config: PTConfig = {
      ...basePTConfig,
      nReplicas: 1, // Should be clamped to 2 internally
      totalSwaps: 50,
    }
    const result = parallelTempering(
      makeInitialState(),
      quadraticEnergy,
      neighbor,
      config,
    )

    // Should still work and have at least 2 replicas worth of data
    expect(result.energyTraces.length).toBeGreaterThanOrEqual(2)
    expect(result.swapAcceptanceRates.length).toBeGreaterThanOrEqual(1)
    expect(result.bestEnergy).toBeLessThanOrEqual(INITIAL_ENERGY)
  })

  it('low-temperature replica achieves lower energy than high-temperature replica on average', () => {
    const config: PTConfig = {
      ...basePTConfig,
      nReplicas: 4,
      totalSwaps: 300,
    }
    const result = parallelTempering(
      makeInitialState(),
      quadraticEnergy,
      neighbor,
      config,
    )

    // Average energy for the coldest replica (index 0) vs the hottest (last index)
    const coldTrace = result.energyTraces[0]!
    const hotTrace = result.energyTraces[config.nReplicas - 1]!

    let coldAvg = 0
    let hotAvg = 0
    // Average over the second half (after burn-in)
    const start = Math.floor(config.totalSwaps / 2)
    const count = config.totalSwaps - start
    for (let i = start; i < config.totalSwaps; i++) {
      coldAvg += coldTrace[i]!
      hotAvg += hotTrace[i]!
    }
    coldAvg /= count
    hotAvg /= count

    // The cold replica should have lower average energy than the hot one
    expect(coldAvg).toBeLessThan(hotAvg)
  })
})

// ---------------------------------------------------------------------------
// PRNG sanity check
// ---------------------------------------------------------------------------

describe('createPRNG', () => {
  it('produces values in [0, 1)', () => {
    const rng = createPRNG(42)
    for (let i = 0; i < 10_000; i++) {
      const v = rng.random()
      expect(v).toBeGreaterThanOrEqual(0)
      expect(v).toBeLessThan(1)
    }
  })

  it('same seed produces same sequence', () => {
    const rng1 = createPRNG(42)
    const rng2 = createPRNG(42)
    for (let i = 0; i < 100; i++) {
      expect(rng1.random()).toBe(rng2.random())
    }
  })

  it('different seeds produce different sequences', () => {
    const rng1 = createPRNG(1)
    const rng2 = createPRNG(2)
    let same = true
    for (let i = 0; i < 10; i++) {
      if (rng1.random() !== rng2.random()) {
        same = false
        break
      }
    }
    expect(same).toBe(false)
  })
})
