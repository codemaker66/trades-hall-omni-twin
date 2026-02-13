/**
 * PS-2: Parallel Tempering / Replica Exchange Monte Carlo
 *
 * Runs multiple replicas at different temperatures and periodically
 * attempts to swap states between adjacent temperature levels.
 * This allows low-temperature replicas to escape local minima by
 * borrowing configurations that explored freely at high temperature.
 *
 * Supports geometric and adaptive (Vousden et al.) temperature spacing.
 */

import type { PTConfig, PTResult, EnergyFunction, NeighborFunction, PRNG } from './types.js'
import { TempSpacing, createPRNG } from './types.js'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a geometric temperature ladder: T_i = tMin * (tMax/tMin)^(i/(n-1)) */
function buildGeometricLadder(tMin: number, tMax: number, n: number): Float64Array {
  const temps = new Float64Array(n)
  if (n === 1) {
    temps[0] = tMin
    return temps
  }
  const ratio = tMax / tMin
  for (let i = 0; i < n; i++) {
    temps[i] = tMin * Math.pow(ratio, i / (n - 1))
  }
  return temps
}

/** Clone a Float64Array */
function cloneState(src: Float64Array): Float64Array {
  const dst = new Float64Array(src.length)
  dst.set(src)
  return dst
}

// ---------------------------------------------------------------------------
// Core: Metropolis sweep at a given temperature
// ---------------------------------------------------------------------------

/**
 * Run `sweeps` Metropolis iterations on `state` at temperature `temp`.
 * Mutates `state` in-place and returns the final energy.
 */
function metropolisSweep(
  state: Float64Array,
  energy: number,
  temp: number,
  sweeps: number,
  energyFn: EnergyFunction,
  neighborFn: NeighborFunction,
  rng: PRNG,
): number {
  let currentEnergy = energy
  for (let s = 0; s < sweeps; s++) {
    const candidate = neighborFn(state, rng)
    const candidateEnergy = energyFn(candidate)
    const deltaE = candidateEnergy - currentEnergy

    if (deltaE < 0 || rng.random() < Math.exp(-deltaE / temp)) {
      state.set(candidate)
      currentEnergy = candidateEnergy
    }
  }
  return currentEnergy
}

// ---------------------------------------------------------------------------
// Adaptive temperature adjustment (Vousden et al., arXiv:1501.05823)
// ---------------------------------------------------------------------------

/** Target swap acceptance rate (optimal for random-walk chains) */
const TARGET_SWAP_RATE = 0.234

/** How often (in swap rounds) to adjust temperatures */
const ADAPT_INTERVAL = 100

/** Blend factor when bringing temperatures closer */
const BLEND_FACTOR = 0.05

/** Multiplicative spread factor when pushing temperatures apart */
const SPREAD_FACTOR = 1.05

/**
 * Adjust temperatures to equalize swap acceptance rates.
 * If a pair swaps too rarely, bring their temperatures closer.
 * If a pair swaps too often, push them apart.
 * Keeps tMin and tMax fixed (only adjusts interior replicas).
 */
function adaptTemperatures(
  temps: Float64Array,
  swapAccepts: Int32Array,
  swapAttempts: Int32Array,
): void {
  const n = temps.length
  if (n <= 2) return

  for (let i = 1; i < n - 1; i++) {
    const attempts = swapAttempts[i - 1]!
    if (attempts === 0) continue

    const rate = swapAccepts[i - 1]! / attempts
    const tLow = temps[i - 1]!
    const tHigh = temps[i + 1]!
    const tCur = temps[i]!

    if (rate < TARGET_SWAP_RATE) {
      // Too few swaps — blend toward geometric mean of neighbors
      const geoMean = Math.sqrt(tLow * tHigh)
      temps[i] = tCur + BLEND_FACTOR * (geoMean - tCur)
    } else if (rate > TARGET_SWAP_RATE + 0.15) {
      // Too many swaps — spread apart from neighbors
      const upper = Math.min(tHigh * 0.98, tCur * SPREAD_FACTOR)
      const lower = Math.max(tLow * 1.02, tCur / SPREAD_FACTOR)
      // Move toward whichever neighbor is closer
      if (tCur - tLow < tHigh - tCur) {
        temps[i] = Math.min(upper, tCur * SPREAD_FACTOR)
      } else {
        temps[i] = Math.max(lower, tCur / SPREAD_FACTOR)
      }
    }

    // Ensure monotonicity
    if (temps[i]! <= tLow) temps[i] = tLow * 1.001
    if (temps[i]! >= tHigh) temps[i] = tHigh * 0.999
  }
}

// ---------------------------------------------------------------------------
// Main: Parallel Tempering
// ---------------------------------------------------------------------------

/**
 * Parallel Tempering (Replica Exchange Monte Carlo).
 *
 * @param initialState  Starting state vector for all replicas
 * @param energyFn      Energy function to minimize
 * @param neighborFn    Neighbor generation function
 * @param config        PT configuration
 * @returns             PTResult with best state, traces, and swap statistics
 */
export function parallelTempering(
  initialState: Float64Array,
  energyFn: EnergyFunction,
  neighborFn: NeighborFunction,
  config: PTConfig,
): PTResult {
  const {
    nReplicas,
    tMin,
    tMax,
    spacing,
    sweepsPerSwap,
    totalSwaps,
    seed = 42,
  } = config

  const rng = createPRNG(seed)
  const n = Math.max(2, nReplicas)

  // Build temperature ladder
  const temps = buildGeometricLadder(tMin, tMax, n)

  // Initialize replicas — each starts from the same state
  const states: Float64Array[] = []
  const energies = new Float64Array(n)
  for (let r = 0; r < n; r++) {
    states.push(cloneState(initialState))
    energies[r] = energyFn(initialState)
  }

  // Track best-ever solution
  let bestEnergy = energies[0]!
  let bestState = cloneState(states[0]!)
  for (let r = 1; r < n; r++) {
    if (energies[r]! < bestEnergy) {
      bestEnergy = energies[r]!
      bestState = cloneState(states[r]!)
    }
  }

  // Swap statistics (for each adjacent pair)
  const swapAccepts = new Int32Array(n - 1)
  const swapAttempts = new Int32Array(n - 1)

  // Energy traces (one per replica, recording energy after each swap round)
  const traces: Float64Array[] = []
  for (let r = 0; r < n; r++) {
    traces.push(new Float64Array(totalSwaps))
  }

  // Main loop
  for (let sw = 0; sw < totalSwaps; sw++) {
    // 1. Run Metropolis sweeps independently on each replica
    for (let r = 0; r < n; r++) {
      energies[r] = metropolisSweep(
        states[r]!, energies[r]!, temps[r]!, sweepsPerSwap,
        energyFn, neighborFn, rng,
      )
    }

    // 2. Attempt swaps between adjacent replicas (even/odd alternating)
    const offset = sw % 2
    for (let i = offset; i < n - 1; i += 2) {
      swapAttempts[i] = (swapAttempts[i] ?? 0) + 1

      // Metropolis swap criterion:
      // Delta = (1/T_i - 1/T_{i+1}) * (E_i - E_{i+1})
      const betaI = 1.0 / temps[i]!
      const betaJ = 1.0 / temps[i + 1]!
      const deltaE = energies[i]! - energies[i + 1]!
      const delta = (betaI - betaJ) * deltaE

      if (delta > 0 || rng.random() < Math.exp(delta)) {
        // Swap states and energies
        const tmpState = states[i]!
        states[i] = states[i + 1]!
        states[i + 1] = tmpState

        const tmpE = energies[i]!
        energies[i] = energies[i + 1]!
        energies[i + 1] = tmpE

        swapAccepts[i] = (swapAccepts[i] ?? 0) + 1
      }
    }

    // 3. Record energy traces
    for (let r = 0; r < n; r++) {
      traces[r]![sw] = energies[r]!
    }

    // 4. Update best
    for (let r = 0; r < n; r++) {
      if (energies[r]! < bestEnergy) {
        bestEnergy = energies[r]!
        bestState = cloneState(states[r]!)
      }
    }

    // 5. Adaptive temperature adjustment (Vousden et al.)
    if (spacing === TempSpacing.Adaptive && sw > 0 && sw % ADAPT_INTERVAL === 0) {
      adaptTemperatures(temps, swapAccepts, swapAttempts)
    }
  }

  // Compute final swap acceptance rates
  const swapAcceptanceRates = new Float64Array(n - 1)
  for (let i = 0; i < n - 1; i++) {
    swapAcceptanceRates[i] = swapAttempts[i]! > 0
      ? swapAccepts[i]! / swapAttempts[i]!
      : 0
  }

  return {
    bestEnergy,
    bestState,
    replicaEnergies: new Float64Array(energies),
    swapAcceptanceRates,
    energyTraces: traces,
  }
}
