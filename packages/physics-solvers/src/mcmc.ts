/**
 * PS-7: MCMC Layout Sampling — Diverse High-Quality Alternatives
 *
 * Rather than finding THE optimal layout, sample from the Boltzmann
 * distribution over good layouts: p(layout) ~ exp(-E(layout)/T).
 * This produces 20-50 diverse high-quality configurations for planners
 * to browse.
 *
 * Implements:
 * - Metropolis-Hastings (gradient-free, for client-side)
 * - Hamiltonian Monte Carlo (gradient via finite differences)
 * - Diversity metrics and effective sample size diagnostics
 *
 * References:
 * - Merrell et al. (2011). "Interactive Furniture Layout." ACM TOG (SIGGRAPH)
 * - Hoffman & Gelman (2014). "NUTS." JMLR 15:1351-1381
 */

import type { MCMCConfig, MCMCResult, EnergyFunction, NeighborFunction, PRNG } from './types.js'
import { createPRNG } from './types.js'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Box-Muller normal random variate */
function normalRandom(rng: PRNG): number {
  const u1 = rng.random() || 1e-10
  const u2 = rng.random()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

// ---------------------------------------------------------------------------
// Metropolis-Hastings Sampler
// ---------------------------------------------------------------------------

/**
 * Metropolis-Hastings sampler for layout configurations.
 * Gradient-free — suitable for < 20 items where NUTS overhead is unnecessary.
 */
export function sampleLayoutsMH(
  initialState: Float64Array,
  config: MCMCConfig,
  energyFn: EnergyFunction,
  neighborFn: NeighborFunction,
): MCMCResult {
  const rng = createPRNG(config.seed ?? 42)
  const { temperature, nSamples, thin, burnIn } = config
  const totalSteps = burnIn + nSamples * thin

  let state = new Float64Array(initialState)
  let energy = energyFn(state)
  let accepts = 0
  let totalProposed = 0

  const samples: Float64Array[] = []
  const energies: number[] = []
  let sampleIdx = 0

  for (let step = 0; step < totalSteps; step++) {
    // Propose neighbor
    const neighbor = neighborFn(state, rng)
    const neighborEnergy = energyFn(neighbor)
    const deltaE = neighborEnergy - energy
    totalProposed++

    // Metropolis acceptance: p = exp(-deltaE / T)
    const accept = deltaE < 0 || rng.random() < Math.exp(-deltaE / temperature)

    if (accept) {
      state = new Float64Array(neighbor)
      energy = neighborEnergy
      accepts++
    }

    // After burn-in, collect samples every thin-th step
    if (step >= burnIn && (step - burnIn) % thin === 0 && sampleIdx < nSamples) {
      samples.push(new Float64Array(state))
      energies.push(energy)
      sampleIdx++
    }
  }

  return {
    samples,
    energies: new Float64Array(energies),
    acceptanceRate: accepts / totalProposed,
  }
}

// ---------------------------------------------------------------------------
// Hamiltonian Monte Carlo
// ---------------------------------------------------------------------------

export interface HMCConfig extends MCMCConfig {
  leapfrogSteps: number
  stepSize: number
}

/**
 * Hamiltonian Monte Carlo with gradient via finite differences.
 *
 * HMC uses Hamiltonian dynamics to propose states, achieving much
 * better acceptance rates in high-dimensional spaces than random-walk MH.
 * For N furniture items × 3 params = 3N dimensions.
 */
export function sampleLayoutsHMC(
  initialState: Float64Array,
  config: HMCConfig,
  energyFn: EnergyFunction,
): MCMCResult {
  const rng = createPRNG(config.seed ?? 42)
  const { temperature, nSamples, thin, burnIn, leapfrogSteps, stepSize } = config
  const n = initialState.length
  const eps = 1e-5 // Finite difference step

  const totalSteps = burnIn + nSamples * thin
  let q = new Float64Array(initialState)
  let currentU = energyFn(q) / temperature

  let accepts = 0
  let totalProposed = 0
  const samples: Float64Array[] = []
  const energies: number[] = []
  let sampleIdx = 0

  // Gradient via central finite differences
  function gradU(pos: Float64Array): Float64Array {
    const grad = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      const fwd = new Float64Array(pos)
      const bwd = new Float64Array(pos)
      fwd[i] = fwd[i]! + eps
      bwd[i] = bwd[i]! - eps
      grad[i] = (energyFn(fwd) - energyFn(bwd)) / (2 * eps * temperature)
    }
    return grad
  }

  for (let step = 0; step < totalSteps; step++) {
    // Sample random momentum: p ~ N(0, I)
    const p = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      p[i] = normalRandom(rng)
    }

    const qNew = new Float64Array(q)
    const pNew = new Float64Array(p)

    // Leapfrog integration
    // Half step for momentum
    const grad0 = gradU(qNew)
    for (let i = 0; i < n; i++) {
      pNew[i] = pNew[i]! - stepSize * 0.5 * grad0[i]!
    }

    for (let l = 0; l < leapfrogSteps; l++) {
      // Full step for position
      for (let i = 0; i < n; i++) {
        qNew[i] = qNew[i]! + stepSize * pNew[i]!
      }

      // Full step for momentum (except at last step)
      if (l < leapfrogSteps - 1) {
        const grad = gradU(qNew)
        for (let i = 0; i < n; i++) {
          pNew[i] = pNew[i]! - stepSize * grad[i]!
        }
      }
    }

    // Half step for momentum at end
    const gradF = gradU(qNew)
    for (let i = 0; i < n; i++) {
      pNew[i] = pNew[i]! - stepSize * 0.5 * gradF[i]!
    }

    // Negate momentum for reversibility
    for (let i = 0; i < n; i++) {
      pNew[i] = -pNew[i]!
    }

    // Compute Hamiltonian: H = U(q) + K(p) where K(p) = p^2 / 2
    const proposedU = energyFn(qNew) / temperature
    let currentK = 0
    let proposedK = 0
    for (let i = 0; i < n; i++) {
      currentK += p[i]! * p[i]! * 0.5
      proposedK += pNew[i]! * pNew[i]! * 0.5
    }

    // Metropolis acceptance on the Hamiltonian
    const deltaH = (proposedU + proposedK) - (currentU + currentK)
    totalProposed++

    if (rng.random() < Math.exp(-deltaH)) {
      q = qNew
      currentU = proposedU
      accepts++
    }

    if (step >= burnIn && (step - burnIn) % thin === 0 && sampleIdx < nSamples) {
      samples.push(new Float64Array(q))
      energies.push(currentU * temperature) // Store actual energy
      sampleIdx++
    }
  }

  return {
    samples,
    energies: new Float64Array(energies),
    acceptanceRate: accepts / totalProposed,
  }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/**
 * Compute average pairwise L2 distance between samples.
 * Higher = more diverse layout collection.
 */
export function layoutDiversity(samples: Float64Array[]): number {
  const n = samples.length
  if (n < 2) return 0

  let totalDist = 0
  let count = 0
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const a = samples[i]!
      const b = samples[j]!
      let dist = 0
      for (let k = 0; k < a.length; k++) {
        const d = a[k]! - b[k]!
        dist += d * d
      }
      totalDist += Math.sqrt(dist)
      count++
    }
  }
  return totalDist / count
}

/**
 * Effective Sample Size via initial monotone sequence estimator.
 * ESS = N / (1 + 2 * sum of autocorrelations)
 *
 * Low ESS indicates high autocorrelation = poor mixing.
 * Target ESS > N/4 for reliable estimates.
 */
export function effectiveSampleSize(energies: Float64Array): number {
  const n = energies.length
  if (n < 4) return n

  // Compute mean
  let mean = 0
  for (let i = 0; i < n; i++) mean += energies[i]!
  mean /= n

  // Compute variance
  let variance = 0
  for (let i = 0; i < n; i++) {
    const d = energies[i]! - mean
    variance += d * d
  }
  variance /= n
  if (variance < 1e-20) return n

  // Compute autocorrelations using initial monotone sequence estimator
  let sumAutoCorr = 0
  const maxLag = Math.floor(n / 2)

  for (let lag = 1; lag < maxLag; lag += 2) {
    // Compute autocorrelation at lag and lag+1
    let rho1 = 0
    let rho2 = 0
    for (let i = 0; i < n - lag; i++) {
      rho1 += (energies[i]! - mean) * (energies[i + lag]! - mean)
    }
    rho1 = rho1 / (n * variance)

    if (lag + 1 < n) {
      for (let i = 0; i < n - lag - 1; i++) {
        rho2 += (energies[i]! - mean) * (energies[i + lag + 1]! - mean)
      }
      rho2 = rho2 / (n * variance)
    }

    // Initial monotone sequence: stop when pair sum is negative
    const pairSum = rho1 + rho2
    if (pairSum < 0) break

    sumAutoCorr += pairSum
  }

  return n / (1 + 2 * sumAutoCorr)
}
