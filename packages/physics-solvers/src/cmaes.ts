/**
 * PS-9: CMA-ES — Covariance Matrix Adaptation Evolution Strategy
 *
 * After discrete structure is fixed (which tables, what arrangement pattern),
 * use CMA-ES for fine-tuning continuous positions and rotations.
 * State-of-the-art for 3-100 dimensional continuous optimization.
 *
 * The covariance update learns second-order landscape structure:
 *   C_{g+1} = (1-c1-cmu)*C_g + c1*pc*pc^T + cmu*sum w_i*(xi-m)(xi-m)^T/sigma^2
 *
 * Quasi-parameter-free: only needs initial sigma (step size).
 * Ideal for 3-100 dimensions -> 1-33 furniture items * 3 params each.
 *
 * Reference: Hansen (2016). "The CMA Evolution Strategy: A Tutorial."
 *   arXiv:1604.00772
 */

import type { CMAESConfig, EnergyFunction, PRNG } from './types.js'
import { createPRNG } from './types.js'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function normalRandom(rng: PRNG): number {
  const u1 = rng.random() || 1e-10
  const u2 = rng.random()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

/**
 * Jacobi eigendecomposition for symmetric matrix.
 * Returns eigenvalues D and eigenvectors B (columns of B are eigenvectors).
 * C = B * diag(D) * B^T
 */
function eigenDecomposition(
  C: Float64Array,
  n: number,
): { B: Float64Array; D: Float64Array } {
  // Copy C to working matrix A
  const A = new Float64Array(C)
  const V = new Float64Array(n * n)
  // Initialize V to identity
  for (let i = 0; i < n; i++) V[i * n + i] = 1.0

  const maxIter = 100 * n * n
  for (let iter = 0; iter < maxIter; iter++) {
    // Find largest off-diagonal element
    let maxVal = 0
    let p = 0
    let q = 1
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const val = Math.abs(A[i * n + j]!)
        if (val > maxVal) {
          maxVal = val
          p = i
          q = j
        }
      }
    }

    if (maxVal < 1e-15) break

    // Compute rotation angle
    const app = A[p * n + p]!
    const aqq = A[q * n + q]!
    const apq = A[p * n + q]!

    let theta: number
    if (Math.abs(app - aqq) < 1e-30) {
      theta = Math.PI / 4
    } else {
      theta = 0.5 * Math.atan2(2 * apq, app - aqq)
    }

    const c = Math.cos(theta)
    const s = Math.sin(theta)

    // Apply Givens rotation to A
    for (let i = 0; i < n; i++) {
      const aip = A[i * n + p]!
      const aiq = A[i * n + q]!
      A[i * n + p] = c * aip + s * aiq
      A[i * n + q] = -s * aip + c * aiq
    }
    for (let j = 0; j < n; j++) {
      const apj = A[p * n + j]!
      const aqj = A[q * n + j]!
      A[p * n + j] = c * apj + s * aqj
      A[q * n + j] = -s * apj + c * aqj
    }

    // Apply to eigenvectors
    for (let i = 0; i < n; i++) {
      const vip = V[i * n + p]!
      const viq = V[i * n + q]!
      V[i * n + p] = c * vip + s * viq
      V[i * n + q] = -s * vip + c * viq
    }
  }

  const D = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    D[i] = Math.max(1e-20, A[i * n + i]!) // Clamp to positive
  }

  return { B: V, D }
}

// ---------------------------------------------------------------------------
// CMA-ES
// ---------------------------------------------------------------------------

export function cmaes(
  initialMean: Float64Array,
  config: CMAESConfig,
  energyFn: EnergyFunction,
  bounds?: { lower: Float64Array; upper: Float64Array },
): { bestState: Float64Array; bestEnergy: number; evaluations: number } {
  const rng = createPRNG(config.seed ?? 42)
  const n = initialMean.length

  // Strategy parameters (Hansen defaults)
  const lambda = 4 + Math.floor(3 * Math.log(n))
  const mu = Math.floor(lambda / 2)

  // Recombination weights
  const rawWeights = new Float64Array(mu)
  for (let i = 0; i < mu; i++) {
    rawWeights[i] = Math.log(mu + 0.5) - Math.log(i + 1)
  }
  let wSum = 0
  for (let i = 0; i < mu; i++) wSum += rawWeights[i]!
  const weights = new Float64Array(mu)
  for (let i = 0; i < mu; i++) weights[i] = rawWeights[i]! / wSum

  let wSqSum = 0
  for (let i = 0; i < mu; i++) wSqSum += weights[i]! * weights[i]!
  const muEff = 1.0 / wSqSum

  // Step-size adaptation
  const cSigma = (muEff + 2) / (n + muEff + 5)
  const dSigma = 1 + 2 * Math.max(0, Math.sqrt((muEff - 1) / (n + 1)) - 1) + cSigma
  const chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

  // Covariance adaptation
  const cC = (4 + muEff / n) / (n + 4 + 2 * muEff / n)
  const c1 = 2 / ((n + 1.3) * (n + 1.3) + muEff)
  const cMu = Math.min(1 - c1, 2 * (muEff - 2 + 1 / muEff) / ((n + 2) * (n + 2) + muEff))

  // State
  let m = new Float64Array(initialMean)
  let sigma = config.initialSigma
  const C = new Float64Array(n * n)
  for (let i = 0; i < n; i++) C[i * n + i] = 1.0 // Identity

  const pSigma = new Float64Array(n)
  const pC = new Float64Array(n)

  // Eigendecomposition cache
  let { B, D } = eigenDecomposition(C, n)
  let eigenUpdateCounter = 0
  const eigenUpdateFreq = Math.max(1, Math.ceil(1 / (c1 + cMu) / n / 10))

  let bestState = new Float64Array(m)
  let bestEnergy = energyFn(m)
  let evaluations = 1
  let noImprovementCount = 0

  while (evaluations < config.maxEvaluations) {
    // Sample lambda offspring: x_k = m + sigma * B * D^{1/2} * z_k
    const offspring: Float64Array[] = []
    const fitnesses: number[] = []

    for (let k = 0; k < lambda; k++) {
      const z = new Float64Array(n)
      for (let i = 0; i < n; i++) z[i] = normalRandom(rng)

      // y = B * D^{1/2} * z
      const y = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        let sum = 0
        for (let j = 0; j < n; j++) {
          sum += B[i * n + j]! * Math.sqrt(D[j]!) * z[j]!
        }
        y[i] = sum
      }

      const x = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        x[i] = m[i]! + sigma * y[i]!
      }

      // Bounds handling with penalty
      let penalty = 0
      if (bounds) {
        for (let i = 0; i < n; i++) {
          if (x[i]! < bounds.lower[i]!) {
            penalty += (bounds.lower[i]! - x[i]!) * (bounds.lower[i]! - x[i]!)
            x[i] = bounds.lower[i]!
          }
          if (x[i]! > bounds.upper[i]!) {
            penalty += (x[i]! - bounds.upper[i]!) * (x[i]! - bounds.upper[i]!)
            x[i] = bounds.upper[i]!
          }
        }
      }

      offspring.push(x)
      fitnesses.push(energyFn(x) + 1e6 * penalty)
      evaluations++

      if (evaluations >= config.maxEvaluations) break
    }

    if (offspring.length === 0) break

    // Sort by fitness
    const indices = Array.from({ length: offspring.length }, (_, i) => i)
    indices.sort((a, b) => fitnesses[a]! - fitnesses[b]!)

    // Track best
    const currentBestEnergy = fitnesses[indices[0]!]!
    if (currentBestEnergy < bestEnergy) {
      bestEnergy = currentBestEnergy
      bestState = new Float64Array(offspring[indices[0]!]!)
      noImprovementCount = 0
    } else {
      noImprovementCount++
    }

    // Compute new mean
    const mOld = new Float64Array(m)
    const mNew = new Float64Array(n)
    for (let k = 0; k < mu && k < indices.length; k++) {
      const x = offspring[indices[k]!]!
      for (let i = 0; i < n; i++) {
        mNew[i] = mNew[i]! + weights[k]! * x[i]!
      }
    }
    m = mNew

    // Displacement
    const dm = new Float64Array(n)
    for (let i = 0; i < n; i++) dm[i] = (m[i]! - mOld[i]!) / sigma

    // Update p_sigma (cumulative step-size path)
    // Compute C^{-1/2} * dm via B * D^{-1/2} * B^T * dm
    const cinvDm = new Float64Array(n)
    // B^T * dm
    const bTdm = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let s = 0
      for (let j = 0; j < n; j++) s += B[j * n + i]! * dm[j]!
      bTdm[i] = s
    }
    // D^{-1/2} * bTdm
    for (let i = 0; i < n; i++) bTdm[i] = bTdm[i]! / Math.sqrt(D[i]!)
    // B * result
    for (let i = 0; i < n; i++) {
      let s = 0
      for (let j = 0; j < n; j++) s += B[i * n + j]! * bTdm[j]!
      cinvDm[i] = s
    }

    const csFactor = Math.sqrt(cSigma * (2 - cSigma) * muEff)
    for (let i = 0; i < n; i++) {
      pSigma[i] = (1 - cSigma) * pSigma[i]! + csFactor * cinvDm[i]!
    }

    // Update sigma
    let pSigmaNorm = 0
    for (let i = 0; i < n; i++) pSigmaNorm += pSigma[i]! * pSigma[i]!
    pSigmaNorm = Math.sqrt(pSigmaNorm)

    sigma *= Math.exp((cSigma / dSigma) * (pSigmaNorm / chiN - 1))

    // h_sigma indicator
    const gen = evaluations / lambda
    const threshold = (1.4 + 2.0 / (n + 1)) * chiN * Math.sqrt(1 - Math.pow(1 - cSigma, 2 * gen))
    const hSigma = pSigmaNorm < threshold ? 1 : 0

    // Update p_c (covariance evolution path)
    const ccFactor = Math.sqrt(cC * (2 - cC) * muEff)
    for (let i = 0; i < n; i++) {
      pC[i] = (1 - cC) * pC[i]! + hSigma * ccFactor * dm[i]!
    }

    // Update covariance matrix C
    const deltaHsigma = (1 - hSigma) * cC * (2 - cC)
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        // Rank-1 update
        let cNew = (1 - c1 - cMu) * C[i * n + j]!
        cNew += c1 * (pC[i]! * pC[j]! + deltaHsigma * C[i * n + j]!)

        // Rank-mu update
        let rankMu = 0
        for (let k = 0; k < mu && k < indices.length; k++) {
          const yi = (offspring[indices[k]!]![i]! - mOld[i]!) / sigma
          const yj = (offspring[indices[k]!]![j]! - mOld[j]!) / sigma
          rankMu += weights[k]! * yi * yj
        }
        cNew += cMu * rankMu

        C[i * n + j] = cNew
      }
    }

    // Periodically update eigendecomposition
    eigenUpdateCounter++
    if (eigenUpdateCounter >= eigenUpdateFreq) {
      // Enforce symmetry
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const avg = (C[i * n + j]! + C[j * n + i]!) / 2
          C[i * n + j] = avg
          C[j * n + i] = avg
        }
      }
      const eig = eigenDecomposition(C, n)
      B = eig.B
      D = eig.D
      eigenUpdateCounter = 0
    }

    // Termination: sigma * max eigenvalue < 1e-12
    let maxD = 0
    for (let i = 0; i < n; i++) maxD = Math.max(maxD, D[i]!)
    if (sigma * Math.sqrt(maxD) < 1e-12) break

    // No improvement for too long
    if (noImprovementCount > 10 + Math.floor(30 * n / lambda)) break
  }

  return { bestState, bestEnergy, evaluations }
}
