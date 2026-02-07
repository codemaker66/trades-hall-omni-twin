/**
 * Merton Jump-Diffusion Model
 *
 * Extends GBM with compound Poisson jumps:
 *   dS/S = (μ - λk)dt + σdW + J·dN
 *
 * where:
 * - N is a Poisson process with intensity λ (jumps per year)
 * - J is the random jump size: log(1+J) ~ N(μ_J, σ_J²)
 * - k = E[J] = exp(μ_J + σ_J²/2) - 1 (compensator)
 *
 * The compensated drift (μ - λk) ensures the process is a martingale
 * under the risk-neutral measure when μ = r.
 *
 * Reference: Merton (1976). "Option pricing when underlying stock returns
 * are discontinuous." Journal of Financial Economics.
 */

import type { GBMParams, JumpParams, SimulationConfig, SimulationResult } from './types'
import { Rng } from './random'

/**
 * Simulate Merton jump-diffusion paths.
 *
 * At each time step:
 * 1. Sample number of jumps N ~ Poisson(λΔt)
 * 2. If N > 0, sum N independent log-normal jump sizes
 * 3. Apply drift + diffusion + jumps
 *
 * Uses exact distribution for the log-price increment:
 *   ln(S(t+dt)/S(t)) = (μ - λk - σ²/2)dt + σ√dt·Z + Σᵢ Jᵢ
 *
 * @param gbm - GBM drift and volatility parameters
 * @param jumps - Jump intensity and size distribution parameters
 * @param config - Simulation configuration
 */
export function simulateMertonJD(
  gbm: GBMParams,
  jumps: JumpParams,
  config: SimulationConfig,
): SimulationResult {
  const { mu, sigma } = gbm
  const { lambda, muJ, sigmaJ } = jumps
  const { s0, tYears, dt, nPaths, seed } = config
  const nSteps = Math.floor(tYears / dt)

  const rng = new Rng(seed)
  const paths = new Float64Array(nPaths * (nSteps + 1))

  // Jump compensator: k = E[e^J - 1] = exp(μ_J + σ_J²/2) - 1
  const k = Math.exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1

  // Compensated drift per time step
  const drift = (mu - lambda * k - 0.5 * sigma * sigma) * dt
  const diffusionScale = sigma * Math.sqrt(dt)

  for (let p = 0; p < nPaths; p++) {
    const offset = p * (nSteps + 1)
    paths[offset] = s0
    let logPrice = Math.log(s0)

    for (let t = 1; t <= nSteps; t++) {
      // Diffusion component
      const z = rng.normal()
      let increment = drift + diffusionScale * z

      // Poisson jumps
      const nJumps = rng.poisson(lambda * dt)
      for (let j = 0; j < nJumps; j++) {
        increment += rng.normal(muJ, sigmaJ)
      }

      logPrice += increment
      paths[offset + t] = Math.exp(logPrice)
    }
  }

  return { paths, nSteps, nPaths, dt }
}

/**
 * Calibrate Merton jump-diffusion parameters from a price series.
 *
 * Algorithm:
 * 1. Compute log returns rₜ = ln(S(t)/S(t-1))
 * 2. Detect jumps using threshold: |rₜ - μ̂| > κσ̂ (κ=3 by default)
 * 3. Estimate diffusion parameters (μ, σ) from non-jump returns
 * 4. Estimate jump parameters (λ, μ_J, σ_J) from jump returns
 *
 * @param prices - Historical price series
 * @param dt - Time step between observations (in years)
 * @param jumpThreshold - Number of standard deviations for jump detection (default 3)
 */
export function calibrateMertonJD(
  prices: number[],
  dt: number,
  jumpThreshold: number = 3,
): { gbm: GBMParams; jumps: JumpParams; jumpIndices: number[] } {
  if (prices.length < 10) {
    throw new Error('Need at least 10 observations for jump-diffusion calibration')
  }

  // Step 1: Log returns
  const logReturns: number[] = []
  for (let i = 1; i < prices.length; i++) {
    logReturns.push(Math.log(prices[i]! / prices[i - 1]!))
  }

  const n = logReturns.length

  // Initial estimates (before jump detection)
  const meanAll = logReturns.reduce((s, r) => s + r, 0) / n
  const varAll = logReturns.reduce((s, r) => s + (r - meanAll) ** 2, 0) / (n - 1)
  const stdAll = Math.sqrt(varAll)

  // Step 2: Detect jumps — returns exceeding κσ threshold
  const jumpIndices: number[] = []
  const normalReturns: number[] = []
  const jumpReturns: number[] = []

  for (let i = 0; i < n; i++) {
    if (Math.abs(logReturns[i]! - meanAll) > jumpThreshold * stdAll) {
      jumpIndices.push(i)
      jumpReturns.push(logReturns[i]!)
    } else {
      normalReturns.push(logReturns[i]!)
    }
  }

  // Step 3: Diffusion parameters from non-jump returns
  const nNormal = normalReturns.length
  const meanNormal = nNormal > 0
    ? normalReturns.reduce((s, r) => s + r, 0) / nNormal
    : meanAll
  const varNormal = nNormal > 1
    ? normalReturns.reduce((s, r) => s + (r - meanNormal) ** 2, 0) / (nNormal - 1)
    : varAll

  const sigma = Math.sqrt(varNormal / dt)
  const mu = meanNormal / dt + 0.5 * sigma * sigma

  // Step 4: Jump parameters
  const nJumps = jumpReturns.length
  const lambda = nJumps / (n * dt) // Jumps per year

  let muJ = 0
  let sigmaJ = 0.01

  if (nJumps > 0) {
    muJ = jumpReturns.reduce((s, r) => s + r, 0) / nJumps
    if (nJumps > 1) {
      sigmaJ = Math.sqrt(
        jumpReturns.reduce((s, r) => s + (r - muJ) ** 2, 0) / (nJumps - 1),
      )
    }
  }

  // Step 5: Re-estimate diffusion σ excluding jump periods
  // Already done in step 3

  return {
    gbm: { mu, sigma },
    jumps: { lambda, muJ, sigmaJ },
    jumpIndices,
  }
}

/**
 * Exact expected value of Merton jump-diffusion at time t.
 * E[S(t)] = S₀ · exp(μt)
 * (same as GBM due to compensator)
 */
export function mertonExpectedValue(
  s0: number,
  mu: number,
  t: number,
): number {
  return s0 * Math.exp(mu * t)
}

/**
 * Merton characteristic function (for Fourier pricing methods).
 * φ(u) = exp(iuX₀ + (iu(μ-λk) - u²σ²/2)T + λT(exp(iuμ_J - u²σ_J²/2) - 1))
 */
export function mertonCharacteristicFunction(
  u: number,
  x0: number,
  mu: number,
  sigma: number,
  lambda: number,
  muJ: number,
  sigmaJ: number,
  T: number,
): { real: number; imag: number } {
  const k = Math.exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1

  // (iu(μ-λk) - u²σ²/2)T
  const driftPhase = -u * u * sigma * sigma * 0.5 * T
  const driftImag = u * (mu - lambda * k) * T

  // λT(exp(iuμ_J - u²σ_J²/2) - 1)
  const jumpExpReal = Math.exp(-u * u * sigmaJ * sigmaJ * 0.5) * Math.cos(u * muJ) - 1
  const jumpExpImag = Math.exp(-u * u * sigmaJ * sigmaJ * 0.5) * Math.sin(u * muJ)
  const jumpReal = lambda * T * jumpExpReal
  const jumpImag = lambda * T * jumpExpImag

  // Total: exp(iuX₀) * exp(drift + jump)
  const totalReal = driftPhase + jumpReal
  const totalImag = u * x0 + driftImag + jumpImag

  // e^{a+bi} = e^a(cos(b) + i·sin(b))
  const magnitude = Math.exp(totalReal)
  return {
    real: magnitude * Math.cos(totalImag),
    imag: magnitude * Math.sin(totalImag),
  }
}
