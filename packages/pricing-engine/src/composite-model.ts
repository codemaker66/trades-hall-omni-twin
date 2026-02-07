/**
 * Composite Venue Pricing Model
 *
 * The complete stochastic model combining OU mean-reversion, seasonal Fourier
 * fair value, and Merton jump-diffusion:
 *
 *   dP(t) = θ(μ(t) - P(t))dt + σdW(t) + ξdJ(t)
 *
 * where:
 * - P(t) is log-price
 * - μ(t) = μ₀ + Σₖ(aₖcos(2πkt/365) + bₖsin(2πkt/365)) is seasonal fair value
 * - θ is mean-reversion speed (half-life = ln(2)/θ)
 * - σ is diffusion volatility
 * - J is compound Poisson process (intensity λ, jump ~ N(μ_J, σ_J²))
 *
 * Exact discretization with jumps:
 *   P(t+dt) = P(t)e^{-θdt} + μ(t)(1-e^{-θdt})
 *             + σ√((1-e^{-2θdt})/(2θ))·Z + Σᵢ Jᵢ
 *
 * References:
 * - Cartea & Figueroa (2005). Mean-reverting jump-diffusion for electricity
 * - Guizzardi et al. (2022). Hotel dynamic pricing under stochastic demand
 */

import type {
  VenuePricingModel,
  SimulationConfig,
  SimulationResult,
  CalibrationResult,
} from './types'
import { Rng } from './random'
import { seasonalMean, calibrateOU } from './ou'
import { calibrateMertonJD } from './merton'

/**
 * Simulate the composite OU + seasonal + jump-diffusion venue pricing model.
 *
 * This is the primary simulation function for venue price paths. It combines:
 * 1. Mean-reversion to a seasonal fair value (OU process)
 * 2. Diffusion noise (Brownian motion)
 * 3. Sudden demand shocks (compound Poisson jumps)
 *
 * @param model - Complete venue pricing model parameters
 * @param config - Simulation configuration
 * @returns SimulationResult with price paths
 */
export function simulateVenuePrice(
  model: VenuePricingModel,
  config: SimulationConfig,
): SimulationResult {
  const { ou, jumps } = model
  const { s0, tYears, dt, nPaths, seed } = config
  const nSteps = Math.floor(tYears / dt)

  const rng = new Rng(seed)
  const paths = new Float64Array(nPaths * (nSteps + 1))

  // OU exact discretization constants
  const decay = Math.exp(-ou.theta * dt)
  const oneMinusDecay = 1 - decay
  const condVar = (ou.sigma * ou.sigma * (1 - Math.exp(-2 * ou.theta * dt))) / (2 * ou.theta)
  const condStd = Math.sqrt(condVar)

  for (let p = 0; p < nPaths; p++) {
    const offset = p * (nSteps + 1)
    paths[offset] = Math.log(s0) // Work in log-price space

    for (let t = 1; t <= nSteps; t++) {
      const time = t * dt
      const muT = seasonalMean(ou, time)

      // OU mean-reversion (exact conditional mean)
      const condMean = paths[offset + t - 1]! * decay + muT * oneMinusDecay

      // Diffusion
      const diffusion = condStd * rng.normal()

      // Compound Poisson jumps
      const nJumps = rng.poisson(jumps.lambda * dt)
      let jumpSum = 0
      for (let j = 0; j < nJumps; j++) {
        jumpSum += rng.normal(jumps.muJ, jumps.sigmaJ)
      }

      paths[offset + t] = condMean + diffusion + jumpSum
    }

    // Convert from log-price to price
    for (let t = 0; t <= nSteps; t++) {
      paths[offset + t] = Math.exp(paths[offset + t]!)
    }
  }

  return { paths, nSteps, nPaths, dt }
}

/**
 * Full 5-step calibration pipeline for the composite venue pricing model.
 *
 * Pipeline:
 * 1. Remove seasonality via Fourier regression → seasonal coefficients
 * 2. Fit OU parameters to residuals via MLE (AR(1) representation)
 * 3. Detect jumps using threshold |rₜ| > 3σ√Δt
 * 4. Estimate jump parameters (λ, μ_J, σ_J) from detected events
 * 5. Re-estimate diffusion σ excluding jump periods
 *
 * @param prices - Historical price series (daily recommended, at least 60 observations)
 * @param dt - Time step between observations in years (e.g., 1/365 for daily)
 * @param nHarmonics - Number of Fourier harmonics (default 2: annual + semi-annual)
 * @param jumpThreshold - Standard deviations for jump detection (default 3)
 */
export function calibrateVenueModel(
  prices: number[],
  dt: number,
  nHarmonics: number = 2,
  jumpThreshold: number = 3,
): CalibrationResult {
  if (prices.length < 60) {
    throw new Error('Need at least 60 price observations for full calibration')
  }

  const n = prices.length
  const logPrices = prices.map(Math.log)

  // Steps 1-2: OU calibration with seasonal decomposition
  const { params: ouParams, residuals } = calibrateOU(prices, dt, nHarmonics)

  // Steps 3-5: Jump detection and estimation on OU residuals
  // Compute first differences of residuals (innovations)
  const innovations: number[] = []
  for (let i = 1; i < residuals.length; i++) {
    innovations.push(residuals[i]! - residuals[i - 1]! * Math.exp(-ouParams.theta * dt))
  }

  const nInno = innovations.length
  const meanInno = innovations.reduce((s, r) => s + r, 0) / nInno
  const stdInno = Math.sqrt(
    innovations.reduce((s, r) => s + (r - meanInno) ** 2, 0) / (nInno - 1),
  )

  // Detect jumps in innovations
  const jumpIndices: number[] = []
  const jumpValues: number[] = []
  const normalValues: number[] = []

  for (let i = 0; i < nInno; i++) {
    if (Math.abs(innovations[i]! - meanInno) > jumpThreshold * stdInno) {
      jumpIndices.push(i)
      jumpValues.push(innovations[i]!)
    } else {
      normalValues.push(innovations[i]!)
    }
  }

  // Jump parameters
  const lambda = jumpValues.length / (nInno * dt)
  let muJ = 0
  let sigmaJ = 0.01

  if (jumpValues.length > 0) {
    muJ = jumpValues.reduce((s, r) => s + r, 0) / jumpValues.length
    if (jumpValues.length > 1) {
      sigmaJ = Math.sqrt(
        jumpValues.reduce((s, r) => s + (r - muJ) ** 2, 0) / (jumpValues.length - 1),
      )
    }
  }

  // Re-estimate diffusion σ from non-jump innovations
  const sigmaClean = normalValues.length > 1
    ? Math.sqrt(
        normalValues.reduce((s, r) => s + r * r, 0) / normalValues.length / dt,
      )
    : ouParams.sigma

  // Diagnostics
  const logLikelihood = computeLogLikelihood(logPrices, ouParams, { lambda, muJ, sigmaJ }, dt)
  const nParams = 3 + 2 * nHarmonics + 3 // theta, mu, sigma + Fourier + jump params
  const aic = -2 * logLikelihood + 2 * nParams
  const bic = -2 * logLikelihood + nParams * Math.log(n)
  const ljungBoxPValue = ljungBoxTest(normalValues, 10)

  return {
    model: {
      ou: { ...ouParams, sigma: sigmaClean },
      jumps: { lambda, muJ, sigmaJ },
    },
    diagnostics: {
      logLikelihood,
      aic,
      bic,
      nJumpsDetected: jumpValues.length,
      ljungBoxPValue,
    },
  }
}

/**
 * Extract percentile bands from simulation paths at each time step.
 * Used for fan chart visualization.
 *
 * @param result - Simulation result
 * @param percentiles - Percentile levels to compute (e.g., [5, 25, 50, 75, 95])
 * @returns Array of percentile arrays, each of length nSteps+1
 */
export function computePercentileBands(
  result: SimulationResult,
  percentiles: number[] = [5, 25, 50, 75, 95],
): number[][] {
  const { paths, nSteps, nPaths } = result
  const bands: number[][] = percentiles.map(() => new Array<number>(nSteps + 1))

  const column = new Float64Array(nPaths)

  for (let t = 0; t <= nSteps; t++) {
    // Extract column t across all paths
    for (let p = 0; p < nPaths; p++) {
      column[p] = paths[p * (nSteps + 1) + t]!
    }

    // Sort for quantile computation
    column.sort()

    for (let i = 0; i < percentiles.length; i++) {
      const pct = percentiles[i]! / 100
      const idx = Math.min(Math.floor(pct * nPaths), nPaths - 1)
      bands[i]![t] = column[idx]!
    }
  }

  return bands
}

/**
 * Compute path statistics: mean, variance, skewness, kurtosis at each time step.
 */
export function computePathStatistics(
  result: SimulationResult,
): { mean: Float64Array; variance: Float64Array; skewness: Float64Array; kurtosis: Float64Array } {
  const { paths, nSteps, nPaths } = result
  const mean = new Float64Array(nSteps + 1)
  const variance = new Float64Array(nSteps + 1)
  const skewness = new Float64Array(nSteps + 1)
  const kurtosis = new Float64Array(nSteps + 1)

  for (let t = 0; t <= nSteps; t++) {
    // Mean
    let sum = 0
    for (let p = 0; p < nPaths; p++) {
      sum += paths[p * (nSteps + 1) + t]!
    }
    mean[t] = sum / nPaths

    // Variance
    let sumSq = 0
    for (let p = 0; p < nPaths; p++) {
      const dev = paths[p * (nSteps + 1) + t]! - mean[t]!
      sumSq += dev * dev
    }
    variance[t] = sumSq / (nPaths - 1)

    // Skewness and kurtosis
    const std = Math.sqrt(variance[t]!)
    let sum3 = 0
    let sum4 = 0
    if (std > 0) {
      for (let p = 0; p < nPaths; p++) {
        const z = (paths[p * (nSteps + 1) + t]! - mean[t]!) / std
        sum3 += z * z * z
        sum4 += z * z * z * z
      }
      skewness[t] = sum3 / nPaths
      kurtosis[t] = sum4 / nPaths - 3 // Excess kurtosis
    }
  }

  return { mean, variance, skewness, kurtosis }
}

// ---------------------------------------------------------------------------
// Internal: Log-likelihood and diagnostics
// ---------------------------------------------------------------------------

function computeLogLikelihood(
  logPrices: number[],
  ou: { theta: number; mu: number; sigma: number; seasonalA: number[]; seasonalB: number[] },
  jumps: { lambda: number; muJ: number; sigmaJ: number },
  dt: number,
): number {
  const n = logPrices.length
  const decay = Math.exp(-ou.theta * dt)
  const condVar = (ou.sigma * ou.sigma * (1 - Math.exp(-2 * ou.theta * dt))) / (2 * ou.theta)
  let ll = 0

  for (let i = 1; i < n; i++) {
    const time = i * dt
    const muT = seasonalMean(ou, time)
    const condMean = logPrices[i - 1]! * decay + muT * (1 - decay)
    const residual = logPrices[i]! - condMean

    // Mixture density: (1-p_jump)*Normal + p_jump*Normal_with_jump
    const pJump = 1 - Math.exp(-jumps.lambda * dt)
    const normalDensity = gaussianPDF(residual, 0, Math.sqrt(condVar))
    const jumpDensity = gaussianPDF(
      residual,
      jumps.muJ,
      Math.sqrt(condVar + jumps.sigmaJ * jumps.sigmaJ),
    )

    const density = (1 - pJump) * normalDensity + pJump * jumpDensity
    ll += Math.log(Math.max(density, 1e-300))
  }

  return ll
}

function gaussianPDF(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI))
}

/**
 * Ljung-Box test for residual autocorrelation.
 * H0: residuals are independently distributed (no autocorrelation).
 * Returns approximate p-value.
 */
function ljungBoxTest(residuals: number[], lags: number): number {
  const n = residuals.length
  if (n < lags + 1) return 1

  const mean = residuals.reduce((s, r) => s + r, 0) / n
  const demeaned = residuals.map((r) => r - mean)
  const c0 = demeaned.reduce((s, r) => s + r * r, 0) / n

  let Q = 0
  for (let k = 1; k <= lags; k++) {
    let ck = 0
    for (let t = k; t < n; t++) {
      ck += demeaned[t]! * demeaned[t - k]!
    }
    ck /= n
    const rho = ck / c0
    Q += (rho * rho) / (n - k)
  }
  Q *= n * (n + 2)

  // Approximate chi-squared p-value (chi2 with lags degrees of freedom)
  return 1 - chi2CDF(Q, lags)
}

/** Regularized incomplete gamma function approximation for chi-squared CDF */
function chi2CDF(x: number, k: number): number {
  if (x <= 0) return 0
  // Use series expansion for regularized lower incomplete gamma
  const a = k / 2
  const z = x / 2
  let sum = 0
  let term = Math.exp(-z) * Math.pow(z, a) / gamma(a + 1)
  for (let n = 0; n < 200; n++) {
    sum += term
    term *= z / (a + n + 1)
    if (Math.abs(term) < 1e-12) break
  }
  return sum
}

/** Stirling approximation for gamma function */
function gamma(n: number): number {
  if (n < 0.5) {
    return Math.PI / (Math.sin(Math.PI * n) * gamma(1 - n))
  }
  n -= 1
  const g = 7
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ]
  let x = c[0]!
  for (let i = 1; i < g + 2; i++) {
    x += c[i]! / (n + i)
  }
  const t = n + g + 0.5
  return Math.sqrt(2 * Math.PI) * Math.pow(t, n + 0.5) * Math.exp(-t) * x
}
