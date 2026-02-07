/**
 * Ornstein-Uhlenbeck Process with Seasonal Component
 *
 * The mean-reverting process for venue pricing:
 *   dX(t) = θ(μ(t) - X(t))dt + σdW(t)
 *
 * where μ(t) = μ₀ + Σₖ(aₖcos(2πkt/365) + bₖsin(2πkt/365)) is the
 * seasonal fair value (Fourier series representation).
 *
 * Key property: mean-reversion half-life = ln(2)/θ
 *
 * Exact discretization (no Euler error):
 *   X(t+dt) = X(t)e^{-θdt} + μ(t)(1 - e^{-θdt}) + σ√((1-e^{-2θdt})/(2θ)) · Z
 *
 * References:
 * - Holý & Tomanová (2018). "Estimation of OU Process Using Ultra-High-Frequency Data"
 * - Cartea & Figueroa (2005). "Pricing in Electricity Markets: Mean Reverting Jump Diffusion"
 */

import type { OUParams, SimulationConfig, SimulationResult } from './types'
import { Rng } from './random'

/**
 * Compute the seasonal fair value at time t (in years).
 *
 * μ(t) = μ₀ + Σₖ(aₖcos(2πk·t·365/365) + bₖsin(2πk·t·365/365))
 *      = μ₀ + Σₖ(aₖcos(2πkt) + bₖsin(2πkt))  [with t in years, freq per year]
 *
 * The Fourier components capture annual seasonality (k=1), semi-annual (k=2), etc.
 */
export function seasonalMean(params: OUParams, tYears: number): number {
  let mu = params.mu
  const tDays = tYears * 365
  for (let k = 0; k < params.seasonalA.length; k++) {
    const freq = (2 * Math.PI * (k + 1)) / 365
    mu += (params.seasonalA[k] ?? 0) * Math.cos(freq * tDays)
          + (params.seasonalB[k] ?? 0) * Math.sin(freq * tDays)
  }
  return mu
}

/**
 * Simulate OU process with seasonal mean-reversion using exact discretization.
 *
 * Uses the exact transition density (not Euler-Maruyama) to avoid discretization
 * error. The OU conditional distribution is:
 *   X(t+dt) | X(t) ~ N(X(t)e^{-θdt} + μ(t)(1 - e^{-θdt}), σ²(1-e^{-2θdt})/(2θ))
 *
 * @param params - OU parameters with seasonal Fourier coefficients
 * @param config - Simulation configuration
 * @returns SimulationResult with price paths (exponentiated from log-price space)
 */
export function simulateOU(
  params: OUParams,
  config: SimulationConfig,
): SimulationResult {
  const { theta, sigma } = params
  const { s0, tYears, dt, nPaths, seed } = config
  const nSteps = Math.floor(tYears / dt)

  const rng = new Rng(seed)
  const paths = new Float64Array(nPaths * (nSteps + 1))

  // Precompute exact discretization constants
  const decay = Math.exp(-theta * dt)
  const oneMinusDecay = 1 - decay
  // Conditional variance: σ²(1 - e^{-2θdt}) / (2θ)
  const condVar = (sigma * sigma * (1 - Math.exp(-2 * theta * dt))) / (2 * theta)
  const condStd = Math.sqrt(condVar)

  for (let p = 0; p < nPaths; p++) {
    const offset = p * (nSteps + 1)
    paths[offset] = Math.log(s0) // Work in log-price space

    for (let t = 1; t <= nSteps; t++) {
      const time = t * dt
      const muT = seasonalMean(params, time)

      // Exact conditional mean: E[X(t+dt) | X(t)] = X(t)e^{-θdt} + μ(t)(1-e^{-θdt})
      const condMean = paths[offset + t - 1]! * decay + muT * oneMinusDecay

      // Sample from exact transition density
      paths[offset + t] = condMean + condStd * rng.normal()
    }

    // Convert from log-price to price
    for (let t = 0; t <= nSteps; t++) {
      paths[offset + t] = Math.exp(paths[offset + t]!)
    }
  }

  return { paths, nSteps, nPaths, dt }
}

/**
 * Calibrate OU parameters from a historical price series via MLE.
 *
 * The OU process has the AR(1) representation:
 *   X(t+1) = a + b·X(t) + ε,  ε ~ N(0, σ²_ε)
 *
 * where:
 *   b = e^{-θΔt}
 *   a = μ(1 - b)
 *   σ²_ε = σ²(1 - b²) / (2θ)
 *
 * Step 1: Remove seasonality via Fourier regression
 * Step 2: Fit AR(1) to deseasonalized residuals via OLS
 * Step 3: Recover OU parameters from AR(1) coefficients
 *
 * @param prices - Historical price series (at least 30 observations)
 * @param dt - Time between observations in years (e.g., 1/365 for daily)
 * @param nHarmonics - Number of Fourier harmonics for seasonality (default 2)
 */
export function calibrateOU(
  prices: number[],
  dt: number,
  nHarmonics: number = 2,
): { params: OUParams; residuals: number[] } {
  if (prices.length < 30) {
    throw new Error('Need at least 30 observations for OU calibration')
  }

  const n = prices.length
  const logPrices = prices.map(Math.log)

  // Step 1: Fourier regression to extract seasonality
  // y = μ₀ + Σₖ(aₖcos(2πkt/365) + bₖsin(2πkt/365)) + residual
  // Build design matrix X for OLS: X = [1, cos(ω₁t), sin(ω₁t), cos(ω₂t), sin(ω₂t), ...]
  const nCols = 1 + 2 * nHarmonics
  const X: number[][] = []
  for (let i = 0; i < n; i++) {
    const tDays = i * dt * 365
    const row = [1]
    for (let k = 1; k <= nHarmonics; k++) {
      const freq = (2 * Math.PI * k) / 365
      row.push(Math.cos(freq * tDays))
      row.push(Math.sin(freq * tDays))
    }
    X.push(row)
  }

  // Solve OLS: β = (X'X)^{-1} X'y
  const beta = solveOLS(X, logPrices, nCols)

  // Extract seasonal coefficients
  const mu0 = beta[0]!
  const seasonalA: number[] = []
  const seasonalB: number[] = []
  for (let k = 0; k < nHarmonics; k++) {
    seasonalA.push(beta[1 + 2 * k]!)
    seasonalB.push(beta[2 + 2 * k]!)
  }

  // Compute residuals (deseasonalized log-prices)
  const residuals: number[] = []
  for (let i = 0; i < n; i++) {
    let seasonal = mu0
    const tDays = i * dt * 365
    for (let k = 0; k < nHarmonics; k++) {
      const freq = (2 * Math.PI * (k + 1)) / 365
      seasonal += seasonalA[k]! * Math.cos(freq * tDays)
                + seasonalB[k]! * Math.sin(freq * tDays)
    }
    residuals.push(logPrices[i]! - seasonal)
  }

  // Step 2: Fit AR(1) to residuals: r(t) = a + b*r(t-1) + ε
  let sumX = 0, sumY = 0, sumXX = 0, sumXY = 0
  const nAR = n - 1
  for (let i = 1; i < n; i++) {
    const x = residuals[i - 1]!
    const y = residuals[i]!
    sumX += x
    sumY += y
    sumXX += x * x
    sumXY += x * y
  }

  const b = (nAR * sumXY - sumX * sumY) / (nAR * sumXX - sumX * sumX)
  const a = (sumY - b * sumX) / nAR

  // AR(1) residual variance
  let ssResid = 0
  for (let i = 1; i < n; i++) {
    const predicted = a + b * residuals[i - 1]!
    ssResid += (residuals[i]! - predicted) ** 2
  }
  const sigmaEps2 = ssResid / (nAR - 2)

  // Step 3: Recover OU parameters
  // b = e^{-θΔt}  →  θ = -ln(b)/Δt
  const theta = -Math.log(Math.abs(b)) / dt
  // σ²_ε = σ²(1 - b²)/(2θ)  →  σ = √(2θ·σ²_ε / (1 - b²))
  const sigma = Math.sqrt((2 * theta * sigmaEps2) / (1 - b * b))

  return {
    params: {
      theta,
      mu: mu0,
      sigma,
      seasonalA,
      seasonalB,
    },
    residuals,
  }
}

/**
 * Compute the half-life of mean reversion in the same time unit as theta.
 * Half-life = ln(2) / theta
 */
export function ouHalfLife(theta: number): number {
  return Math.LN2 / theta
}

/**
 * Stationary (long-run) variance of OU process: σ²/(2θ)
 */
export function ouStationaryVariance(theta: number, sigma: number): number {
  return (sigma * sigma) / (2 * theta)
}

// ---------------------------------------------------------------------------
// Internal: OLS solver for small systems (Fourier regression)
// ---------------------------------------------------------------------------

/** Solve (X'X)β = X'y via Cholesky decomposition */
function solveOLS(X: number[][], y: number[], nCols: number): number[] {
  const n = X.length

  // X'X
  const XtX: number[][] = Array.from({ length: nCols }, () => new Array(nCols).fill(0) as number[])
  for (let i = 0; i < nCols; i++) {
    for (let j = i; j < nCols; j++) {
      let sum = 0
      for (let k = 0; k < n; k++) {
        sum += X[k]![i]! * X[k]![j]!
      }
      XtX[i]![j] = sum
      XtX[j]![i] = sum
    }
  }

  // X'y
  const Xty: number[] = new Array(nCols).fill(0)
  for (let i = 0; i < nCols; i++) {
    let sum = 0
    for (let k = 0; k < n; k++) {
      sum += X[k]![i]! * y[k]!
    }
    Xty[i] = sum
  }

  // Cholesky decomposition: XtX = LL'
  const L: number[][] = Array.from({ length: nCols }, () => new Array(nCols).fill(0) as number[])
  for (let i = 0; i < nCols; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0
      for (let k = 0; k < j; k++) {
        sum += L[i]![k]! * L[j]![k]!
      }
      if (i === j) {
        L[i]![j] = Math.sqrt(XtX[i]![i]! - sum)
      } else {
        L[i]![j] = (XtX[i]![j]! - sum) / L[j]![j]!
      }
    }
  }

  // Forward substitution: Lz = Xty
  const z: number[] = new Array(nCols).fill(0)
  for (let i = 0; i < nCols; i++) {
    let sum = 0
    for (let k = 0; k < i; k++) {
      sum += L[i]![k]! * z[k]!
    }
    z[i] = (Xty[i]! - sum) / L[i]![i]!
  }

  // Backward substitution: L'β = z
  const beta: number[] = new Array(nCols).fill(0)
  for (let i = nCols - 1; i >= 0; i--) {
    let sum = 0
    for (let k = i + 1; k < nCols; k++) {
      sum += L[k]![i]! * beta[k]!
    }
    beta[i] = (z[i]! - sum) / L[i]![i]!
  }

  return beta
}
