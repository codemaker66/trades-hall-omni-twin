/**
 * Geometric Brownian Motion (GBM) Simulation
 *
 * The fundamental stochastic process for price modelling:
 *   dS = μS dt + σS dW
 *
 * Exact solution (no discretization error):
 *   S(t+dt) = S(t) · exp((μ - σ²/2)dt + σ√dt Z)
 *
 * where Z ~ N(0,1).
 */

import type { GBMParams, SimulationConfig, SimulationResult } from './types'
import { Rng } from './random'

/**
 * Simulate GBM price paths using the exact log-normal solution.
 *
 * @param params - GBM drift and volatility parameters
 * @param config - Simulation configuration (s0, time horizon, paths, etc.)
 * @returns SimulationResult with row-major price matrix
 */
export function simulateGBM(
  params: GBMParams,
  config: SimulationConfig,
): SimulationResult {
  const { mu, sigma } = params
  const { s0, tYears, dt, nPaths, seed } = config
  const nSteps = Math.floor(tYears / dt)

  const rng = new Rng(seed)
  const paths = new Float64Array(nPaths * (nSteps + 1))

  // Precompute constants
  const drift = (mu - 0.5 * sigma * sigma) * dt
  const diffusion = sigma * Math.sqrt(dt)

  for (let p = 0; p < nPaths; p++) {
    const offset = p * (nSteps + 1)
    paths[offset] = s0

    for (let t = 1; t <= nSteps; t++) {
      const z = rng.normal()
      // Exact solution: S(t+dt) = S(t) * exp(drift + diffusion * Z)
      paths[offset + t] = paths[offset + t - 1]! * Math.exp(drift + diffusion * z)
    }
  }

  return { paths, nSteps, nPaths, dt }
}

/**
 * Compute the exact expected value of GBM at time t.
 * E[S(t)] = S(0) * exp(μt)
 */
export function gbmExpectedValue(s0: number, mu: number, t: number): number {
  return s0 * Math.exp(mu * t)
}

/**
 * Compute the exact variance of GBM at time t.
 * Var[S(t)] = S(0)² * exp(2μt) * (exp(σ²t) - 1)
 */
export function gbmVariance(s0: number, mu: number, sigma: number, t: number): number {
  return s0 * s0 * Math.exp(2 * mu * t) * (Math.exp(sigma * sigma * t) - 1)
}

/**
 * Calibrate GBM parameters from historical log-returns.
 *
 * Given a series of prices, compute:
 *   μ̂ = (1/n)Σ rₜ / dt + σ̂²/2  (annualized drift)
 *   σ̂ = std(rₜ) / √dt             (annualized volatility)
 *
 * where rₜ = ln(S(t)/S(t-1))
 *
 * @param prices - Historical price series
 * @param dt - Time step between observations (in years)
 */
export function calibrateGBM(prices: number[], dt: number): GBMParams {
  if (prices.length < 3) {
    throw new Error('Need at least 3 price observations for calibration')
  }

  // Compute log returns
  const logReturns: number[] = []
  for (let i = 1; i < prices.length; i++) {
    logReturns.push(Math.log(prices[i]! / prices[i - 1]!))
  }

  const n = logReturns.length
  const meanReturn = logReturns.reduce((s, r) => s + r, 0) / n

  // Variance of log returns
  const variance = logReturns.reduce((s, r) => s + (r - meanReturn) ** 2, 0) / (n - 1)

  // Annualize
  const sigma = Math.sqrt(variance / dt)
  const mu = meanReturn / dt + 0.5 * sigma * sigma

  return { mu, sigma }
}
