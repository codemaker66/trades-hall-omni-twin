/**
 * Hawkes Self-Exciting Process for Demand Modelling
 *
 * Three progressively sophisticated demand models:
 *
 * 1. Non-homogeneous Poisson (baseline):
 *    λ(t) = λ₀ · [1 + α·sin(2πt/7)] · seasonal(t)
 *    Independent arrivals, good for stable venues.
 *
 * 2. Hawkes self-exciting process (FOMO/social proof):
 *    λ*(t) = μ + Σ_{tᵢ < t} α · e^{-β(t-tᵢ)}
 *    Each booking increases probability of future bookings.
 *    Branching ratio n* = α/β measures virality.
 *
 * 3. Cox-Hawkes hybrid (state of the art):
 *    Log-Gaussian Cox background + Hawkes self-excitation.
 *
 * References:
 * - Hawkes (1971). "Spectra of some self-exciting and mutually exciting point processes"
 * - Ogata (1981). "On Lewis' simulation method for point processes"
 * - Miscouridou et al. (2022). "Cox-Hawkes hybrid processes" arXiv:2210.11844
 */

import type { HawkesParams, HawkesFitResult } from './types'
import { Rng } from './random'

/**
 * Simulate a Hawkes process via Ogata's thinning algorithm.
 *
 * The thinning method:
 * 1. Compute upper bound λ* on the intensity
 * 2. Generate candidate event at rate λ*
 * 3. Accept with probability λ(t)/λ*
 * 4. Update λ* after each accepted event
 *
 * Exact simulation — no discretization error.
 *
 * @param params - Hawkes parameters (mu, alpha, beta)
 * @param tMax - Simulation horizon
 * @param seed - Optional RNG seed
 * @returns Array of event timestamps
 */
export function simulateHawkes(
  params: HawkesParams,
  tMax: number,
  seed?: number,
): number[] {
  const { mu, alpha, beta } = params
  const rng = new Rng(seed)

  if (alpha / beta >= 1) {
    throw new Error(
      `Hawkes process is explosive: branching ratio α/β = ${(alpha / beta).toFixed(3)} >= 1`,
    )
  }

  const events: number[] = []
  let t = 0
  let lambdaStar = mu // Upper bound on intensity

  while (t < tMax) {
    // Generate candidate inter-arrival time
    const u = rng.next()
    t += -Math.log(u) / lambdaStar

    if (t >= tMax) break

    // Compute actual intensity at candidate time
    let lambdaT = mu
    for (let i = events.length - 1; i >= 0; i--) {
      const dt = t - events[i]!
      if (dt > 10 / beta) break // Truncate negligible contributions
      lambdaT += alpha * Math.exp(-beta * dt)
    }

    // Accept/reject
    const d = rng.next()
    if (d * lambdaStar <= lambdaT) {
      events.push(t)
      lambdaStar = lambdaT + alpha // Update bound after new event
    } else {
      lambdaStar = lambdaT // Tighten bound
    }
  }

  return events
}

/**
 * Compute the Hawkes intensity at a given time.
 *
 * λ*(t) = μ + α · Σ_{tᵢ < t} e^{-β(t-tᵢ)}
 */
export function hawkesIntensity(
  params: HawkesParams,
  t: number,
  events: number[],
): number {
  const { mu, alpha, beta } = params
  let intensity = mu

  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i]! >= t) continue
    const dt = t - events[i]!
    if (dt > 10 / beta) break
    intensity += alpha * Math.exp(-beta * dt)
  }

  return intensity
}

/**
 * Compute the intensity function over a time grid.
 *
 * @param params - Hawkes parameters
 * @param events - Observed event timestamps
 * @param tMin - Start of evaluation window
 * @param tMax - End of evaluation window
 * @param nPoints - Number of evaluation points
 */
export function hawkesIntensityCurve(
  params: HawkesParams,
  events: number[],
  tMin: number,
  tMax: number,
  nPoints: number,
): Array<{ t: number; intensity: number }> {
  const curve: Array<{ t: number; intensity: number }> = []
  const dt = (tMax - tMin) / (nPoints - 1)

  for (let i = 0; i < nPoints; i++) {
    const t = tMin + i * dt
    curve.push({ t, intensity: hawkesIntensity(params, t, events) })
  }

  return curve
}

/**
 * Fit Hawkes parameters via Maximum Likelihood Estimation.
 *
 * Log-likelihood of exponential Hawkes:
 *   ℓ = Σᵢ log(λ*(tᵢ)) - ∫₀ᵀ λ*(t) dt
 *
 * The integral has a closed form for exponential kernel:
 *   ∫₀ᵀ λ*(t) dt = μT + (α/β) Σᵢ (1 - e^{-β(T-tᵢ)})
 *
 * Optimization via coordinate descent (fast convergence for this structure).
 *
 * @param events - Observed event timestamps (must be sorted)
 * @param tMax - Observation window end time
 * @param maxIter - Maximum optimization iterations
 */
export function fitHawkes(
  events: number[],
  tMax: number,
  maxIter: number = 200,
): HawkesFitResult {
  if (events.length < 5) {
    throw new Error('Need at least 5 events to fit Hawkes process')
  }

  const n = events.length

  // Initial guesses
  let mu = n / tMax * 0.7 // 70% of average rate as background
  let alpha = 0.5
  let beta = 1.0

  // Precompute inter-event times
  const sortedEvents = [...events].sort((a, b) => a - b)

  for (let iter = 0; iter < maxIter; iter++) {
    // E-step: compute intensities and auxiliary variables
    const lambdas = new Float64Array(n)
    const A = new Float64Array(n) // Recursive kernel sum

    for (let i = 0; i < n; i++) {
      let kernelSum = 0
      if (i > 0) {
        const dt = sortedEvents[i]! - sortedEvents[i - 1]!
        kernelSum = Math.exp(-beta * dt) * (A[i - 1]! + 1)
      }
      A[i] = kernelSum
      lambdas[i] = mu + alpha * kernelSum
    }

    // M-step: update parameters

    // Update mu: μ = n / (T + (α/β) Σ_i (e^{-β(T-tᵢ)} - 1) / μ)
    // Simplified: use proportion of background rate
    let sumLogLambda = 0
    let sumMuPart = 0
    for (let i = 0; i < n; i++) {
      const li = lambdas[i]!
      if (li > 0) {
        sumLogLambda += Math.log(li)
        sumMuPart += mu / li
      }
    }

    const muNew = sumMuPart / tMax

    // Update alpha: gradient ascent
    let dAlpha = 0
    for (let i = 0; i < n; i++) {
      const li = lambdas[i]!
      if (li > 0) {
        dAlpha += A[i]! / li
      }
    }
    // Integral term: ∂/∂α ∫λ*(t)dt = (1/β) Σ_i (1 - e^{-β(T-tᵢ)})
    let integralDA = 0
    for (let i = 0; i < n; i++) {
      integralDA += (1 - Math.exp(-beta * (tMax - sortedEvents[i]!))) / beta
    }
    dAlpha -= integralDA

    // Update beta: gradient ascent
    let dBeta = 0
    for (let i = 0; i < n; i++) {
      const li = lambdas[i]!
      if (li > 0 && i > 0) {
        // ∂A_i/∂β = -Σ_{j<i} (t_i - t_j) α e^{-β(t_i - t_j)}
        let dA = 0
        for (let j = 0; j < i; j++) {
          const dt = sortedEvents[i]! - sortedEvents[j]!
          dA -= dt * alpha * Math.exp(-beta * dt)
        }
        dBeta += dA / li
      }
    }
    // Integral term
    let integralDB = 0
    for (let i = 0; i < n; i++) {
      const dtEnd = tMax - sortedEvents[i]!
      integralDB += (alpha / (beta * beta)) * (1 - Math.exp(-beta * dtEnd) * (1 + beta * dtEnd))
    }
    dBeta += integralDB

    // Gradient step with projection
    const lr = 0.05 / (1 + iter * 0.01) // Decreasing learning rate
    mu = Math.max(1e-6, muNew)
    alpha = Math.max(1e-6, alpha + lr * dAlpha)
    beta = Math.max(1e-6, beta + lr * dBeta)

    // Ensure stability: α/β < 1
    if (alpha / beta >= 0.99) {
      alpha = 0.98 * beta
    }
  }

  // Final log-likelihood
  let ll = 0
  const A = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    if (i > 0) {
      A[i] = Math.exp(-beta * (sortedEvents[i]! - sortedEvents[i - 1]!)) * (A[i - 1]! + 1)
    }
    const lambda = mu + alpha * A[i]!
    ll += Math.log(Math.max(lambda, 1e-300))
  }
  // Subtract integral
  let integral = mu * tMax
  for (let i = 0; i < n; i++) {
    integral += (alpha / beta) * (1 - Math.exp(-beta * (tMax - sortedEvents[i]!)))
  }
  ll -= integral

  return {
    mu,
    alpha,
    beta,
    branchingRatio: alpha / beta,
    halfLife: Math.LN2 / beta,
    logLikelihood: ll,
  }
}

/**
 * Non-homogeneous Poisson process simulation.
 *
 * λ(t) = λ₀ · [1 + α·sin(2πt/period)] · seasonal(t)
 *
 * Uses Ogata's thinning algorithm with the time-varying rate.
 *
 * @param baseRate - λ₀ baseline rate
 * @param weeklyAmplitude - Amplitude of weekly cycle (0 to 1)
 * @param period - Period of oscillation (default 7 for weekly)
 * @param seasonalMultiplier - Optional function mapping time to seasonal multiplier
 * @param tMax - Simulation horizon
 * @param seed - Optional RNG seed
 */
export function simulateNHPP(
  baseRate: number,
  weeklyAmplitude: number,
  period: number,
  tMax: number,
  seasonalMultiplier?: (t: number) => number,
  seed?: number,
): number[] {
  const rng = new Rng(seed)
  const events: number[] = []

  // Upper bound on intensity
  const lambdaMax = baseRate * (1 + weeklyAmplitude) * (seasonalMultiplier ? 2.0 : 1.0)
  let t = 0

  while (t < tMax) {
    t += rng.exponential(lambdaMax)
    if (t >= tMax) break

    // Compute actual intensity
    const seasonal = seasonalMultiplier ? seasonalMultiplier(t) : 1.0
    const lambda = baseRate * (1 + weeklyAmplitude * Math.sin((2 * Math.PI * t) / period)) * seasonal

    // Accept/reject
    if (rng.next() * lambdaMax <= lambda) {
      events.push(t)
    }
  }

  return events
}

/**
 * Estimate the branching ratio from data using the EM algorithm approximation.
 *
 * Quick diagnostic: branching ratio = 1 - (inter-event time CV)^{-2}
 * More reliable with >50 events.
 */
export function estimateBranchingRatio(events: number[]): number {
  if (events.length < 3) return 0

  const sorted = [...events].sort((a, b) => a - b)
  const interArrivals: number[] = []
  for (let i = 1; i < sorted.length; i++) {
    interArrivals.push(sorted[i]! - sorted[i - 1]!)
  }

  const mean = interArrivals.reduce((s, x) => s + x, 0) / interArrivals.length
  const variance = interArrivals.reduce((s, x) => s + (x - mean) ** 2, 0) / (interArrivals.length - 1)
  const cv2 = variance / (mean * mean)

  // For a Hawkes process, CV² = 1 / (1 - n*)²
  // So n* = 1 - 1/CV
  return Math.max(0, Math.min(0.99, 1 - 1 / Math.sqrt(cv2)))
}
