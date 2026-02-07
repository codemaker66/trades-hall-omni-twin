/**
 * Monte Carlo Engine with Variance Reduction + QMC
 *
 * Efficient Monte Carlo for pricing path-dependent booking options
 * and simulating revenue scenarios.
 *
 * Variance reduction techniques:
 * 1. Antithetic variates: use (Z, -Z) pairs → reduces variance by ~50%
 * 2. Control variates: adjust using known E[N(T)] = λT → variance reduction >90%
 * 3. Sobol quasi-MC: fills unit hypercube more uniformly
 *    Koksma-Hlawka bound: error O((log N)^s / N) vs O(1/√N) for standard MC
 *    For smooth venue revenue functions: 10-1000× speedup
 *
 * References:
 * - Glasserman (2003). "Monte Carlo Methods in Financial Engineering"
 * - sobol_burley: Owen-scrambled Sobol sequences
 */

import type {
  MCConfig,
  MCResult,
  VenuePricingModel,
  SimulationConfig,
} from './types'
import { Rng, SobolSequence } from './random'
import { seasonalMean } from './ou'

/**
 * Monte Carlo revenue simulation with all variance reduction techniques.
 *
 * Simulates venue price paths using the composite model and computes
 * terminal revenue statistics including VaR and CVaR.
 *
 * @param model - Venue pricing model parameters
 * @param simConfig - Simulation time horizon and step size
 * @param mcConfig - MC configuration (paths, variance reduction, confidence)
 */
export function simulateRevenueMC(
  model: VenuePricingModel,
  simConfig: Omit<SimulationConfig, 'nPaths' | 'seed'>,
  mcConfig: MCConfig,
): MCResult {
  const { ou, jumps } = model
  const { s0, tYears, dt } = simConfig
  const { nPaths, useAntithetic, useControlVariate, useSobol, confidenceLevel, seed } = mcConfig

  const nSteps = Math.floor(tYears / dt)
  const effectivePaths = useAntithetic ? Math.ceil(nPaths / 2) * 2 : nPaths

  const rng = new Rng(seed)
  const sobol = useSobol ? new SobolSequence(1) : null

  // OU exact discretization constants
  const decay = Math.exp(-ou.theta * dt)
  const oneMinusDecay = 1 - decay
  const condVar = (ou.sigma * ou.sigma * (1 - Math.exp(-2 * ou.theta * dt))) / (2 * ou.theta)
  const condStd = Math.sqrt(condVar)

  const terminalValues = new Float64Array(effectivePaths)
  const jumpCounts = new Float64Array(effectivePaths) // For control variate

  for (let p = 0; p < effectivePaths; p += useAntithetic ? 2 : 1) {
    // Generate normal variates
    const normals: number[] = []
    for (let t = 0; t < nSteps; t++) {
      if (useSobol && sobol) {
        const u = sobol.next()[0]!
        // Inverse normal CDF approximation (Beasley-Springer-Moro)
        normals.push(inverseCDFApprox(u))
      } else {
        normals.push(rng.normal())
      }
    }

    // Simulate path
    const simulate = (sign: number): { terminal: number; jumpCount: number } => {
      let logPrice = Math.log(s0)
      let totalJumps = 0

      for (let t = 1; t <= nSteps; t++) {
        const time = t * dt
        const muT = seasonalMean(ou, time)

        // OU mean-reversion
        const condMean = logPrice * decay + muT * oneMinusDecay

        // Diffusion (with antithetic sign)
        const z = sign * normals[t - 1]!
        const diffusion = condStd * z

        // Jumps
        const nJumps = rng.poisson(jumps.lambda * dt)
        totalJumps += nJumps
        let jumpSum = 0
        for (let j = 0; j < nJumps; j++) {
          jumpSum += rng.normal(jumps.muJ, jumps.sigmaJ)
        }

        logPrice = condMean + diffusion + jumpSum
      }

      return { terminal: Math.exp(logPrice), jumpCount: totalJumps }
    }

    // Primary path
    const primary = simulate(1)
    terminalValues[p] = primary.terminal
    jumpCounts[p] = primary.jumpCount

    // Antithetic path
    if (useAntithetic && p + 1 < effectivePaths) {
      const anti = simulate(-1)
      terminalValues[p + 1] = anti.terminal
      jumpCounts[p + 1] = anti.jumpCount
    }
  }

  // Control variate adjustment
  if (useControlVariate) {
    // E[N(T)] = λT (expected total jumps)
    const expectedJumps = jumps.lambda * tYears
    const meanJumps = jumpCounts.reduce((s, j) => s + j, 0) / effectivePaths
    const meanTerminal = terminalValues.reduce((s, v) => s + v, 0) / effectivePaths

    // Compute optimal control variate coefficient c*
    let covXY = 0
    let varY = 0
    for (let i = 0; i < effectivePaths; i++) {
      const dx = terminalValues[i]! - meanTerminal
      const dy = jumpCounts[i]! - meanJumps
      covXY += dx * dy
      varY += dy * dy
    }

    const c = varY > 0 ? -covXY / varY : 0

    // Adjust: θ̂_CV = θ̂ - c*(N̄(T) - λT)
    for (let i = 0; i < effectivePaths; i++) {
      terminalValues[i] = terminalValues[i]! + c * (jumpCounts[i]! - expectedJumps)
    }
  }

  // Compute statistics
  return computeMCStatistics(terminalValues, effectivePaths, confidenceLevel)
}

/**
 * General-purpose Monte Carlo with a custom payoff function.
 *
 * @param payoffFn - Function that generates one random payoff
 * @param config - MC configuration
 */
export function monteCarloGeneric(
  payoffFn: (rng: Rng) => number,
  config: MCConfig,
): MCResult {
  const rng = new Rng(config.seed)
  const terminalValues = new Float64Array(config.nPaths)

  for (let i = 0; i < config.nPaths; i++) {
    terminalValues[i] = payoffFn(rng)
  }

  return computeMCStatistics(terminalValues, config.nPaths, config.confidenceLevel)
}

/**
 * Compute MC statistics from terminal values.
 */
function computeMCStatistics(
  values: Float64Array,
  n: number,
  confidenceLevel: number,
): MCResult {
  // Mean
  let sum = 0
  for (let i = 0; i < n; i++) sum += values[i]!
  const mean = sum / n

  // Standard error
  let sumSq = 0
  for (let i = 0; i < n; i++) sumSq += (values[i]! - mean) ** 2
  const stdError = Math.sqrt(sumSq / (n * (n - 1)))

  // Sort for quantile computation
  const sorted = Float64Array.from(values.subarray(0, n))
  sorted.sort()

  // VaR: worst loss at confidence level
  // For revenue, VaR is the low percentile
  const varIdx = Math.floor((1 - confidenceLevel) * n)
  const varValue = sorted[Math.max(0, varIdx)]!

  // CVaR: expected value below VaR (Expected Shortfall)
  let cvarSum = 0
  const cvarCount = varIdx + 1
  for (let i = 0; i <= varIdx; i++) {
    cvarSum += sorted[i]!
  }
  const cvarValue = cvarCount > 0 ? cvarSum / cvarCount : varValue

  // Percentiles: 5th, 25th, 50th, 75th, 95th
  const percentiles: [number, number, number, number, number] = [
    sorted[Math.floor(0.05 * n)]!,
    sorted[Math.floor(0.25 * n)]!,
    sorted[Math.floor(0.50 * n)]!,
    sorted[Math.floor(0.75 * n)]!,
    sorted[Math.min(Math.floor(0.95 * n), n - 1)]!,
  ]

  return {
    mean,
    stdError,
    var: varValue,
    cvar: cvarValue,
    percentiles,
    terminalValues: Float64Array.from(sorted),
  }
}

/**
 * Beasley-Springer-Moro approximation for inverse standard normal CDF.
 * Used to convert uniform Sobol samples to normal variates.
 * Maximum absolute error < 3.5e-4.
 */
function inverseCDFApprox(u: number): number {
  const a = [
    -3.969683028665376e1, 2.209460984245205e2,
    -2.759285104469687e2, 1.383577518672690e2,
    -3.066479806614716e1, 2.506628277459239e0,
  ]
  const b = [
    -5.447609879822406e1, 1.615858368580409e2,
    -1.556989798598866e2, 6.680131188771972e1,
    -1.328068155288572e1,
  ]
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1,
    -2.400758277161838e0, -2.549732539343734e0,
    4.374664141464968e0, 2.938163982698783e0,
  ]
  const d = [
    7.784695709041462e-3, 3.224671290700398e-1,
    2.445134137142996e0, 3.754408661907416e0,
  ]

  const pLow = 0.02425
  const pHigh = 1 - pLow

  if (u < pLow) {
    const q = Math.sqrt(-2 * Math.log(u))
    return (((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) /
           ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1)
  }

  if (u <= pHigh) {
    const q = u - 0.5
    const r = q * q
    return (((((a[0]! * r + a[1]!) * r + a[2]!) * r + a[3]!) * r + a[4]!) * r + a[5]!) * q /
           (((((b[0]! * r + b[1]!) * r + b[2]!) * r + b[3]!) * r + b[4]!) * r + 1)
  }

  const q = Math.sqrt(-2 * Math.log(1 - u))
  return -(((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) /
          ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1)
}
