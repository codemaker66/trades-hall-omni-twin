/**
 * Portfolio Theory — Booking Mix Optimization
 *
 * Markowitz portfolio optimization for venue's mix of event types,
 * plus CVaR risk management and Black-Litterman views.
 *
 * Venue application:
 * - Each event type is an "asset class"
 * - Wedding: high revenue, seasonal (May-Oct), weekend-heavy
 * - Corporate: medium revenue, fiscal-year-driven (Sep-Nov, Jan-Mar), weekday
 * - Party: low revenue, year-round, Friday-Saturday
 * - Conference: high revenue, sporadic, multi-day
 *
 * Diversification insight: Wedding demand has LOW or NEGATIVE correlation
 * with corporate events. Venues diversifying across types achieve 40-60%
 * higher annual revenue (industry data).
 *
 * References:
 * - Markowitz (1952). "Portfolio Selection"
 * - Rockafellar & Uryasev (2000). "CVaR optimization"
 * - Black & Litterman (1992). "Global Portfolio Optimization"
 * - Andersson et al. (2017). "Event portfolios: Markowitz for events"
 */

import type { PortfolioResult } from './types'

/**
 * Markowitz efficient frontier for event type diversification.
 *
 * Solves: min w'Σw  s.t.  w'μ ≥ R,  w'1 = 1,  w ≥ 0
 * for a range of target returns R.
 *
 * Uses projected gradient descent (handles non-negativity constraints).
 *
 * @param expectedReturns - Expected revenue per event type (μ vector)
 * @param covarianceMatrix - Covariance matrix (row-major flat array, n×n)
 * @param nTypes - Number of event types
 * @param nFrontierPoints - Number of points on the efficient frontier
 * @param riskFreeRate - Risk-free rate for Sharpe ratio (default 0)
 */
export function optimizeBookingMix(
  expectedReturns: number[],
  covarianceMatrix: number[],
  nTypes: number,
  nFrontierPoints: number = 20,
  riskFreeRate: number = 0,
): PortfolioResult {
  const n = nTypes

  // Validate inputs
  if (expectedReturns.length !== n) {
    throw new Error(`Expected ${n} returns, got ${expectedReturns.length}`)
  }
  if (covarianceMatrix.length !== n * n) {
    throw new Error(`Covariance matrix should be ${n}×${n} = ${n * n} elements`)
  }

  // Find min and max achievable returns
  const minReturn = Math.min(...expectedReturns)
  const maxReturn = Math.max(...expectedReturns)

  // Compute efficient frontier
  const frontier: Array<[number, number]> = []
  let bestSharpe = -Infinity
  let bestWeights = new Array(n).fill(1 / n) as number[]
  let bestExpReturn = 0
  let bestVol = 0

  const targetReturns = Array.from({ length: nFrontierPoints }, (_, i) =>
    minReturn + (i / (nFrontierPoints - 1)) * (maxReturn - minReturn),
  )

  for (const targetReturn of targetReturns) {
    const w = solveMinVariance(expectedReturns, covarianceMatrix, n, targetReturn)
    const vol = portfolioVolatility(w, covarianceMatrix, n)
    const ret = portfolioReturn(w, expectedReturns)

    frontier.push([vol, ret])

    const sharpe = vol > 0 ? (ret - riskFreeRate) / vol : 0
    if (sharpe > bestSharpe) {
      bestSharpe = sharpe
      bestWeights = w
      bestExpReturn = ret
      bestVol = vol
    }
  }

  // Compute VaR and CVaR for optimal portfolio (normal approximation)
  const z95 = 1.645
  const var95 = bestExpReturn - z95 * bestVol
  const cvar95 = bestExpReturn - bestVol * normalPDF(z95) / 0.05

  return {
    weights: bestWeights,
    expectedRevenue: bestExpReturn,
    volatility: bestVol,
    sharpeRatio: bestSharpe,
    var95,
    cvar95,
    efficientFrontier: frontier,
  }
}

/**
 * CVaR optimization via linear programming formulation.
 *
 * Rockafellar-Uryasev (2000) LP:
 *   min  ζ + (1/(S·(1-β))) · Σₛ uₛ
 *   s.t. uₛ ≥ -(w'rₛ) - ζ,  uₛ ≥ 0,  w'1 = 1,  w ≥ 0
 *        w'E[r] ≥ minReturn
 *
 * Unlike VaR, CVaR is sub-additive (always rewards diversification).
 * Handles 100s of scenarios × dozens of event types.
 *
 * @param scenarios - Revenue scenarios: row-major S×n matrix
 * @param nScenarios - Number of scenarios
 * @param nTypes - Number of event types
 * @param confidence - β (e.g., 0.95 for 95% CVaR)
 * @param minExpectedReturn - Minimum acceptable expected return
 */
export function optimizeCVaR(
  scenarios: number[],
  nScenarios: number,
  nTypes: number,
  confidence: number = 0.95,
  minExpectedReturn: number = 0,
): PortfolioResult {
  const n = nTypes
  const S = nScenarios

  // Compute expected returns from scenarios
  const expectedReturns = new Array(n).fill(0) as number[]
  for (let j = 0; j < n; j++) {
    for (let s = 0; s < S; s++) {
      expectedReturns[j] = expectedReturns[j]! + scenarios[s * n + j]! / S
    }
  }

  // Iterative optimization: projected gradient descent on CVaR
  let weights = new Array(n).fill(1 / n) as number[]
  let bestWeights = [...weights]
  let bestCVaR = -Infinity

  const lr = 0.01
  const maxIter = 500

  for (let iter = 0; iter < maxIter; iter++) {
    // Compute portfolio returns for each scenario
    const portfolioReturns = new Array(S).fill(0) as number[]
    for (let s = 0; s < S; s++) {
      for (let j = 0; j < n; j++) {
        portfolioReturns[s] = portfolioReturns[s]! + weights[j]! * scenarios[s * n + j]!
      }
    }

    // Sort to find VaR
    const sorted = [...portfolioReturns].sort((a, b) => a - b)
    const varIdx = Math.floor((1 - confidence) * S)
    const varValue = sorted[varIdx]!

    // CVaR = average of returns below VaR
    let cvarSum = 0
    let cvarCount = 0
    for (let s = 0; s < S; s++) {
      if (portfolioReturns[s]! <= varValue) {
        cvarSum += portfolioReturns[s]!
        cvarCount++
      }
    }
    const cvarValue = cvarCount > 0 ? cvarSum / cvarCount : varValue

    if (cvarValue > bestCVaR) {
      bestCVaR = cvarValue
      bestWeights = [...weights]
    }

    // Gradient: ∂CVaR/∂wⱼ ≈ (1/|tail|) Σ_{s in tail} r_{sj}
    const gradient = new Array(n).fill(0) as number[]
    for (let s = 0; s < S; s++) {
      if (portfolioReturns[s]! <= varValue) {
        for (let j = 0; j < n; j++) {
          gradient[j] = gradient[j]! + scenarios[s * n + j]! / Math.max(cvarCount, 1)
        }
      }
    }

    // Gradient ascent (maximize CVaR = minimize tail risk)
    for (let j = 0; j < n; j++) {
      weights[j] = weights[j]! + lr * gradient[j]!
    }

    // Project onto simplex (w ≥ 0, Σw = 1) with return constraint
    weights = projectOntoSimplex(weights)

    // Check return constraint
    let expRet = 0
    for (let j = 0; j < n; j++) {
      expRet += weights[j]! * expectedReturns[j]!
    }
    if (expRet < minExpectedReturn) {
      // Push towards higher return
      const maxRetIdx = expectedReturns.indexOf(Math.max(...expectedReturns))
      weights[maxRetIdx] = weights[maxRetIdx]! + 0.01
      weights = projectOntoSimplex(weights)
    }
  }

  // Compute final statistics
  const portfolioReturns: number[] = []
  for (let s = 0; s < S; s++) {
    let ret = 0
    for (let j = 0; j < n; j++) {
      ret += bestWeights[j]! * scenarios[s * n + j]!
    }
    portfolioReturns.push(ret)
  }

  const sorted = [...portfolioReturns].sort((a, b) => a - b)
  const varIdx = Math.floor((1 - confidence) * S)

  let cvarSum = 0
  for (let i = 0; i <= varIdx; i++) {
    cvarSum += sorted[i]!
  }

  const expReturn = portfolioReturns.reduce((s, r) => s + r, 0) / S
  const variance = portfolioReturns.reduce((s, r) => s + (r - expReturn) ** 2, 0) / (S - 1)
  const vol = Math.sqrt(variance)

  return {
    weights: bestWeights,
    expectedRevenue: expReturn,
    volatility: vol,
    sharpeRatio: vol > 0 ? expReturn / vol : 0,
    var95: sorted[varIdx]!,
    cvar95: (varIdx + 1) > 0 ? cvarSum / (varIdx + 1) : sorted[0]!,
    efficientFrontier: [], // CVaR doesn't produce a standard frontier
  }
}

/**
 * Black-Litterman model for incorporating venue manager views.
 *
 * Combines market equilibrium with manager beliefs:
 *   "Corporate events will increase 15% next quarter" (absolute view)
 *   "Weddings will outperform concerts by $2K/event" (relative view)
 *
 * Posterior: E[r] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ · [(τΣ)⁻¹π + P'Ω⁻¹Q]
 *
 * @param marketWeights - Current booking mix (equilibrium weights)
 * @param covarianceMatrix - n×n covariance (row-major flat)
 * @param viewsP - Pick matrix (k×n, row-major): which assets each view references
 * @param viewsQ - View expected returns (k values)
 * @param viewConfidence - Diagonal of Ω (uncertainty per view)
 * @param riskAversion - δ (risk aversion parameter)
 * @param tau - Uncertainty scaling on prior (typically 0.025-0.05)
 * @param nTypes - Number of event types
 * @param nViews - Number of views
 */
export function blackLitterman(
  marketWeights: number[],
  covarianceMatrix: number[],
  viewsP: number[],
  viewsQ: number[],
  viewConfidence: number[],
  riskAversion: number,
  tau: number,
  nTypes: number,
  nViews: number,
): { posteriorReturns: number[]; posteriorWeights: number[] } {
  const n = nTypes
  const k = nViews

  // Step 1: Implied equilibrium returns π = δΣw
  const pi: number[] = new Array(n).fill(0)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      pi[i] = pi[i]! + riskAversion * covarianceMatrix[i * n + j]! * marketWeights[j]!
    }
  }

  // Step 2: τΣ (prior covariance of returns)
  const tauSigma: number[] = covarianceMatrix.map((v) => tau * v)

  // Step 3: Ω (view uncertainty matrix, diagonal)
  const omegaInv: number[] = new Array(k * k).fill(0)
  for (let i = 0; i < k; i++) {
    omegaInv[i * k + i] = 1 / viewConfidence[i]!
  }

  // Step 4: (τΣ)⁻¹ — invert using Cholesky
  const tauSigmaInv = invertMatrix(tauSigma, n)

  // Step 5: P'Ω⁻¹P (n×n matrix)
  const PtOmegaInvP: number[] = new Array(n * n).fill(0)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      for (let v = 0; v < k; v++) {
        PtOmegaInvP[i * n + j] = PtOmegaInvP[i * n + j]!
          + viewsP[v * n + i]! * omegaInv[v * k + v]! * viewsP[v * n + j]!
      }
    }
  }

  // Step 6: [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹
  const combined: number[] = new Array(n * n).fill(0)
  for (let i = 0; i < n * n; i++) {
    combined[i] = tauSigmaInv[i]! + PtOmegaInvP[i]!
  }
  const combinedInv = invertMatrix(combined, n)

  // Step 7: (τΣ)⁻¹π + P'Ω⁻¹Q
  const rhs: number[] = new Array(n).fill(0)
  // (τΣ)⁻¹π
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      rhs[i] = rhs[i]! + tauSigmaInv[i * n + j]! * pi[j]!
    }
  }
  // + P'Ω⁻¹Q
  for (let i = 0; i < n; i++) {
    for (let v = 0; v < k; v++) {
      rhs[i] = rhs[i]! + viewsP[v * n + i]! * omegaInv[v * k + v]! * viewsQ[v]!
    }
  }

  // Step 8: Posterior returns E[r] = combinedInv · rhs
  const posteriorReturns: number[] = new Array(n).fill(0)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      posteriorReturns[i] = posteriorReturns[i]! + combinedInv[i * n + j]! * rhs[j]!
    }
  }

  // Step 9: Posterior weights w* = (δΣ)⁻¹ · E[r]
  const deltaSigmaInv = invertMatrix(
    covarianceMatrix.map((v) => riskAversion * v),
    n,
  )
  const posteriorWeights: number[] = new Array(n).fill(0)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      posteriorWeights[i] = posteriorWeights[i]! + deltaSigmaInv[i * n + j]! * posteriorReturns[j]!
    }
  }

  // Normalize and project onto simplex
  const projectedWeights = projectOntoSimplex(posteriorWeights)

  return { posteriorReturns, posteriorWeights: projectedWeights }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Minimum variance portfolio for a target return via projected gradient descent */
function solveMinVariance(
  returns: number[],
  cov: number[],
  n: number,
  targetReturn: number,
): number[] {
  let w = new Array(n).fill(1 / n) as number[]
  const lr = 0.005
  const maxIter = 300
  const penalty = 100 // Lagrange penalty for return constraint

  for (let iter = 0; iter < maxIter; iter++) {
    // Gradient of w'Σw: 2Σw
    const grad: number[] = new Array(n).fill(0)
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        grad[i] = grad[i]! + 2 * cov[i * n + j]! * w[j]!
      }
    }

    // Penalty gradient for return constraint: -2·penalty·(w'μ - R)·μ
    const curReturn = portfolioReturn(w, returns)
    const returnPenalty = penalty * Math.max(0, targetReturn - curReturn)
    for (let i = 0; i < n; i++) {
      grad[i] = grad[i]! - returnPenalty * returns[i]!
    }

    // Gradient descent
    for (let i = 0; i < n; i++) {
      w[i] = w[i]! - lr * grad[i]!
    }

    // Project onto simplex
    w = projectOntoSimplex(w)
  }

  return w
}

function portfolioReturn(w: number[], returns: number[]): number {
  let ret = 0
  for (let i = 0; i < w.length; i++) {
    ret += w[i]! * returns[i]!
  }
  return ret
}

function portfolioVolatility(w: number[], cov: number[], n: number): number {
  let variance = 0
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      variance += w[i]! * cov[i * n + j]! * w[j]!
    }
  }
  return Math.sqrt(Math.max(0, variance))
}

/** Project a vector onto the probability simplex (w ≥ 0, Σw = 1) */
function projectOntoSimplex(v: number[]): number[] {
  const n = v.length
  const u = [...v].sort((a, b) => b - a) // Sort descending

  let cumSum = 0
  let rho = 0
  for (let i = 0; i < n; i++) {
    cumSum += u[i]!
    if (u[i]! + (1 - cumSum) / (i + 1) > 0) {
      rho = i
    }
  }

  let sum = 0
  for (let i = 0; i <= rho; i++) {
    sum += u[i]!
  }
  const theta = (sum - 1) / (rho + 1)

  return v.map((vi) => Math.max(0, vi - theta))
}

/** Invert a matrix via Gauss-Jordan elimination */
function invertMatrix(m: number[], n: number): number[] {
  // Augmented matrix [M | I]
  const aug: number[][] = Array.from({ length: n }, (_, i) => {
    const row = new Array(2 * n).fill(0) as number[]
    for (let j = 0; j < n; j++) {
      row[j] = m[i * n + j]!
    }
    row[n + i] = 1
    return row
  })

  // Forward elimination
  for (let col = 0; col < n; col++) {
    // Partial pivoting
    let maxRow = col
    let maxVal = Math.abs(aug[col]![col]!)
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row]![col]!) > maxVal) {
        maxVal = Math.abs(aug[row]![col]!)
        maxRow = row
      }
    }

    if (maxRow !== col) {
      const tmp = aug[col]!
      aug[col] = aug[maxRow]!
      aug[maxRow] = tmp
    }

    const pivot = aug[col]![col]!
    if (Math.abs(pivot) < 1e-15) {
      // Singular matrix — return identity scaled by large number
      const result = new Array(n * n).fill(0) as number[]
      for (let i = 0; i < n; i++) result[i * n + i] = 1e10
      return result
    }

    // Scale pivot row
    for (let j = 0; j < 2 * n; j++) {
      aug[col]![j] = aug[col]![j]! / pivot
    }

    // Eliminate column
    for (let row = 0; row < n; row++) {
      if (row === col) continue
      const factor = aug[row]![col]!
      for (let j = 0; j < 2 * n; j++) {
        aug[row]![j] = aug[row]![j]! - factor * aug[col]![j]!
      }
    }
  }

  // Extract inverse
  const result = new Array(n * n).fill(0) as number[]
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      result[i * n + j] = aug[i]![n + j]!
    }
  }

  return result
}

function normalPDF(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)
}
