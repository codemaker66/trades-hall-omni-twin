/**
 * Black-Scholes Option Pricing for Venue Booking Hold Fees
 *
 * Maps financial options to venue booking:
 * - Underlying S: Expected revenue from the slot at current demand
 * - Strike K: Locked-in booking price
 * - Expiry T: Confirmation deadline (days to decision)
 * - Volatility σ: Demand uncertainty for that date/venue type
 * - Risk-free rate r: Opportunity cost of holding slot unavailable
 * - Premium C: Non-refundable hold fee
 * - Delta Δ: Hold fee sensitivity to demand shift
 * - Theta Θ: Daily time decay — justifies declining refund schedules
 * - Vega ν: Sensitivity to uncertainty — hold fees rise during volatile periods
 * - Gamma Γ: Signals when to re-price hold fees more frequently
 *
 * Includes:
 * - European call/put pricing with all Greeks
 * - Implied volatility via Newton-Raphson with bisection fallback
 * - CRR binomial tree for American-style early exercise
 * - Floor price computation (Anderson et al. 2004)
 *
 * References:
 * - Black & Scholes (1973). "The Pricing of Options and Corporate Liabilities"
 * - Anderson, Davison & Rasmussen (2004). "Real options in RM"
 */

import type { OptionResult, OptionType } from './types'

// ---------------------------------------------------------------------------
// Normal Distribution Functions
// ---------------------------------------------------------------------------

/**
 * Standard normal CDF using Abramowitz & Stegun approximation.
 * Maximum absolute error < 1.5e-7.
 */
export function normCDF(x: number): number {
  const a1 = 0.254829592
  const a2 = -0.284496736
  const a3 = 1.421413741
  const a4 = -1.453152027
  const a5 = 1.061405429
  const p = 0.3275911

  const sign = x < 0 ? -1 : 1
  const absX = Math.abs(x) / Math.SQRT2
  const t = 1.0 / (1.0 + p * absX)
  const y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-absX * absX)
  return 0.5 * (1.0 + sign * y)
}

/** Standard normal PDF */
export function normPDF(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)
}

// ---------------------------------------------------------------------------
// Black-Scholes Pricing
// ---------------------------------------------------------------------------

/**
 * European call option price with all Greeks.
 *
 * C = S·N(d₁) - K·e^{-rT}·N(d₂)
 *
 * where:
 *   d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
 *   d₂ = d₁ - σ√T
 *
 * @param s - Current underlying price (demand value of slot)
 * @param k - Strike price (locked-in booking price)
 * @param t - Time to expiry in years
 * @param r - Risk-free rate (opportunity cost)
 * @param sigma - Volatility (demand uncertainty)
 */
export function blackScholesCall(
  s: number,
  k: number,
  t: number,
  r: number,
  sigma: number,
): OptionResult {
  if (t <= 0) {
    const intrinsic = Math.max(s - k, 0)
    return {
      price: intrinsic,
      delta: intrinsic > 0 ? 1 : 0,
      gamma: 0,
      theta: 0,
      vega: 0,
      rho: 0,
    }
  }

  const sqrtT = Math.sqrt(t)
  const d1 = (Math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrtT)
  const d2 = d1 - sigma * sqrtT
  const discount = Math.exp(-r * t)

  return {
    price: s * normCDF(d1) - k * discount * normCDF(d2),
    delta: normCDF(d1),
    gamma: normPDF(d1) / (s * sigma * sqrtT),
    theta: -(s * normPDF(d1) * sigma) / (2 * sqrtT) - r * k * discount * normCDF(d2),
    vega: s * sqrtT * normPDF(d1),
    rho: k * t * discount * normCDF(d2),
  }
}

/**
 * European put option price with all Greeks.
 *
 * P = K·e^{-rT}·N(-d₂) - S·N(-d₁)
 */
export function blackScholesPut(
  s: number,
  k: number,
  t: number,
  r: number,
  sigma: number,
): OptionResult {
  if (t <= 0) {
    const intrinsic = Math.max(k - s, 0)
    return {
      price: intrinsic,
      delta: intrinsic > 0 ? -1 : 0,
      gamma: 0,
      theta: 0,
      vega: 0,
      rho: 0,
    }
  }

  const sqrtT = Math.sqrt(t)
  const d1 = (Math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrtT)
  const d2 = d1 - sigma * sqrtT
  const discount = Math.exp(-r * t)

  return {
    price: k * discount * normCDF(-d2) - s * normCDF(-d1),
    delta: normCDF(d1) - 1,
    gamma: normPDF(d1) / (s * sigma * sqrtT),
    theta: -(s * normPDF(d1) * sigma) / (2 * sqrtT) + r * k * discount * normCDF(-d2),
    vega: s * sqrtT * normPDF(d1),
    rho: -k * t * discount * normCDF(-d2),
  }
}

/**
 * Price a European option (call or put).
 */
export function blackScholes(
  s: number,
  k: number,
  t: number,
  r: number,
  sigma: number,
  type: OptionType,
): OptionResult {
  return type === 'call'
    ? blackScholesCall(s, k, t, r, sigma)
    : blackScholesPut(s, k, t, r, sigma)
}

// ---------------------------------------------------------------------------
// Implied Volatility
// ---------------------------------------------------------------------------

/**
 * Compute implied volatility via Newton-Raphson with bisection fallback.
 *
 * Newton-Raphson: σ_{n+1} = σ_n - (C(σ_n) - C_market) / vega(σ_n)
 * Falls back to bisection if Newton diverges (vega too small or oscillation).
 *
 * @param marketPrice - Observed option premium
 * @param s - Underlying price
 * @param k - Strike price
 * @param t - Time to expiry (years)
 * @param r - Risk-free rate
 * @param type - 'call' or 'put'
 * @param tol - Convergence tolerance (default 1e-8)
 * @param maxIter - Maximum iterations (default 100)
 */
export function impliedVolatility(
  marketPrice: number,
  s: number,
  k: number,
  t: number,
  r: number,
  type: OptionType,
  tol: number = 1e-8,
  maxIter: number = 100,
): number {
  const pricer = type === 'call' ? blackScholesCall : blackScholesPut

  // Initial guess using Brenner-Subrahmanyam approximation
  let sigma = Math.sqrt((2 * Math.PI) / t) * (marketPrice / s)
  sigma = Math.max(0.01, Math.min(sigma, 5.0))

  // Newton-Raphson
  for (let i = 0; i < maxIter; i++) {
    const result = pricer(s, k, t, r, sigma)
    const diff = result.price - marketPrice

    if (Math.abs(diff) < tol) return sigma

    // If vega is too small, switch to bisection
    if (Math.abs(result.vega) < 1e-10) break

    const newSigma = sigma - diff / result.vega
    if (newSigma <= 0.001 || newSigma >= 10.0) break

    sigma = newSigma
  }

  // Bisection fallback
  let lo = 0.001
  let hi = 5.0
  for (let i = 0; i < maxIter; i++) {
    const mid = (lo + hi) / 2
    const result = pricer(s, k, t, r, mid)
    const diff = result.price - marketPrice

    if (Math.abs(diff) < tol) return mid

    if (diff > 0) {
      hi = mid
    } else {
      lo = mid
    }
  }

  return (lo + hi) / 2
}

// ---------------------------------------------------------------------------
// CRR Binomial Tree (American Options)
// ---------------------------------------------------------------------------

/**
 * Price an American-style option using the Cox-Ross-Rubinstein binomial tree.
 *
 * American options (exercisable anytime before deadline) are important for
 * venue bookings where the client can confirm at any point, not just at expiry.
 *
 * At each node, compare continuation value vs. immediate exercise value.
 *
 * @param s - Current underlying price
 * @param k - Strike price
 * @param t - Time to expiry (years)
 * @param r - Risk-free rate
 * @param sigma - Volatility
 * @param nSteps - Number of tree steps (higher = more accurate, default 200)
 * @param type - 'call' or 'put'
 * @returns American option price
 */
export function americanOptionBinomial(
  s: number,
  k: number,
  t: number,
  r: number,
  sigma: number,
  nSteps: number = 200,
  type: OptionType = 'put',
): number {
  const dt = t / nSteps
  const u = Math.exp(sigma * Math.sqrt(dt))
  const d = 1 / u
  const p = (Math.exp(r * dt) - d) / (u - d)
  const disc = Math.exp(-r * dt)

  const isCall = type === 'call'
  const n = nSteps

  // Terminal payoffs
  const values = new Float64Array(n + 1)
  for (let i = 0; i <= n; i++) {
    const spot = s * Math.pow(u, i) * Math.pow(d, n - i)
    values[i] = isCall ? Math.max(spot - k, 0) : Math.max(k - spot, 0)
  }

  // Backward induction with early exercise check
  for (let step = n - 1; step >= 0; step--) {
    for (let i = 0; i <= step; i++) {
      const hold = disc * (p * values[i + 1]! + (1 - p) * values[i]!)
      const spot = s * Math.pow(u, i) * Math.pow(d, step - i)
      const exercise = isCall ? Math.max(spot - k, 0) : Math.max(k - spot, 0)
      values[i] = Math.max(hold, exercise) // American: take the better of hold vs exercise
    }
  }

  return values[0]!
}

/**
 * Full American option result with early exercise boundary.
 *
 * Returns the option price plus the critical price at each time step
 * below (for puts) or above (for calls) which early exercise is optimal.
 */
export function americanOptionWithBoundary(
  s: number,
  k: number,
  t: number,
  r: number,
  sigma: number,
  nSteps: number = 200,
  type: OptionType = 'put',
): { price: number; exerciseBoundary: number[] } {
  const dt = t / nSteps
  const u = Math.exp(sigma * Math.sqrt(dt))
  const d = 1 / u
  const p = (Math.exp(r * dt) - d) / (u - d)
  const disc = Math.exp(-r * dt)

  const isCall = type === 'call'
  const n = nSteps

  // Terminal payoffs
  const values = new Float64Array(n + 1)
  for (let i = 0; i <= n; i++) {
    const spot = s * Math.pow(u, i) * Math.pow(d, n - i)
    values[i] = isCall ? Math.max(spot - k, 0) : Math.max(k - spot, 0)
  }

  // Track exercise boundary
  const exerciseBoundary: number[] = new Array(n + 1)
  exerciseBoundary[n] = k // At expiry, exercise boundary is the strike

  // Backward induction
  for (let step = n - 1; step >= 0; step--) {
    let boundary = isCall ? Infinity : 0

    for (let i = 0; i <= step; i++) {
      const hold = disc * (p * values[i + 1]! + (1 - p) * values[i]!)
      const spot = s * Math.pow(u, i) * Math.pow(d, step - i)
      const exercise = isCall ? Math.max(spot - k, 0) : Math.max(k - spot, 0)

      if (exercise > hold) {
        // Early exercise is optimal here
        values[i] = exercise
        if (isCall) {
          boundary = Math.min(boundary, spot)
        } else {
          boundary = Math.max(boundary, spot)
        }
      } else {
        values[i] = hold
      }
    }

    exerciseBoundary[step] = boundary
  }

  return { price: values[0]!, exerciseBoundary }
}

// ---------------------------------------------------------------------------
// Floor Price (Anderson et al. 2004)
// ---------------------------------------------------------------------------

/**
 * Compute the minimum price the venue should accept for a slot.
 *
 * Anderson et al. (2004) proved venues systematically discount too deeply.
 * The floor price equals current demand value minus the option value of waiting.
 * Below this price, it's better to wait for a higher-paying customer.
 *
 * Floor = max(0, S - C_ATM)
 *
 * where C_ATM is the at-the-money call option value (the value of waiting).
 *
 * @param currentDemandValue - S: what the slot is worth at current demand
 * @param daysUntilEvent - Time until the event date
 * @param demandVolatility - σ: how uncertain is future demand
 * @param opportunityRate - r: rate of return on alternative use of the slot
 */
export function computeFloorPrice(
  currentDemandValue: number,
  daysUntilEvent: number,
  demandVolatility: number,
  opportunityRate: number,
): number {
  const t = daysUntilEvent / 365
  if (t <= 0) return currentDemandValue

  const option = blackScholesCall(
    currentDemandValue,
    currentDemandValue, // ATM option
    t,
    opportunityRate,
    demandVolatility,
  )

  // Don't sell below: current demand minus the option value of waiting
  return Math.max(0, currentDemandValue - option.price)
}

/**
 * Compute the optimal hold fee (premium) for a venue booking option.
 *
 * The hold fee is the price a client pays for the right to confirm later.
 * It's an out-of-the-money call option premium where:
 * - S = current demand value
 * - K = locked-in price (typically at a premium to current)
 * - T = hold period (days until confirmation deadline)
 *
 * @param currentDemandValue - Current demand-implied price
 * @param lockedPrice - The price the client will pay if they confirm
 * @param holdDays - Number of days they can hold the reservation
 * @param demandVolatility - Demand uncertainty
 * @param opportunityRate - Venue's opportunity cost
 */
export function computeHoldFee(
  currentDemandValue: number,
  lockedPrice: number,
  holdDays: number,
  demandVolatility: number,
  opportunityRate: number,
): OptionResult {
  return blackScholesCall(
    currentDemandValue,
    lockedPrice,
    holdDays / 365,
    opportunityRate,
    demandVolatility,
  )
}

/**
 * Generate a volatility surface: implied vol across strikes and expiries.
 * Useful for understanding the term structure of venue booking uncertainty.
 *
 * @param s - Current underlying
 * @param strikes - Array of strike prices
 * @param expiries - Array of expiry times (years)
 * @param r - Risk-free rate
 * @param pricer - Function that returns market price for (K, T)
 */
export function computeVolSurface(
  s: number,
  strikes: number[],
  expiries: number[],
  r: number,
  pricer: (k: number, t: number) => number,
): Array<{ strike: number; expiry: number; impliedVol: number }> {
  const surface: Array<{ strike: number; expiry: number; impliedVol: number }> = []

  for (const k of strikes) {
    for (const t of expiries) {
      const marketPrice = pricer(k, t)
      const iv = impliedVolatility(marketPrice, s, k, t, r, 'call')
      surface.push({ strike: k, expiry: t, impliedVol: iv })
    }
  }

  return surface
}
