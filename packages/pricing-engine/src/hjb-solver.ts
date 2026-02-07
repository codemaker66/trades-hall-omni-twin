/**
 * Hamilton-Jacobi-Bellman Dynamic Pricing Solver
 *
 * Finds the optimal price for each (remaining capacity, time-to-event) pair.
 * The venue owner maximizes expected revenue subject to capacity constraints
 * and stochastic demand.
 *
 * The HJB PDE with remaining capacity n and time-to-event t:
 *   ∂V/∂t + max_p { D(p,t) · [p + V(n-1,t) - V(n,t)] } = 0
 *   V(n, 0) = 0  (unsold slots have zero salvage value)
 *
 * Optimal price from first-order condition:
 *   p*(n,t) = 1/α + [V(n,t) - V(n-1,t)]
 *           = markup + shadow price of capacity
 *
 * As capacity becomes scarce, shadow price rises → prices increase automatically.
 *
 * References:
 * - Gallego & van Ryzin (1994). "Optimal Dynamic Pricing of Inventories"
 * - Talluri & van Ryzin (2004). "The Theory and Practice of Revenue Management"
 */

import type { HJBConfig, HJBResult } from './types'

/**
 * Solve the HJB dynamic pricing equation via policy iteration.
 *
 * Policy iteration converges in 5-10 iterations (quadratic convergence)
 * vs. value iteration which may need 100+ iterations.
 *
 * Demand model: D(p,t) = λ₀(t) · e^{-αp} (exponential demand)
 *
 * The solver computes:
 * 1. Value function V(n,t) — expected revenue-to-go
 * 2. Optimal price p*(n,t) — price to charge
 * 3. Shadow price ∂V/∂n — marginal value of one more unit of capacity
 *
 * @param config - HJB solver configuration
 * @returns HJBResult with optimal prices, value function, and shadow prices
 */
export function solveHJBPricing(config: HJBConfig): HJBResult {
  const {
    maxCapacity,
    timeHorizonDays,
    dt,
    baseDemandRate,
    priceSensitivity,
    seasonalFactors,
  } = config

  const nCap = maxCapacity
  const nSteps = Math.floor(timeHorizonDays / dt)
  const alpha = priceSensitivity

  // V[n][t] = value function
  // Stored as flat arrays for performance
  const vSize = (nCap + 1) * (nSteps + 1)
  const v = new Float64Array(vSize)
  const p = new Float64Array(vSize)

  // Helper to index into flat arrays: v[n * (nSteps+1) + t]
  const idx = (n: number, t: number) => n * (nSteps + 1) + t

  // Terminal condition: V(n, 0) = 0 for all n (already zero-initialized)

  const maxPolicyIter = 20
  const convergenceTol = 1e-6
  let iterations = 0

  // Previous value function for convergence check
  const vOld = new Float64Array(vSize)

  for (let iter = 0; iter < maxPolicyIter; iter++) {
    // Copy current value function
    vOld.set(v)
    iterations = iter + 1

    // Backward in time (from horizon back to now)
    for (let t = nSteps - 1; t >= 0; t--) {
      const lambdaT = baseDemandRate * (seasonalFactors?.[t] ?? 1.0)

      for (let n = 1; n <= nCap; n++) {
        // Shadow price: marginal value of capacity
        const shadow = v[idx(n, t + 1)]! - v[idx(n - 1, t + 1)]!

        // Optimal price from FOC: p* = 1/α + shadow_price
        const pStar = Math.max(0, 1 / alpha + shadow)

        // Demand at optimal price: λ(t) · e^{-αp*}
        const demand = lambdaT * Math.exp(-alpha * pStar)

        // Value function update (Bellman equation):
        // V(n,t) = D(p*,t)·dt·[p* + V(n-1,t+1) - V(n,t+1)] + V(n,t+1)
        v[idx(n, t)] = demand * dt * (pStar + v[idx(n - 1, t + 1)]! - v[idx(n, t + 1)]!)
                      + v[idx(n, t + 1)]!

        p[idx(n, t)] = pStar
      }

      // V(0, t) = 0 for all t (no capacity left, no revenue possible)
      // Already zero-initialized
    }

    // Check convergence
    let maxDiff = 0
    for (let i = 0; i < vSize; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(v[i]! - vOld[i]!))
    }
    if (maxDiff < convergenceTol) break
  }

  // Compute shadow prices
  const shadows = new Float64Array(vSize)
  for (let n = 1; n <= nCap; n++) {
    for (let t = 0; t <= nSteps; t++) {
      shadows[idx(n, t)] = v[idx(n, t)]! - v[idx(n - 1, t)]!
    }
  }

  return {
    optimalPrices: p,
    valueFunction: v,
    shadowPrices: shadows,
    nCapacity: nCap,
    nTimeSteps: nSteps,
    iterations,
  }
}

/**
 * Get the optimal price for a specific (capacity, time) state.
 *
 * @param result - HJB solver result
 * @param remainingCapacity - Current remaining capacity
 * @param timeStep - Current time step (0 = now, nTimeSteps = event time)
 */
export function getOptimalPrice(
  result: HJBResult,
  remainingCapacity: number,
  timeStep: number,
): number {
  const n = Math.min(remainingCapacity, result.nCapacity)
  const t = Math.min(timeStep, result.nTimeSteps)
  return result.optimalPrices[n * (result.nTimeSteps + 1) + t]!
}

/**
 * Get the shadow price (marginal value of one more unit of capacity).
 */
export function getShadowPrice(
  result: HJBResult,
  remainingCapacity: number,
  timeStep: number,
): number {
  const n = Math.min(remainingCapacity, result.nCapacity)
  const t = Math.min(timeStep, result.nTimeSteps)
  return result.shadowPrices[n * (result.nTimeSteps + 1) + t]!
}

/**
 * Get the value function (expected revenue-to-go).
 */
export function getValueFunction(
  result: HJBResult,
  remainingCapacity: number,
  timeStep: number,
): number {
  const n = Math.min(remainingCapacity, result.nCapacity)
  const t = Math.min(timeStep, result.nTimeSteps)
  return result.valueFunction[n * (result.nTimeSteps + 1) + t]!
}

/**
 * Compute the accept/reject decision for a booking request.
 *
 * Accept if: offered price >= shadow price of capacity
 * (Equivalent to bid-price control in revenue management)
 *
 * @param result - HJB solver result
 * @param offeredPrice - Price the customer is willing to pay
 * @param remainingCapacity - Current remaining capacity
 * @param timeStep - Current time step
 * @returns { accept: boolean, shadowPrice: number, surplus: number }
 */
export function shouldAcceptBooking(
  result: HJBResult,
  offeredPrice: number,
  remainingCapacity: number,
  timeStep: number,
): { accept: boolean; shadowPrice: number; surplus: number } {
  const shadow = getShadowPrice(result, remainingCapacity, timeStep)
  return {
    accept: offeredPrice >= shadow,
    shadowPrice: shadow,
    surplus: offeredPrice - shadow,
  }
}

/**
 * Generate a complete pricing schedule: optimal price at each
 * (capacity level, days before event) combination.
 *
 * Useful for building a pricing lookup table.
 */
export function generatePricingSchedule(
  result: HJBResult,
  dtDays: number,
): Array<{
  daysBeforeEvent: number
  capacity: number
  optimalPrice: number
  shadowPrice: number
  expectedRevenue: number
}> {
  const schedule: Array<{
    daysBeforeEvent: number
    capacity: number
    optimalPrice: number
    shadowPrice: number
    expectedRevenue: number
  }> = []

  for (let n = 1; n <= result.nCapacity; n++) {
    for (let t = 0; t < result.nTimeSteps; t++) {
      schedule.push({
        daysBeforeEvent: (result.nTimeSteps - t) * dtDays,
        capacity: n,
        optimalPrice: getOptimalPrice(result, n, t),
        shadowPrice: getShadowPrice(result, n, t),
        expectedRevenue: getValueFunction(result, n, t),
      })
    }
  }

  return schedule
}

/**
 * Solve HJB with multiple customer segments (heterogeneous demand).
 *
 * Extends the basic solver to handle multiple demand classes with different
 * price sensitivities and arrival rates. This is the multi-class version
 * from Talluri & van Ryzin (2004).
 *
 * @param config - Base HJB config
 * @param segments - Array of { demandRate, priceSensitivity } per customer segment
 */
export function solveHJBMultiSegment(
  config: Omit<HJBConfig, 'baseDemandRate' | 'priceSensitivity'>,
  segments: Array<{ demandRate: number; priceSensitivity: number; name: string }>,
): HJBResult & { segmentPrices: Float64Array[] } {
  const { maxCapacity, timeHorizonDays, dt, seasonalFactors } = config
  const nCap = maxCapacity
  const nSteps = Math.floor(timeHorizonDays / dt)
  const nSeg = segments.length

  const vSize = (nCap + 1) * (nSteps + 1)
  const v = new Float64Array(vSize)
  const p = new Float64Array(vSize)
  const segmentPrices: Float64Array[] = segments.map(() => new Float64Array(vSize))

  const idx = (n: number, t: number) => n * (nSteps + 1) + t

  const maxPolicyIter = 20
  const convergenceTol = 1e-6
  let iterations = 0
  const vOld = new Float64Array(vSize)

  for (let iter = 0; iter < maxPolicyIter; iter++) {
    vOld.set(v)
    iterations = iter + 1

    for (let t = nSteps - 1; t >= 0; t--) {
      const seasonal = seasonalFactors?.[t] ?? 1.0

      for (let n = 1; n <= nCap; n++) {
        const shadow = v[idx(n, t + 1)]! - v[idx(n - 1, t + 1)]!

        // For each segment, compute optimal price and contribution
        let totalContribution = 0
        let bestPrice = 0
        let bestContribution = -Infinity

        for (let s = 0; s < nSeg; s++) {
          const seg = segments[s]!
          const lambdaT = seg.demandRate * seasonal
          const alpha = seg.priceSensitivity

          const pStar = Math.max(0, 1 / alpha + shadow)
          const demand = lambdaT * Math.exp(-alpha * pStar)
          const contribution = demand * dt * (pStar + v[idx(n - 1, t + 1)]! - v[idx(n, t + 1)]!)

          segmentPrices[s]![idx(n, t)] = pStar

          if (contribution > bestContribution) {
            bestContribution = contribution
            bestPrice = pStar
          }

          totalContribution += contribution
        }

        v[idx(n, t)] = totalContribution + v[idx(n, t + 1)]!
        p[idx(n, t)] = bestPrice
      }
    }

    let maxDiff = 0
    for (let i = 0; i < vSize; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(v[i]! - vOld[i]!))
    }
    if (maxDiff < convergenceTol) break
  }

  const shadows = new Float64Array(vSize)
  for (let n = 1; n <= nCap; n++) {
    for (let t = 0; t <= nSteps; t++) {
      shadows[idx(n, t)] = v[idx(n, t)]! - v[idx(n - 1, t)]!
    }
  }

  return {
    optimalPrices: p,
    valueFunction: v,
    shadowPrices: shadows,
    nCapacity: nCap,
    nTimeSteps: nSteps,
    iterations,
    segmentPrices,
  }
}
