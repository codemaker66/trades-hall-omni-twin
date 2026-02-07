/**
 * Log-domain stabilized Sinkhorn (Schmitzer 2019, arXiv:1610.06519).
 *
 * Works with dual potentials f, g instead of scaling vectors u, v:
 *   f_i ← -ε · LSE_j((g_j - C_ij) / ε) + ε·log(a_i)
 *   g_j ← -ε · LSE_i((f_i - C_ij) / ε) + ε·log(b_j)
 *
 * Stable for ANY ε, including very small values where standard Sinkhorn overflows.
 */

import type { SinkhornConfig, TransportResult } from './types'
import { DEFAULT_SINKHORN_CONFIG } from './types'
import { dot } from './utils'

/**
 * Log-domain Sinkhorn solver.
 */
export function sinkhornLog(
  a: Float64Array,
  b: Float64Array,
  C: Float64Array,
  config: Partial<SinkhornConfig> = {},
): TransportResult {
  const cfg = { ...DEFAULT_SINKHORN_CONFIG, method: 'log' as const, ...config }
  const N = a.length
  const M = b.length
  const eps = cfg.epsilon

  // Pre-compute log of marginals
  const logA = new Float64Array(N)
  const logB = new Float64Array(M)
  for (let i = 0; i < N; i++) {
    logA[i] = Math.log(Math.max(a[i]!, 1e-300))
  }
  for (let j = 0; j < M; j++) {
    logB[j] = Math.log(Math.max(b[j]!, 1e-300))
  }

  // Dual potentials
  const f = new Float64Array(N)  // initialized to 0
  const g = new Float64Array(M)

  // Temp buffer for LSE computation
  const lseBuffer = new Float64Array(Math.max(N, M))

  let converged = false
  let iter = 0

  for (iter = 0; iter < cfg.maxIterations; iter++) {
    // Update f: f_i = -ε · LSE_j((g_j - C_ij) / ε) + ε·log(a_i)
    for (let i = 0; i < N; i++) {
      // Compute (g_j - C_ij) / ε for all j
      let maxVal = -Infinity
      for (let j = 0; j < M; j++) {
        const val = (g[j]! - C[i * M + j]!) / eps
        lseBuffer[j] = val
        if (val > maxVal) maxVal = val
      }

      // LSE = max + log(Σ exp(val - max))
      let sumExp = 0
      for (let j = 0; j < M; j++) {
        sumExp += Math.exp(lseBuffer[j]! - maxVal)
      }
      const lse = maxVal + Math.log(sumExp)

      f[i] = -eps * lse + eps * logA[i]!
    }

    // Update g: g_j = -ε · LSE_i((f_i - C_ij) / ε) + ε·log(b_j)
    for (let j = 0; j < M; j++) {
      let maxVal = -Infinity
      for (let i = 0; i < N; i++) {
        const val = (f[i]! - C[i * M + j]!) / eps
        lseBuffer[i] = val
        if (val > maxVal) maxVal = val
      }

      let sumExp = 0
      for (let i = 0; i < N; i++) {
        sumExp += Math.exp(lseBuffer[i]! - maxVal)
      }
      const lse = maxVal + Math.log(sumExp)

      g[j] = -eps * lse + eps * logB[j]!
    }

    // Check convergence every 5 iterations
    if ((iter + 1) % 5 === 0) {
      // Check marginal error by computing row sums of the implicit plan
      let maxErr = 0
      for (let i = 0; i < N; i++) {
        let rowSum = 0
        for (let j = 0; j < M; j++) {
          rowSum += Math.exp((f[i]! + g[j]! - C[i * M + j]!) / eps)
        }
        const err = Math.abs(rowSum - a[i]!)
        if (err > maxErr) maxErr = err
      }
      if (maxErr < cfg.tolerance) {
        converged = true
        iter++
        break
      }
    }
  }

  // Recover transport plan: T_ij = exp((f_i + g_j - C_ij) / ε)
  const plan = new Float64Array(N * M)
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < M; j++) {
      plan[i * M + j] = Math.exp((f[i]! + g[j]! - C[i * M + j]!) / eps)
    }
  }

  // Transport cost: <T, C>
  const cost = dot(plan, C)

  // Dual potentials are already f and g
  const dualF = new Float64Array(f)
  const dualG = new Float64Array(g)

  return { plan, cost, dualF, dualG, iterations: iter, converged, N, M }
}

/**
 * Convenience: just return the transport cost from log-domain solver.
 */
export function sinkhornLogCost(
  a: Float64Array,
  b: Float64Array,
  C: Float64Array,
  epsilon: number,
): number {
  return sinkhornLog(a, b, C, { epsilon }).cost
}
