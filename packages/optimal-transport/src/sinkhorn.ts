/**
 * Standard Sinkhorn algorithm (OT-1).
 *
 * T* = diag(u) · K · diag(v)
 * where K = exp(-C/ε), iterating u = a/(Kv), v = b/(Kᵀu)
 *
 * Stable for ε > 0.01·median(C). For smaller ε, use sinkhornLog.
 */

import type { SinkhornConfig, TransportResult } from './types'
import { DEFAULT_SINKHORN_CONFIG } from './types'
import { computeRowSums, computeColSums, l1Distance, dot } from './utils'

const EPSILON_FLOOR = 1e-30

/**
 * Standard Sinkhorn in the multiplicative (scaling) domain.
 */
export function sinkhorn(
  a: Float64Array,
  b: Float64Array,
  C: Float64Array,
  config: Partial<SinkhornConfig> = {},
): TransportResult {
  const cfg = { ...DEFAULT_SINKHORN_CONFIG, ...config }
  const N = a.length
  const M = b.length

  // Build Gibbs kernel K_ij = exp(-C_ij / ε)
  const K = new Float64Array(N * M)
  const invEps = 1 / cfg.epsilon
  for (let k = 0; k < N * M; k++) {
    K[k] = Math.exp(-C[k]! * invEps)
  }

  // Scaling vectors
  const u = new Float64Array(N).fill(1)
  const v = new Float64Array(M).fill(1)

  // Temp buffers
  const Kv = new Float64Array(N)
  const Ktu = new Float64Array(M)

  let converged = false
  let iter = 0

  for (iter = 0; iter < cfg.maxIterations; iter++) {
    // Kv = K · v
    for (let i = 0; i < N; i++) {
      let s = 0
      for (let j = 0; j < M; j++) {
        s += K[i * M + j]! * v[j]!
      }
      Kv[i] = s
    }

    // u = a / Kv
    for (let i = 0; i < N; i++) {
      u[i] = a[i]! / Math.max(Kv[i]!, EPSILON_FLOOR)
    }

    // Kᵀu = Kᵀ · u
    Ktu.fill(0)
    for (let i = 0; i < N; i++) {
      const ui = u[i]!
      for (let j = 0; j < M; j++) {
        Ktu[j]! += K[i * M + j]! * ui
      }
    }

    // v = b / Kᵀu
    for (let j = 0; j < M; j++) {
      v[j] = b[j]! / Math.max(Ktu[j]!, EPSILON_FLOOR)
    }

    // Check convergence: marginal error
    if ((iter + 1) % 5 === 0) {
      // Recompute Kv for marginal check
      for (let i = 0; i < N; i++) {
        let s = 0
        for (let j = 0; j < M; j++) {
          s += K[i * M + j]! * v[j]!
        }
        Kv[i] = s
      }
      // Row marginal: u * Kv should ≈ a
      let maxErr = 0
      for (let i = 0; i < N; i++) {
        const err = Math.abs(u[i]! * Kv[i]! - a[i]!)
        if (err > maxErr) maxErr = err
      }
      if (maxErr < cfg.tolerance) {
        converged = true
        iter++
        break
      }
    }
  }

  // Recover transport plan: T_ij = u_i · K_ij · v_j
  const plan = new Float64Array(N * M)
  for (let i = 0; i < N; i++) {
    const ui = u[i]!
    for (let j = 0; j < M; j++) {
      plan[i * M + j] = ui * K[i * M + j]! * v[j]!
    }
  }

  // Compute dual potentials: f = ε·log(u), g = ε·log(v)
  const dualF = new Float64Array(N)
  const dualG = new Float64Array(M)
  for (let i = 0; i < N; i++) {
    dualF[i] = cfg.epsilon * Math.log(Math.max(u[i]!, EPSILON_FLOOR))
  }
  for (let j = 0; j < M; j++) {
    dualG[j] = cfg.epsilon * Math.log(Math.max(v[j]!, EPSILON_FLOOR))
  }

  // Transport cost: <T, C>
  const cost = dot(plan, C)

  return { plan, cost, dualF, dualG, iterations: iter, converged, N, M }
}

/**
 * Convenience: just return the transport cost (used by sinkhornDivergence).
 */
export function sinkhornCost(
  a: Float64Array,
  b: Float64Array,
  C: Float64Array,
  epsilon: number,
): number {
  return sinkhorn(a, b, C, { epsilon }).cost
}
