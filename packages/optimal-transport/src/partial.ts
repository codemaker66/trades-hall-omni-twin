/**
 * Partial & Unbalanced Optimal Transport (OT-4).
 *
 * Standard OT forces all supply to meet all demand. These variants relax that:
 * - Partial OT (Chapel et al. 2020): only transport a fraction m of total mass
 * - Unbalanced OT (Chizat et al. 2016): soft KL-divergence marginal penalties
 */

import type { SinkhornConfig, TransportResult } from './types'
import { DEFAULT_SINKHORN_CONFIG } from './types'
import { dot } from './utils'

const EPSILON_FLOOR = 1e-30

/**
 * Partial Optimal Transport via augmented Sinkhorn.
 *
 * min_T <T, C>  s.t.  T≥0, T·1 ≤ a, Tᵀ·1 ≤ b, Σ_ij T_ij = m
 *
 * Only transport mass m ≤ min(||a||₁, ||b||₁).
 * Implemented by augmenting with slack variables (a dummy bin that absorbs
 * untransported mass at cost = maxCost * penaltyFactor).
 *
 * @param a - Source distribution (N), sums to 1
 * @param b - Target distribution (M), sums to 1
 * @param C - Cost matrix (N×M, row-major)
 * @param mass - Total mass to transport (0 < m ≤ 1)
 * @param epsilon - Regularization parameter
 * @param config - Additional Sinkhorn config
 */
export function partialSinkhorn(
  a: Float64Array,
  b: Float64Array,
  C: Float64Array,
  mass: number,
  epsilon: number,
  config: Partial<SinkhornConfig> = {},
): TransportResult {
  const cfg = { ...DEFAULT_SINKHORN_CONFIG, epsilon, ...config }
  const N = a.length
  const M = b.length

  // Clamp mass
  const totalA = a.reduce((s, v) => s + v, 0)
  const totalB = b.reduce((s, v) => s + v, 0)
  const m = Math.min(mass, totalA, totalB)

  // Find max cost for slack penalty
  let maxC = 0
  for (let k = 0; k < N * M; k++) {
    if (C[k]! > maxC) maxC = C[k]!
  }
  const slackCost = maxC * 2  // Heavy penalty to discourage using slack unnecessarily

  // Augment: add a dummy row (N+1) and dummy column (M+1)
  // The augmented distributions absorb the untransported mass
  const Na = N + 1
  const Ma = M + 1

  const aAug = new Float64Array(Na)
  const bAug = new Float64Array(Ma)
  for (let i = 0; i < N; i++) aAug[i] = a[i]!
  aAug[N] = Math.max(totalB - m, 0)  // slack absorbs from target
  for (let j = 0; j < M; j++) bAug[j] = b[j]!
  bAug[M] = Math.max(totalA - m, 0)  // slack absorbs from source

  // Augmented cost matrix (Na × Ma)
  const Ca = new Float64Array(Na * Ma)
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < M; j++) {
      Ca[i * Ma + j] = C[i * M + j]!
    }
    Ca[i * Ma + M] = slackCost  // source i → slack
  }
  for (let j = 0; j < M; j++) {
    Ca[N * Ma + j] = slackCost  // slack → target j
  }
  Ca[N * Ma + M] = 0  // slack → slack (free)

  // Run standard Sinkhorn on augmented problem
  const invEps = 1 / cfg.epsilon
  const K = new Float64Array(Na * Ma)
  for (let k = 0; k < Na * Ma; k++) {
    K[k] = Math.exp(-Ca[k]! * invEps)
  }

  const u = new Float64Array(Na).fill(1)
  const v = new Float64Array(Ma).fill(1)
  const Kv = new Float64Array(Na)
  const Ktu = new Float64Array(Ma)

  let converged = false
  let iter = 0

  for (iter = 0; iter < cfg.maxIterations; iter++) {
    // Kv = K · v
    for (let i = 0; i < Na; i++) {
      let s = 0
      for (let j = 0; j < Ma; j++) {
        s += K[i * Ma + j]! * v[j]!
      }
      Kv[i] = s
    }
    for (let i = 0; i < Na; i++) {
      u[i] = aAug[i]! / Math.max(Kv[i]!, EPSILON_FLOOR)
    }

    // Kᵀu = Kᵀ · u
    Ktu.fill(0)
    for (let i = 0; i < Na; i++) {
      const ui = u[i]!
      for (let j = 0; j < Ma; j++) {
        Ktu[j]! += K[i * Ma + j]! * ui
      }
    }
    for (let j = 0; j < Ma; j++) {
      v[j] = bAug[j]! / Math.max(Ktu[j]!, EPSILON_FLOOR)
    }

    // Convergence check
    if ((iter + 1) % 5 === 0) {
      for (let i = 0; i < Na; i++) {
        let s = 0
        for (let j = 0; j < Ma; j++) {
          s += K[i * Ma + j]! * v[j]!
        }
        Kv[i] = s
      }
      let maxErr = 0
      for (let i = 0; i < Na; i++) {
        const err = Math.abs(u[i]! * Kv[i]! - aAug[i]!)
        if (err > maxErr) maxErr = err
      }
      if (maxErr < cfg.tolerance) {
        converged = true
        iter++
        break
      }
    }
  }

  // Extract the N×M sub-plan (exclude slack row/column)
  const plan = new Float64Array(N * M)
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < M; j++) {
      plan[i * M + j] = u[i]! * K[i * Ma + j]! * v[j]!
    }
  }

  // Dual potentials for the original dimensions
  const dualF = new Float64Array(N)
  const dualG = new Float64Array(M)
  for (let i = 0; i < N; i++) {
    dualF[i] = cfg.epsilon * Math.log(Math.max(u[i]!, EPSILON_FLOOR))
  }
  for (let j = 0; j < M; j++) {
    dualG[j] = cfg.epsilon * Math.log(Math.max(v[j]!, EPSILON_FLOOR))
  }

  const cost = dot(plan, C)

  return { plan, cost, dualF, dualG, iterations: iter, converged, N, M }
}

/**
 * Unbalanced Optimal Transport via modified Sinkhorn.
 *
 * min_T <T,C> + ε·Ω(T) + ρ₁·KL(T·1 || a) + ρ₂·KL(Tᵀ·1 || b)
 *
 * ρ (regMarginal) controls marginal relaxation:
 *   ρ → 0:   nearly ignores marginals (full mass destruction)
 *   ρ → ∞:   enforces marginals exactly (recovers balanced OT)
 *   ρ = 0.1: good default for venue matching
 *
 * Sinkhorn iterations with modified updates:
 *   u_i = (a_i / (K·v)_i)^(ρ/(ρ+ε))
 *   v_j = (b_j / (Kᵀ·u)_j)^(ρ/(ρ+ε))
 *
 * @param a - Source distribution (N)
 * @param b - Target distribution (M)
 * @param C - Cost matrix (N×M, row-major)
 * @param epsilon - Entropic regularization
 * @param regMarginal - ρ, marginal relaxation strength
 * @param config - Additional Sinkhorn config
 */
export function unbalancedSinkhorn(
  a: Float64Array,
  b: Float64Array,
  C: Float64Array,
  epsilon: number,
  regMarginal: number,
  config: Partial<SinkhornConfig> = {},
): TransportResult {
  const cfg = { ...DEFAULT_SINKHORN_CONFIG, epsilon, ...config }
  const N = a.length
  const M = b.length

  // Gibbs kernel
  const invEps = 1 / cfg.epsilon
  const K = new Float64Array(N * M)
  for (let k = 0; k < N * M; k++) {
    K[k] = Math.exp(-C[k]! * invEps)
  }

  // Scaling exponent: ρ / (ρ + ε)
  const tau = regMarginal / (regMarginal + cfg.epsilon)

  const u = new Float64Array(N).fill(1)
  const v = new Float64Array(M).fill(1)
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

    // u_i = (a_i / Kv_i)^τ
    for (let i = 0; i < N; i++) {
      u[i] = Math.pow(a[i]! / Math.max(Kv[i]!, EPSILON_FLOOR), tau)
    }

    // Kᵀu = Kᵀ · u
    Ktu.fill(0)
    for (let i = 0; i < N; i++) {
      const ui = u[i]!
      for (let j = 0; j < M; j++) {
        Ktu[j]! += K[i * M + j]! * ui
      }
    }

    // v_j = (b_j / Kᵀu_j)^τ
    for (let j = 0; j < M; j++) {
      v[j] = Math.pow(b[j]! / Math.max(Ktu[j]!, EPSILON_FLOOR), tau)
    }

    // Convergence check
    if ((iter + 1) % 5 === 0) {
      // Compute Kv again
      for (let i = 0; i < N; i++) {
        let s = 0
        for (let j = 0; j < M; j++) {
          s += K[i * M + j]! * v[j]!
        }
        Kv[i] = s
      }
      // Check if u^(1/τ) * Kv ≈ a
      let maxErr = 0
      for (let i = 0; i < N; i++) {
        const rowMarg = Math.pow(u[i]!, 1 / tau) * Kv[i]!
        const err = Math.abs(rowMarg - a[i]!)
        if (err > maxErr) maxErr = err
      }
      if (maxErr < cfg.tolerance) {
        converged = true
        iter++
        break
      }
    }
  }

  // Recover plan: T_ij = u_i · K_ij · v_j
  const plan = new Float64Array(N * M)
  for (let i = 0; i < N; i++) {
    const ui = u[i]!
    for (let j = 0; j < M; j++) {
      plan[i * M + j] = ui * K[i * M + j]! * v[j]!
    }
  }

  // Dual potentials
  const dualF = new Float64Array(N)
  const dualG = new Float64Array(M)
  for (let i = 0; i < N; i++) {
    dualF[i] = cfg.epsilon * Math.log(Math.max(u[i]!, EPSILON_FLOOR))
  }
  for (let j = 0; j < M; j++) {
    dualG[j] = cfg.epsilon * Math.log(Math.max(v[j]!, EPSILON_FLOOR))
  }

  const cost = dot(plan, C)

  return { plan, cost, dualF, dualG, iterations: iter, converged, N, M }
}
