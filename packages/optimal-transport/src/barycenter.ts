/**
 * Wasserstein Barycenters (OT-3).
 *
 * Computes the Wasserstein barycenter via Iterative Bregman Projections
 * (Benamou et al. 2015, arXiv:1412.5154).
 *
 * The barycenter of past successful venue bookings creates an "ideal venue"
 * profile. Score new candidates by distance to this centroid.
 */

import type { BarycenterConfig } from './types'
import { DEFAULT_BARYCENTER_CONFIG } from './types'
import { sinkhornDivergenceSymmetric } from './cost-matrix'
import { l1Distance, normalizeDistribution } from './utils'

/**
 * Fixed-support Wasserstein barycenter via Iterative Bregman Projections.
 *
 * Given N input distributions μ_1, ..., μ_N on the same support of size n,
 * finds bar(μ) minimizing Σ_i λ_i · W_ε²(bar, μ_i).
 *
 * @param distributions - N distributions, each of length n (support size)
 * @param costMatrix - n×n cost matrix on the shared support
 * @param weights - N weights (sum to 1), how much each input matters
 * @param config - Barycenter configuration
 */
export function fixedSupportBarycenter(
  distributions: Float64Array[],
  costMatrix: Float64Array,
  weights: Float64Array,
  config: Partial<BarycenterConfig> = {},
): Float64Array {
  const cfg = { ...DEFAULT_BARYCENTER_CONFIG, ...config }
  const N = distributions.length
  if (N === 0) return new Float64Array(0)

  const n = distributions[0]!.length
  const eps = cfg.epsilon

  // Build Gibbs kernel K = exp(-C/ε)
  const K = new Float64Array(n * n)
  for (let k = 0; k < n * n; k++) {
    K[k] = Math.exp(-costMatrix[k]! / eps)
  }

  // Initialize barycenter as uniform
  let bary = new Float64Array(n).fill(1.0 / n)

  // Scaling vectors: v_i for each input distribution (length n each)
  const v: Float64Array[] = []
  for (let i = 0; i < N; i++) {
    v.push(new Float64Array(n).fill(1.0))
  }

  // Temp buffers
  const Kv = new Float64Array(n)
  const Ktu = new Float64Array(n)

  for (let iter = 0; iter < cfg.maxIterations; iter++) {
    const prevBary = new Float64Array(bary)

    // Accumulate log-barycenter via weighted geometric mean
    const logBary = new Float64Array(n).fill(0)

    for (let i = 0; i < N; i++) {
      const vi = v[i]!
      const mu_i = distributions[i]!

      // Kv_i: K · v_i
      for (let p = 0; p < n; p++) {
        let s = 0
        for (let q = 0; q < n; q++) {
          s += K[p * n + q]! * vi[q]!
        }
        Kv[p] = Math.max(s, 1e-300)
      }

      // u_i = bary / Kv_i
      // Then Kᵀu_i
      Ktu.fill(0)
      for (let p = 0; p < n; p++) {
        const u_p = bary[p]! / Kv[p]!
        for (let q = 0; q < n; q++) {
          Ktu[q]! += K[p * n + q]! * u_p
        }
      }

      // v_i = mu_i / Kᵀu_i
      for (let q = 0; q < n; q++) {
        vi[q] = mu_i[q]! / Math.max(Ktu[q]!, 1e-300)
      }

      // Accumulate: logBary += w_i * log(K · v_i)
      // Recompute K · v_i with updated v_i
      for (let p = 0; p < n; p++) {
        let s = 0
        for (let q = 0; q < n; q++) {
          s += K[p * n + q]! * vi[q]!
        }
        logBary[p]! += weights[i]! * Math.log(Math.max(s, 1e-300))
      }
    }

    // bary = exp(logBary), then normalize
    for (let p = 0; p < n; p++) {
      bary[p] = Math.exp(logBary[p]!)
    }
    normalizeDistribution(bary)

    // Check convergence
    if (l1Distance(bary, prevBary) < cfg.tolerance) {
      break
    }
  }

  return bary
}

/**
 * Score a candidate venue against the ideal barycenter.
 * Lower score = better match to historical successful bookings.
 *
 * Uses Sinkhorn divergence for a debiased, positive-definite metric.
 */
export function scoreAgainstBarycenter(
  idealBarycenter: Float64Array,
  candidateDistribution: Float64Array,
  costMatrix: Float64Array,
  epsilon: number = 0.01,
): number {
  return sinkhornDivergenceSymmetric(
    idealBarycenter,
    candidateDistribution,
    costMatrix,
    epsilon,
  )
}

/**
 * Convert venue features to a histogram distribution over discretized bins.
 * Used to represent venues as distributions for barycenter computation.
 *
 * @param features - Numeric feature values (e.g., capacity, price, area)
 * @param binEdges - Bin edges for discretization
 * @returns Normalized histogram distribution
 */
export function featuresToDistribution(
  features: number[],
  binEdges: number[],
): Float64Array {
  const nBins = binEdges.length - 1
  if (nBins <= 0) return new Float64Array(0)

  const hist = new Float64Array(nBins)

  for (const val of features) {
    // Find bin
    let bin = nBins - 1
    for (let b = 0; b < nBins; b++) {
      if (val < binEdges[b + 1]!) {
        bin = b
        break
      }
    }
    hist[bin]! += 1
  }

  // Normalize to distribution
  normalizeDistribution(hist)

  // Ensure no zeros (add small epsilon for numerical stability)
  for (let i = 0; i < nBins; i++) {
    hist[i] = Math.max(hist[i]!, 1e-10)
  }
  normalizeDistribution(hist)

  return hist
}
