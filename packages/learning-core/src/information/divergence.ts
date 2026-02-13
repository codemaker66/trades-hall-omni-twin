// ---------------------------------------------------------------------------
// KL Divergence, Jensen-Shannon Divergence, and Related Measures
// ---------------------------------------------------------------------------

/**
 * KL Divergence: KL(P || Q) = Σ p_i * log(p_i / q_i)
 *
 * Both p and q must be valid probability distributions (sum to 1, non-negative).
 * Uses additive smoothing (epsilon) to avoid log(0) and division by zero.
 *
 * @param p - Distribution P (reference/true distribution)
 * @param q - Distribution Q (approximate distribution)
 * @returns KL divergence in nats (always >= 0)
 */
export function klDivergence(p: number[], q: number[]): number {
  const n = Math.min(p.length, q.length);
  if (n === 0) return 0;

  const eps = 1e-12;
  let kl = 0;
  for (let i = 0; i < n; i++) {
    const pi = (p[i] ?? 0) + eps;
    const qi = (q[i] ?? 0) + eps;
    if (pi > eps) {
      kl += pi * Math.log(pi / qi);
    }
  }
  return Math.max(0, kl);
}

/**
 * Reverse KL Divergence: KL(Q || P) = Σ q_i * log(q_i / p_i)
 *
 * @param p - Distribution P
 * @param q - Distribution Q
 * @returns Reverse KL divergence in nats (always >= 0)
 */
export function reverseKL(p: number[], q: number[]): number {
  return klDivergence(q, p);
}

/**
 * Jensen-Shannon Divergence: JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
 * where M = 0.5 * (P + Q)
 *
 * JSD is symmetric and always finite (bounded by log(2) for distributions).
 *
 * @param p - Distribution P
 * @param q - Distribution Q
 * @returns JSD in nats (always in [0, log(2)])
 */
export function jsDivergence(p: number[], q: number[]): number {
  const n = Math.min(p.length, q.length);
  if (n === 0) return 0;

  // Compute M = 0.5*(P + Q)
  const m: number[] = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    m[i] = 0.5 * ((p[i] ?? 0) + (q[i] ?? 0));
  }

  return 0.5 * klDivergence(p, m) + 0.5 * klDivergence(q, m);
}

/**
 * JSD metric: √JSD(P || Q)
 *
 * The square root of the Jensen-Shannon Divergence is a true metric
 * (satisfies triangle inequality, symmetry, identity of indiscernibles).
 *
 * @param p - Distribution P
 * @param q - Distribution Q
 * @returns √JSD — a value in [0, √log(2)]
 */
export function jsdMetric(p: number[], q: number[]): number {
  return Math.sqrt(jsDivergence(p, q));
}

/**
 * Create a normalized probability distribution from continuous samples
 * using equal-width histogram binning.
 *
 * @param samples - Array of continuous sample values
 * @param nBins - Number of histogram bins
 * @returns Normalized probability distribution (sums to 1)
 */
export function histogramFromSamples(samples: number[], nBins: number): number[] {
  const n = samples.length;
  if (n === 0 || nBins <= 0) {
    return [];
  }

  // Find range
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = samples[i] ?? 0;
    if (v < min) min = v;
    if (v > max) max = v;
  }

  // Initialize bins
  const counts = new Array<number>(nBins);
  for (let i = 0; i < nBins; i++) {
    counts[i] = 0;
  }

  // Fill bins
  const range = max - min;
  if (range <= 0) {
    // All values are the same — put everything in the first bin
    counts[0] = n;
  } else {
    for (let i = 0; i < n; i++) {
      const v = samples[i] ?? 0;
      let bin = Math.floor(((v - min) / range) * nBins);
      // Clamp to [0, nBins-1]
      bin = Math.max(0, Math.min(nBins - 1, bin));
      counts[bin]! += 1;
    }
  }

  // Normalize to probability distribution
  const dist = new Array<number>(nBins);
  for (let i = 0; i < nBins; i++) {
    dist[i] = (counts[i] ?? 0) / n;
  }
  return dist;
}
