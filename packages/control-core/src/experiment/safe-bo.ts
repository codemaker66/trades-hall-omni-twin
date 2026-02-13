// ---------------------------------------------------------------------------
// OC-10: Optimal Experiment Design -- Safe Bayesian Optimization
// ---------------------------------------------------------------------------
//
// SafeOpt-style Bayesian optimization with a Gaussian process surrogate.
//
// Key ideas:
// - Maintain a GP posterior over the objective function.
// - A point x is *safe* if its lower confidence bound (LCB) is at or above
//   the safety threshold:  mu(x) - beta * sigma(x) >= threshold.
// - Among safe candidate points, select the one with the highest expected
//   improvement (EI) over the current best safe observation.
//
// The GP uses a squared-exponential (SE) kernel:
//   k(x, x') = variance * exp( -||x - x'||^2 / (2 * lengthscale^2) )
//
// GP prediction is done via direct kernel matrix inversion (suitable for
// moderate numbers of observations).
// ---------------------------------------------------------------------------

import type { SafeBOConfig, SafeBOResult } from '../types.js';

// ---------------------------------------------------------------------------
// Gaussian CDF / PDF approximations
// ---------------------------------------------------------------------------

/**
 * Approximate the error function erf(x) using Abramowitz & Stegun 7.1.26.
 *
 * Maximum error: ~ 1.5e-7.
 */
function erf(x: number): number {
  const sign = x >= 0 ? 1 : -1;
  const ax = Math.abs(x);

  // Constants from A&S 7.1.26
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const t = 1.0 / (1.0 + p * ax);
  const t2 = t * t;
  const t3 = t2 * t;
  const t4 = t3 * t;
  const t5 = t4 * t;

  const y = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * Math.exp(-ax * ax);
  return sign * y;
}

/** Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2))). */
function normalCDF(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

/** Standard normal PDF: phi(x) = (1 / sqrt(2*pi)) * exp(-0.5 * x^2). */
function normalPDF(x: number): number {
  return (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * x * x);
}

// ---------------------------------------------------------------------------
// Squared-exponential kernel
// ---------------------------------------------------------------------------

/**
 * Compute the SE kernel between two points.
 *
 * k(x, x') = variance * exp( -||x - x'||^2 / (2 * lengthscale^2) )
 */
function seKernel(
  x1: Float64Array,
  x2: Float64Array,
  lengthscale: number,
  variance: number,
): number {
  let sqDist = 0;
  const d = x1.length;
  for (let i = 0; i < d; i++) {
    const diff = x1[i]! - x2[i]!;
    sqDist += diff * diff;
  }
  return variance * Math.exp(-sqDist / (2 * lengthscale * lengthscale));
}

// ---------------------------------------------------------------------------
// GP prediction
// ---------------------------------------------------------------------------

/**
 * Compute GP posterior mean and variance at a single test point.
 *
 * mu(x*) = K_* (K + sigma_n^2 I)^{-1} y
 * var(x*) = k(x*,x*) - K_* (K + sigma_n^2 I)^{-1} K_*^T
 *
 * @param xStar         Test point (d)
 * @param obsX          Observation inputs (n x d, stored as array of Float64Array)
 * @param obsY          Observation outputs (n)
 * @param Kinv          Precomputed (K + sigma_n^2 I)^{-1}  (n x n, row-major)
 * @param alpha         Precomputed (K + sigma_n^2 I)^{-1} y  (n)
 * @param lengthscale   Kernel lengthscale
 * @param kVariance     Kernel signal variance
 * @returns { mu, sigma2 }  posterior mean and variance
 */
function gpPredict(
  xStar: Float64Array,
  obsX: Float64Array[],
  alpha: Float64Array,
  Kinv: Float64Array,
  lengthscale: number,
  kVariance: number,
): { mu: number; sigma2: number } {
  const n = obsX.length;

  // K_* : kernel between x* and each observation
  const kStar = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    kStar[i] = seKernel(xStar, obsX[i]!, lengthscale, kVariance);
  }

  // mu = K_* alpha
  let mu = 0;
  for (let i = 0; i < n; i++) {
    mu += kStar[i]! * alpha[i]!;
  }

  // var = k(x*,x*) - K_* Kinv K_*^T
  const kss = seKernel(xStar, xStar, lengthscale, kVariance);
  let vReduction = 0;
  for (let i = 0; i < n; i++) {
    let kinvKstar_i = 0;
    for (let j = 0; j < n; j++) {
      kinvKstar_i += Kinv[i * n + j]! * kStar[j]!;
    }
    vReduction += kStar[i]! * kinvKstar_i;
  }
  const sigma2 = Math.max(0, kss - vReduction);

  return { mu, sigma2 };
}

// ---------------------------------------------------------------------------
// Matrix inversion for the GP kernel matrix (small, dense)
// ---------------------------------------------------------------------------

function invertSymmetric(data: Float64Array, n: number): Float64Array {
  // Gauss-Jordan with partial pivoting
  const aug = new Float64Array(n * 2 * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      aug[i * 2 * n + j] = data[i * n + j]!;
    }
    aug[i * 2 * n + n + i] = 1;
  }

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    let maxVal = Math.abs(aug[col * 2 * n + col]!);
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(aug[row * 2 * n + col]!);
      if (val > maxVal) {
        maxVal = val;
        maxRow = row;
      }
    }
    if (maxRow !== col) {
      for (let j = 0; j < 2 * n; j++) {
        const tmp = aug[col * 2 * n + j]!;
        aug[col * 2 * n + j] = aug[maxRow * 2 * n + j]!;
        aug[maxRow * 2 * n + j] = tmp;
      }
    }

    const pivot = aug[col * 2 * n + col]!;
    if (Math.abs(pivot) < 1e-15) {
      // Regularise and retry would be ideal; for now, return identity
      const fallback = new Float64Array(n * n);
      for (let i = 0; i < n; i++) {
        fallback[i * n + i] = 1;
      }
      return fallback;
    }

    for (let j = 0; j < 2 * n; j++) {
      aug[col * 2 * n + j] = aug[col * 2 * n + j]! / pivot;
    }

    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row * 2 * n + col]!;
      for (let j = 0; j < 2 * n; j++) {
        aug[row * 2 * n + j] = aug[row * 2 * n + j]! - factor * aug[col * 2 * n + j]!;
      }
    }
  }

  const inv = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      inv[i * n + j] = aug[i * 2 * n + n + j]!;
    }
  }
  return inv;
}

// ---------------------------------------------------------------------------
// Safe Expected Improvement
// ---------------------------------------------------------------------------

/**
 * Expected improvement for a point with Gaussian predictive distribution,
 * relative to the current best safe observation.
 *
 *   EI = (mu - best) * Phi(z) + sigma * phi(z)
 *
 * where  z = (mu - best) / sigma.
 *
 * @param mean     GP posterior mean at the candidate
 * @param std      GP posterior standard deviation at the candidate
 * @param bestSafe Current best objective value among safe points
 * @returns Expected improvement (non-negative)
 */
export function safeExpectedImprovement(
  mean: number,
  std: number,
  bestSafe: number,
): number {
  if (std < 1e-15) {
    // Deterministic: EI = max(0, mean - bestSafe)
    return Math.max(0, mean - bestSafe);
  }
  const z = (mean - bestSafe) / std;
  const ei = (mean - bestSafe) * normalCDF(z) + std * normalPDF(z);
  return Math.max(0, ei);
}

// ---------------------------------------------------------------------------
// Safe Bayesian Optimization: next point selection
// ---------------------------------------------------------------------------

/**
 * Select the next evaluation point using SafeOpt-style Bayesian optimisation.
 *
 * 1. Build a GP posterior from the observations.
 * 2. Generate a grid of candidate points within the input bounds.
 * 3. For each candidate, compute the GP posterior mean and variance.
 * 4. Classify candidates as safe/unsafe using the LCB criterion:
 *      safe iff mu(x) - beta * sigma(x) >= safetyThreshold
 * 5. Among safe candidates, find the one with the highest expected
 *    improvement over the best safe observation so far.
 *
 * If no observations are provided (cold start), the center of the bounds
 * is returned with zero EI and safety probability 0.5.
 *
 * @param config        Safe BO configuration
 * @param observations  Array of past evaluations { x, y, safe }
 * @returns SafeBOResult with next evaluation point and diagnostics
 */
export function safeBONext(
  config: SafeBOConfig,
  observations: { x: Float64Array; y: number; safe: boolean }[],
): SafeBOResult {
  const { bounds, safetyThreshold, kernelLengthscale, kernelVariance, noiseVariance, beta } = config;
  const dim = bounds.length;

  // -------------------------------------------------------------------------
  // Cold start: no observations => return centre of bounds
  // -------------------------------------------------------------------------
  if (observations.length === 0) {
    const center = new Float64Array(dim);
    for (let i = 0; i < dim; i++) {
      center[i] = (bounds[i]!.min + bounds[i]!.max) / 2;
    }
    return {
      nextPoint: center,
      expectedImprovement: 0,
      safetyProbability: 0.5,
      isSafe: false,
    };
  }

  const n = observations.length;

  // -------------------------------------------------------------------------
  // Build kernel matrix K + sigma_n^2 I  and invert
  // -------------------------------------------------------------------------
  const obsX = observations.map((o) => o.x);
  const obsY = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    obsY[i] = observations[i]!.y;
  }

  const Kmat = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      Kmat[i * n + j] = seKernel(obsX[i]!, obsX[j]!, kernelLengthscale, kernelVariance);
    }
    // Add noise to diagonal
    Kmat[i * n + i] = Kmat[i * n + i]! + noiseVariance;
  }

  const Kinv = invertSymmetric(Kmat, n);

  // alpha = Kinv * y
  const alpha = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let val = 0;
    for (let j = 0; j < n; j++) {
      val += Kinv[i * n + j]! * obsY[j]!;
    }
    alpha[i] = val;
  }

  // -------------------------------------------------------------------------
  // Best safe observation so far
  // -------------------------------------------------------------------------
  let bestSafe = -Infinity;
  for (let i = 0; i < n; i++) {
    if (observations[i]!.safe && observations[i]!.y > bestSafe) {
      bestSafe = observations[i]!.y;
    }
  }
  if (bestSafe === -Infinity) {
    // No safe observations yet — use threshold as baseline
    bestSafe = safetyThreshold;
  }

  // -------------------------------------------------------------------------
  // Generate candidate grid
  // -------------------------------------------------------------------------
  // Use a regular grid with ~20 points per dimension (capped at 10k total)
  const nPerDim = Math.max(2, Math.min(20, Math.floor(Math.pow(10000, 1 / dim))));
  const candidates = generateGrid(bounds, nPerDim);

  // -------------------------------------------------------------------------
  // Evaluate GP at each candidate and select best safe point
  // -------------------------------------------------------------------------
  let bestEI = -Infinity;
  let bestPoint = new Float64Array(dim);
  let bestSafetyProb = 0;
  let bestIsSafe = false;

  for (let c = 0; c < candidates.length; c++) {
    const xc = candidates[c]!;
    const { mu, sigma2 } = gpPredict(
      xc,
      obsX,
      alpha,
      Kinv,
      kernelLengthscale,
      kernelVariance,
    );
    const sigma = Math.sqrt(sigma2);

    // Safety check: LCB >= safetyThreshold
    const lcb = mu - beta * sigma;
    const isSafe = lcb >= safetyThreshold;

    // Safety probability: P(f(x) >= threshold) = Phi((mu - threshold) / sigma)
    const safetyProb = sigma > 1e-15
      ? normalCDF((mu - safetyThreshold) / sigma)
      : (mu >= safetyThreshold ? 1.0 : 0.0);

    if (!isSafe) continue;

    // Expected improvement among safe points
    const ei = safeExpectedImprovement(mu, sigma, bestSafe);

    if (ei > bestEI) {
      bestEI = ei;
      bestPoint = new Float64Array(xc) as Float64Array<ArrayBuffer>;
      bestSafetyProb = safetyProb;
      bestIsSafe = true;
    }
  }

  // -------------------------------------------------------------------------
  // Fallback: if no safe candidate was found, return the one with the
  // highest safety probability (most likely to be safe).
  // -------------------------------------------------------------------------
  if (!bestIsSafe) {
    let highestSafeProb = -1;
    for (let c = 0; c < candidates.length; c++) {
      const xc = candidates[c]!;
      const { mu, sigma2 } = gpPredict(
        xc,
        obsX,
        alpha,
        Kinv,
        kernelLengthscale,
        kernelVariance,
      );
      const sigma = Math.sqrt(sigma2);
      const safetyProb = sigma > 1e-15
        ? normalCDF((mu - safetyThreshold) / sigma)
        : (mu >= safetyThreshold ? 1.0 : 0.0);

      if (safetyProb > highestSafeProb) {
        highestSafeProb = safetyProb;
        bestPoint = new Float64Array(xc) as Float64Array<ArrayBuffer>;
        bestSafetyProb = safetyProb;
      }
    }
  }

  return {
    nextPoint: bestPoint,
    expectedImprovement: Math.max(0, bestEI),
    safetyProbability: bestSafetyProb,
    isSafe: bestIsSafe,
  };
}

// ---------------------------------------------------------------------------
// Internal: grid generation
// ---------------------------------------------------------------------------

/**
 * Generate a regular grid of candidate points within the given bounds.
 *
 * For d dimensions with nPerDim points each, generates nPerDim^d candidates.
 * Capped at 10,000 points to keep computation tractable.
 */
function generateGrid(
  bounds: Array<{ min: number; max: number }>,
  nPerDim: number,
): Float64Array[] {
  const dim = bounds.length;
  const total = Math.min(Math.pow(nPerDim, dim), 10000);

  // Pre-compute coordinate values for each dimension
  const coords: number[][] = [];
  for (let d = 0; d < dim; d++) {
    const lo = bounds[d]!.min;
    const hi = bounds[d]!.max;
    const vals: number[] = [];
    for (let i = 0; i < nPerDim; i++) {
      vals.push(nPerDim > 1 ? lo + (hi - lo) * i / (nPerDim - 1) : (lo + hi) / 2);
    }
    coords.push(vals);
  }

  // Generate all combinations via mixed-radix enumeration
  const candidates: Float64Array[] = [];
  for (let idx = 0; idx < total; idx++) {
    const point = new Float64Array(dim);
    let remainder = idx;
    for (let d = dim - 1; d >= 0; d--) {
      const coordIdx = remainder % nPerDim;
      point[d] = coords[d]![coordIdx]!;
      remainder = Math.floor(remainder / nPerDim);
    }
    candidates.push(point);
  }

  return candidates;
}
