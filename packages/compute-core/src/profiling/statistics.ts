// ---------------------------------------------------------------------------
// HPC-10: Profiling — Statistics computation
// ---------------------------------------------------------------------------
// Pure functions for computing descriptive statistics from profiling samples.
// All heavy-lifting uses Float64Array for cache-friendly, GC-free numerics.
// Includes Welford's online algorithm for streaming mean/variance.
// ---------------------------------------------------------------------------

import type { ProfileSample, ProfileStatistics } from '../types.js';

// ---------------------------------------------------------------------------
// computeMean
// ---------------------------------------------------------------------------

/**
 * Arithmetic mean of a Float64Array. Returns 0 for empty arrays.
 */
export function computeMean(values: Float64Array): number {
  const n = values.length;
  if (n === 0) return 0;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += values[i]!;
  }
  return sum / n;
}

// ---------------------------------------------------------------------------
// computeMedian
// ---------------------------------------------------------------------------

/**
 * Median of a Float64Array. Non-destructive (sorts a copy).
 * Returns 0 for empty arrays.
 */
export function computeMedian(values: Float64Array): number {
  const n = values.length;
  if (n === 0) return 0;
  const sorted = new Float64Array(values).sort();
  const mid = n >>> 1;
  if (n % 2 === 0) {
    return (sorted[mid - 1]! + sorted[mid]!) / 2;
  }
  return sorted[mid]!;
}

// ---------------------------------------------------------------------------
// computePercentile
// ---------------------------------------------------------------------------

/**
 * Compute the p-th percentile using the nearest-rank method.
 *
 * @param values - source array (not modified)
 * @param p      - percentile in [0, 1], e.g. 0.95 for p95
 *
 * Returns 0 for empty arrays.
 */
export function computePercentile(values: Float64Array, p: number): number {
  const n = values.length;
  if (n === 0) return 0;
  const sorted = new Float64Array(values).sort();
  // Nearest-rank: index = ceil(p * n) - 1, clamped to [0, n-1]
  const rank = Math.ceil(p * n);
  const idx = Math.max(0, Math.min(rank - 1, n - 1));
  return sorted[idx]!;
}

// ---------------------------------------------------------------------------
// computeStdDev
// ---------------------------------------------------------------------------

/**
 * Population standard deviation of a Float64Array.
 * Returns 0 for arrays with fewer than 2 elements.
 */
export function computeStdDev(values: Float64Array): number {
  const n = values.length;
  if (n < 2) return 0;
  const mean = computeMean(values);
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const d = values[i]! - mean;
    sumSq += d * d;
  }
  return Math.sqrt(sumSq / n);
}

// ---------------------------------------------------------------------------
// isOutlier
// ---------------------------------------------------------------------------

/**
 * Check whether a value is an outlier using the z-score method.
 *
 * @param threshold - number of standard deviations (default 3, i.e. 3-sigma)
 */
export function isOutlier(
  value: number,
  mean: number,
  stdDev: number,
  threshold: number = 3,
): boolean {
  if (stdDev === 0) return value !== mean;
  return Math.abs(value - mean) > threshold * stdDev;
}

// ---------------------------------------------------------------------------
// runningStatistics (Welford's online algorithm)
// ---------------------------------------------------------------------------

/**
 * Incrementally update running mean and M2 (sum of squares of differences
 * from the current mean) using Welford's online algorithm.
 *
 * @param prev  - previous accumulator `{ count, mean, m2 }`
 * @param value - new observation
 * @returns     - updated accumulator
 */
export function runningStatistics(
  prev: { count: number; mean: number; m2: number },
  value: number,
): { count: number; mean: number; m2: number } {
  const count = prev.count + 1;
  const delta = value - prev.mean;
  const mean = prev.mean + delta / count;
  const delta2 = value - mean;
  const m2 = prev.m2 + delta * delta2;
  return { count, mean, m2 };
}

// ---------------------------------------------------------------------------
// finalizeRunningStats
// ---------------------------------------------------------------------------

/**
 * Finalise a Welford accumulator into mean, population variance, and
 * population standard deviation.
 *
 * If count < 2, variance and stdDev are 0.
 */
export function finalizeRunningStats(
  stats: { count: number; mean: number; m2: number },
): { mean: number; variance: number; stdDev: number } {
  if (stats.count < 2) {
    return { mean: stats.mean, variance: 0, stdDev: 0 };
  }
  const variance = stats.m2 / stats.count;
  return { mean: stats.mean, variance, stdDev: Math.sqrt(variance) };
}

// ---------------------------------------------------------------------------
// computeProfileStatistics
// ---------------------------------------------------------------------------

/**
 * Compute aggregate profiling statistics from an array of samples.
 *
 * The name is taken from the first sample. If the array is empty, all
 * numeric fields default to 0.
 */
export function computeProfileStatistics(
  samples: ProfileSample[],
): ProfileStatistics {
  const n = samples.length;
  if (n === 0) {
    return {
      name: '',
      count: 0,
      meanMs: 0,
      medianMs: 0,
      p95Ms: 0,
      p99Ms: 0,
      minMs: 0,
      maxMs: 0,
      stdDevMs: 0,
    };
  }

  const durations = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    durations[i] = samples[i]!.durationMs;
  }

  const sorted = new Float64Array(durations).sort();

  return {
    name: samples[0]!.name,
    count: n,
    meanMs: computeMean(durations),
    medianMs: computeMedian(durations),
    p95Ms: computePercentile(durations, 0.95),
    p99Ms: computePercentile(durations, 0.99),
    minMs: sorted[0]!,
    maxMs: sorted[n - 1]!,
    stdDevMs: computeStdDev(durations),
  };
}
