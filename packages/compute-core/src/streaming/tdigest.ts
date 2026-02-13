// ---------------------------------------------------------------------------
// HPC-6: Streaming Algorithms — T-Digest
// ---------------------------------------------------------------------------
// Streaming quantile estimation using the t-digest algorithm (Dunning 2019).
// Maintains a sorted list of centroids whose sizes are limited by a scale
// function that allocates more resolution to the tails (q near 0 or 1).
// Pure TypeScript, zero external dependencies.
// ---------------------------------------------------------------------------

import type { TDigestCentroid, TDigestConfig, TDigestState } from '../types.js';

// ---------------------------------------------------------------------------
// Create
// ---------------------------------------------------------------------------

/**
 * Create an empty t-digest with the given compression parameter.
 *
 * @param config.compression  Delta parameter controlling accuracy vs. memory.
 *                            Higher = more centroids = better accuracy.
 *                            Default 100 (~500 centroids max).
 */
export function createTDigest(config?: TDigestConfig): TDigestState {
  const compression = config?.compression ?? 100;
  return {
    centroids: [],
    totalCount: 0,
    compression,
    min: Infinity,
    max: -Infinity,
  };
}

// ---------------------------------------------------------------------------
// Scale function
// ---------------------------------------------------------------------------

/**
 * Scale function k(q, delta) = (delta / (2 * pi)) * arcsin(2q - 1)
 *
 * This maps quantile q in [0,1] to a "k-space" value. The derivative
 * k'(q) determines the maximum centroid size at quantile q.
 * The function provides higher resolution near q=0 and q=1 (the tails).
 */
function scaleFunction(q: number, delta: number): number {
  return (delta / (2 * Math.PI)) * Math.asin(2 * q - 1);
}

/**
 * Maximum weight a centroid can have at quantile position q,
 * derived from the scale function. A centroid at quantile q can hold
 * at most delta * k'(q) = delta * 1 / (2*pi * sqrt(q*(1-q))) weight,
 * bounded to avoid infinities at q=0 and q=1.
 *
 * In practice, we check whether adding weight to a centroid would cause
 * it to span too large a range in k-space.
 */
function maxWeight(q: number, totalCount: number, delta: number): number {
  // k'(q) = delta / (2*pi * sqrt(q*(1-q)))
  // max weight ~ 4 * totalCount * q * (1-q) / delta for the arcsin scale
  const qBound = Math.max(1e-10, Math.min(1 - 1e-10, q));
  return Math.max(1, Math.floor(4 * totalCount * qBound * (1 - qBound) / delta));
}

// ---------------------------------------------------------------------------
// Add
// ---------------------------------------------------------------------------

/**
 * Add a value (with optional count) to the t-digest. Mutates state in-place.
 *
 * Algorithm:
 * 1. Update min/max.
 * 2. Find the closest centroid.
 * 3. If the closest centroid can absorb the new value without violating the
 *    scale function constraint, merge into it.
 * 4. Otherwise, create a new centroid.
 * 5. If centroids exceed ~5 * compression, trigger compression.
 *
 * @param state The t-digest state.
 * @param value The data value to add.
 * @param count Number of occurrences of this value (default 1).
 */
export function tdigestAdd(state: TDigestState, value: number, count?: number): void {
  const c = count ?? 1;
  const mutable = state as unknown as {
    centroids: TDigestCentroid[];
    totalCount: number;
    min: number;
    max: number;
  };

  // Update bounds
  if (value < mutable.min) mutable.min = value;
  if (value > mutable.max) mutable.max = value;

  const centroids = mutable.centroids;

  // Empty digest: just add the centroid
  if (centroids.length === 0) {
    centroids.push({ mean: value, count: c });
    mutable.totalCount += c;
    return;
  }

  // Find the insertion point (sorted by mean) using binary search
  const insertIdx = binarySearchInsert(centroids, value);

  // Find closest centroid
  let bestIdx = insertIdx;
  let bestDist = Infinity;

  // Check the centroid at insertIdx and its neighbors
  for (let i = Math.max(0, insertIdx - 1); i <= Math.min(centroids.length - 1, insertIdx + 1); i++) {
    const dist = Math.abs(centroids[i]!.mean - value);
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = i;
    }
  }

  // Ensure bestIdx is within bounds
  bestIdx = Math.min(bestIdx, centroids.length - 1);

  // Check if we can merge into the closest centroid
  const newTotal = mutable.totalCount + c;
  // Quantile of the closest centroid
  let cumWeight = 0;
  for (let i = 0; i < bestIdx; i++) {
    cumWeight += centroids[i]!.count;
  }
  const centroidQ = (cumWeight + centroids[bestIdx]!.count / 2) / newTotal;
  const limit = maxWeight(centroidQ, newTotal, state.compression);

  if (centroids[bestIdx]!.count + c <= limit) {
    // Merge: update mean as weighted average
    const oldMean = centroids[bestIdx]!.mean;
    const oldCount = centroids[bestIdx]!.count;
    const newCount = oldCount + c;
    const newMean = (oldMean * oldCount + value * c) / newCount;
    centroids[bestIdx] = { mean: newMean, count: newCount };
  } else {
    // Insert a new centroid at the correct sorted position
    const newCentroid: TDigestCentroid = { mean: value, count: c };
    centroids.splice(insertIdx, 0, newCentroid);
  }

  mutable.totalCount = newTotal;

  // Compress if we have too many centroids
  if (centroids.length > 5 * state.compression) {
    tdigestCompress(state);
  }
}

/**
 * Binary search for the insertion index to keep centroids sorted by mean.
 */
function binarySearchInsert(centroids: readonly TDigestCentroid[], value: number): number {
  let lo = 0;
  let hi = centroids.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (centroids[mid]!.mean < value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// ---------------------------------------------------------------------------
// Compress
// ---------------------------------------------------------------------------

/**
 * Compress the t-digest by merging adjacent centroids where the scale
 * function permits. This reduces memory usage while preserving accuracy
 * at the tails.
 *
 * After compression, centroids are still sorted by mean.
 */
export function tdigestCompress(state: TDigestState): void {
  const mutable = state as unknown as { centroids: TDigestCentroid[]; totalCount: number };
  const centroids = mutable.centroids;

  if (centroids.length <= 1) return;

  // Sort by mean (should already be sorted, but ensure)
  centroids.sort((a, b) => a.mean - b.mean);

  const totalCount = mutable.totalCount;
  const delta = state.compression;

  const merged: TDigestCentroid[] = [];
  let current = centroids[0]!;
  let cumWeight = 0;

  for (let i = 1; i < centroids.length; i++) {
    const next = centroids[i]!;
    const proposedCount = current.count + next.count;
    const proposedQ = (cumWeight + proposedCount / 2) / totalCount;
    const limit = maxWeight(proposedQ, totalCount, delta);

    if (proposedCount <= limit) {
      // Merge current and next
      const newMean =
        (current.mean * current.count + next.mean * next.count) / proposedCount;
      current = { mean: newMean, count: proposedCount };
    } else {
      // Can't merge: commit current and start new
      merged.push(current);
      cumWeight += current.count;
      current = next;
    }
  }
  merged.push(current);

  mutable.centroids = merged;
}

// ---------------------------------------------------------------------------
// Quantile estimation
// ---------------------------------------------------------------------------

/**
 * Estimate the value at quantile q (0 = minimum, 1 = maximum).
 *
 * Interpolates between centroids based on cumulative weight. At the
 * extreme tails, interpolates between the min/max and the first/last centroid.
 *
 * @param state The t-digest state.
 * @param q     Quantile in [0, 1].
 * @returns Estimated value at the given quantile, or NaN if the digest is empty.
 */
export function tdigestQuantile(state: TDigestState, q: number): number {
  const centroids = state.centroids;
  if (centroids.length === 0) return NaN;
  if (q <= 0) return state.min;
  if (q >= 1) return state.max;

  const totalCount = state.totalCount;
  const targetWeight = q * totalCount;

  // Accumulate weight through centroids, interpolating
  let cumWeight = 0;

  for (let i = 0; i < centroids.length; i++) {
    const centroid = centroids[i]!;
    const halfCount = centroid.count / 2;

    // Weight at the midpoint of this centroid
    const midWeight = cumWeight + halfCount;

    if (midWeight >= targetWeight) {
      // Target is within or before this centroid
      if (i === 0) {
        // Interpolate between min and first centroid's midpoint
        if (halfCount === 0) return centroid.mean;
        const innerQ = (targetWeight - cumWeight) / halfCount;
        return state.min + (centroid.mean - state.min) * Math.max(0, Math.min(1, innerQ));
      }

      // Interpolate between previous centroid and this one
      const prev = centroids[i - 1]!;
      const prevMidWeight = cumWeight - prev.count / 2;
      const span = midWeight - prevMidWeight;
      if (span <= 0) return centroid.mean;
      const t = (targetWeight - prevMidWeight) / span;
      return prev.mean + (centroid.mean - prev.mean) * Math.max(0, Math.min(1, t));
    }

    cumWeight += centroid.count;
  }

  // Target is past the last centroid: interpolate to max
  const last = centroids[centroids.length - 1]!;
  return last.mean + (state.max - last.mean) * Math.min(1, (targetWeight - (cumWeight - last.count / 2)) / (last.count / 2));
}

// ---------------------------------------------------------------------------
// CDF estimation
// ---------------------------------------------------------------------------

/**
 * Estimate the cumulative distribution function (CDF) at a given value.
 * Returns the estimated fraction of items <= value.
 *
 * @param state The t-digest state.
 * @param value The value to evaluate.
 * @returns Estimated CDF in [0, 1], or NaN if the digest is empty.
 */
export function tdigestCDF(state: TDigestState, value: number): number {
  const centroids = state.centroids;
  if (centroids.length === 0) return NaN;
  if (value <= state.min) return 0;
  if (value >= state.max) return 1;

  const totalCount = state.totalCount;
  let cumWeight = 0;

  for (let i = 0; i < centroids.length; i++) {
    const centroid = centroids[i]!;

    if (value < centroid.mean) {
      // Value falls before this centroid's midpoint
      if (i === 0) {
        // Between min and first centroid
        if (centroid.mean === state.min) return 0;
        const t = (value - state.min) / (centroid.mean - state.min);
        return (t * centroid.count / 2) / totalCount;
      }

      // Between previous centroid and this one
      const prev = centroids[i - 1]!;
      const prevWeight = cumWeight - prev.count / 2;
      const span = centroid.mean - prev.mean;
      if (span <= 0) return cumWeight / totalCount;
      const t = (value - prev.mean) / span;
      const interpolatedWeight = prevWeight + t * (cumWeight + centroid.count / 2 - prevWeight);
      return Math.max(0, Math.min(1, interpolatedWeight / totalCount));
    }

    cumWeight += centroid.count;
  }

  // Value is past the last centroid but <= max
  const last = centroids[centroids.length - 1]!;
  if (last.mean === state.max) return 1;
  const t = (value - last.mean) / (state.max - last.mean);
  const weight = (cumWeight - last.count / 2) + t * (last.count / 2);
  return Math.max(0, Math.min(1, weight / totalCount));
}

// ---------------------------------------------------------------------------
// Merge
// ---------------------------------------------------------------------------

/**
 * Merge two t-digests into a new one. The resulting digest contains all
 * data from both inputs.
 *
 * Algorithm: combine all centroids, sort by mean, then compress.
 */
export function tdigestMerge(a: TDigestState, b: TDigestState): TDigestState {
  // Combine all centroids
  const allCentroids: TDigestCentroid[] = [];
  for (let i = 0; i < a.centroids.length; i++) {
    allCentroids.push(a.centroids[i]!);
  }
  for (let i = 0; i < b.centroids.length; i++) {
    allCentroids.push(b.centroids[i]!);
  }

  // Sort by mean
  allCentroids.sort((x, y) => x.mean - y.mean);

  const merged: TDigestState = {
    centroids: allCentroids,
    totalCount: a.totalCount + b.totalCount,
    compression: Math.max(a.compression, b.compression),
    min: Math.min(a.min, b.min),
    max: Math.max(a.max, b.max),
  };

  // Compress to maintain the target centroid count
  tdigestCompress(merged);

  return merged;
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/** Compute the weighted mean of all values in the digest. */
export function tdigestMean(state: TDigestState): number {
  if (state.totalCount === 0) return NaN;

  let weightedSum = 0;
  for (let i = 0; i < state.centroids.length; i++) {
    const c = state.centroids[i]!;
    weightedSum += c.mean * c.count;
  }

  return weightedSum / state.totalCount;
}

/** Return the minimum value seen by the digest. */
export function tdigestMin(state: TDigestState): number {
  return state.min;
}

/** Return the maximum value seen by the digest. */
export function tdigestMax(state: TDigestState): number {
  return state.max;
}

/** Return the total count of values added to the digest. */
export function tdigestCount(state: TDigestState): number {
  return state.totalCount;
}
