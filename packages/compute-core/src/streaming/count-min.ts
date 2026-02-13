// ---------------------------------------------------------------------------
// HPC-6: Streaming Algorithms — Count-Min Sketch
// ---------------------------------------------------------------------------
// Probabilistic frequency estimator. Overestimates are possible, but the
// minimum over all rows gives a tight upper bound. Pure TypeScript, zero deps.
// ---------------------------------------------------------------------------

import type { CountMinSketchConfig, CountMinSketchState } from '../types.js';
import { murmurHash3_32 } from '../types.js';

// ---------------------------------------------------------------------------
// Create
// ---------------------------------------------------------------------------

/**
 * Create a Count-Min Sketch with the given width and depth.
 *
 * - `width` controls accuracy: error bound is totalCount / width.
 * - `depth` controls confidence: probability of exceeding error bound is (1/2)^depth.
 *
 * The table is stored as a flat Int32Array of width * depth for cache efficiency.
 *
 * @param config.width  Number of counters per row (columns).
 * @param config.depth  Number of hash functions (rows).
 */
export function createCountMinSketch(config: CountMinSketchConfig): CountMinSketchState {
  const width = Math.max(1, config.width);
  const depth = Math.max(1, config.depth);

  return {
    table: new Int32Array(width * depth),
    width,
    depth,
    totalCount: 0,
  };
}

// ---------------------------------------------------------------------------
// Internal: hash to column index for a given row
// ---------------------------------------------------------------------------

/**
 * Compute the column index for `item` in row `row`.
 * Each row uses a different seed to produce an independent hash.
 */
function hashColumn(item: string, row: number, width: number): number {
  // Use row index as part of the seed to get independent hashes per row
  return murmurHash3_32(item, row * 0x9e3779b9) % width;
}

// ---------------------------------------------------------------------------
// Add
// ---------------------------------------------------------------------------

/**
 * Increment the count for `item` by `count` (default 1).
 * Mutates the sketch state in-place.
 *
 * @param state The CMS state.
 * @param item  The item to add.
 * @param count How many occurrences to add (default 1).
 */
export function cmsAdd(state: CountMinSketchState, item: string, count?: number): void {
  const c = count ?? 1;

  for (let row = 0; row < state.depth; row++) {
    const col = hashColumn(item, row, state.width);
    const idx = row * state.width + col;
    state.table[idx] = (state.table[idx]! + c) | 0;
  }

  (state as { totalCount: number }).totalCount += c;
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

/**
 * Estimate the frequency of `item`.
 *
 * Returns the minimum counter value across all rows, which is an upper bound
 * on the true frequency. The true count is guaranteed to be <= this value.
 *
 * @returns Estimated frequency (non-negative integer).
 */
export function cmsQuery(state: CountMinSketchState, item: string): number {
  let minCount = Infinity;

  for (let row = 0; row < state.depth; row++) {
    const col = hashColumn(item, row, state.width);
    const idx = row * state.width + col;
    const val = state.table[idx]!;
    if (val < minCount) minCount = val;
  }

  return minCount === Infinity ? 0 : minCount;
}

// ---------------------------------------------------------------------------
// Heavy hitters (threshold validation)
// ---------------------------------------------------------------------------

/**
 * Validate a heavy-hitter threshold against the sketch.
 *
 * Note: Count-Min Sketches cannot enumerate items (they are summary-only
 * structures). This function validates that the threshold is within a
 * meaningful range for the current state.
 *
 * To actually find heavy hitters, maintain a separate candidate set and
 * use `cmsQuery` to filter.
 *
 * @param state     The CMS state.
 * @param threshold Minimum count to qualify as a heavy hitter.
 * @throws If threshold is negative or exceeds totalCount.
 */
export function cmsHeavyHitters(state: CountMinSketchState, threshold: number): void {
  if (threshold < 0) {
    throw new Error('Heavy hitter threshold must be non-negative');
  }
  if (threshold > state.totalCount) {
    throw new Error(
      `Threshold ${threshold} exceeds total count ${state.totalCount}`,
    );
  }
}

// ---------------------------------------------------------------------------
// Merge
// ---------------------------------------------------------------------------

/**
 * Merge two Count-Min Sketches by element-wise addition of their tables.
 * Both sketches must have the same width and depth.
 *
 * @throws If the sketches have different dimensions.
 */
export function cmsMerge(
  a: CountMinSketchState,
  b: CountMinSketchState,
): CountMinSketchState {
  if (a.width !== b.width || a.depth !== b.depth) {
    throw new Error('Cannot merge Count-Min Sketches with different dimensions');
  }

  const merged = new Int32Array(a.table.length);
  for (let i = 0; i < merged.length; i++) {
    merged[i] = (a.table[i]! + b.table[i]!) | 0;
  }

  return {
    table: merged,
    width: a.width,
    depth: a.depth,
    totalCount: a.totalCount + b.totalCount,
  };
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

/** Clear all counters and reset the total count to zero. */
export function cmsReset(state: CountMinSketchState): void {
  state.table.fill(0);
  (state as { totalCount: number }).totalCount = 0;
}

// ---------------------------------------------------------------------------
// Error bound
// ---------------------------------------------------------------------------

/**
 * Compute the error bound for frequency estimates.
 *
 * For any item, the estimated count is at most the true count + (totalCount / width).
 * This is the additive error introduced by hash collisions.
 *
 * @returns The error bound value.
 */
export function cmsEstimateError(state: CountMinSketchState): number {
  if (state.width === 0) return Infinity;
  return state.totalCount / state.width;
}
