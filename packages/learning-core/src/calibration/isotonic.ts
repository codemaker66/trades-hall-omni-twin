// ---------------------------------------------------------------------------
// Isotonic Regression (Zadrozny & Elkan 2002)
// Pool Adjacent Violators Algorithm (PAVA)
// ---------------------------------------------------------------------------

import type { IsotonicParams } from '../types.js';

/**
 * Fit isotonic regression using the Pool Adjacent Violators Algorithm (PAVA).
 *
 * Given (prediction, label) pairs, fits a non-decreasing step function
 * that minimizes mean squared error subject to monotonicity constraints.
 *
 * Algorithm:
 * 1. Sort pairs by prediction value (ascending).
 * 2. Initialize each point as its own block with value = label.
 * 3. Scan left to right: if adjacent blocks violate monotonicity
 *    (block[i] > block[i+1]), merge them by taking their weighted average.
 * 4. Repeat merging until no violations remain.
 *
 * @param predictions - Uncalibrated model scores
 * @param labels - Binary labels (0 or 1)
 * @returns IsotonicParams with breakpoints (xs) and values (ys)
 */
export function isotonicFit(
  predictions: number[],
  labels: number[],
): IsotonicParams {
  const n = Math.min(predictions.length, labels.length);
  if (n === 0) return { xs: [], ys: [] };

  // Create (prediction, label) pairs and sort by prediction
  const indices: number[] = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    indices[i] = i;
  }
  indices.sort((a, b) => (predictions[a] ?? 0) - (predictions[b] ?? 0));

  // Initialize blocks: each point is its own block
  // A block has: sum of values, count, and representative x value
  const blockSum: number[] = new Array<number>(n);
  const blockCount: number[] = new Array<number>(n);
  const blockMinX: number[] = new Array<number>(n);
  const blockMaxX: number[] = new Array<number>(n);

  for (let i = 0; i < n; i++) {
    const idx = indices[i]!;
    blockSum[i] = labels[idx] ?? 0;
    blockCount[i] = 1;
    blockMinX[i] = predictions[idx] ?? 0;
    blockMaxX[i] = predictions[idx] ?? 0;
  }

  // PAVA: Pool Adjacent Violators
  // Use a stack-like approach: we process blocks left to right,
  // merging when the current block's value is less than the previous.
  let nBlocks = n;
  let merged = true;

  while (merged) {
    merged = false;
    let writeIdx = 0;

    for (let i = 0; i < nBlocks; i++) {
      // Copy current block to write position
      if (writeIdx !== i) {
        blockSum[writeIdx] = blockSum[i] ?? 0;
        blockCount[writeIdx] = blockCount[i] ?? 0;
        blockMinX[writeIdx] = blockMinX[i] ?? 0;
        blockMaxX[writeIdx] = blockMaxX[i] ?? 0;
      }

      // Merge with previous block while violation exists
      while (
        writeIdx > 0 &&
        (blockSum[writeIdx - 1] ?? 0) / (blockCount[writeIdx - 1] ?? 1) >
          (blockSum[writeIdx] ?? 0) / (blockCount[writeIdx] ?? 1)
      ) {
        // Pool: merge current block into previous
        blockSum[writeIdx - 1] = (blockSum[writeIdx - 1] ?? 0) + (blockSum[writeIdx] ?? 0);
        blockCount[writeIdx - 1] = (blockCount[writeIdx - 1] ?? 0) + (blockCount[writeIdx] ?? 0);
        blockMaxX[writeIdx - 1] = blockMaxX[writeIdx] ?? 0;
        writeIdx--;
        merged = true;
      }

      writeIdx++;
    }

    nBlocks = writeIdx;
  }

  // Build breakpoint representation
  // Each block maps an x-range to a constant y-value
  const xs: number[] = [];
  const ys: number[] = [];

  for (let i = 0; i < nBlocks; i++) {
    const value = (blockSum[i] ?? 0) / Math.max(blockCount[i] ?? 1, 1);
    // Use the midpoint of each block's x-range as the breakpoint
    // But for interpolation, we store both min and max
    xs.push(blockMinX[i] ?? 0);
    ys.push(value);

    // If the block spans multiple x values, also store the max
    if ((blockMaxX[i] ?? 0) > (blockMinX[i] ?? 0)) {
      xs.push(blockMaxX[i] ?? 0);
      ys.push(value);
    }
  }

  return { xs, ys };
}

/**
 * Apply isotonic calibration via piecewise constant interpolation.
 *
 * For a score s:
 * - If s <= xs[0], return ys[0]
 * - If s >= xs[last], return ys[last]
 * - Otherwise, find the interval [xs[i], xs[i+1]] containing s and
 *   linearly interpolate between ys[i] and ys[i+1]
 *
 * @param scores - Uncalibrated model scores
 * @param params - Fitted isotonic parameters with breakpoints
 * @returns Calibrated probabilities
 */
export function isotonicTransform(scores: number[], params: IsotonicParams): number[] {
  const { xs, ys } = params;
  const nBreak = xs.length;

  if (nBreak === 0) {
    // No breakpoints — return 0.5 as default
    const result: number[] = new Array<number>(scores.length);
    for (let i = 0; i < scores.length; i++) {
      result[i] = 0.5;
    }
    return result;
  }

  const result: number[] = new Array<number>(scores.length);

  for (let i = 0; i < scores.length; i++) {
    const s = scores[i] ?? 0;

    // Clamp to breakpoint range
    if (s <= (xs[0] ?? 0)) {
      result[i] = ys[0] ?? 0;
      continue;
    }
    if (s >= (xs[nBreak - 1] ?? 0)) {
      result[i] = ys[nBreak - 1] ?? 0;
      continue;
    }

    // Binary search for the interval containing s
    let lo = 0;
    let hi = nBreak - 1;
    while (lo < hi - 1) {
      const mid = (lo + hi) >> 1;
      if ((xs[mid] ?? 0) <= s) {
        lo = mid;
      } else {
        hi = mid;
      }
    }

    // Linear interpolation between breakpoints
    const x0 = xs[lo] ?? 0;
    const x1 = xs[hi] ?? 0;
    const y0 = ys[lo] ?? 0;
    const y1 = ys[hi] ?? 0;

    if (x1 <= x0) {
      result[i] = y0;
    } else {
      const t = (s - x0) / (x1 - x0);
      result[i] = y0 + t * (y1 - y0);
    }
  }

  return result;
}
