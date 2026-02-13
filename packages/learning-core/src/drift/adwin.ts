// ---------------------------------------------------------------------------
// ADWIN (Adaptive Windowing) Drift Detector (Bifet & Gavalda 2007)
// ---------------------------------------------------------------------------
//
// Maintains a variable-length window of recent observations using a
// bucket list data structure. When the difference in means between any
// two subwindows exceeds a statistically-derived threshold, drift is
// detected and the older portion of the window is dropped.
// ---------------------------------------------------------------------------

import type { DriftDetectorState } from '../types.js';

/**
 * A bucket stores a compressed summary of consecutive observations.
 * Each bucket at level k holds 2^k original observations.
 */
interface Bucket {
  total: number;      // Sum of values in this bucket
  variance: number;   // Sum of squared deviations within this bucket
  count: number;      // Number of original observations (always a power of 2)
}

/**
 * ADWIN drift detector.
 *
 * The bucket list is organized in rows: row k contains buckets of size 2^k.
 * When a row accumulates more than maxBuckets entries, the two oldest are
 * merged into one bucket in row k+1. This gives O(log W) memory for a
 * window of size W.
 *
 * On each update, ADWIN checks all possible binary splits of the window
 * (at bucket boundaries) and detects drift when the Hoeffding-inspired bound
 * |mu_0 - mu_1| >= epsilon_cut is violated.
 */
export class ADWIN {
  private readonly delta: number;
  private readonly maxBuckets: number;

  // Bucket list: rows[k] is an array of buckets, each summarizing 2^k values
  private rows: Bucket[][];

  // Global stats
  private total: number;
  private variance: number;
  private width: number;      // Total observations in the window
  private _driftDetected: boolean;

  constructor(delta: number = 0.002) {
    this.delta = delta;
    this.maxBuckets = 5;  // M parameter: max buckets per row before merging
    this.rows = [];
    this.total = 0;
    this.variance = 0;
    this.width = 0;
    this._driftDetected = false;
  }

  /**
   * Add a new value and check for drift.
   * @returns true if drift was detected on this update
   */
  update(value: number): boolean {
    this._driftDetected = false;

    // Insert new value as a single-element bucket at row 0
    this.insertBucket(value);

    // Compress: merge buckets if any row exceeds maxBuckets
    this.compress();

    // Check for drift across all possible splits
    this._driftDetected = this.checkDrift();

    return this._driftDetected;
  }

  /** Reset the detector to its initial state. */
  reset(): void {
    this.rows = [];
    this.total = 0;
    this.variance = 0;
    this.width = 0;
    this._driftDetected = false;
  }

  /** Current mean of the window. */
  get mean(): number {
    return this.width > 0 ? this.total / this.width : 0;
  }

  /** Whether drift was detected on the most recent update. */
  get driftDetected(): boolean {
    return this._driftDetected;
  }

  /** Current window width (number of observations). */
  get windowWidth(): number {
    return this.width;
  }

  // ---- Internal Methods ----

  /** Insert a new observation as a bucket at row 0. */
  private insertBucket(value: number): void {
    if (this.rows.length === 0) {
      this.rows.push([]);
    }
    const row0 = this.rows[0]!;
    row0.push({ total: value, variance: 0, count: 1 });
    this.total += value;
    this.width += 1;

    // Update running variance using Welford-like approach:
    // variance tracks sum of (x_i - mean)^2 across the window,
    // but we approximate by simply tracking incremental variance
    if (this.width > 1) {
      const oldMean = (this.total - value) / (this.width - 1);
      this.variance += (value - oldMean) * (value - this.total / this.width);
    }
  }

  /**
   * Compress: if any row has more than maxBuckets buckets, merge the two
   * oldest (first two) into one bucket in the next row.
   */
  private compress(): void {
    for (let k = 0; k < this.rows.length; k++) {
      const row = this.rows[k]!;
      if (row.length > this.maxBuckets) {
        // Merge the two oldest buckets (indices 0 and 1)
        const b1 = row[0]!;
        const b2 = row[1]!;

        // Merged bucket stats
        const mergedCount = b1.count + b2.count;
        const mergedTotal = b1.total + b2.total;
        // Variance of merged = var(b1) + var(b2) + n1*n2/(n1+n2) * (mu1-mu2)^2
        const mu1 = b1.total / b1.count;
        const mu2 = b2.total / b2.count;
        const mergedVariance = b1.variance + b2.variance +
          (b1.count * b2.count / mergedCount) * (mu1 - mu2) * (mu1 - mu2);

        const merged: Bucket = {
          total: mergedTotal,
          variance: mergedVariance,
          count: mergedCount,
        };

        // Remove the two oldest from this row
        row.splice(0, 2);

        // Push merged bucket to the next row
        if (k + 1 >= this.rows.length) {
          this.rows.push([]);
        }
        this.rows[k + 1]!.push(merged);
      }
    }
  }

  /**
   * Check all possible binary splits of the window for drift.
   * The split is between a "left" (newer) and "right" (older) subwindow.
   * If |mu_left - mu_right| >= epsilon_cut, drift is detected and the
   * older portion is removed.
   *
   * @returns true if drift was detected
   */
  private checkDrift(): boolean {
    if (this.width < 2) return false;

    let driftFound = false;

    // We iterate through buckets from newest to oldest, accumulating
    // the "left" subwindow and checking against the "right" remainder.
    let leftTotal = 0;
    let leftWidth = 0;

    // Traverse rows from 0 (smallest buckets) upward, and within each
    // row from newest (end) to oldest (start).
    for (let k = 0; k < this.rows.length; k++) {
      const row = this.rows[k]!;
      for (let j = row.length - 1; j >= 0; j--) {
        const bucket = row[j]!;
        leftTotal += bucket.total;
        leftWidth += bucket.count;

        const rightWidth = this.width - leftWidth;
        if (rightWidth < 1 || leftWidth < 1) continue;

        const rightTotal = this.total - leftTotal;
        const leftMean = leftTotal / leftWidth;
        const rightMean = rightTotal / rightWidth;

        const absDiff = Math.abs(leftMean - rightMean);
        const epsCut = this.computeEpsilonCut(leftWidth, rightWidth);

        if (absDiff >= epsCut) {
          // Drift detected: remove older (right) portion
          this.removeBucketsFromOldest(rightWidth);
          driftFound = true;
          break;
        }
      }
      if (driftFound) break;
    }

    return driftFound;
  }

  /**
   * Compute the epsilon_cut threshold for two subwindows of sizes n0 and n1.
   *
   * epsilon_cut = sqrt( (1/(2m)) * ln(4/delta') )
   *
   * where m = 1/(1/n0 + 1/n1) = n0*n1/(n0+n1) is the harmonic mean,
   * and delta' = delta / ln(n) to account for multiple testing over
   * different split points (Bonferroni-like correction).
   */
  private computeEpsilonCut(n0: number, n1: number): number {
    const n = n0 + n1;
    const logN = Math.log(n);
    if (logN <= 0) return Infinity;

    const deltaPrime = this.delta / logN;
    if (deltaPrime <= 0) return Infinity;

    // Harmonic mean of n0 and n1
    const m = (n0 * n1) / n;

    // Hoeffding-style bound
    const eps = Math.sqrt((1 / (2 * m)) * Math.log(4 / deltaPrime));
    return eps;
  }

  /**
   * Remove `count` observations from the oldest end of the bucket list.
   * After removal, update total and width.
   */
  private removeBucketsFromOldest(count: number): void {
    let remaining = count;

    // Oldest buckets are at the front (index 0) of the highest-level rows.
    // We traverse from the highest row down to row 0.
    for (let k = this.rows.length - 1; k >= 0 && remaining > 0; k--) {
      const row = this.rows[k]!;
      while (row.length > 0 && remaining > 0) {
        const oldest = row[0]!;
        if (oldest.count <= remaining) {
          // Remove entire bucket
          this.total -= oldest.total;
          this.width -= oldest.count;
          remaining -= oldest.count;
          row.shift();
        } else {
          // Partial removal not possible at bucket level; just break
          // (ADWIN operates at bucket boundaries)
          break;
        }
      }
    }

    // Clean up empty rows at the top
    while (this.rows.length > 0 && this.rows[this.rows.length - 1]!.length === 0) {
      this.rows.pop();
    }

    // Recompute variance from scratch (necessary after removing buckets)
    this.recomputeVariance();
  }

  /** Recompute the running variance from bucket-level statistics. */
  private recomputeVariance(): void {
    if (this.width <= 1) {
      this.variance = 0;
      return;
    }

    // We approximate: rebuild variance from bucket variances plus
    // between-bucket variance
    const globalMean = this.total / this.width;
    let totalVar = 0;

    for (let k = 0; k < this.rows.length; k++) {
      const row = this.rows[k]!;
      for (let j = 0; j < row.length; j++) {
        const b = row[j]!;
        const bucketMean = b.total / b.count;
        // Within-bucket variance
        totalVar += b.variance;
        // Between-bucket variance contribution
        totalVar += b.count * (bucketMean - globalMean) * (bucketMean - globalMean);
      }
    }

    this.variance = totalVar;
  }

  /** Get current state as a DriftDetectorState. */
  getState(): DriftDetectorState {
    return {
      driftDetected: this._driftDetected,
      warningDetected: false,  // ADWIN does not have a warning zone
      nObservations: this.width,
    };
  }
}
