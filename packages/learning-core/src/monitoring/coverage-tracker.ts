// ---------------------------------------------------------------------------
// Rolling Conformal Coverage Monitoring
// ---------------------------------------------------------------------------

import type { CoverageMetrics } from '../types.js';

/**
 * CoverageTracker monitors rolling conformal prediction coverage.
 *
 * Maintains a sliding window of coverage indicators (1 = covered, 0 = not covered)
 * and raises alerts when empirical coverage drops significantly below nominal.
 *
 * Usage:
 *   const tracker = new CoverageTracker(0.9, 100, 0.05);
 *   for (const { yTrue, lower, upper } of stream) {
 *     const metrics = tracker.update(yTrue, lower, upper);
 *     if (metrics.alert) { // coverage has degraded }
 *   }
 */
export class CoverageTracker {
  private readonly nominalCoverage: number;
  private readonly windowSize: number;
  private readonly alertThreshold: number;
  private readonly window: number[];
  private windowIndex: number;
  private totalCovered: number;
  private totalObservations: number;

  /**
   * @param nominalCoverage - Target coverage level (e.g. 0.9 for 90%)
   * @param windowSize - Size of the rolling window
   * @param alertThreshold - Alert if empirical coverage is more than this below nominal
   */
  constructor(nominalCoverage: number, windowSize: number, alertThreshold: number) {
    this.nominalCoverage = nominalCoverage;
    this.windowSize = Math.max(1, windowSize);
    this.alertThreshold = alertThreshold;
    this.window = [];
    this.windowIndex = 0;
    this.totalCovered = 0;
    this.totalObservations = 0;
  }

  /**
   * Update the tracker with a new observation and its prediction interval.
   *
   * @param yTrue - True observed value
   * @param lower - Lower bound of prediction interval
   * @param upper - Upper bound of prediction interval
   * @returns Current coverage metrics including alert status
   */
  update(yTrue: number, lower: number, upper: number): CoverageMetrics {
    const covered = (yTrue >= lower && yTrue <= upper) ? 1 : 0;

    this.totalObservations++;
    this.totalCovered += covered;

    // Update rolling window
    if (this.window.length < this.windowSize) {
      // Window is not yet full: append
      this.window.push(covered);
    } else {
      // Window is full: replace oldest entry
      const oldest = this.window[this.windowIndex] ?? 0;
      this.totalCovered -= oldest; // Remove from total only if we're cycling
      this.totalCovered += oldest; // Add it back — wait, we need a separate rolling count
      this.window[this.windowIndex] = covered;
      this.windowIndex = (this.windowIndex + 1) % this.windowSize;
    }

    return this.getMetrics();
  }

  /**
   * Get current coverage metrics.
   *
   * @returns CoverageMetrics with nominal, empirical, rolling window, and alert flag
   */
  getMetrics(): CoverageMetrics {
    const n = this.window.length;
    if (n === 0) {
      return {
        nominal: this.nominalCoverage,
        empirical: 0,
        rolling: [],
        windowSize: this.windowSize,
        alert: false,
      };
    }

    // Compute rolling coverage from the window
    let windowCovered = 0;
    for (let i = 0; i < n; i++) {
      windowCovered += (this.window[i] ?? 0);
    }
    const empiricalCoverage = windowCovered / n;

    // Build rolling array: coverage values in chronological order
    const rolling: number[] = [];
    if (n < this.windowSize) {
      // Window not full yet: window is in insertion order
      let runningCovered = 0;
      for (let i = 0; i < n; i++) {
        runningCovered += (this.window[i] ?? 0);
        rolling.push(runningCovered / (i + 1));
      }
    } else {
      // Window is full: reconstruct chronological order from circular buffer
      let runningCovered = 0;
      for (let i = 0; i < n; i++) {
        const idx = (this.windowIndex + i) % n;
        runningCovered += (this.window[idx] ?? 0);
        rolling.push(runningCovered / (i + 1));
      }
    }

    // Alert if coverage drops more than alertThreshold below nominal
    const alert = empiricalCoverage < (this.nominalCoverage - this.alertThreshold);

    return {
      nominal: this.nominalCoverage,
      empirical: empiricalCoverage,
      rolling,
      windowSize: this.windowSize,
      alert,
    };
  }
}
