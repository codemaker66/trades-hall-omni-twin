// ---------------------------------------------------------------------------
// Drift → Retraining Pipeline Trigger
// ---------------------------------------------------------------------------

import type {
  CoverageMetrics,
  RetrainingTriggerResult,
} from '../types.js';

/**
 * RetrainingTrigger evaluates whether a model should be retrained based on
 * coverage degradation and drift scores, with a minimum interval between
 * retraining events to avoid excessive churn.
 *
 * Usage:
 *   const trigger = new RetrainingTrigger(0.05, 0.1, 86400000);
 *   const result = trigger.evaluate(coverageMetrics, driftScore, Date.now());
 *   if (result.shouldRetrain) { // initiate retraining }
 */
export class RetrainingTrigger {
  private readonly coverageThreshold: number;
  private readonly driftThreshold: number;
  private readonly minInterval: number;
  private lastRetrainedAt: number;

  /**
   * @param coverageThreshold - Maximum acceptable coverage drop below nominal (e.g. 0.05)
   * @param driftThreshold - Maximum acceptable drift score before triggering retraining
   * @param minInterval - Minimum time (in ms or arbitrary time units) between retraining events
   */
  constructor(coverageThreshold: number, driftThreshold: number, minInterval: number) {
    this.coverageThreshold = coverageThreshold;
    this.driftThreshold = driftThreshold;
    this.minInterval = minInterval;
    this.lastRetrainedAt = 0;
  }

  /**
   * Evaluate whether retraining should be triggered.
   *
   * Retraining is triggered when:
   * 1. Sufficient time has elapsed since last retraining (>= minInterval)
   * 2. AND either:
   *    a. Coverage has dropped more than coverageThreshold below nominal
   *    b. Drift score exceeds driftThreshold
   *
   * @param coverage - Current coverage metrics from CoverageTracker
   * @param driftScore - Current drift score (e.g. from ADWIN or DDM)
   * @param now - Current timestamp (ms or arbitrary time units)
   * @returns RetrainingTriggerResult indicating whether to retrain and why
   */
  evaluate(
    coverage: CoverageMetrics,
    driftScore: number,
    now: number,
  ): RetrainingTriggerResult {
    const coverageDrop = coverage.nominal - coverage.empirical;
    const timeSinceLastRetrain = now - this.lastRetrainedAt;
    const intervalOk = timeSinceLastRetrain >= this.minInterval;

    // Check conditions
    const coverageDegraded = coverageDrop > this.coverageThreshold;
    const driftExceeded = driftScore > this.driftThreshold;

    let shouldRetrain = false;
    let reason: string | null = null;

    if (intervalOk) {
      if (coverageDegraded && driftExceeded) {
        shouldRetrain = true;
        reason = `Coverage dropped by ${coverageDrop.toFixed(4)} (threshold: ${this.coverageThreshold}) and drift score ${driftScore.toFixed(4)} exceeds threshold ${this.driftThreshold}`;
      } else if (coverageDegraded) {
        shouldRetrain = true;
        reason = `Coverage dropped by ${coverageDrop.toFixed(4)} (threshold: ${this.coverageThreshold})`;
      } else if (driftExceeded) {
        shouldRetrain = true;
        reason = `Drift score ${driftScore.toFixed(4)} exceeds threshold ${this.driftThreshold}`;
      }
    }

    // Update last retrained timestamp if we're triggering retraining
    if (shouldRetrain) {
      this.lastRetrainedAt = now;
    }

    return {
      shouldRetrain,
      reason,
      driftScore,
      coverageDrop: Math.max(0, coverageDrop),
      lastRetrainedAt: this.lastRetrainedAt,
    };
  }
}
