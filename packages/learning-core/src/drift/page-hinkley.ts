// ---------------------------------------------------------------------------
// Page-Hinkley Test for Change Detection
// ---------------------------------------------------------------------------
//
// A sequential change detection test that monitors the cumulative sum of
// deviations from the running mean. Detects drift when the maximum
// cumulative sum minus the current cumulative sum exceeds a threshold.
//
// Parameters:
//   delta  - Magnitude allowance (tolerance to gradual changes)
//   lambda - Detection threshold
//   alpha  - Forgetting factor for exponentially weighted mean (0 < alpha < 1)
// ---------------------------------------------------------------------------

import type { DriftDetectorState, PageHinkleyConfig } from '../types.js';

/** Default Page-Hinkley configuration. */
const PH_DEFAULTS: PageHinkleyConfig = {
  delta: 0.005,
  lambda: 50,
  alpha: 0.9999,
};

/**
 * Page-Hinkley change detection test.
 *
 * Monitors a cumulative sum:
 *   m_T = sum_{t=1}^{T} (x_t - x_bar_T - delta)
 *   M_T = max_{t=1}^{T} m_t
 *
 * Drift is signaled when M_T - m_T > lambda.
 */
export class PageHinkley {
  private readonly config: PageHinkleyConfig;

  private nObservations: number;
  private sum: number;          // Sum of all observed values
  private cumulativeSum: number; // m_T: cumulative sum of (x_t - x_bar - delta)
  private maxCumulativeSum: number; // M_T: maximum of m_t over all t

  private _driftDetected: boolean;

  constructor(config?: Partial<PageHinkleyConfig>) {
    this.config = { ...PH_DEFAULTS, ...config };
    this.nObservations = 0;
    this.sum = 0;
    this.cumulativeSum = 0;
    this.maxCumulativeSum = 0;
    this._driftDetected = false;
  }

  /**
   * Update with a new observation value.
   * @param value The new observed value
   * @returns Current drift detector state
   */
  update(value: number): DriftDetectorState {
    this._driftDetected = false;

    this.nObservations += 1;

    // Update exponentially weighted sum for running mean
    // Using alpha-weighted sum: sum = alpha * sum + value
    this.sum = this.config.alpha * this.sum + value;

    // Compute running mean
    // For the exponentially weighted version, the effective count is
    // approximately 1 / (1 - alpha) for large T, but we use a simpler
    // approach: maintain actual running mean
    const xBar = this.sum / this.nObservations;

    // Update cumulative sum: m_T += (x_t - x_bar - delta)
    this.cumulativeSum += (value - xBar - this.config.delta);

    // Update maximum cumulative sum
    if (this.cumulativeSum > this.maxCumulativeSum) {
      this.maxCumulativeSum = this.cumulativeSum;
    }

    // Drift detection: M_T - m_T > lambda
    if (this.maxCumulativeSum - this.cumulativeSum > this.config.lambda) {
      this._driftDetected = true;
    }

    return this.getState();
  }

  /** Reset the detector to its initial state. */
  reset(): void {
    this.nObservations = 0;
    this.sum = 0;
    this.cumulativeSum = 0;
    this.maxCumulativeSum = 0;
    this._driftDetected = false;
  }

  /** Get current state. */
  getState(): DriftDetectorState {
    return {
      driftDetected: this._driftDetected,
      warningDetected: false, // Page-Hinkley does not have a warning zone
      nObservations: this.nObservations,
    };
  }

  /** Current PH test statistic: M_T - m_T. */
  get testStatistic(): number {
    return this.maxCumulativeSum - this.cumulativeSum;
  }
}
