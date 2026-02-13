// ---------------------------------------------------------------------------
// DDM (Drift Detection Method) (Gama et al. 2004)
// ---------------------------------------------------------------------------
//
// Monitors a stream of binary predictions (0 = correct, 1 = error).
// Tracks running error rate p_i and standard deviation s_i = sqrt(p_i(1-p_i)/i).
// Warning is raised when p_i + s_i >= p_min + warningLevel * s_min.
// Drift is detected when p_i + s_i >= p_min + driftLevel * s_min.
// ---------------------------------------------------------------------------

import type { DDMConfig, DriftDetectorState } from '../types.js';

/** Default DDM configuration. */
const DDM_DEFAULTS: DDMConfig = {
  minInstances: 30,
  warningLevel: 2,
  driftLevel: 3,
};

/**
 * DDM (Drift Detection Method) drift detector.
 *
 * Monitors the error rate of a classifier and detects concept drift
 * by watching for statistically significant increases in error rate.
 */
export class DDM {
  private readonly config: DDMConfig;

  // Running statistics
  private nObservations: number;
  private errorSum: number;    // Sum of errors (0/1)
  private pi: number;          // Running error rate
  private si: number;          // Running std dev: sqrt(pi*(1-pi)/n)

  // Minimum statistics
  private pMin: number;        // Minimum error rate
  private sMin: number;        // Std dev at minimum

  // State flags
  private _driftDetected: boolean;
  private _warningDetected: boolean;

  constructor(config?: Partial<DDMConfig>) {
    this.config = { ...DDM_DEFAULTS, ...config };
    this.nObservations = 0;
    this.errorSum = 0;
    this.pi = 0;
    this.si = 0;
    this.pMin = Infinity;
    this.sMin = Infinity;
    this._driftDetected = false;
    this._warningDetected = false;
  }

  /**
   * Update with a new prediction result.
   * @param prediction 0 for correct prediction, 1 for error
   * @returns Current drift detector state
   */
  update(prediction: number): DriftDetectorState {
    this._driftDetected = false;
    this._warningDetected = false;

    this.nObservations += 1;
    this.errorSum += prediction;

    // Compute running error rate and standard deviation
    this.pi = this.errorSum / this.nObservations;
    this.si = Math.sqrt(this.pi * (1 - this.pi) / this.nObservations);

    // Only detect after minimum instances
    if (this.nObservations >= this.config.minInstances) {
      // Track minimum p + s
      const psPlusS = this.pi + this.si;
      const pMinPlusSMin = this.pMin + this.sMin;

      // Update minimum if current p+s is lower
      if (psPlusS < pMinPlusSMin) {
        this.pMin = this.pi;
        this.sMin = this.si;
      }

      // Check for drift: p_i + s_i >= p_min + driftLevel * s_min
      if (psPlusS >= this.pMin + this.config.driftLevel * this.sMin) {
        this._driftDetected = true;
        // Reset after drift detection
        this.resetStats();
      }
      // Check for warning: p_i + s_i >= p_min + warningLevel * s_min
      else if (psPlusS >= this.pMin + this.config.warningLevel * this.sMin) {
        this._warningDetected = true;
      }
    }

    return this.getState();
  }

  /** Reset the detector to its initial state. */
  reset(): void {
    this.resetStats();
    this._driftDetected = false;
    this._warningDetected = false;
  }

  /** Get current state. */
  getState(): DriftDetectorState {
    return {
      driftDetected: this._driftDetected,
      warningDetected: this._warningDetected,
      nObservations: this.nObservations,
    };
  }

  /** Reset running statistics (called after drift). */
  private resetStats(): void {
    this.nObservations = 0;
    this.errorSum = 0;
    this.pi = 0;
    this.si = 0;
    this.pMin = Infinity;
    this.sMin = Infinity;
  }
}
