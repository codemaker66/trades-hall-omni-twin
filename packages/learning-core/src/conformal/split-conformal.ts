// ---------------------------------------------------------------------------
// Split Conformal Prediction (Vovk et al. 2005) + Jackknife+ (Barber et al. 2019)
// ---------------------------------------------------------------------------

import type { PredictionInterval } from '../types.js';

/**
 * Compute absolute residuals |y_true - y_pred|.
 * These are the non-conformity scores used in split conformal prediction.
 */
export function computeResiduals(yTrue: number[], yPred: number[]): number[] {
  const n = Math.min(yTrue.length, yPred.length);
  const residuals: number[] = [];
  for (let i = 0; i < n; i++) {
    residuals.push(Math.abs((yTrue[i] ?? 0) - (yPred[i] ?? 0)));
  }
  return residuals;
}

/**
 * Compute the conformal quantile at level ceil((n+1)(1-alpha))/n.
 *
 * For finite-sample validity, we need the ceil((n+1)(1-alpha))/n-th quantile
 * of the calibration residuals. This guarantees:
 *   P(Y_{n+1} in C(X_{n+1})) >= 1 - alpha
 * under exchangeability.
 *
 * @param residuals - Absolute residuals from calibration set
 * @param alpha - Miscoverage rate (e.g. 0.1 for 90% coverage)
 * @returns The conformal quantile value
 */
export function conformalQuantile(residuals: number[], alpha: number): number {
  const n = residuals.length;
  if (n === 0) return 0;

  // Sort residuals in ascending order
  const sorted = [...residuals].sort((a, b) => a - b);

  // Compute the finite-sample corrected quantile level
  // q = ceil((n+1)(1-alpha)) / n
  const level = Math.ceil((n + 1) * (1 - alpha)) / n;

  if (level >= 1) {
    // If level >= 1, return the maximum residual (or +Infinity for full coverage)
    return sorted[n - 1] ?? 0;
  }

  // Find the quantile at the computed level
  // Index = floor(level * n) but capped at n-1
  const idx = Math.min(Math.ceil(level * n) - 1, n - 1);
  return sorted[Math.max(idx, 0)] ?? 0;
}

/**
 * Split conformal prediction: construct prediction intervals as pred +/- quantile.
 *
 * After computing the conformal quantile Q from calibration residuals,
 * intervals are: [y_pred - Q, y_pred + Q] for each test point.
 *
 * @param yPred - Point predictions for new data
 * @param quantile - The conformal quantile from conformalQuantile()
 * @param alpha - Miscoverage rate used to annotate confidence metadata
 * @returns Prediction intervals with guaranteed coverage
 */
export function splitConformalPredict(
  yPred: number[],
  quantile: number,
  alpha: number,
): PredictionInterval[] {
  const intervals: PredictionInterval[] = [];
  for (let i = 0; i < yPred.length; i++) {
    const pred = yPred[i] ?? 0;
    intervals.push({
      lower: pred - quantile,
      upper: pred + quantile,
      confidenceLevel: 1 - alpha,
    });
  }
  return intervals;
}

/**
 * Jackknife+ prediction intervals (Barber, Candes, Ramdas & Tibshirani 2019).
 *
 * Given LOO (leave-one-out) predictions, constructs tighter intervals than
 * split conformal by leveraging all training data.
 *
 * Algorithm:
 * 1. For each training point i, we have a model trained without i.
 *    predictions[i] contains that model's predictions on all training points + new point.
 * 2. Compute LOO residuals: R_i = |y_i - hat{y}_{-i}(X_i)|
 * 3. For each new test point j, compute:
 *    - R_{-i,j}^+ = y_i + R_{-i,j} and R_{-i,j}^- = y_i - R_{-i,j}
 *    Actually: intervals are the (alpha)-quantile of {hat{y}_{-i}(X_new) - R_i}
 *    to the (1-alpha)-quantile of {hat{y}_{-i}(X_new) + R_i}
 *
 * Coverage guarantee: P(Y_{n+1} in C(X_{n+1})) >= 1 - 2*alpha
 *
 * @param predictions - predictions[i][j] = prediction of model trained without i
 *                      on j-th point. Last column is the new test point.
 *                      Shape: nTrain x (nTrain + nNew)
 * @param yTrain - Training labels
 * @param yNew - Predictions for new points (unused in LOO version, kept for API compat)
 * @param alpha - Miscoverage rate
 * @returns Prediction intervals for the new test points
 */
export function jackknifePlusPredict(
  predictions: number[][],
  yTrain: number[],
  yNew: number[],
  alpha: number,
): PredictionInterval[] {
  const nTrain = yTrain.length;
  const nNew = yNew.length;

  if (nTrain === 0 || predictions.length === 0) {
    return yNew.map(() => ({
      lower: -Infinity,
      upper: Infinity,
      confidenceLevel: 1 - 2 * alpha,
    }));
  }

  // Step 1: Compute LOO residuals
  // R_i = |y_i - hat{y}_{-i}(X_i)|
  // predictions[i][i] = prediction of model-without-i on point i
  const looResiduals: number[] = [];
  for (let i = 0; i < nTrain; i++) {
    const predRow = predictions[i];
    if (!predRow) {
      looResiduals.push(0);
      continue;
    }
    const predOnSelf = predRow[i] ?? 0;
    looResiduals.push(Math.abs((yTrain[i] ?? 0) - predOnSelf));
  }

  // Step 2: For each new test point, compute jackknife+ intervals
  const intervals: PredictionInterval[] = [];

  for (let j = 0; j < nNew; j++) {
    // Collect {hat{y}_{-i}(X_new_j) +/- R_i} for all i
    const lowerValues: number[] = [];
    const upperValues: number[] = [];

    for (let i = 0; i < nTrain; i++) {
      const predRow = predictions[i];
      if (!predRow) continue;
      // predictions[i][nTrain + j] = prediction of model-without-i on new point j
      const predOnNew = predRow[nTrain + j] ?? 0;
      const ri = looResiduals[i] ?? 0;
      lowerValues.push(predOnNew - ri);
      upperValues.push(predOnNew + ri);
    }

    // Sort to find quantiles
    lowerValues.sort((a, b) => a - b);
    upperValues.sort((a, b) => a - b);

    const nValues = lowerValues.length;
    if (nValues === 0) {
      intervals.push({
        lower: -Infinity,
        upper: Infinity,
        confidenceLevel: 1 - 2 * alpha,
      });
      continue;
    }

    // Lower bound: alpha-quantile of lower values (take the floor)
    // Upper bound: (1-alpha)-quantile of upper values (take the ceil)
    const lowerIdx = Math.max(Math.floor(alpha * (nValues + 1)) - 1, 0);
    const upperIdx = Math.min(
      Math.ceil((1 - alpha) * (nValues + 1)) - 1,
      nValues - 1,
    );

    intervals.push({
      lower: lowerValues[lowerIdx] ?? 0,
      upper: upperValues[upperIdx] ?? 0,
      confidenceLevel: 1 - 2 * alpha,
    });
  }

  return intervals;
}
