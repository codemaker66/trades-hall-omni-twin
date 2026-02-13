// ---------------------------------------------------------------------------
// Weighted Conformal Prediction for Distribution Shift
// (Tibshirani, Barber, Candes & Ramdas 2019, arXiv:1904.06019)
// ---------------------------------------------------------------------------

import type { PredictionInterval } from '../types.js';

/**
 * Compute the weighted quantile of a set of values.
 *
 * For covariate shift, standard conformal prediction can lose coverage
 * because the exchangeability assumption is violated. Weighted conformal
 * uses importance weights w_i = p_test(X_i) / p_train(X_i) to correct
 * for the distributional mismatch.
 *
 * The weighted quantile is defined as the smallest q such that:
 *   sum_{i: v_i <= q} w_i / sum_i w_i >= 1 - alpha
 *
 * An additional "infinite" weight is implicitly added for the test point:
 *   w_{n+1} = 1 (normalized). This accounts for the finite-sample correction.
 *
 * @param values - The values to compute the quantile of (e.g. residuals)
 * @param weights - Importance weights, one per value. Must be non-negative.
 * @param alpha - Miscoverage rate (e.g. 0.1 for 90% coverage)
 * @returns The weighted (1-alpha) quantile
 */
export function weightedQuantile(
  values: number[],
  weights: number[],
  alpha: number,
): number {
  const n = Math.min(values.length, weights.length);
  if (n === 0) return 0;

  // Create (value, weight) pairs and sort by value
  const pairs: Array<{ value: number; weight: number }> = [];
  for (let i = 0; i < n; i++) {
    const w = weights[i] ?? 0;
    if (w < 0) continue; // Skip negative weights
    pairs.push({ value: values[i] ?? 0, weight: w });
  }

  pairs.sort((a, b) => a.value - b.value);

  if (pairs.length === 0) return 0;

  // Compute total weight including the implicit test point weight
  // Following Tibshirani et al.: add w_{n+1} = 1 for the test point
  // to the denominator (but not to the sum, since we search for the
  // threshold among calibration points)
  let totalWeight = 1; // w_{n+1} = 1 for the test point
  for (let i = 0; i < pairs.length; i++) {
    totalWeight += pairs[i]!.weight;
  }

  // Find the smallest value q such that:
  // sum_{v_i <= q} w_i / totalWeight >= 1 - alpha
  const threshold = (1 - alpha) * totalWeight;
  let cumulativeWeight = 0;

  for (let i = 0; i < pairs.length; i++) {
    cumulativeWeight += pairs[i]!.weight;
    if (cumulativeWeight >= threshold) {
      return pairs[i]!.value;
    }
  }

  // If threshold not reached, return the maximum value
  return pairs[pairs.length - 1]!.value;
}

/**
 * Weighted conformal prediction intervals.
 *
 * Adjusts split conformal for covariate shift using importance weights.
 * The interval for each test point is: [y_pred - q_w, y_pred + q_w]
 * where q_w is the weighted conformal quantile of the calibration residuals.
 *
 * For venue models: the weights correct for seasonal distributional changes
 * or geographic domain shifts. For example, if the model was trained mostly
 * on summer data but deployed in winter, the weights up-weight winter-like
 * calibration points.
 *
 * @param yPred - Point predictions for test points
 * @param residuals - Absolute residuals from calibration set
 * @param weights - Importance weights for calibration points.
 *                  w_i = p_test(X_i) / p_train(X_i)
 * @param alpha - Miscoverage rate
 * @returns Prediction intervals with weighted coverage guarantee
 */
export function weightedConformalPredict(
  yPred: number[],
  residuals: number[],
  weights: number[],
  alpha: number,
): PredictionInterval[] {
  // Compute the weighted conformal quantile
  const qW = weightedQuantile(residuals, weights, alpha);

  // Construct intervals: pred +/- q_w
  const intervals: PredictionInterval[] = [];
  for (let i = 0; i < yPred.length; i++) {
    const pred = yPred[i] ?? 0;
    intervals.push({
      lower: pred - qW,
      upper: pred + qW,
      confidenceLevel: 1 - alpha,
    });
  }

  return intervals;
}
