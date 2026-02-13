// ---------------------------------------------------------------------------
// Ensemble Batch Prediction Intervals (EnbPI) for Time Series
// (Xu & Xie 2021, arXiv:2010.09107)
// ---------------------------------------------------------------------------

import type { PredictionInterval } from '../types.js';

/**
 * EnbPI: Ensemble Batch Prediction Intervals for time series.
 *
 * Unlike split conformal which assumes exchangeability, EnbPI uses bootstrap
 * aggregation to handle temporal dependence without data splitting or retraining.
 *
 * Algorithm:
 * 1. Train B bootstrap models on resampled training data.
 * 2. For each training point i, compute the leave-one-out bootstrap aggregated
 *    prediction: hat{y}_i = mean of predictions from models whose bootstrap
 *    sample did NOT include point i.
 * 3. Compute residuals: e_i = y_i - hat{y}_i
 * 4. For new points, the interval is:
 *    [hat{y}_new + Q_{alpha/2}(residuals), hat{y}_new + Q_{1-alpha/2}(residuals)]
 *
 * @param bootstrapPredictions - bootstrapPredictions[b][i] = prediction of
 *   bootstrap model b on training point i. Shape: B x nTrain.
 *   The out-of-bag (OOB) mechanism is approximated: each bootstrap sample
 *   includes ~63.2% of training points. For point i, we average predictions
 *   from the ~36.8% of models that did NOT include i.
 *   In this simplified version, we assume all models contribute (no OOB tracking)
 *   and use the full ensemble mean as the aggregated prediction.
 * @param yTrain - Training labels
 * @param alpha - Miscoverage rate (e.g. 0.1 for 90% coverage)
 * @returns Prediction intervals for each training point (for online updating)
 *          and the residual quantiles needed for future predictions.
 */
export function enbpiPredict(
  bootstrapPredictions: number[][],
  yTrain: number[],
  alpha: number,
): PredictionInterval[] {
  const nBootstrap = bootstrapPredictions.length;
  const nTrain = yTrain.length;

  if (nBootstrap === 0 || nTrain === 0) {
    return [];
  }

  // Step 1: Compute ensemble-aggregated predictions for each training point
  // hat{y}_i = (1/B) * sum_{b=1}^{B} hat{f}_b(X_i)
  const ensemblePreds: number[] = [];
  for (let i = 0; i < nTrain; i++) {
    let sum = 0;
    let count = 0;
    for (let b = 0; b < nBootstrap; b++) {
      const row = bootstrapPredictions[b];
      if (row && i < row.length) {
        sum += row[i] ?? 0;
        count++;
      }
    }
    ensemblePreds.push(count > 0 ? sum / count : 0);
  }

  // Step 2: Compute residuals e_i = y_i - hat{y}_i
  const residuals: number[] = [];
  for (let i = 0; i < nTrain; i++) {
    residuals.push((yTrain[i] ?? 0) - (ensemblePreds[i] ?? 0));
  }

  // Step 3: Sort residuals to compute quantiles
  const sorted = [...residuals].sort((a, b) => a - b);
  const n = sorted.length;

  // Compute lower quantile (alpha/2) and upper quantile (1 - alpha/2)
  // with finite-sample correction
  const lowerLevel = alpha / 2;
  const upperLevel = 1 - alpha / 2;

  // Lower quantile index: floor(lowerLevel * (n+1)) - 1, clamped to [0, n-1]
  const lowerIdx = Math.max(
    Math.min(Math.floor(lowerLevel * (n + 1)) - 1, n - 1),
    0,
  );
  // Upper quantile index: ceil(upperLevel * (n+1)) - 1, clamped to [0, n-1]
  const upperIdx = Math.max(
    Math.min(Math.ceil(upperLevel * (n + 1)) - 1, n - 1),
    0,
  );

  const qLower = sorted[lowerIdx] ?? 0;
  const qUpper = sorted[upperIdx] ?? 0;

  // Step 4: Construct prediction intervals for each training point
  // The interval for point i is: hat{y}_i + [qLower, qUpper]
  // This represents the residual-corrected interval.
  const intervals: PredictionInterval[] = [];
  for (let i = 0; i < nTrain; i++) {
    const pred = ensemblePreds[i] ?? 0;
    intervals.push({
      lower: pred + qLower,
      upper: pred + qUpper,
      confidenceLevel: 1 - alpha,
    });
  }

  return intervals;
}
