// ---------------------------------------------------------------------------
// Conformalized Quantile Regression (Romano, Patterson & Candes 2019)
// arXiv:1905.03222
// ---------------------------------------------------------------------------

import type { PredictionInterval } from '../types.js';

/**
 * Compute CQR non-conformity scores.
 *
 * The CQR score measures how much the true value falls outside the
 * estimated quantile interval:
 *   E_i = max(qLow_i - y_i, y_i - qHigh_i)
 *
 * If the true value is inside [qLow, qHigh], the score is negative.
 * If outside, the score is positive (indicating miscoverage).
 *
 * This enables heteroscedastic intervals: wider where the quantile model
 * is uncertain, tighter where it is confident.
 *
 * @param yTrue - True labels from calibration set
 * @param qLow - Predicted lower quantile (e.g. alpha/2 quantile)
 * @param qHigh - Predicted upper quantile (e.g. 1-alpha/2 quantile)
 * @returns Non-conformity scores E_i
 */
export function cqrScores(
  yTrue: number[],
  qLow: number[],
  qHigh: number[],
): number[] {
  const n = Math.min(yTrue.length, qLow.length, qHigh.length);
  const scores: number[] = [];
  for (let i = 0; i < n; i++) {
    const y = yTrue[i] ?? 0;
    const lo = qLow[i] ?? 0;
    const hi = qHigh[i] ?? 0;
    // E_i = max(qLow_i - y_i, y_i - qHigh_i)
    scores.push(Math.max(lo - y, y - hi));
  }
  return scores;
}

/**
 * CQR prediction: adjust quantile intervals using calibration scores.
 *
 * Given predicted quantile intervals [qLow, qHigh] for test points,
 * and the non-conformity scores from the calibration set, construct
 * conformalized intervals:
 *   [qLow_new - Q_{1-alpha}(E), qHigh_new + Q_{1-alpha}(E)]
 *
 * where Q_{1-alpha}(E) is the ceil((n+1)(1-alpha))/n quantile of scores.
 *
 * This gives finite-sample coverage guarantees while preserving the
 * heteroscedastic structure of the quantile regression intervals.
 *
 * @param qLow - Predicted lower quantiles for test points
 * @param qHigh - Predicted upper quantiles for test points
 * @param scores - Non-conformity scores from cqrScores() on calibration set
 * @param alpha - Miscoverage rate (e.g. 0.1 for 90% coverage)
 * @returns Adjusted prediction intervals with coverage guarantee
 */
export function cqrPredict(
  qLow: number[],
  qHigh: number[],
  scores: number[],
  alpha: number,
): PredictionInterval[] {
  const nCal = scores.length;
  if (nCal === 0) {
    // No calibration data: return raw quantile intervals
    const nTest = Math.min(qLow.length, qHigh.length);
    const intervals: PredictionInterval[] = [];
    for (let i = 0; i < nTest; i++) {
      intervals.push({
        lower: qLow[i] ?? 0,
        upper: qHigh[i] ?? 0,
        confidenceLevel: 1 - alpha,
      });
    }
    return intervals;
  }

  // Sort scores to find the quantile
  const sorted = [...scores].sort((a, b) => a - b);

  // Compute the conformal quantile with finite-sample correction
  // Level = ceil((n+1)(1-alpha)) / n
  const level = Math.ceil((nCal + 1) * (1 - alpha)) / nCal;

  let qHat: number;
  if (level >= 1) {
    // Need to cover everything: use max score (could be +Infinity in theory)
    qHat = sorted[nCal - 1] ?? 0;
  } else {
    const idx = Math.min(Math.ceil(level * nCal) - 1, nCal - 1);
    qHat = sorted[Math.max(idx, 0)] ?? 0;
  }

  // Construct adjusted intervals
  const nTest = Math.min(qLow.length, qHigh.length);
  const intervals: PredictionInterval[] = [];
  for (let i = 0; i < nTest; i++) {
    intervals.push({
      lower: (qLow[i] ?? 0) - qHat,
      upper: (qHigh[i] ?? 0) + qHat,
      confidenceLevel: 1 - alpha,
    });
  }

  return intervals;
}
