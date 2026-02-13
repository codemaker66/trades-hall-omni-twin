// ---------------------------------------------------------------------------
// Fairness Metrics
// ---------------------------------------------------------------------------

/**
 * Demographic Parity: |P(Ŷ=1|A=0) - P(Ŷ=1|A=1)|
 *
 * Measures the absolute difference in positive prediction rates between
 * two groups defined by a sensitive attribute. A value of 0 indicates
 * perfect demographic parity.
 *
 * @param predictions - Binary predictions (0 or 1), or probabilities thresholded at 0.5
 * @param sensitiveAttr - Binary sensitive attribute (0 or 1) for each sample
 * @returns Absolute difference in positive prediction rates
 */
export function demographicParity(
  predictions: number[],
  sensitiveAttr: number[],
): number {
  const n = Math.min(predictions.length, sensitiveAttr.length);
  if (n === 0) return 0;

  let posGroup0 = 0;
  let countGroup0 = 0;
  let posGroup1 = 0;
  let countGroup1 = 0;

  for (let i = 0; i < n; i++) {
    const pred = (predictions[i] ?? 0) >= 0.5 ? 1 : 0;
    const attr = sensitiveAttr[i] ?? 0;

    if (attr < 0.5) {
      countGroup0++;
      posGroup0 += pred;
    } else {
      countGroup1++;
      posGroup1 += pred;
    }
  }

  const rate0 = countGroup0 > 0 ? posGroup0 / countGroup0 : 0;
  const rate1 = countGroup1 > 0 ? posGroup1 / countGroup1 : 0;

  return Math.abs(rate0 - rate1);
}

/**
 * Equalized Odds: TPR and FPR differences across groups.
 *
 * Measures the difference in True Positive Rate and False Positive Rate
 * between two groups. Equalized odds requires both to be zero.
 *
 * TPR_diff = |TPR(A=0) - TPR(A=1)|
 * FPR_diff = |FPR(A=0) - FPR(A=1)|
 *
 * @param predictions - Binary predictions (0 or 1), or probabilities thresholded at 0.5
 * @param labels - True binary labels (0 or 1)
 * @param sensitiveAttr - Binary sensitive attribute (0 or 1)
 * @returns Object with tprDiff and fprDiff
 */
export function equalizedOdds(
  predictions: number[],
  labels: number[],
  sensitiveAttr: number[],
): { tprDiff: number; fprDiff: number } {
  const n = Math.min(predictions.length, labels.length, sensitiveAttr.length);
  if (n === 0) return { tprDiff: 0, fprDiff: 0 };

  // Group 0: tp, fn, fp, tn
  let tp0 = 0, fn0 = 0, fp0 = 0, tn0 = 0;
  // Group 1: tp, fn, fp, tn
  let tp1 = 0, fn1 = 0, fp1 = 0, tn1 = 0;

  for (let i = 0; i < n; i++) {
    const pred = (predictions[i] ?? 0) >= 0.5 ? 1 : 0;
    const label = (labels[i] ?? 0) >= 0.5 ? 1 : 0;
    const attr = sensitiveAttr[i] ?? 0;

    if (attr < 0.5) {
      // Group 0
      if (label === 1 && pred === 1) tp0++;
      else if (label === 1 && pred === 0) fn0++;
      else if (label === 0 && pred === 1) fp0++;
      else tn0++;
    } else {
      // Group 1
      if (label === 1 && pred === 1) tp1++;
      else if (label === 1 && pred === 0) fn1++;
      else if (label === 0 && pred === 1) fp1++;
      else tn1++;
    }
  }

  // TPR = TP / (TP + FN)
  const tpr0 = (tp0 + fn0) > 0 ? tp0 / (tp0 + fn0) : 0;
  const tpr1 = (tp1 + fn1) > 0 ? tp1 / (tp1 + fn1) : 0;

  // FPR = FP / (FP + TN)
  const fpr0 = (fp0 + tn0) > 0 ? fp0 / (fp0 + tn0) : 0;
  const fpr1 = (fp1 + tn1) > 0 ? fp1 / (fp1 + tn1) : 0;

  return {
    tprDiff: Math.abs(tpr0 - tpr1),
    fprDiff: Math.abs(fpr0 - fpr1),
  };
}

/**
 * Individual Fairness: maximum violation of the Lipschitz condition.
 *
 * For all pairs (i, j), checks |f(x_i) - f(x_j)| <= K * d(x_i, x_j).
 * Returns the maximum violation (how much the condition is exceeded).
 * A value of 0 means the model is individually fair with Lipschitz constant K.
 *
 * @param predictions - Model predictions (continuous values)
 * @param distances - Pairwise distance matrix, distances[i][j] = d(x_i, x_j)
 * @param lipschitzK - Lipschitz constant K
 * @returns Maximum violation: max(|f(x_i)-f(x_j)| - K*d(x_i,x_j), 0) over all pairs
 */
export function individualFairness(
  predictions: number[],
  distances: number[][],
  lipschitzK: number,
): number {
  const n = predictions.length;
  if (n <= 1) return 0;

  let maxViolation = 0;

  for (let i = 0; i < n; i++) {
    const distRow = distances[i];
    if (!distRow) continue;

    for (let j = i + 1; j < n; j++) {
      const predDiff = Math.abs((predictions[i] ?? 0) - (predictions[j] ?? 0));
      const dist = distRow[j] ?? 0;
      const allowed = lipschitzK * dist;
      const violation = predDiff - allowed;

      if (violation > maxViolation) {
        maxViolation = violation;
      }
    }
  }

  return Math.max(0, maxViolation);
}

/**
 * Disparate Impact: min(P(Ŷ=1|A=0)/P(Ŷ=1|A=1), P(Ŷ=1|A=1)/P(Ŷ=1|A=0))
 *
 * The ratio of positive prediction rates between groups. A value of 1
 * indicates perfect parity. The "four-fifths rule" considers ratios
 * below 0.8 as evidence of disparate impact.
 *
 * We take the minimum of both directions to make the metric symmetric.
 *
 * @param predictions - Binary predictions (0 or 1), or probabilities thresholded at 0.5
 * @param sensitiveAttr - Binary sensitive attribute (0 or 1)
 * @returns Disparate impact ratio in [0, 1]
 */
export function disparateImpact(
  predictions: number[],
  sensitiveAttr: number[],
): number {
  const n = Math.min(predictions.length, sensitiveAttr.length);
  if (n === 0) return 1;

  let posGroup0 = 0;
  let countGroup0 = 0;
  let posGroup1 = 0;
  let countGroup1 = 0;

  for (let i = 0; i < n; i++) {
    const pred = (predictions[i] ?? 0) >= 0.5 ? 1 : 0;
    const attr = sensitiveAttr[i] ?? 0;

    if (attr < 0.5) {
      countGroup0++;
      posGroup0 += pred;
    } else {
      countGroup1++;
      posGroup1 += pred;
    }
  }

  const rate0 = countGroup0 > 0 ? posGroup0 / countGroup0 : 0;
  const rate1 = countGroup1 > 0 ? posGroup1 / countGroup1 : 0;

  // Avoid division by zero
  if (rate0 <= 0 && rate1 <= 0) return 1; // Both zero: no disparity
  if (rate0 <= 0 || rate1 <= 0) return 0; // One zero, other nonzero: max disparity

  return Math.min(rate0 / rate1, rate1 / rate0);
}
