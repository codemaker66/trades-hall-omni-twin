// ---------------------------------------------------------------------------
// Multi-Calibration (Hébert-Johnson et al. 2018)
// Achieves calibration across multiple overlapping subgroups simultaneously.
// ---------------------------------------------------------------------------

import type { CalibrationResult } from '../types.js';

/**
 * Evaluate calibration quality using a reliability diagram approach.
 *
 * Bins predictions into nBins equal-width bins, computes:
 * - Per-bin: predicted mean, observed frequency, and count
 * - ECE (Expected Calibration Error): Σ (n_b/n) * |acc_b - conf_b|
 * - MCE (Maximum Calibration Error): max_b |acc_b - conf_b|
 * - Brier score: (1/n) Σ (p_i - y_i)²
 *
 * @param predictions - Predicted probabilities in [0, 1]
 * @param labels - Binary labels (0 or 1)
 * @param nBins - Number of bins for the reliability diagram
 * @returns CalibrationResult with bins, ECE, MCE, and Brier score
 */
export function evaluateCalibration(
  predictions: number[],
  labels: number[],
  nBins: number,
): CalibrationResult {
  const n = Math.min(predictions.length, labels.length);
  if (n === 0 || nBins <= 0) {
    return {
      bins: [],
      ece: 0,
      mce: 0,
      brier: 0,
    };
  }

  // Initialize bins
  const binPredSum: number[] = new Array<number>(nBins).fill(0);
  const binLabelSum: number[] = new Array<number>(nBins).fill(0);
  const binCount: number[] = new Array<number>(nBins).fill(0);

  // Assign each prediction to a bin
  for (let i = 0; i < n; i++) {
    const p = predictions[i] ?? 0;
    // Bin index: floor(p * nBins), clamped to [0, nBins-1]
    let bin = Math.floor(p * nBins);
    bin = Math.max(0, Math.min(nBins - 1, bin));
    binPredSum[bin] = (binPredSum[bin] ?? 0) + p;
    binLabelSum[bin] = (binLabelSum[bin] ?? 0) + (labels[i] ?? 0);
    binCount[bin] = (binCount[bin] ?? 0) + 1;
  }

  // Compute per-bin statistics and ECE/MCE
  const bins: CalibrationResult['bins'] = [];
  let ece = 0;
  let mce = 0;

  for (let b = 0; b < nBins; b++) {
    const count = binCount[b] ?? 0;
    if (count === 0) {
      bins.push({
        predictedMean: (b + 0.5) / nBins,
        observedFrequency: 0,
        count: 0,
      });
      continue;
    }

    const predictedMean = (binPredSum[b] ?? 0) / count;
    const observedFrequency = (binLabelSum[b] ?? 0) / count;
    const gap = Math.abs(observedFrequency - predictedMean);

    bins.push({ predictedMean, observedFrequency, count });

    ece += (count / n) * gap;
    if (gap > mce) mce = gap;
  }

  // Compute Brier score: (1/n) Σ (p_i - y_i)²
  let brier = 0;
  for (let i = 0; i < n; i++) {
    const diff = (predictions[i] ?? 0) - (labels[i] ?? 0);
    brier += diff * diff;
  }
  brier /= n;

  return { bins, ece, mce, brier };
}

/**
 * Multi-calibrate predictions across multiple overlapping subgroups.
 *
 * Hébert-Johnson et al. (2018) iterative algorithm:
 * 1. For each subgroup S and bin b, check if |E[Y|S,b] - E[p|S,b]| > alpha
 * 2. If so, adjust predictions in that subgroup/bin toward the observed frequency
 * 3. Repeat until no subgroup/bin pair violates alpha-calibration
 *
 * This ensures that the predictions are simultaneously calibrated on every
 * subgroup in the collection, not just the overall population.
 *
 * @param predictions - Initial predicted probabilities in [0, 1]
 * @param labels - Binary labels (0 or 1)
 * @param subgroupMasks - Boolean masks, subgroupMasks[g][i] = true if sample i is in subgroup g
 * @param alpha - Calibration tolerance (violations below alpha are acceptable)
 * @param maxIter - Maximum number of passes
 * @returns Adjusted predictions that are alpha-calibrated on all subgroups
 */
export function multiCalibrate(
  predictions: number[],
  labels: number[],
  subgroupMasks: boolean[][],
  alpha: number,
  maxIter: number,
): number[] {
  const n = Math.min(predictions.length, labels.length);
  if (n === 0) return [];

  const nGroups = subgroupMasks.length;
  const nBins = 10; // Standard number of calibration bins

  // Work on a copy
  const adjusted: number[] = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    adjusted[i] = predictions[i] ?? 0;
  }

  for (let iter = 0; iter < maxIter; iter++) {
    let anyViolation = false;

    // For each subgroup
    for (let g = 0; g < nGroups; g++) {
      const mask = subgroupMasks[g];
      if (!mask) continue;

      // Bin the predictions for this subgroup
      const binPredSum: number[] = new Array<number>(nBins).fill(0);
      const binLabelSum: number[] = new Array<number>(nBins).fill(0);
      const binCount: number[] = new Array<number>(nBins).fill(0);

      for (let i = 0; i < n; i++) {
        if (!(mask[i] ?? false)) continue;
        const p = adjusted[i] ?? 0;
        let bin = Math.floor(p * nBins);
        bin = Math.max(0, Math.min(nBins - 1, bin));
        binPredSum[bin] = (binPredSum[bin] ?? 0) + p;
        binLabelSum[bin] = (binLabelSum[bin] ?? 0) + (labels[i] ?? 0);
        binCount[bin] = (binCount[bin] ?? 0) + 1;
      }

      // Check each bin for violations and adjust
      for (let b = 0; b < nBins; b++) {
        const count = binCount[b] ?? 0;
        if (count === 0) continue;

        const predMean = (binPredSum[b] ?? 0) / count;
        const obsMean = (binLabelSum[b] ?? 0) / count;
        const gap = obsMean - predMean;

        if (Math.abs(gap) <= alpha) continue;

        // Violation found: adjust predictions in this subgroup/bin
        anyViolation = true;

        // Move predictions toward observed frequency by a fraction of the gap
        // Use a dampened step to ensure convergence
        const step = gap * 0.5;

        for (let i = 0; i < n; i++) {
          if (!(mask[i] ?? false)) continue;
          const p = adjusted[i] ?? 0;
          let bin = Math.floor(p * nBins);
          bin = Math.max(0, Math.min(nBins - 1, bin));
          if (bin === b) {
            // Adjust and clamp to [0, 1]
            adjusted[i] = Math.max(0, Math.min(1, p + step));
          }
        }
      }
    }

    if (!anyViolation) break;
  }

  return adjusted;
}
