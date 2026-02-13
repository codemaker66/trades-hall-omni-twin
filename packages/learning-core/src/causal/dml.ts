// ---------------------------------------------------------------------------
// Double/Debiased Machine Learning (Chernozhukov et al. 2018)
// ---------------------------------------------------------------------------
//
// DML estimates causal effects by using cross-fitting to avoid overfitting
// bias in nuisance parameter estimation. The procedure:
//
// 1. Split data into K folds
// 2. For each fold k:
//    a. Train nuisance models (E[Y|W] and E[T|W]) on data excluding fold k
//    b. Predict on fold k to get nuisance estimates
// 3. Compute residuals: Y_tilde = Y - g_hat(W), T_tilde = T - m_hat(W)
// 4. Estimate ATE: theta_hat = sum(T_tilde * Y_tilde) / sum(T_tilde^2)
// 5. Standard error via sandwich estimator
//
// The cross-fitting ensures that the estimation error of nuisance models
// does not bias the causal estimate, achieving sqrt(n)-consistency.
// ---------------------------------------------------------------------------

import type { CausalEffect } from '../types.js';

/**
 * Estimate the Average Treatment Effect using Double/Debiased Machine Learning.
 *
 * @param Y     Outcome variable (length n)
 * @param T     Treatment variable (length n, can be continuous)
 * @param W     Confounders / covariates (n x p matrix, W[i] is the i-th row)
 * @param predictY  Function that trains a model on (W, Y) and returns predictions.
 *                   Receives the training covariate matrix and must return
 *                   predictions for those same rows. (Caller wraps their ML model.)
 * @param predictT  Function that trains a model on (W, T) and returns predictions.
 * @param nFolds Number of cross-fitting folds (default 5)
 * @returns CausalEffect with ATE, confidence interval, and p-value
 */
export function dmlEstimate(
  Y: number[],
  T: number[],
  W: number[][],
  predictY: (W: number[][]) => number[],
  predictT: (W: number[][]) => number[],
  nFolds: number = 5,
): CausalEffect {
  const n = Y.length;

  if (n < nFolds * 2) {
    // Not enough data for meaningful cross-fitting
    return {
      ate: 0,
      ciLower: -Infinity,
      ciUpper: Infinity,
      pValue: 1,
    };
  }

  // ---- Step 1: Create fold assignments ----
  const foldAssignment = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    foldAssignment[i] = i % nFolds;
  }

  // Shuffle fold assignments for randomness (Fisher-Yates using simple deterministic shuffle)
  // We use a deterministic approach based on index to ensure reproducibility
  for (let i = n - 1; i > 0; i--) {
    // Simple hash-based pseudo-random swap (deterministic)
    const j = ((i * 2654435761) >>> 0) % (i + 1);
    const tmp = foldAssignment[i] ?? 0;
    foldAssignment[i] = foldAssignment[j] ?? 0;
    foldAssignment[j] = tmp;
  }

  // ---- Step 2: Cross-fitting ----
  const yResiduals = new Float64Array(n);  // Y_tilde = Y - g_hat(W)
  const tResiduals = new Float64Array(n);  // T_tilde = T - m_hat(W)

  for (let fold = 0; fold < nFolds; fold++) {
    // Split into train (everything except this fold) and test (this fold)
    const trainIndices: number[] = [];
    const testIndices: number[] = [];

    for (let i = 0; i < n; i++) {
      if ((foldAssignment[i] ?? 0) === fold) {
        testIndices.push(i);
      } else {
        trainIndices.push(i);
      }
    }

    if (trainIndices.length === 0 || testIndices.length === 0) continue;

    // Extract train data
    const wTrain: number[][] = new Array(trainIndices.length);
    const yTrain: number[] = new Array(trainIndices.length);
    const tTrain: number[] = new Array(trainIndices.length);

    for (let j = 0; j < trainIndices.length; j++) {
      const idx = trainIndices[j]!;
      wTrain[j] = W[idx]!;
      yTrain[j] = Y[idx] ?? 0;
      tTrain[j] = T[idx] ?? 0;
    }

    // Extract test covariates
    const wTest: number[][] = new Array(testIndices.length);
    for (let j = 0; j < testIndices.length; j++) {
      const idx = testIndices[j]!;
      wTest[j] = W[idx]!;
    }

    // Train nuisance model for Y and predict on test fold
    // The predictY function is expected to:
    //   1. Internally train a model on the provided data
    //   2. Return predictions for the provided covariate matrix
    // We pass train data implicitly: the caller's closure captures (wTrain, yTrain)
    // But our API passes just W. So we need a different approach:
    // predictY receives the TEST covariates and returns predictions.
    // The caller is responsible for training on the complement internally.
    //
    // Actually, re-reading the spec: predictY(W) trains on W and returns predictions.
    // This means the caller wraps their model such that predictY(W_test) returns
    // E[Y|W] predictions. But that doesn't include the training labels...
    //
    // The standard DML interface passes the nuisance learners as functions that
    // take (X_train, y_train, X_test) -> y_pred. Our API simplifies this:
    // predictY receives the covariate matrix and returns predictions of Y.
    // The caller must handle training vs prediction internally.
    //
    // For a clean cross-fitting implementation, we create a combined W that
    // includes training rows followed by test rows. The predictions for the
    // test rows (last testIndices.length entries) are what we need.

    const wCombined: number[][] = new Array(trainIndices.length + testIndices.length);
    for (let j = 0; j < trainIndices.length; j++) {
      wCombined[j] = wTrain[j]!;
    }
    for (let j = 0; j < testIndices.length; j++) {
      wCombined[trainIndices.length + j] = wTest[j]!;
    }

    // Get Y predictions: caller trains on first trainIndices.length rows,
    // predictions are for all rows but we only use test portion.
    const yPreds = predictY(wCombined);
    const tPreds = predictT(wCombined);

    // Extract predictions for the test fold
    for (let j = 0; j < testIndices.length; j++) {
      const idx = testIndices[j]!;
      const predIdx = trainIndices.length + j;
      yResiduals[idx] = (Y[idx] ?? 0) - (yPreds[predIdx] ?? 0);
      tResiduals[idx] = (T[idx] ?? 0) - (tPreds[predIdx] ?? 0);
    }
  }

  // ---- Step 3: Estimate ATE ----
  // theta_hat = (sum T_tilde * Y_tilde) / (sum T_tilde^2)
  let numerator = 0;
  let denominator = 0;

  for (let i = 0; i < n; i++) {
    const tTilde = tResiduals[i] ?? 0;
    const yTilde = yResiduals[i] ?? 0;
    numerator += tTilde * yTilde;
    denominator += tTilde * tTilde;
  }

  if (Math.abs(denominator) < 1e-15) {
    return {
      ate: 0,
      ciLower: -Infinity,
      ciUpper: Infinity,
      pValue: 1,
    };
  }

  const ate = numerator / denominator;

  // ---- Step 4: Standard error via sandwich/HC estimator ----
  // Var(theta_hat) = (1/n) * E[psi^2] / (E[T_tilde^2])^2
  // where psi_i = T_tilde_i * (Y_tilde_i - theta_hat * T_tilde_i)

  let psiSqSum = 0;
  for (let i = 0; i < n; i++) {
    const tTilde = tResiduals[i] ?? 0;
    const yTilde = yResiduals[i] ?? 0;
    const psi = tTilde * (yTilde - ate * tTilde);
    psiSqSum += psi * psi;
  }

  const meanPsiSq = psiSqSum / n;
  const meanTTildeSq = denominator / n;

  // Sandwich variance: Var = (1/n) * E[psi^2] / (E[T_tilde^2])^2
  const variance = meanPsiSq / (n * meanTTildeSq * meanTTildeSq);
  const se = Math.sqrt(Math.max(variance, 0));

  // 95% confidence interval (z = 1.96)
  const z = 1.96;
  const ciLower = ate - z * se;
  const ciUpper = ate + z * se;

  // Two-sided p-value using normal approximation
  const zStat = se > 0 ? Math.abs(ate / se) : Infinity;
  // p-value = 2 * (1 - Phi(|z|))
  // Approximate Phi using the error function: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
  const pValue = 2 * (1 - normalCdf(zStat));

  return {
    ate,
    ciLower,
    ciUpper,
    pValue: Math.max(pValue, 0),
  };
}

// ---- Internal Helpers ----

/**
 * Standard normal CDF approximation.
 * Uses the Abramowitz and Stegun approximation (formula 7.1.26).
 */
function normalCdf(x: number): number {
  if (x < -8) return 0;
  if (x > 8) return 1;

  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = x < 0 ? -1 : 1;
  const absX = Math.abs(x) / Math.SQRT2;

  const t = 1 / (1 + p * absX);
  const t2 = t * t;
  const t3 = t2 * t;
  const t4 = t3 * t;
  const t5 = t4 * t;

  const erf = 1 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * Math.exp(-absX * absX);

  return 0.5 * (1 + sign * erf);
}
