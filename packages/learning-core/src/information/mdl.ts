// ---------------------------------------------------------------------------
// Minimum Description Length (MDL) Model Selection
// ---------------------------------------------------------------------------

import type { MDLResult } from '../types.js';

/**
 * MDL score (BIC equivalent): -loglik + (k/2) * log(n)
 *
 * The MDL principle selects the model that minimizes the total description
 * length, which is the sum of the model complexity and the data misfit.
 *
 * @param logLikelihood - Log-likelihood of the data given the model
 * @param nParams - Number of free parameters (k)
 * @param n - Number of data points
 * @returns MDL score (lower is better)
 */
export function mdlScore(logLikelihood: number, nParams: number, n: number): number {
  if (n <= 0) return Infinity;
  // MDL = -log L + (k/2) * log(n)
  return -logLikelihood + (nParams / 2) * Math.log(n);
}

/**
 * Select the best model according to MDL (BIC equivalent).
 *
 * Given a set of candidate models with their log-likelihoods and parameter counts,
 * returns the model with the smallest MDL score.
 *
 * @param models - Array of candidate models with name, logLikelihood, and nParams
 * @param n - Number of data points
 * @returns MDLResult with the winning model's index, complexity, data fit, and total length
 */
export function mdlSelect(
  models: Array<{ name: string; logLikelihood: number; nParams: number }>,
  n: number,
): MDLResult {
  if (models.length === 0) {
    return {
      modelIndex: -1,
      modelComplexity: 0,
      dataFit: 0,
      totalLength: Infinity,
    };
  }

  let bestIdx = 0;
  let bestScore = Infinity;

  for (let i = 0; i < models.length; i++) {
    const model = models[i]!;
    const score = mdlScore(model.logLikelihood, model.nParams, n);
    if (score < bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  }

  const best = models[bestIdx]!;
  const complexity = (best.nParams / 2) * Math.log(Math.max(n, 1));
  const dataFit = -best.logLikelihood;

  return {
    modelIndex: bestIdx,
    modelComplexity: complexity,
    dataFit,
    totalLength: bestScore,
  };
}

/**
 * Normalized Maximum Likelihood (NML) score with parametric complexity.
 *
 * NML is the minimax optimal universal code for a parametric model family.
 * For Gaussian linear regression:
 *   NML ≈ -log L_max + (k/2) * log(n/(2π)) + log(C_k)
 *
 * where C_k is the parametric complexity (we approximate with the stochastic
 * complexity penalty term).
 *
 * We compute:
 *   - Data fit: (n/2) * log(RSS/n) where RSS = Σ residuals²
 *   - Parametric complexity: (k/2) * log(n / (2π)) + (1/2) * log(k * π)
 *
 * @param residuals - Model residuals (y_true - y_pred)
 * @param nParams - Number of free parameters
 * @returns NML score (lower is better)
 */
export function normalizedMaximumLikelihood(
  residuals: number[],
  nParams: number,
): number {
  const n = residuals.length;
  if (n === 0) return Infinity;

  // Compute residual sum of squares
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const r = residuals[i] ?? 0;
    rss += r * r;
  }

  // Variance estimate (MLE)
  const sigmaSquared = rss / n;

  // Log-likelihood for Gaussian: -(n/2)*log(2π) - (n/2)*log(σ²) - n/2
  // Data fit component (negative log-likelihood):
  const dataFit = (n / 2) * Math.log(Math.max(sigmaSquared, 1e-300)) +
    (n / 2) * Math.log(2 * Math.PI) +
    n / 2;

  // Parametric complexity (Fisher information based):
  // (k/2) * log(n / (2π)) captures how much the parameter space is "usable"
  // given n data points. Add (1/2)*log(k*π) as the Rissanen correction.
  const k = Math.max(nParams, 1);
  const parametricComplexity = (k / 2) * Math.log(n / (2 * Math.PI)) +
    0.5 * Math.log(k * Math.PI);

  return dataFit + parametricComplexity;
}
