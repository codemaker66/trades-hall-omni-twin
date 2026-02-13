// ---------------------------------------------------------------------------
// Model Selection & Sample Complexity via PAC-Bayes / VC Theory
// ---------------------------------------------------------------------------

import type { ModelCandidate } from '../types.js';
import { pacBayesKLBound } from './bounds.js';

/**
 * Structural Risk Minimization (SRM) via PAC-Bayes bounds.
 *
 * Selects the model from a set of candidates that minimizes the
 * PAC-Bayes-kl generalization bound. This implements the SRM principle:
 * choose the model that best trades off empirical risk against complexity.
 *
 * For each candidate model, the PAC-Bayes-kl bound provides:
 *   R(rho) <= invert_kl(R_hat, [KL + ln(n/delta)] / n)
 *
 * The model with the smallest bound is selected.
 *
 * Practical guidelines for venue data:
 *   n=1K   -> linear models (low KL)
 *   n=5-10K -> small NNs / tree ensembles
 *   n=50K+ -> deep architectures with PAC-Bayes regularization
 *
 * @param candidates - Array of model candidates with empirical risk and KL divergence
 * @param n - Number of training samples
 * @param delta - Confidence parameter (default 0.05). Split across K models using
 *                union bound: delta_i = delta / K.
 * @returns The name of the selected model (with smallest PAC-Bayes bound)
 */
export function selectModelComplexity(
  candidates: ModelCandidate[],
  n: number,
  delta: number = 0.05,
): string {
  if (candidates.length === 0) return '';

  const k = candidates.length;
  // Union bound: use delta/k for each candidate
  const deltaPerModel = delta / k;

  let bestName = '';
  let bestBound = Infinity;

  for (let i = 0; i < k; i++) {
    const c = candidates[i]!;
    const bound = pacBayesKLBound(c.empRisk, c.klDiv, n, deltaPerModel);
    if (bound < bestBound) {
      bestBound = bound;
      bestName = c.name;
    }
  }

  return bestName;
}

/**
 * Compute minimum sample complexity for (epsilon, delta)-PAC learning.
 *
 * From VC theory, the number of samples needed to achieve:
 *   P(R(h) - R_hat(h) > epsilon) <= delta
 * for any hypothesis class with VC dimension d is:
 *
 *   m >= (1/epsilon) * (d * ln(1/epsilon) + ln(1/delta))
 *
 * This is the standard PAC learning sample complexity bound
 * (Blumer et al. 1989, Vapnik 1998).
 *
 * VC dimensions for common venue models:
 *   - Logistic regression (20 features): d = 21 -> m >= ~1,320
 *   - Decision tree (depth 5): d = 32 -> m >= ~640
 *   - Neural net: d = O(W*L*log W) -> VACUOUS
 *     (PAC-Bayes gives better bounds for NNs)
 *
 * @param vcDimension - VC dimension of the hypothesis class
 * @param epsilon - Desired accuracy (e.g. 0.05 for 5% error tolerance)
 * @param delta - Desired confidence (e.g. 0.05 for 95% confidence)
 * @returns Minimum number of training samples needed
 */
export function computeSampleComplexity(
  vcDimension: number,
  epsilon: number,
  delta: number,
): number {
  if (epsilon <= 0 || delta <= 0 || vcDimension < 0) {
    return Infinity;
  }

  // m >= (1/epsilon) * (d * ln(1/epsilon) + ln(1/delta))
  const m =
    (1 / epsilon) *
    (vcDimension * Math.log(1 / epsilon) + Math.log(1 / delta));

  return Math.ceil(Math.max(m, 1));
}

/**
 * Rademacher complexity-based generalization bound.
 *
 * Data-dependent bound that can be tighter than VC theory for
 * specific datasets. For linear models with bounded norms:
 *
 *   R(h) <= R_hat(h) + 2 * Rad_S(H) + 3 * sqrt(ln(2/delta) / (2n))
 *
 * where the Rademacher complexity for norm-bounded linear models is:
 *   Rad_S(H) = (normBound * dataRadius) / sqrt(n)
 *
 * This is norm-based, NOT dimension-based, so it can be meaningful
 * even for high-dimensional models (unlike VC bounds for NNs).
 *
 * @param empRisk - Empirical risk R_hat(h)
 * @param normBound - Upper bound on the hypothesis norm ||w|| (e.g. L2 norm of weights)
 * @param dataRadius - Upper bound on data norm ||x|| (maximum norm of feature vectors)
 * @param n - Number of training samples
 * @param delta - Confidence parameter (default 0.05)
 * @returns Upper bound on the true risk R(h)
 */
export function rademacherBound(
  empRisk: number,
  normBound: number,
  dataRadius: number,
  n: number,
  delta: number = 0.05,
): number {
  if (n <= 0) return Infinity;

  // Rademacher complexity for norm-bounded linear models
  const radComplexity = (normBound * dataRadius) / Math.sqrt(n);

  // Concentration term (from McDiarmid's inequality)
  const concentration = 3 * Math.sqrt(Math.log(2 / delta) / (2 * n));

  return empRisk + 2 * radComplexity + concentration;
}

/**
 * Recommend a model class based on available sample size.
 *
 * SRM decision table for venue platforms (from technique doc):
 *   n < 1,000       -> "linear": Linear/logistic regression
 *   1,000 <= n < 5,000 -> "tree-ensemble": Small tree ensemble (RF, GBM)
 *   5,000 <= n < 50,000 -> "small-nn": Small neural network or large ensemble
 *   n >= 50,000     -> "deep": Deep architectures with PAC-Bayes regularization
 *
 * These thresholds are derived from PAC-Bayes bound analysis:
 * - At n=1K with KL=50, PAC-Bayes-kl gives R <= ~0.30 (only linear viable)
 * - At n=10K with KL=50, PAC-Bayes-kl gives R <= ~0.133 (small NNs viable)
 * - At n=50K+, even large models get non-vacuous bounds
 *
 * @param n - Number of available training samples
 * @returns A recommendation string describing the appropriate model class
 */
export function recommendModel(n: number): string {
  if (n < 1000) {
    return 'linear: Use linear/logistic regression. With n<1000, complex models yield vacuous generalization bounds. PAC-Bayes with KL=50 gives R<=0.30 — only low-complexity models are justified.';
  }
  if (n < 5000) {
    return 'tree-ensemble: Use random forest or gradient boosting (depth<=5). Moderate sample size supports shallow tree ensembles. VC dimension ~32 for depth-5 trees requires ~640 samples at epsilon=0.05.';
  }
  if (n < 50000) {
    return 'small-nn: Use small neural network or large tree ensemble. PAC-Bayes bounds become non-vacuous (R<=0.133 at n=10K, KL=50). Validate with PAC-Bayes-kl to confirm generalization.';
  }
  return 'deep: Deep architectures are viable with PAC-Bayes regularization. At n>=50K, even models with KL=200 yield R<0.10. Use Bayes by Backprop or data-dependent priors to optimize the bound.';
}
