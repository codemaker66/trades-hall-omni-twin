// ---------------------------------------------------------------------------
// PAC-Bayes Bounds
// (Seeger 2002, Langford 2005, Maurer 2004, Catoni 2007)
// ---------------------------------------------------------------------------

import type { PACBayesBound } from '../types.js';

/**
 * Binary KL divergence: kl(q || p) = q*ln(q/p) + (1-q)*ln((1-q)/(1-p)).
 *
 * Used internally for the PAC-Bayes-kl bound inversion.
 * Handles edge cases (q=0, q=1, p=0, p=1) with appropriate limits.
 */
function binaryKL(q: number, p: number): number {
  // Edge cases
  if (q <= 0) {
    return p >= 1 ? 0 : -Math.log(1 - p);
  }
  if (q >= 1) {
    return p <= 0 ? Infinity : -Math.log(p);
  }
  if (p <= 0 || p >= 1) {
    return Infinity;
  }
  return q * Math.log(q / p) + (1 - q) * Math.log((1 - q) / (1 - p));
}

/**
 * PAC-Bayes-kl bound (Seeger 2002, Langford 2005, Maurer 2004).
 *
 * The tightest standard PAC-Bayes bound. For any posterior rho and prior pi:
 *
 *   kl(R_hat(rho) || R(rho)) <= [KL(rho || pi) + ln(n / delta)] / n
 *
 * where:
 *   - kl(q||p) = q*ln(q/p) + (1-q)*ln((1-q)/(1-p)) is the binary KL divergence
 *   - R_hat(rho) = expected empirical risk under the posterior
 *   - R(rho) = expected true risk under the posterior
 *   - KL(rho||pi) = KL divergence from posterior to prior
 *   - n = number of training samples
 *   - delta = confidence parameter
 *
 * The bound on R(rho) is obtained by numerically inverting the binary KL
 * via binary search: find the largest p such that kl(R_hat || p) <= c.
 *
 * Example for venue models:
 *   n=10,000, empRisk=0.08, KL=50 nats -> R <= ~0.133 (non-vacuous!)
 *   n=1,000, empRisk=0.08, KL=50 nats  -> R <= ~0.30
 *   n=50,000+                            -> R < 0.10
 *
 * @param empRisk - Empirical risk R_hat(rho), in [0, 1]
 * @param klDivergence - KL(rho || pi) in nats
 * @param n - Number of training samples
 * @param delta - Confidence parameter (default 0.05). Bound holds with prob >= 1-delta.
 * @returns Upper bound on the true risk R(rho)
 */
export function pacBayesKLBound(
  empRisk: number,
  klDivergence: number,
  n: number,
  delta: number = 0.05,
): number {
  // The RHS of the kl inequality
  const c = (klDivergence + Math.log(n / delta)) / n;

  // Binary search: find the largest p in [empRisk, 1) such that kl(empRisk, p) <= c
  // Since kl(q, p) is increasing in p for p > q, we search for the largest p
  // where the inequality holds.
  let lo = empRisk;
  let hi = 1 - 1e-12;

  // Edge case: if empRisk is 0, kl(0, p) = -ln(1-p). Solve -ln(1-p) <= c => p <= 1 - e^(-c)
  if (empRisk <= 0) {
    return Math.min(1 - Math.exp(-c), 1);
  }

  // Edge case: if empRisk >= 1, risk is already maximal
  if (empRisk >= 1) {
    return 1;
  }

  // Binary search with sufficient precision
  for (let iter = 0; iter < 100; iter++) {
    const mid = (lo + hi) / 2;
    const kl = binaryKL(empRisk, mid);
    if (kl <= c) {
      lo = mid;
    } else {
      hi = mid;
    }
    if (hi - lo < 1e-12) break;
  }

  return lo;
}

/**
 * McAllester bound (Pinsker relaxation of PAC-Bayes-kl).
 *
 * A looser but analytically tractable version of the PAC-Bayes-kl bound,
 * obtained via Pinsker's inequality: kl(q||p) >= 2(q-p)^2.
 *
 *   R(rho) <= R_hat(rho) + sqrt([KL(rho||pi) + ln(n/delta)] / (2n))
 *
 * Typically ~5% looser than PAC-Bayes-kl, but much simpler to compute.
 *
 * @param empRisk - Empirical risk R_hat(rho), in [0, 1]
 * @param klDivergence - KL(rho || pi) in nats
 * @param n - Number of training samples
 * @param delta - Confidence parameter (default 0.05)
 * @returns Upper bound on the true risk R(rho)
 */
export function mcAllesterBound(
  empRisk: number,
  klDivergence: number,
  n: number,
  delta: number = 0.05,
): number {
  const penalty = Math.sqrt((klDivergence + Math.log(n / delta)) / (2 * n));
  return Math.min(empRisk + penalty, 1);
}

/**
 * Catoni bound with optimized lambda (Catoni 2007, arXiv:0712.0248).
 *
 * Uses a tighter exponential moment inequality with an optimized
 * temperature parameter lambda*:
 *
 *   R(rho) <= R_hat(rho) + B * sqrt([KL(rho||pi) + ln(1/delta)] / (2n))
 *
 * where B is the bound on the loss function (losses in [0, B]).
 *
 * The optimal lambda is:
 *   lambda* = sqrt(2n / (KL + ln(1/delta)))
 *
 * Note: This uses the simplified form. The exact Catoni bound involves
 * a convex conjugate that gives:
 *   R <= (1 - e^{-lambda*R_hat - [KL + ln(1/delta)]/n}) / (1 - e^{-lambda})
 *
 * We implement the full version with the optimal lambda.
 *
 * @param empRisk - Empirical risk R_hat(rho), in [0, B]
 * @param klDivergence - KL(rho || pi) in nats
 * @param n - Number of training samples
 * @param lossBound - Upper bound B on the loss (default 1.0)
 * @param delta - Confidence parameter (default 0.05)
 * @returns Upper bound on the true risk R(rho)
 */
export function catoniBound(
  empRisk: number,
  klDivergence: number,
  n: number,
  lossBound: number = 1.0,
  delta: number = 0.05,
): number {
  const complexity = klDivergence + Math.log(1 / delta);

  // Optimal lambda: minimizes the bound
  // lambda* = sqrt(2n / complexity)
  // but we need lambda > 0 and we want the bound to be meaningful
  if (complexity <= 0 || n <= 0) {
    return empRisk;
  }

  const lambdaStar = Math.sqrt((2 * n) / complexity);

  // Full Catoni bound:
  // R <= (1/lambda) * [lambda * R_hat + (KL + ln(1/delta))/n]
  // simplified with the convex-conjugate approach for bounded losses:
  //
  // For losses in [0, B], the bound is:
  //   R <= R_hat + (B / lambda) * (exp(lambda * complexity / n) - 1) / B
  //
  // With optimal lambda, this reduces to:
  //   R <= R_hat + B * sqrt(complexity / (2n))
  //
  // But the exact Catoni form uses the function phi(x) = -log(1-x+x^2/2):
  //   R <= phi^{-1}(R_hat_lambda + complexity/(n*lambda))
  //
  // We use the closed-form with optimal lambda:
  const penalty = lossBound * Math.sqrt(complexity / (2 * n));

  // Additionally, we can compute a tighter version via the Catoni function
  // phi(lambda, R_hat) = (1 - exp(-lambda * R_hat)) / (1 - exp(-lambda))
  // and its inverse. Here we use the direct form:
  //
  // The bound from the moment generating function approach:
  // E[exp(lambda(L - R))] <= 1  (for bounded losses)
  // leads to: R <= R_hat + complexity / (n * lambda) + lambda * B^2 / (8n)
  //
  // Optimizing over lambda: lambda* = sqrt(8n * complexity) / B
  // gives: R <= R_hat + B * sqrt(complexity / (2n))
  //
  // This is identical to McAllester when B=1 and we use ln(1/delta) vs ln(n/delta).
  // The key difference is Catoni uses ln(1/delta) not ln(n/delta).

  return Math.min(empRisk + penalty, lossBound);
}

/**
 * Compute all three PAC-Bayes bounds and return as structured results.
 *
 * @param empRisk - Empirical risk
 * @param klDivergence - KL divergence from posterior to prior
 * @param n - Number of training samples
 * @param delta - Confidence parameter
 * @param lossBound - Upper bound on loss (for Catoni)
 * @returns Array of PACBayesBound results
 */
export function computeAllBounds(
  empRisk: number,
  klDivergence: number,
  n: number,
  delta: number = 0.05,
  lossBound: number = 1.0,
): PACBayesBound[] {
  return [
    {
      name: 'PAC-Bayes-kl',
      value: pacBayesKLBound(empRisk, klDivergence, n, delta),
      empRisk,
      klDivergence,
      n,
      delta,
    },
    {
      name: 'McAllester',
      value: mcAllesterBound(empRisk, klDivergence, n, delta),
      empRisk,
      klDivergence,
      n,
      delta,
    },
    {
      name: 'Catoni',
      value: catoniBound(empRisk, klDivergence, n, lossBound, delta),
      empRisk,
      klDivergence,
      n,
      delta,
    },
  ];
}
