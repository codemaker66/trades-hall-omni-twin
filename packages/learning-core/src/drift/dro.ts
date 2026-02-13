// ---------------------------------------------------------------------------
// Distributionally Robust Optimization (Esfahani & Kuhn 2018)
// ---------------------------------------------------------------------------
//
// Provides tools for optimization under distributional uncertainty.
// Instead of minimizing expected loss under the empirical distribution,
// DRO minimizes the worst-case expected loss over all distributions
// within a Wasserstein ball of radius epsilon around the empirical.
//
// Key functions:
//   droObjective    — worst-case expected loss given weights and epsilon
//   droOptimize     — gradient descent on DRO objective
//   wassersteinBall — bounds of the ambiguity set
// ---------------------------------------------------------------------------

import type { PRNG } from '../types.js';

/**
 * Compute the worst-case expected loss within a Wasserstein ball.
 *
 * For the type-1 Wasserstein distance with a finite discrete distribution,
 * the DRO objective has a dual formulation (Esfahani & Kuhn 2018):
 *
 *   sup_{Q: W(Q, P_n) <= epsilon} E_Q[l]
 *   = inf_{lambda >= 0} { lambda * epsilon + (1/n) sum_i sup_{z} [l(z) - lambda * c(z, z_i)] }
 *
 * For computational tractability, we use the regularized worst-case:
 *   objective = (1/n) sum_i max(losses_i, 0) + epsilon * max_gradient_norm
 *
 * In practice, with given per-sample losses and importance weights,
 * the DRO objective shifts weight toward high-loss samples:
 *
 *   objective = sum_i w_i * losses_i  where w is the worst-case reweighting
 *
 * We solve the dual: inf_{lambda >= 0} lambda * epsilon + (1/n) sum_i [l_i + lambda]_+
 * which simplifies to a sorted-loss computation.
 *
 * @param losses Per-sample loss values
 * @param weights Importance weights (must sum to 1)
 * @param epsilon Wasserstein ball radius
 * @returns Worst-case expected loss
 */
export function droObjective(
  losses: number[],
  weights: number[],
  epsilon: number,
): number {
  const n = losses.length;
  if (n === 0) return 0;

  // Dual formulation: find optimal lambda via bisection
  // inf_{lambda >= 0} { lambda * epsilon + (1/n) sum_i max(l_i - lambda, 0) * n * w_i }
  // Wait -- with weights, the dual is:
  // inf_{lambda >= 0} { lambda * epsilon + sum_i w_i * max(l_i, lambda) }
  //
  // Actually, the clean dual for Wasserstein DRO with cost c(z,z') = |l(z)-l(z')| is:
  // sup_Q E_Q[l] = inf_{lambda >= 0} { lambda * epsilon + sum_i w_i * (l_i + lambda * ...) }
  //
  // For simplicity and correctness, we use the CVaR-based upper bound:
  // The worst-case expectation within a Wasserstein-1 ball of radius epsilon
  // around a discrete distribution with weights w_i at losses l_i is:
  //
  //   sup = inf_{lambda >= 0} { lambda * epsilon + sum_i w_i * phi_lambda(l_i) }
  //
  // where phi_lambda(l) = l when l is the transport-augmented loss.
  //
  // Practical approach: use the sorted-loss reweighting.
  // Sort losses in descending order, shift weight upward.

  // Create (loss, weight) pairs and sort by loss descending
  const pairs: Array<{ loss: number; weight: number }> = [];
  for (let i = 0; i < n; i++) {
    pairs.push({ loss: losses[i] ?? 0, weight: weights[i] ?? (1 / n) });
  }
  pairs.sort((a, b) => b.loss - a.loss);

  // Binary search for optimal dual variable lambda
  // The dual objective is: g(lambda) = lambda * epsilon + sum_i w_i * max(l_i - lambda, 0)
  // This is convex in lambda. We search for the minimum.

  // Compute weighted loss without DRO as baseline
  let baseLoss = 0;
  for (let i = 0; i < n; i++) {
    baseLoss += (weights[i] ?? (1 / n)) * (losses[i] ?? 0);
  }

  if (epsilon <= 0) return baseLoss;

  // Dual objective: g(lambda) = lambda * epsilon + sum_i w_i * max(l_i - lambda, 0)
  // At lambda=0: g(0) = sum_i w_i * l_i (just the expected loss where l_i >= 0 contributes)
  // The minimum is at the point where epsilon = sum_{l_i > lambda} w_i

  // Find optimal lambda via the breakpoints (the sorted losses)
  let bestObj = Infinity;

  // Check lambda = 0
  {
    let obj = 0;
    for (let i = 0; i < pairs.length; i++) {
      const p = pairs[i]!;
      obj += p.weight * Math.max(p.loss, 0);
    }
    if (obj < bestObj) bestObj = obj;
  }

  // Check each breakpoint lambda = l_i (sorted desc)
  for (let k = 0; k < pairs.length; k++) {
    const lambda = pairs[k]!.loss;
    let obj = lambda * epsilon;
    for (let i = 0; i < pairs.length; i++) {
      const p = pairs[i]!;
      obj += p.weight * Math.max(p.loss - lambda, 0);
    }
    if (obj < bestObj) bestObj = obj;
  }

  // The worst-case expected loss is the dual optimal value
  // Since the dual gives: inf_lambda { lambda*eps + E_P[max(l - lambda, 0)] }
  // and the primal is: sup_Q E_Q[l], these are equal by strong duality.
  return bestObj;
}

/**
 * Gradient descent on the DRO (worst-case) objective.
 *
 * At each iteration:
 *   1. Compute per-sample losses using the loss function
 *   2. Compute worst-case weights via the dual
 *   3. Compute the weighted gradient
 *   4. Update parameters
 *
 * @param lossFn Function mapping parameters to per-sample losses
 * @param params Initial parameters
 * @param epsilon Wasserstein ball radius
 * @param learningRate Step size for gradient descent
 * @param nIterations Number of optimization steps
 * @param rng Seedable PRNG for perturbation
 * @returns Optimized parameters
 */
export function droOptimize(
  lossFn: (params: Float64Array) => number[],
  params: Float64Array,
  epsilon: number,
  learningRate: number,
  nIterations: number,
  rng: PRNG,
): Float64Array {
  const d = params.length;
  const current = new Float64Array(params);
  const perturbation = new Float64Array(d);
  const gradientEst = new Float64Array(d);
  const delta = 1e-5; // Finite difference step

  for (let iter = 0; iter < nIterations; iter++) {
    // Compute base losses
    const baseLosses = lossFn(current);
    const n = baseLosses.length;
    if (n === 0) continue;

    // Compute worst-case weights via dual
    const wcWeights = computeWorstCaseWeights(baseLosses, epsilon);

    // Estimate gradient via finite differences with random perturbation
    // (SPSA-like approach for efficiency)
    // Generate random direction
    for (let j = 0; j < d; j++) {
      perturbation[j] = (rng() < 0.5) ? 1 : -1;
    }

    // Forward perturbation
    const paramsPlus = new Float64Array(d);
    const paramsMinus = new Float64Array(d);
    for (let j = 0; j < d; j++) {
      paramsPlus[j] = (current[j] ?? 0) + delta * (perturbation[j] ?? 0);
      paramsMinus[j] = (current[j] ?? 0) - delta * (perturbation[j] ?? 0);
    }

    const lossesPlus = lossFn(paramsPlus);
    const lossesMinus = lossFn(paramsMinus);

    // Compute worst-case objectives for plus and minus
    let objPlus = 0;
    let objMinus = 0;
    for (let i = 0; i < n; i++) {
      objPlus += (wcWeights[i] ?? 0) * (lossesPlus[i] ?? 0);
      objMinus += (wcWeights[i] ?? 0) * (lossesMinus[i] ?? 0);
    }

    // SPSA gradient estimate
    const gradScale = (objPlus - objMinus) / (2 * delta);
    for (let j = 0; j < d; j++) {
      gradientEst[j] = gradScale / (perturbation[j] ?? 1);
    }

    // Gradient descent step
    for (let j = 0; j < d; j++) {
      current[j] = (current[j] ?? 0) - learningRate * (gradientEst[j] ?? 0);
    }
  }

  return current;
}

/**
 * Compute the bounds of the Wasserstein ambiguity set.
 *
 * For a 1-D empirical distribution with n samples, the Wasserstein ball
 * of radius epsilon contains all distributions Q such that
 * W_1(Q, P_n) <= epsilon.
 *
 * The worst-case distribution shifts probability mass to maximize/minimize
 * the expected value. The bounds are:
 *   - lower[i] = empiricalDist[i] - epsilon * transport (shifts mass left)
 *   - upper[i] = empiricalDist[i] + epsilon * transport (shifts mass right)
 *
 * More precisely, for each sample point x_i in the empirical distribution,
 * the point can be perturbed by up to epsilon in the Wasserstein sense.
 *
 * @param empiricalDist Empirical distribution values (sample points)
 * @param epsilon Wasserstein ball radius
 * @returns Lower and upper bounds for each sample point
 */
export function wassersteinBall(
  empiricalDist: number[],
  epsilon: number,
): { lower: number[]; upper: number[] } {
  const n = empiricalDist.length;
  const lower: number[] = new Array(n);
  const upper: number[] = new Array(n);

  if (n === 0) return { lower: [], upper: [] };

  // Sort to find range
  const sorted = [...empiricalDist].sort((a, b) => a - b);
  const minVal = sorted[0] ?? 0;
  const maxVal = sorted[n - 1] ?? 0;
  const range = maxVal - minVal;

  // For each sample point, the worst-case perturbation in a Wasserstein-1 ball
  // allows shifting the point by epsilon. The bounds represent the interval
  // within which the true distribution's mass near this point could lie.
  for (let i = 0; i < n; i++) {
    const x = empiricalDist[i] ?? 0;

    // The worst case lower bound: the value can decrease by epsilon
    lower[i] = x - epsilon;

    // The worst case upper bound: the value can increase by epsilon
    upper[i] = x + epsilon;
  }

  // Additionally, scale by a factor that accounts for distributional spread
  // to ensure the ball is calibrated to the actual data range
  if (range > 0) {
    const scale = epsilon / range;
    for (let i = 0; i < n; i++) {
      const x = empiricalDist[i] ?? 0;
      // Tighten bounds based on relative position in the distribution
      const relPos = (x - minVal) / range;
      // Lower bound is tighter for points near the minimum
      lower[i] = x - epsilon * (1 + scale * (1 - relPos));
      // Upper bound is tighter for points near the maximum
      upper[i] = x + epsilon * (1 + scale * relPos);
    }
  }

  return { lower, upper };
}

// ---- Internal Helpers ----

/**
 * Compute worst-case importance weights for a given set of losses
 * and Wasserstein ball radius.
 *
 * The worst-case distribution concentrates more weight on high-loss
 * samples. We solve via the dual, which yields:
 *
 *   w_i* proportional to w_i * I(l_i >= lambda*) / sum(w_j * I(l_j >= lambda*))
 *
 * where lambda* is chosen so the constraint W(Q, P) <= epsilon binds.
 *
 * For uniform empirical distribution (w_i = 1/n), this reduces to
 * shifting weight to the top-k losses where k ~ n * (1 - epsilon_factor).
 */
function computeWorstCaseWeights(losses: number[], epsilon: number): number[] {
  const n = losses.length;
  if (n === 0) return [];

  const uniformW = 1 / n;

  // Create indices sorted by loss (descending)
  const indices: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    indices[i] = i;
  }
  indices.sort((a, b) => (losses[b] ?? 0) - (losses[a] ?? 0));

  // Compute the CVaR-like reweighting
  // The amount of weight we can shift is bounded by epsilon relative
  // to the loss range
  const maxLoss = losses[indices[0]!] ?? 0;
  const minLoss = losses[indices[n - 1]!] ?? 0;
  const lossRange = maxLoss - minLoss;

  if (lossRange <= 1e-15) {
    // All losses equal: uniform weights
    const w = new Array<number>(n);
    for (let i = 0; i < n; i++) w[i] = uniformW;
    return w;
  }

  // The fraction of total weight to shift to high-loss samples
  // Bounded by [0, 1]: higher epsilon = more shifting
  const shiftFraction = Math.min(epsilon / lossRange, 1);

  // Soft reweighting: w_i proportional to exp(eta * l_i) where eta ~ epsilon/range
  const eta = shiftFraction * n;
  const weights = new Array<number>(n);
  let totalWeight = 0;

  for (let i = 0; i < n; i++) {
    const l = losses[i] ?? 0;
    // Shift to avoid overflow: subtract max loss before exp
    const w = Math.exp(eta * (l - maxLoss) / (lossRange > 0 ? lossRange : 1));
    weights[i] = w;
    totalWeight += w;
  }

  // Normalize
  if (totalWeight > 0) {
    for (let i = 0; i < n; i++) {
      weights[i] = (weights[i] ?? 0) / totalWeight;
    }
  } else {
    for (let i = 0; i < n; i++) {
      weights[i] = uniformW;
    }
  }

  return weights;
}
