// ---------------------------------------------------------------------------
// Synthetic Control Method (Abadie, Diamond & Hainmueller 2010)
// ---------------------------------------------------------------------------
//
// Estimates the causal effect of an intervention on a single treated unit
// by constructing a synthetic counterfactual from a weighted combination
// of untreated "donor" units.
//
// The weights w* are found by minimizing:
//   ||Y_treated_pre - sum_j w_j * Y_j_pre||_2
//   subject to: w_j >= 0 for all j, sum_j w_j = 1
//
// The causal effect is estimated as:
//   effect_t = Y_treated_post_t - sum_j w*_j * Y_j_post_t
//
// Statistical inference is via placebo tests: apply the same procedure
// to each donor unit as if it were treated, and compare the treated
// unit's effect to the placebo distribution.
// ---------------------------------------------------------------------------

import type { PRNG, SyntheticControlResult } from '../types.js';

/**
 * Estimate the causal effect using the Synthetic Control Method.
 *
 * @param treatedPre    Pre-intervention outcomes for the treated unit (length T_pre)
 * @param treatedPost   Post-intervention outcomes for the treated unit (length T_post)
 * @param donorsPre     Pre-intervention outcomes for donor units (J x T_pre)
 *                      donorsPre[j][t] = outcome for donor j at pre-period t
 * @param donorsPost    Post-intervention outcomes for donor units (J x T_post)
 * @param nPlacebo      Number of placebo tests to run (default: all donors)
 * @param rng           Optional PRNG (not used in deterministic version)
 * @returns SyntheticControlResult with weights, effects, and confidence intervals
 */
export function syntheticControl(
  treatedPre: number[],
  treatedPost: number[],
  donorsPre: number[][],
  donorsPost: number[][],
  nPlacebo?: number,
  rng?: PRNG,
): SyntheticControlResult {
  const J = donorsPre.length;     // Number of donor units
  const Tpre = treatedPre.length; // Pre-intervention periods
  const Tpost = treatedPost.length; // Post-intervention periods

  if (J === 0 || Tpre === 0) {
    return {
      weights: [],
      preEffect: 0,
      postEffect: 0,
      ciLower: -Infinity,
      ciUpper: Infinity,
      placeboEffects: [],
    };
  }

  // ---- Step 1: Find optimal weights via simplex-constrained optimization ----
  // Minimize ||Y_treated_pre - W' * Y_donors_pre||^2
  // subject to w >= 0, sum(w) = 1
  const weights = findOptimalWeights(treatedPre, donorsPre);

  // ---- Step 2: Compute synthetic control and treatment effects ----

  // Pre-period: synthetic = sum_j w_j * Y_j_pre
  let preEffect = 0;
  for (let t = 0; t < Tpre; t++) {
    let synthetic = 0;
    for (let j = 0; j < J; j++) {
      synthetic += (weights[j] ?? 0) * (donorsPre[j]![t] ?? 0);
    }
    preEffect += (treatedPre[t] ?? 0) - synthetic;
  }
  preEffect = Tpre > 0 ? preEffect / Tpre : 0;

  // Post-period: effect = Y_treated_post - synthetic_post
  let postEffect = 0;
  for (let t = 0; t < Tpost; t++) {
    let synthetic = 0;
    for (let j = 0; j < J; j++) {
      synthetic += (weights[j] ?? 0) * (donorsPost[j]![t] ?? 0);
    }
    postEffect += (treatedPost[t] ?? 0) - synthetic;
  }
  postEffect = Tpost > 0 ? postEffect / Tpost : 0;

  // ---- Step 3: Placebo tests for inference ----
  const nPlaceboTests = nPlacebo ?? J;
  const placeboEffects: number[] = [];

  for (let p = 0; p < Math.min(nPlaceboTests, J); p++) {
    // Treat donor p as the "treated" unit
    const placeboTreatedPre = donorsPre[p]!;
    const placeboTreatedPost = donorsPost[p]!;

    // Donors for this placebo: all donors except p, plus the original treated
    const placeboDonoPreList: number[][] = [];
    const placeboDonoPostList: number[][] = [];

    for (let j = 0; j < J; j++) {
      if (j === p) continue;
      placeboDonoPreList.push(donorsPre[j]!);
      placeboDonoPostList.push(donorsPost[j]!);
    }
    // Optionally include the original treated unit as a donor
    placeboDonoPreList.push(treatedPre);
    placeboDonoPostList.push(treatedPost);

    if (placeboDonoPreList.length === 0) continue;

    // Find placebo weights
    const placeboWeights = findOptimalWeights(placeboTreatedPre, placeboDonoPreList);

    // Compute placebo post-effect
    let placeboPostEffect = 0;
    for (let t = 0; t < Tpost; t++) {
      let synthetic = 0;
      for (let j = 0; j < placeboDonoPostList.length; j++) {
        synthetic += (placeboWeights[j] ?? 0) * (placeboDonoPostList[j]![t] ?? 0);
      }
      placeboPostEffect += (placeboTreatedPost[t] ?? 0) - synthetic;
    }
    placeboPostEffect = Tpost > 0 ? placeboPostEffect / Tpost : 0;

    placeboEffects.push(placeboPostEffect);
  }

  // ---- Step 4: Confidence interval from placebo distribution ----
  let ciLower = -Infinity;
  let ciUpper = Infinity;

  if (placeboEffects.length >= 2) {
    // Sort placebo effects
    const sorted = [...placeboEffects].sort((a, b) => a - b);
    const nP = sorted.length;

    // Use 2.5th and 97.5th percentiles for 95% CI
    const lowerIdx = Math.max(Math.floor(0.025 * nP), 0);
    const upperIdx = Math.min(Math.ceil(0.975 * nP) - 1, nP - 1);

    // The CI for the treated effect is based on the placebo distribution:
    // We use the treated effect +/- the spread of placebo effects
    const placeboMean = placeboEffects.reduce((a, b) => a + b, 0) / nP;
    const placeboStd = Math.sqrt(
      placeboEffects.reduce((sum, e) => sum + (e - placeboMean) * (e - placeboMean), 0) / nP,
    );

    // CI: postEffect +/- 1.96 * placeboStd (normal approximation)
    // Or use the empirical quantiles of (treated_effect - placebo_effects)
    ciLower = postEffect - 1.96 * Math.max(placeboStd, 1e-10);
    ciUpper = postEffect + 1.96 * Math.max(placeboStd, 1e-10);
  }

  return {
    weights,
    preEffect,
    postEffect,
    ciLower,
    ciUpper,
    placeboEffects,
  };
}

// ---- Internal: Constrained Optimization ----

/**
 * Find optimal weights w* that minimize ||y - D @ w||^2
 * subject to w >= 0 and sum(w) = 1.
 *
 * Uses projected gradient descent with simplex projection.
 *
 * @param target  Target vector (length T)
 * @param donors  Donor matrix (J x T), donors[j][t] = donor j at time t
 * @returns Optimal weight vector (length J)
 */
function findOptimalWeights(
  target: number[],
  donors: number[][],
): number[] {
  const J = donors.length;
  const T = target.length;

  if (J === 0) return [];
  if (J === 1) return [1]; // Only one donor: full weight

  // Initialize with uniform weights
  const w = new Array<number>(J);
  for (let j = 0; j < J; j++) {
    w[j] = 1 / J;
  }

  // Precompute D'D (J x J) and D'y (J x 1)
  // D[j][t] = donors[j][t], D is J x T (each donor is a row)
  const DtD = new Float64Array(J * J);
  const Dty = new Float64Array(J);

  for (let j1 = 0; j1 < J; j1++) {
    for (let j2 = j1; j2 < J; j2++) {
      let dot = 0;
      for (let t = 0; t < T; t++) {
        dot += (donors[j1]![t] ?? 0) * (donors[j2]![t] ?? 0);
      }
      DtD[j1 * J + j2] = dot;
      DtD[j2 * J + j1] = dot; // Symmetric
    }
    let dot = 0;
    for (let t = 0; t < T; t++) {
      dot += (donors[j1]![t] ?? 0) * (target[t] ?? 0);
    }
    Dty[j1] = dot;
  }

  // Projected gradient descent
  const maxIter = 1000;
  const tol = 1e-10;
  let learningRate = 0.001;

  // Adaptive learning rate: use 1 / (max eigenvalue of D'D) as a starting point
  // Approximate max eigenvalue by the Frobenius norm / J
  let frobSq = 0;
  for (let j1 = 0; j1 < J; j1++) {
    for (let j2 = 0; j2 < J; j2++) {
      const v = DtD[j1 * J + j2] ?? 0;
      frobSq += v * v;
    }
  }
  const maxEig = Math.sqrt(frobSq); // Upper bound
  if (maxEig > 0) {
    learningRate = 1 / maxEig;
  }

  for (let iter = 0; iter < maxIter; iter++) {
    // Gradient: g = D'D @ w - D'y
    // This is the gradient of 0.5 * ||y - D@w||^2
    const grad = new Array<number>(J);
    for (let j = 0; j < J; j++) {
      let g = 0;
      for (let j2 = 0; j2 < J; j2++) {
        g += (DtD[j * J + j2] ?? 0) * (w[j2] ?? 0);
      }
      g -= Dty[j] ?? 0;
      grad[j] = g;
    }

    // Gradient step
    const wNew = new Array<number>(J);
    for (let j = 0; j < J; j++) {
      wNew[j] = (w[j] ?? 0) - learningRate * (grad[j] ?? 0);
    }

    // Project onto simplex: w >= 0, sum(w) = 1
    simplexProjection(wNew);

    // Check convergence
    let maxChange = 0;
    for (let j = 0; j < J; j++) {
      maxChange = Math.max(maxChange, Math.abs((wNew[j] ?? 0) - (w[j] ?? 0)));
    }

    // Update weights
    for (let j = 0; j < J; j++) {
      w[j] = wNew[j] ?? 0;
    }

    if (maxChange < tol) break;
  }

  return w;
}

/**
 * Project a vector onto the probability simplex: {w | w >= 0, sum(w) = 1}.
 *
 * Uses the efficient O(n log n) algorithm of Duchi et al. (2008):
 * 1. Sort the vector in descending order
 * 2. Find the largest k such that u_k - (1/k)(sum_{j<=k} u_j - 1) > 0
 * 3. Set threshold tau = (1/k*)(sum_{j<=k*} u_j - 1)
 * 4. Project: w_i = max(v_i - tau, 0)
 *
 * Modifies the array in-place.
 *
 * @param v Input vector (modified in-place)
 */
function simplexProjection(v: number[]): void {
  const n = v.length;
  if (n === 0) return;

  // Sort a copy in descending order
  const sorted = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    sorted[i] = v[i] ?? 0;
  }
  sorted.sort((a, b) => b - a);

  // Find the Lagrange multiplier (threshold tau)
  let cumSum = 0;
  let kStar = 0;
  let tau = 0;

  for (let k = 0; k < n; k++) {
    cumSum += sorted[k] ?? 0;
    const candidate = (cumSum - 1) / (k + 1);
    if ((sorted[k] ?? 0) - candidate > 0) {
      kStar = k + 1;
      tau = candidate;
    }
  }

  // Project: w_i = max(v_i - tau, 0)
  for (let i = 0; i < n; i++) {
    v[i] = Math.max((v[i] ?? 0) - tau, 0);
  }

  // Ensure exact summation to 1 (handle floating point)
  let total = 0;
  for (let i = 0; i < n; i++) {
    total += v[i] ?? 0;
  }
  if (total > 0) {
    for (let i = 0; i < n; i++) {
      v[i] = (v[i] ?? 0) / total;
    }
  } else {
    // Fallback: uniform
    for (let i = 0; i < n; i++) {
      v[i] = 1 / n;
    }
  }
}
