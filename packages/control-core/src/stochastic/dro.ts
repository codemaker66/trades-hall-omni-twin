// ---------------------------------------------------------------------------
// OC-9  Stochastic Optimal Control -- Distributionally Robust Optimization
// ---------------------------------------------------------------------------

import type { DROConfig } from '../types.js';

// ---------------------------------------------------------------------------
// 1D Wasserstein Distance
// ---------------------------------------------------------------------------

/**
 * Compute the 1-Wasserstein (Earth Mover's) distance between two
 * 1-dimensional discrete distributions.
 *
 * For 1D distributions, W_1 equals the area between the two CDFs,
 * which can be computed by sorting samples and summing absolute
 * differences of cumulative probabilities.
 *
 * Both p and q are assumed to be equal-weight empirical distributions
 * (i.e., each element is a sample, all with weight 1/n).
 *
 * @param p - Samples from distribution P (sorted internally)
 * @param q - Samples from distribution Q (sorted internally)
 * @returns W_1(P, Q) distance
 */
export function wasserstein1D(p: Float64Array, q: Float64Array): number {
  const np = p.length;
  const nq = q.length;

  if (np === 0 || nq === 0) return 0;

  // Sort copies of both arrays
  const pSorted = new Float64Array(p).sort();
  const qSorted = new Float64Array(q).sort();

  // If both have the same length, the 1-Wasserstein distance between
  // equal-weight empirical distributions is:
  //   W_1 = (1/n) * sum_i |p_sorted[i] - q_sorted[i]|
  if (np === nq) {
    let dist = 0;
    for (let i = 0; i < np; i++) {
      dist += Math.abs(pSorted[i]! - qSorted[i]!);
    }
    return dist / np;
  }

  // General case: compute area between CDFs using a merge-like approach.
  // Each step of p advances by 1/np in CDF, each step of q by 1/nq.
  let dist = 0;
  let ip = 0;
  let iq = 0;
  let cdfP = 0;
  let cdfQ = 0;
  let prevX = Math.min(pSorted[0]!, qSorted[0]!);

  while (ip < np || iq < nq) {
    // Determine next event point
    const nextP = ip < np ? pSorted[ip]! : Infinity;
    const nextQ = iq < nq ? qSorted[iq]! : Infinity;
    const nextX = Math.min(nextP, nextQ);

    // Area of rectangle from prevX to nextX with height |cdfP - cdfQ|
    dist += Math.abs(cdfP - cdfQ) * (nextX - prevX);
    prevX = nextX;

    // Advance the CDF that has the next sample at this point
    if (nextP <= nextQ) {
      cdfP += 1 / np;
      ip++;
    }
    if (nextQ <= nextP) {
      cdfQ += 1 / nq;
      iq++;
    }
  }

  return dist;
}

// ---------------------------------------------------------------------------
// Wasserstein Distributionally Robust Optimization
// ---------------------------------------------------------------------------

/**
 * Distributionally robust optimization over a Wasserstein ambiguity set.
 *
 * For each candidate action, computes the worst-case expected cost over
 * all distributions within a Wasserstein ball of radius epsilon centred
 * at the empirical distribution of observed samples.
 *
 * Uses a simplified conservative bound:
 *
 *   worst-case cost(a) = E_empirical[ c(a, xi) ] + epsilon * L(a)
 *
 * where L(a) is an estimate of the Lipschitz constant of c(a, .) with
 * respect to the scenario argument, computed as the maximum absolute
 * cost difference over all sample pairs divided by their distance.
 *
 * The action minimising this worst-case bound is selected.
 *
 * @param config  - DRO configuration with nSamples, epsilon, and costFn
 * @param samples - Observed scenario samples (nSamples vectors)
 * @param actions - Candidate action vectors
 * @returns Best action and its worst-case cost
 */
export function wassersteinDRO(
  config: DROConfig,
  samples: Float64Array[],
  actions: Float64Array[],
): { bestAction: Float64Array; worstCaseCost: number } {
  const { epsilon, costFn } = config;
  const nSamples = samples.length;
  const nActions = actions.length;

  if (nActions === 0) {
    return { bestAction: new Float64Array(0), worstCaseCost: Infinity };
  }

  if (nSamples === 0) {
    // No data: return first action with infinite worst-case cost
    return { bestAction: new Float64Array(actions[0]!), worstCaseCost: Infinity };
  }

  let bestAction = actions[0]!;
  let bestWorstCaseCost = Infinity;

  for (let ai = 0; ai < nActions; ai++) {
    const action = actions[ai]!;

    // Compute cost for each sample
    const costs = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      costs[i] = costFn(action, samples[i]!);
    }

    // Empirical expected cost
    let meanCost = 0;
    for (let i = 0; i < nSamples; i++) {
      meanCost += costs[i]!;
    }
    meanCost /= nSamples;

    // Estimate Lipschitz constant of costFn(action, .) over sample pairs
    let lipschitz = 0;

    for (let i = 0; i < nSamples; i++) {
      for (let j = i + 1; j < nSamples; j++) {
        // Euclidean distance between samples i and j
        const si = samples[i]!;
        const sj = samples[j]!;
        let distSq = 0;
        for (let d = 0; d < si.length; d++) {
          const diff = si[d]! - sj[d]!;
          distSq += diff * diff;
        }
        const dist = Math.sqrt(distSq);

        if (dist > 1e-15) {
          const costDiff = Math.abs(costs[i]! - costs[j]!);
          const ratio = costDiff / dist;
          if (ratio > lipschitz) {
            lipschitz = ratio;
          }
        }
      }
    }

    // Worst-case cost = empirical mean + epsilon * Lipschitz
    const worstCaseCost = meanCost + epsilon * lipschitz;

    if (worstCaseCost < bestWorstCaseCost) {
      bestWorstCaseCost = worstCaseCost;
      bestAction = action;
    }
  }

  return {
    bestAction: new Float64Array(bestAction),
    worstCaseCost: bestWorstCaseCost,
  };
}
