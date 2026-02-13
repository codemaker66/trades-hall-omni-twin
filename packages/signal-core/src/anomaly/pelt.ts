// ---------------------------------------------------------------------------
// SP-6: PELT Changepoint Detection (Killick et al., JASA 2012)
// ---------------------------------------------------------------------------
// F(t) = min_{s<t} [F(s) + C(x_{s+1:t}) + β]
// O(n) expected time via dynamic programming with pruning.
// C uses Gaussian cost: sum of squared residuals from segment mean.

import type { ChangePointResult } from '../types.js';

/**
 * Gaussian cost function: sum of squared deviations from segment mean.
 * C(x_{a:b}) = Σ(x_i - x̄)² = Σx_i² - (Σx_i)²/n
 */
function gaussianCost(cumSum: Float64Array, cumSum2: Float64Array, start: number, end: number): number {
  const n = end - start;
  if (n <= 0) return 0;
  const sum = cumSum[end]! - cumSum[start]!;
  const sum2 = cumSum2[end]! - cumSum2[start]!;
  return sum2 - (sum * sum) / n;
}

/**
 * PELT (Pruned Exact Linear Time) changepoint detection.
 *
 * @param signal Input time series
 * @param penalty Penalty per changepoint (BIC: log(n)·dim, default: 2·log(n))
 * @param minSize Minimum segment length (default 2)
 */
export function pelt(
  signal: Float64Array,
  penalty?: number,
  minSize: number = 2,
): ChangePointResult {
  const N = signal.length;
  const pen = penalty ?? 2 * Math.log(N);

  // Precompute cumulative sums for O(1) cost evaluation
  const cumSum = new Float64Array(N + 1);
  const cumSum2 = new Float64Array(N + 1);
  for (let i = 0; i < N; i++) {
    cumSum[i + 1] = cumSum[i]! + signal[i]!;
    cumSum2[i + 1] = cumSum2[i]! + signal[i]! * signal[i]!;
  }

  // DP: F[t] = min cost for segmenting x[0:t]
  const F = new Float64Array(N + 1);
  F.fill(Infinity);
  F[0] = -pen; // so that first segment pays exactly one penalty

  const lastChange = new Int32Array(N + 1);
  lastChange.fill(-1);

  // Candidate set for pruning
  let candidates = [0];

  for (let t = minSize; t <= N; t++) {
    let bestCost = Infinity;
    let bestTau = 0;
    const newCandidates: number[] = [];

    for (const tau of candidates) {
      if (t - tau < minSize) {
        // Keep candidate for future evaluation when segment is long enough
        newCandidates.push(tau);
        continue;
      }

      const cost = F[tau]! + gaussianCost(cumSum, cumSum2, tau, t) + pen;

      if (cost < bestCost) {
        bestCost = cost;
        bestTau = tau;
      }

      // PELT pruning: keep candidate if it could still be optimal
      if (cost <= bestCost + pen) {
        newCandidates.push(tau);
      }
    }

    F[t] = bestCost;
    lastChange[t] = bestTau;

    newCandidates.push(t);
    candidates = newCandidates;
  }

  // Backtrack to find changepoints
  const changepoints: number[] = [];
  let idx = N;
  while (idx > 0) {
    const cp = lastChange[idx]!;
    if (cp > 0) changepoints.push(cp);
    idx = cp;
  }

  changepoints.sort((a, b) => a - b);

  return { changepoints, penalty: pen };
}

/**
 * Binary segmentation: simpler O(n log n) changepoint detection.
 * Recursively splits the signal at the point of maximum cost reduction.
 */
export function binarySegmentation(
  signal: Float64Array,
  penalty?: number,
  minSize: number = 2,
): ChangePointResult {
  const N = signal.length;
  const pen = penalty ?? 2 * Math.log(N);

  const cumSum = new Float64Array(N + 1);
  const cumSum2 = new Float64Array(N + 1);
  for (let i = 0; i < N; i++) {
    cumSum[i + 1] = cumSum[i]! + signal[i]!;
    cumSum2[i + 1] = cumSum2[i]! + signal[i]! * signal[i]!;
  }

  const changepoints: number[] = [];

  function search(left: number, right: number): void {
    if (right - left < 2 * minSize) return;

    const baseCost = gaussianCost(cumSum, cumSum2, left, right);
    let bestGain = 0;
    let bestSplit = -1;

    for (let t = left + minSize; t <= right - minSize; t++) {
      const leftCost = gaussianCost(cumSum, cumSum2, left, t);
      const rightCost = gaussianCost(cumSum, cumSum2, t, right);
      const gain = baseCost - leftCost - rightCost;

      if (gain > bestGain) {
        bestGain = gain;
        bestSplit = t;
      }
    }

    if (bestGain > pen && bestSplit > 0) {
      changepoints.push(bestSplit);
      search(left, bestSplit);
      search(bestSplit, right);
    }
  }

  search(0, N);
  changepoints.sort((a, b) => a - b);

  return { changepoints, penalty: pen };
}
