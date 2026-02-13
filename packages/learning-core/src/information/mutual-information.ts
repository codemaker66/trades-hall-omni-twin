// ---------------------------------------------------------------------------
// Mutual Information & mRMR Feature Selection
// ---------------------------------------------------------------------------

import type { MutualInformationResult } from '../types.js';

/**
 * Shannon entropy: H = -Σ p_i log(p_i)
 * Input is a vector of counts (not probabilities).
 * @param counts - Array of non-negative counts
 * @returns Entropy in nats
 */
export function entropy(counts: number[]): number {
  let total = 0;
  for (let i = 0; i < counts.length; i++) {
    total += (counts[i] ?? 0);
  }
  if (total <= 0) return 0;

  let h = 0;
  for (let i = 0; i < counts.length; i++) {
    const c = counts[i] ?? 0;
    if (c > 0) {
      const p = c / total;
      h -= p * Math.log(p);
    }
  }
  return h;
}

/**
 * Joint entropy: H(X,Y) = -Σ_ij p_ij log(p_ij)
 * Input is a 2D array of joint counts.
 * @param jointCounts - 2D array where jointCounts[i][j] is the count for (x=i, y=j)
 * @returns Joint entropy in nats
 */
export function jointEntropy(jointCounts: number[][]): number {
  let total = 0;
  for (let i = 0; i < jointCounts.length; i++) {
    const row = jointCounts[i];
    if (!row) continue;
    for (let j = 0; j < row.length; j++) {
      total += (row[j] ?? 0);
    }
  }
  if (total <= 0) return 0;

  let h = 0;
  for (let i = 0; i < jointCounts.length; i++) {
    const row = jointCounts[i];
    if (!row) continue;
    for (let j = 0; j < row.length; j++) {
      const c = row[j] ?? 0;
      if (c > 0) {
        const p = c / total;
        h -= p * Math.log(p);
      }
    }
  }
  return h;
}

/**
 * Bin a continuous value into one of nBins equal-width bins.
 * Returns an index in [0, nBins-1].
 */
function binValue(value: number, min: number, max: number, nBins: number): number {
  if (max <= min) return 0;
  const idx = Math.floor(((value - min) / (max - min)) * nBins);
  // Clamp to [0, nBins-1] to handle edge case where value === max
  return Math.max(0, Math.min(nBins - 1, idx));
}

/**
 * Mutual information via histogram binning: I(X;Y) = H(X) + H(Y) - H(X,Y)
 *
 * Bins both x and y into nBins equal-width bins, then computes MI from
 * the marginal and joint histograms.
 *
 * @param x - First variable samples
 * @param y - Second variable samples (same length as x)
 * @param nBins - Number of histogram bins for each variable
 * @returns Mutual information in nats (always >= 0)
 */
export function mutualInformation(x: number[], y: number[], nBins: number): number {
  const n = Math.min(x.length, y.length);
  if (n === 0 || nBins <= 0) return 0;

  // Find min/max for both variables
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (let i = 0; i < n; i++) {
    const xv = x[i] ?? 0;
    const yv = y[i] ?? 0;
    if (xv < xMin) xMin = xv;
    if (xv > xMax) xMax = xv;
    if (yv < yMin) yMin = yv;
    if (yv > yMax) yMax = yv;
  }

  // Build joint histogram
  const joint: number[][] = [];
  for (let i = 0; i < nBins; i++) {
    const row = new Array<number>(nBins);
    for (let j = 0; j < nBins; j++) {
      row[j] = 0;
    }
    joint[i] = row;
  }

  for (let i = 0; i < n; i++) {
    const xBin = binValue(x[i] ?? 0, xMin, xMax, nBins);
    const yBin = binValue(y[i] ?? 0, yMin, yMax, nBins);
    joint[xBin]![yBin]! += 1;
  }

  // Marginal counts for X
  const xCounts = new Array<number>(nBins);
  for (let i = 0; i < nBins; i++) {
    let sum = 0;
    const row = joint[i]!;
    for (let j = 0; j < nBins; j++) {
      sum += (row[j] ?? 0);
    }
    xCounts[i] = sum;
  }

  // Marginal counts for Y
  const yCounts = new Array<number>(nBins);
  for (let j = 0; j < nBins; j++) {
    let sum = 0;
    for (let i = 0; i < nBins; i++) {
      sum += (joint[i]![j] ?? 0);
    }
    yCounts[j] = sum;
  }

  // I(X;Y) = H(X) + H(Y) - H(X,Y)
  const hx = entropy(xCounts);
  const hy = entropy(yCounts);
  const hxy = jointEntropy(joint);

  // MI should be non-negative; clamp for numerical stability
  return Math.max(0, hx + hy - hxy);
}

/**
 * Extract column j from a 2D array X[n][d].
 */
function extractColumn(X: number[][], j: number): number[] {
  const col: number[] = [];
  for (let i = 0; i < X.length; i++) {
    const row = X[i];
    col.push(row ? (row[j] ?? 0) : 0);
  }
  return col;
}

/**
 * mRMR (Minimum Redundancy Maximum Relevance) feature selection.
 *
 * Iteratively selects features that maximize:
 *   score(f) = I(f; Y) - (1 / |S|) * Σ_{s in S} I(f; s)
 *
 * where S is the set of already-selected features.
 *
 * @param X - Feature matrix, X[i][j] = value of feature j for sample i
 * @param y - Target variable
 * @param featureNames - Names for each feature
 * @param k - Number of features to select
 * @param nBins - Number of histogram bins for MI estimation
 * @returns MutualInformationResult with selected features, scores, and indices
 */
export function mrmrSelect(
  X: number[][],
  y: number[],
  featureNames: string[],
  k: number,
  nBins: number,
): MutualInformationResult {
  const nFeatures = featureNames.length;
  const kClamped = Math.min(k, nFeatures);

  if (kClamped <= 0 || X.length === 0) {
    return { features: [], scores: [], selectedIndices: [] };
  }

  // Precompute I(f; Y) for all features
  const relevance: number[] = new Array<number>(nFeatures);
  const columns: number[][] = new Array<number[]>(nFeatures);
  for (let j = 0; j < nFeatures; j++) {
    columns[j] = extractColumn(X, j);
    relevance[j] = mutualInformation(columns[j]!, y, nBins);
  }

  // Track which features have been selected
  const selected: boolean[] = new Array<boolean>(nFeatures).fill(false);
  const selectedIndices: number[] = [];
  const selectedScores: number[] = [];
  const selectedNames: string[] = [];

  // Precompute redundancy cache: redundancy[f][s] = I(f; s)
  // We'll compute lazily as features are selected
  // Store cumulative redundancy for each candidate feature
  const cumulativeRedundancy: number[] = new Array<number>(nFeatures).fill(0);

  for (let iter = 0; iter < kClamped; iter++) {
    let bestIdx = -1;
    let bestScore = -Infinity;

    for (let f = 0; f < nFeatures; f++) {
      if (selected[f]) continue;

      // Compute mRMR score: relevance - mean_redundancy
      const rel = relevance[f] ?? 0;
      const redundancy = iter > 0
        ? (cumulativeRedundancy[f] ?? 0) / iter
        : 0;
      const score = rel - redundancy;

      if (score > bestScore) {
        bestScore = score;
        bestIdx = f;
      }
    }

    if (bestIdx < 0) break;

    selected[bestIdx] = true;
    selectedIndices.push(bestIdx);
    selectedScores.push(bestScore);
    selectedNames.push(featureNames[bestIdx] ?? `feature_${bestIdx}`);

    // Update cumulative redundancy for remaining candidates
    const selectedCol = columns[bestIdx]!;
    for (let f = 0; f < nFeatures; f++) {
      if (selected[f]) continue;
      cumulativeRedundancy[f] = (cumulativeRedundancy[f] ?? 0) +
        mutualInformation(columns[f]!, selectedCol, nBins);
    }
  }

  return {
    features: selectedNames,
    scores: selectedScores,
    selectedIndices,
  };
}
