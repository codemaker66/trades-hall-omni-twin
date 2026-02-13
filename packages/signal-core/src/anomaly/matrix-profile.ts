// ---------------------------------------------------------------------------
// SP-6: Matrix Profile for Subsequence Anomaly Detection
// ---------------------------------------------------------------------------
// P_i = min_j d(T_{i:i+m}, T_{j:j+m})
// Discords = max MP values (most unusual subsequences)
// Motifs = min MP values (most recurring patterns)
// O(n²) naive implementation (STUMPY uses MASS for O(n log n)).

import type { AnomalyFlags, MatrixProfileResult } from '../types.js';

/**
 * Z-normalized Euclidean distance between two subsequences.
 */
function zNormalizedDistance(a: Float64Array, b: Float64Array, m: number): number {
  // Compute mean and std for both
  let meanA = 0, meanB = 0;
  for (let i = 0; i < m; i++) {
    meanA += a[i]!;
    meanB += b[i]!;
  }
  meanA /= m;
  meanB /= m;

  let varA = 0, varB = 0;
  for (let i = 0; i < m; i++) {
    varA += (a[i]! - meanA) * (a[i]! - meanA);
    varB += (b[i]! - meanB) * (b[i]! - meanB);
  }
  const stdA = Math.sqrt(varA / m);
  const stdB = Math.sqrt(varB / m);

  if (stdA < 1e-10 || stdB < 1e-10) return Infinity;

  let dist = 0;
  for (let i = 0; i < m; i++) {
    const na = (a[i]! - meanA) / stdA;
    const nb = (b[i]! - meanB) / stdB;
    dist += (na - nb) * (na - nb);
  }
  return Math.sqrt(dist);
}

/**
 * Compute the Matrix Profile using STAMP (naive) algorithm.
 * For each subsequence of length m, finds the nearest non-trivial match.
 *
 * @param signal Input time series
 * @param m Subsequence length (default 14 for biweekly patterns)
 * @param exclusionZone Minimum distance between query and match (default m/4)
 */
export function matrixProfile(
  signal: Float64Array,
  m: number = 14,
  exclusionZone?: number,
): MatrixProfileResult {
  const N = signal.length;
  const nSubseq = N - m + 1;
  const ez = exclusionZone ?? Math.floor(m / 4);

  const profile = new Float64Array(nSubseq);
  const profileIndex = new Int32Array(nSubseq);
  profile.fill(Infinity);
  profileIndex.fill(-1);

  for (let i = 0; i < nSubseq; i++) {
    const subI = signal.subarray(i, i + m);

    for (let j = 0; j < nSubseq; j++) {
      // Skip exclusion zone
      if (Math.abs(i - j) <= ez) continue;

      const subJ = signal.subarray(j, j + m);
      const dist = zNormalizedDistance(subI, subJ, m);

      if (dist < profile[i]!) {
        profile[i] = dist;
        profileIndex[i] = j;
      }
    }
  }

  return { profile, profileIndex, windowSize: m };
}

/**
 * Detect anomalies from Matrix Profile (discord detection).
 * Discords = subsequences with highest MP values (most unusual).
 */
export function matrixProfileAnomalies(
  signal: Float64Array,
  m: number = 14,
  percentile: number = 99,
): AnomalyFlags {
  const { profile, windowSize } = matrixProfile(signal, m);
  const N = signal.length;

  // Compute threshold at given percentile
  const sorted = Float64Array.from(profile).sort();
  const nonInf = sorted.filter(v => isFinite(v));
  const threshIdx = Math.floor(nonInf.length * percentile / 100);
  const threshold = nonInf[Math.min(threshIdx, nonInf.length - 1)] ?? Infinity;

  const scores = new Float64Array(N);
  const anomalies: boolean[] = new Array(N).fill(false);

  for (let i = 0; i < profile.length; i++) {
    const val = profile[i]!;
    if (!isFinite(val)) continue;

    // Spread score across the subsequence window
    for (let j = i; j < Math.min(i + windowSize, N); j++) {
      scores[j] = Math.max(scores[j]!, val);
    }

    if (val > threshold) {
      for (let j = i; j < Math.min(i + windowSize, N); j++) {
        anomalies[j] = true;
      }
    }
  }

  return { anomalies, scores, method: 'matrix-profile' };
}
