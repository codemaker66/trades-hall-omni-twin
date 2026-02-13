// ---------------------------------------------------------------------------
// SP-6: Anomaly Detection Ensemble — Majority Vote
// ---------------------------------------------------------------------------
// Runs STL + Matrix Profile + CUSUM + Spectral Residual in parallel.
// Flags points where ≥2 methods agree (majority vote).

import type { EnsembleAnomalyResult } from '../types.js';
import { spectralResidual } from './spectral-residual.js';
import { matrixProfileAnomalies } from './matrix-profile.js';
import { cusum } from './cusum.js';
import { stlAnomalies } from './stl.js';

/**
 * Ensemble anomaly detection with majority vote.
 *
 * @param signal Input time series
 * @param period Seasonal period for STL (default 7)
 * @param subsequenceLen Subsequence length for Matrix Profile (default 14)
 * @param minVotes Minimum number of methods that must agree (default 2)
 */
export function ensembleAnomalyDetection(
  signal: Float64Array,
  period: number = 7,
  subsequenceLen: number = 14,
  minVotes: number = 2,
): EnsembleAnomalyResult {
  const N = signal.length;

  // Run all detectors
  const stl = stlAnomalies(signal, period);
  const mp = matrixProfileAnomalies(signal, subsequenceLen);
  const cusumResult = cusum(signal);
  const sr = spectralResidual(signal);

  // Vote count per point
  const voteCount = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    if (stl.anomalies[i]) voteCount[i] = (voteCount[i] ?? 0) + 1;
    if (mp.anomalies[i]) voteCount[i] = (voteCount[i] ?? 0) + 1;
    if (cusumResult.anomalies[i]) voteCount[i] = (voteCount[i] ?? 0) + 1;
    if (sr.anomalies[i]) voteCount[i] = (voteCount[i] ?? 0) + 1;
  }

  // Consensus: ≥ minVotes agree
  const consensus: boolean[] = [];
  for (let i = 0; i < N; i++) {
    consensus.push(voteCount[i]! >= minVotes);
  }

  return {
    consensus,
    stl,
    matrixProfile: mp,
    cusum: cusumResult,
    spectralResidual: sr,
    voteCount,
  };
}
