// ---------------------------------------------------------------------------
// SP-6: STL Decomposition (Seasonal-Trend via LOESS)
// ---------------------------------------------------------------------------
// Decomposes: Y_t = T_t + S_t + R_t  (trend + seasonal + remainder)
// Anomalies flagged when |R_t| > k·MAD(R) / 0.6745
// Simplified STL using iterative moving average (no LOESS for pure TS).

import type { STLResult, AnomalyFlags } from '../types.js';

/**
 * Moving average with specified window.
 */
function movingAverage(signal: Float64Array, window: number): Float64Array {
  const N = signal.length;
  const result = new Float64Array(N);
  const half = Math.floor(window / 2);

  for (let i = 0; i < N; i++) {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(N - 1, i + half); j++) {
      sum += signal[j]!;
      count++;
    }
    result[i] = sum / count;
  }
  return result;
}

/**
 * STL-like decomposition using iterative smoothing.
 *
 * @param signal Input time series
 * @param period Seasonal period (e.g., 7 for weekly)
 * @param nIterations Number of STL iterations (default 3)
 * @param trendWindow Moving average window for trend (default 2*period+1)
 */
export function stlDecompose(
  signal: Float64Array,
  period: number = 7,
  nIterations: number = 3,
  trendWindow?: number,
): STLResult {
  const N = signal.length;
  const tw = trendWindow ?? (2 * period + 1);
  let seasonal: Float64Array = new Float64Array(N);
  let trend: Float64Array = new Float64Array(N);

  for (let iter = 0; iter < nIterations; iter++) {
    // Step 1: Deseasonalize
    const deseason = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      deseason[i] = signal[i]! - seasonal[i]!;
    }

    // Step 2: Estimate trend via moving average
    trend = movingAverage(deseason, tw);

    // Step 3: Detrend
    const detrended = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      detrended[i] = signal[i]! - trend[i]!;
    }

    // Step 4: Estimate seasonal component
    // Average detrended values at each seasonal position
    const seasonalAvg = new Float64Array(period);
    const seasonalCount = new Float64Array(period);
    for (let i = 0; i < N; i++) {
      const pos = i % period;
      seasonalAvg[pos] = seasonalAvg[pos]! + detrended[i]!;
      seasonalCount[pos] = seasonalCount[pos]! + 1;
    }
    for (let p = 0; p < period; p++) {
      if (seasonalCount[p]! > 0) {
        seasonalAvg[p] = seasonalAvg[p]! / seasonalCount[p]!;
      }
    }

    // Center seasonal (subtract mean)
    let seasonMean = 0;
    for (let p = 0; p < period; p++) seasonMean += seasonalAvg[p]!;
    seasonMean /= period;
    for (let p = 0; p < period; p++) seasonalAvg[p] = seasonalAvg[p]! - seasonMean;

    // Tile seasonal component
    seasonal = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      seasonal[i] = seasonalAvg[i % period]!;
    }
  }

  // Remainder
  const remainder = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    remainder[i] = signal[i]! - trend[i]! - seasonal[i]!;
  }

  return { trend, seasonal, remainder };
}

/**
 * Anomaly detection via STL remainder exceedance.
 * Flags points where |R_t| > k·σ̂_R
 * σ̂ estimated via MAD for robustness.
 */
export function stlAnomalies(
  signal: Float64Array,
  period: number = 7,
  threshold: number = 3,
): AnomalyFlags {
  const { remainder } = stlDecompose(signal, period);
  const N = remainder.length;

  // Robust scale estimate: MAD / 0.6745
  const absRemainder = Float64Array.from(remainder).map(Math.abs);
  const sorted = Float64Array.from(absRemainder).sort();
  const median = sorted[Math.floor(sorted.length / 2)]!;
  const sigma = median / 0.6745;

  const cutoff = threshold * sigma;
  const scores = new Float64Array(N);
  const anomalies: boolean[] = [];

  for (let i = 0; i < N; i++) {
    scores[i] = Math.abs(remainder[i]!) / Math.max(sigma, 1e-10);
    anomalies.push(Math.abs(remainder[i]!) > cutoff);
  }

  return { anomalies, scores, method: 'stl-remainder' };
}
