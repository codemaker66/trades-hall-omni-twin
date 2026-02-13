// ---------------------------------------------------------------------------
// SP-6: CUSUM (Cumulative Sum) Change Detection
// ---------------------------------------------------------------------------
// S⁺_t = max(0, S⁺_{t-1} + (x_t - μ₀ - k))
// S⁻_t = max(0, S⁻_{t-1} + (μ₀ - k - x_t))
// Signal when S⁺_t > h·σ or S⁻_t > h·σ
// With k=0.5σ, h=5σ: ARL₀ ≈ 465 (false alarm every ~465 samples)

import type { AnomalyFlags, CUSUMConfig } from '../types.js';

/**
 * CUSUM (Cumulative Sum) for mean shift detection.
 *
 * @param signal Input time series
 * @param config CUSUM parameters (k: allowable slack in σ, h: decision threshold in σ)
 */
export function cusum(
  signal: Float64Array,
  config: CUSUMConfig = { k: 0.5, h: 5.0 },
): AnomalyFlags {
  const N = signal.length;
  const { k, h } = config;

  // Estimate baseline mean and std from the signal
  let mean = 0;
  for (let i = 0; i < N; i++) mean += signal[i]!;
  mean /= N;

  let variance = 0;
  for (let i = 0; i < N; i++) {
    const diff = signal[i]! - mean;
    variance += diff * diff;
  }
  const sigma = Math.sqrt(variance / N);

  const sPos = new Float64Array(N);
  const sNeg = new Float64Array(N);
  const scores = new Float64Array(N);
  const anomalies: boolean[] = [];

  for (let t = 0; t < N; t++) {
    if (t === 0) {
      sPos[0] = Math.max(0, signal[0]! - mean - k * sigma);
      sNeg[0] = Math.max(0, mean - k * sigma - signal[0]!);
    } else {
      sPos[t] = Math.max(0, sPos[t - 1]! + (signal[t]! - mean - k * sigma));
      sNeg[t] = Math.max(0, sNeg[t - 1]! + (mean - k * sigma - signal[t]!));
    }

    scores[t] = Math.max(sPos[t]!, sNeg[t]!);
    anomalies.push(sPos[t]! > h * sigma || sNeg[t]! > h * sigma);
  }

  return { anomalies, scores, method: 'cusum' };
}

/**
 * Two-sided CUSUM with automatic baseline estimation.
 * Uses a running window for local baseline instead of global mean.
 */
export function adaptiveCUSUM(
  signal: Float64Array,
  windowSize: number = 50,
  k: number = 0.5,
  h: number = 5.0,
): AnomalyFlags {
  const N = signal.length;
  const scores = new Float64Array(N);
  const anomalies: boolean[] = [];
  let sPos = 0;
  let sNeg = 0;

  for (let t = 0; t < N; t++) {
    // Local baseline: mean and std of preceding window
    const start = Math.max(0, t - windowSize);
    const end = t;
    let localMean = 0;
    let count = 0;
    for (let i = start; i < end; i++) {
      localMean += signal[i]!;
      count++;
    }
    localMean = count > 0 ? localMean / count : signal[t]!;

    let localVar = 0;
    for (let i = start; i < end; i++) {
      const diff = signal[i]! - localMean;
      localVar += diff * diff;
    }
    const localSigma = count > 1 ? Math.sqrt(localVar / (count - 1)) : 1;

    sPos = Math.max(0, sPos + (signal[t]! - localMean - k * localSigma));
    sNeg = Math.max(0, sNeg + (localMean - k * localSigma - signal[t]!));

    scores[t] = Math.max(sPos, sNeg);
    const isAnomaly = sPos > h * localSigma || sNeg > h * localSigma;
    anomalies.push(isAnomaly);

    // Reset after alarm
    if (isAnomaly) {
      sPos = 0;
      sNeg = 0;
    }
  }

  return { anomalies, scores, method: 'adaptive-cusum' };
}
