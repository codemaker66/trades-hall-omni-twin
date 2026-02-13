// ---------------------------------------------------------------------------
// SP-6: Spectral Residual Anomaly Detection (Microsoft, KDD 2019)
// ---------------------------------------------------------------------------
// L(f) = ln|FFT(x)| → AL(f) = h_q * L(f) → R(f) = L - AL
// S(t) = |IFFT(e^{R+jφ})|²
// 36-69% F1 improvement over best baselines.

import type { AnomalyFlags } from '../types.js';
import { fft, ifft, nextPow2 } from '../fourier/fft.js';

/**
 * Spectral Residual method for time series anomaly detection.
 *
 * @param signal Input time series
 * @param q Averaging kernel size for spectral smoothing (default 3)
 * @param threshold Number of standard deviations for anomaly threshold (default 3)
 */
export function spectralResidual(
  signal: Float64Array,
  q: number = 3,
  threshold: number = 3,
): AnomalyFlags {
  const N = signal.length;
  const Npad = nextPow2(N);
  const { re, im } = fft(signal, Npad);

  // Log amplitude spectrum
  const logAmp = new Float64Array(Npad);
  const phase = new Float64Array(Npad);
  for (let k = 0; k < Npad; k++) {
    const mag = Math.sqrt(re[k]! * re[k]! + im[k]! * im[k]!);
    logAmp[k] = Math.log(mag + 1e-10);
    phase[k] = Math.atan2(im[k]!, re[k]!);
  }

  // Average log amplitude (moving average of size q)
  const avgLogAmp = new Float64Array(Npad);
  const halfQ = Math.floor(q / 2);
  for (let k = 0; k < Npad; k++) {
    let sum = 0;
    let count = 0;
    for (let j = -halfQ; j <= halfQ; j++) {
      const idx = ((k + j) % Npad + Npad) % Npad;
      sum += logAmp[idx]!;
      count++;
    }
    avgLogAmp[k] = sum / count;
  }

  // Spectral residual: R(f) = L(f) - AL(f)
  const residualRe = new Float64Array(Npad);
  const residualIm = new Float64Array(Npad);
  for (let k = 0; k < Npad; k++) {
    const r = logAmp[k]! - avgLogAmp[k]!;
    // Reconstruct: e^{R+jφ}
    const mag = Math.exp(r);
    residualRe[k] = mag * Math.cos(phase[k]!);
    residualIm[k] = mag * Math.sin(phase[k]!);
  }

  // Saliency map: S(t) = |IFFT(e^{R+jφ})|²
  const saliency = ifft(residualRe, residualIm);
  const scores = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    scores[i] = saliency[i]! * saliency[i]!;
  }

  // Adaptive threshold: mean + threshold * std
  let mean = 0;
  for (let i = 0; i < N; i++) mean += scores[i]!;
  mean /= N;

  let variance = 0;
  for (let i = 0; i < N; i++) {
    const diff = scores[i]! - mean;
    variance += diff * diff;
  }
  variance /= N;
  const std = Math.sqrt(variance);

  const cutoff = mean + threshold * std;
  const anomalies: boolean[] = [];
  for (let i = 0; i < N; i++) {
    anomalies.push(scores[i]! > cutoff);
  }

  return { anomalies, scores, method: 'spectral-residual' };
}
