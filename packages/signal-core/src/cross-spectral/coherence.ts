// ---------------------------------------------------------------------------
// SP-5: Magnitude-Squared Coherence
// ---------------------------------------------------------------------------
// γ²(f) = |P_xy(f)|² / (P_xx(f)·P_yy(f))
// Range [0,1]. Near 1 = linear relationship at frequency f.
// Phase spectrum reveals lead/lag: τ(f) = -phase(f)/(2πf)

import type { CoherenceResult, WindowFunction } from '../types.js';
import { fft, applyWindow, detrend, nextPow2 } from '../fourier/fft.js';

/**
 * Compute magnitude-squared coherence between two signals.
 * Uses Welch-like segmented approach for stable estimates.
 */
export function coherence(
  signalX: Float64Array,
  signalY: Float64Array,
  fs: number = 1,
  nperseg: number = 256,
  noverlap?: number,
  windowType: WindowFunction = 'hann',
): CoherenceResult {
  if (signalX.length !== signalY.length) {
    throw new Error('Signals must have equal length');
  }

  const overlap = noverlap ?? Math.floor(nperseg / 2);
  const hop = nperseg - overlap;
  const N = nextPow2(nperseg);
  const nFreqs = Math.floor(N / 2) + 1;

  const dtX = detrend(signalX);
  const dtY = detrend(signalY);

  // Accumulators for cross-spectral and auto-spectral densities
  const pxxAccum = new Float64Array(nFreqs);
  const pyyAccum = new Float64Array(nFreqs);
  const pxyReAccum = new Float64Array(nFreqs);
  const pxyImAccum = new Float64Array(nFreqs);
  let nSegments = 0;

  for (let start = 0; start + nperseg <= dtX.length; start += hop) {
    const segX = applyWindow(dtX.subarray(start, start + nperseg), windowType);
    const segY = applyWindow(dtY.subarray(start, start + nperseg), windowType);

    const fftX = fft(segX, N);
    const fftY = fft(segY, N);

    for (let k = 0; k < nFreqs; k++) {
      // Auto-spectral: Pxx = |X|², Pyy = |Y|²
      pxxAccum[k] = pxxAccum[k]! + fftX.re[k]! * fftX.re[k]! + fftX.im[k]! * fftX.im[k]!;
      pyyAccum[k] = pyyAccum[k]! + fftY.re[k]! * fftY.re[k]! + fftY.im[k]! * fftY.im[k]!;

      // Cross-spectral: Pxy = conj(X)·Y
      pxyReAccum[k] = pxyReAccum[k]! + fftX.re[k]! * fftY.re[k]! + fftX.im[k]! * fftY.im[k]!;
      pxyImAccum[k] = pxyImAccum[k]! + (-fftX.im[k]! * fftY.re[k]! + fftX.re[k]! * fftY.im[k]!);
    }
    nSegments++;
  }

  const frequencies = new Float64Array(nFreqs);
  const coh = new Float64Array(nFreqs);
  const phase = new Float64Array(nFreqs);

  for (let k = 0; k < nFreqs; k++) {
    frequencies[k] = (k * fs) / N;

    if (nSegments > 0) {
      const pxx = pxxAccum[k]! / nSegments;
      const pyy = pyyAccum[k]! / nSegments;
      const pxyRe = pxyReAccum[k]! / nSegments;
      const pxyIm = pxyImAccum[k]! / nSegments;

      // γ²(f) = |Pxy|² / (Pxx·Pyy)
      const pxyMag2 = pxyRe * pxyRe + pxyIm * pxyIm;
      const denom = pxx * pyy;
      coh[k] = denom > 1e-20 ? pxyMag2 / denom : 0;

      // Phase: arg(Pxy)
      phase[k] = Math.atan2(pxyIm, pxyRe);
    }
  }

  return { frequencies, coherence: coh, phase };
}

/**
 * Estimate time delay from phase spectrum at a specific frequency.
 * τ(f) = -phase(f) / (2πf) in same units as 1/fs
 */
export function estimateDelay(phase: number, frequency: number): number {
  if (frequency <= 0) return 0;
  return -phase / (2 * Math.PI * frequency);
}
