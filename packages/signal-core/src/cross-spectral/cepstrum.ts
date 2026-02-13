// ---------------------------------------------------------------------------
// SP-5: Cepstral Analysis
// ---------------------------------------------------------------------------
// Real cepstrum: c[n] = IFFT(log|FFT(x)|²)
// Peaks at quefrency q → periodicity of period q.
// Detects biweekly corporate events, monthly recurring bookings.

import { fft, ifft, nextPow2, detrend } from '../fourier/fft.js';
import type { CepstrumResult } from '../types.js';

/**
 * Compute the real cepstrum of a signal.
 * c[n] = IFFT(log|FFT(x)|²)
 * Quefrency peaks indicate periodic structure in the signal.
 */
export function realCepstrum(signal: Float64Array, fs: number = 1): CepstrumResult {
  const N = nextPow2(signal.length) * 2;
  const detrended = detrend(signal);
  const { re, im } = fft(detrended, N);

  // Log magnitude spectrum
  const logMagRe = new Float64Array(N);
  const logMagIm = new Float64Array(N);
  for (let k = 0; k < N; k++) {
    const mag2 = re[k]! * re[k]! + im[k]! * im[k]!;
    logMagRe[k] = Math.log(Math.max(mag2, 1e-20));
  }

  // IFFT of log magnitude
  const cepstrum = ifft(logMagRe, logMagIm);

  // Only positive quefrencies are meaningful
  const nQ = Math.floor(N / 2);
  const quefrencies = new Float64Array(nQ);
  const cepstrumOut = new Float64Array(nQ);
  for (let n = 0; n < nQ; n++) {
    quefrencies[n] = n / fs; // quefrency in same units as sample period
    cepstrumOut[n] = cepstrum[n]!;
  }

  // Find dominant peaks (quefrency > 2 samples to avoid DC)
  const dominantQuefrencies: number[] = [];
  const maxVal = cepstrumOut.reduce((a, b) => Math.max(a, Math.abs(b)), 0);
  const threshold = maxVal * 0.1;

  for (let n = 3; n < nQ - 1; n++) {
    if (
      Math.abs(cepstrumOut[n]!) > threshold &&
      Math.abs(cepstrumOut[n]!) > Math.abs(cepstrumOut[n - 1]!) &&
      Math.abs(cepstrumOut[n]!) > Math.abs(cepstrumOut[n + 1]!)
    ) {
      dominantQuefrencies.push(quefrencies[n]!);
    }
  }

  // Sort by magnitude descending
  dominantQuefrencies.sort(
    (a, b) => {
      const idxA = Math.round(a * fs);
      const idxB = Math.round(b * fs);
      return Math.abs(cepstrumOut[idxB] ?? 0) - Math.abs(cepstrumOut[idxA] ?? 0);
    },
  );

  return {
    quefrencies,
    cepstrum: cepstrumOut,
    dominantQuefrencies: dominantQuefrencies.slice(0, 10),
  };
}

/**
 * Power cepstrum: |IFFT(log|FFT(x)|²)|²
 * Better for periodic detection in noisy signals.
 */
export function powerCepstrum(signal: Float64Array, fs: number = 1): CepstrumResult {
  const result = realCepstrum(signal, fs);
  // Square the cepstrum values
  const squared = new Float64Array(result.cepstrum.length);
  for (let i = 0; i < squared.length; i++) {
    squared[i] = result.cepstrum[i]! * result.cepstrum[i]!;
  }
  return { ...result, cepstrum: squared };
}
