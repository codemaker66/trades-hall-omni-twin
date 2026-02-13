// ---------------------------------------------------------------------------
// SP-1: Welch's Method — averaged periodogram for PSD estimation
// ---------------------------------------------------------------------------
// K overlapping segments → window → FFT → average power.
// Variance drops ~1/K compared to raw periodogram.
// 50% overlap with Hann window is the standard choice.

import type { WelchResult, WindowFunction } from '../types.js';
import { fft, applyWindow, detrend, nextPow2 } from './fft.js';

/**
 * Welch's method for power spectral density estimation.
 *
 * @param signal Input time series
 * @param fs Sampling frequency (default 1.0 = daily)
 * @param nperseg Samples per segment (default 256)
 * @param noverlap Overlap samples (default nperseg/2)
 * @param windowType Window function (default 'hann')
 * @param nfft FFT size (default nperseg, rounded to next pow2)
 */
export function welchPSD(
  signal: Float64Array,
  fs: number = 1,
  nperseg: number = 256,
  noverlap?: number,
  windowType: WindowFunction = 'hann',
  nfft?: number,
): WelchResult {
  const overlap = noverlap ?? Math.floor(nperseg / 2);
  const step = nperseg - overlap;
  const N = nfft ?? nextPow2(nperseg);
  const detrended = detrend(signal);

  // Compute window energy for normalization
  const winTemplate = new Float64Array(nperseg);
  winTemplate.fill(1);
  const window = applyWindow(winTemplate, windowType);
  let winEnergy = 0;
  for (let i = 0; i < nperseg; i++) {
    winEnergy += window[i]! * window[i]!;
  }

  const nFreqs = Math.floor(N / 2) + 1;
  const psdAccum = new Float64Array(nFreqs);
  let nSegments = 0;

  for (let start = 0; start + nperseg <= detrended.length; start += step) {
    const segment = detrended.subarray(start, start + nperseg);
    const windowed = applyWindow(segment, windowType);
    const { re, im } = fft(windowed, N);

    for (let k = 0; k < nFreqs; k++) {
      const power = re[k]! * re[k]! + im[k]! * im[k]!;
      // Scale: 1/(fs·winEnergy) for one-sided PSD
      const scale = (k === 0 || k === N / 2) ? 1 : 2;
      psdAccum[k] = psdAccum[k]! + (scale * power) / (fs * winEnergy);
    }
    nSegments++;
  }

  if (nSegments === 0) {
    return {
      frequencies: new Float64Array(nFreqs),
      psd: new Float64Array(nFreqs),
    };
  }

  // Average across segments
  const frequencies = new Float64Array(nFreqs);
  const psd = new Float64Array(nFreqs);
  for (let k = 0; k < nFreqs; k++) {
    frequencies[k] = (k * fs) / N;
    psd[k] = psdAccum[k]! / nSegments;
  }

  return { frequencies, psd };
}

/**
 * Multitaper spectral estimation (Thomson, 1982).
 * Uses K = 2*NW - 1 Slepian (DPSS) tapers for superior bias-variance tradeoff.
 * Approximated here using sine tapers: V_k[n] = √(2/(N+1)) · sin(πk(n+1)/(N+1))
 * which approach DPSS properties for moderate NW.
 */
export function multitaperPSD(
  signal: Float64Array,
  fs: number = 1,
  nw: number = 4,
  nfft?: number,
): WelchResult {
  const N = signal.length;
  const K = 2 * nw - 1; // Number of tapers
  const Nfft = nfft ?? nextPow2(N) * 2;
  const nFreqs = Math.floor(Nfft / 2) + 1;
  const detrended = detrend(signal);

  const psdAccum = new Float64Array(nFreqs);

  for (let k = 1; k <= K; k++) {
    // Sine taper (approximation to Slepian/DPSS)
    const tapered = new Float64Array(N);
    let taperEnergy = 0;
    for (let n = 0; n < N; n++) {
      const taper = Math.sqrt(2 / (N + 1)) * Math.sin((Math.PI * k * (n + 1)) / (N + 1));
      tapered[n] = detrended[n]! * taper;
      taperEnergy += taper * taper;
    }

    const { re, im } = fft(tapered, Nfft);

    for (let f = 0; f < nFreqs; f++) {
      const power = re[f]! * re[f]! + im[f]! * im[f]!;
      const scale = (f === 0 || f === Nfft / 2) ? 1 : 2;
      psdAccum[f] = psdAccum[f]! + (scale * power) / (fs * taperEnergy);
    }
  }

  const frequencies = new Float64Array(nFreqs);
  const psd = new Float64Array(nFreqs);
  for (let k = 0; k < nFreqs; k++) {
    frequencies[k] = (k * fs) / Nfft;
    psd[k] = psdAccum[k]! / K;
  }

  return { frequencies, psd };
}
