// ---------------------------------------------------------------------------
// SP-1: Short-Time Fourier Transform (STFT)
// ---------------------------------------------------------------------------
// Time-frequency uncertainty: Δt·Δf ≥ 1/4π
// Longer windows → better frequency resolution, worse time resolution.
// Typical: 90-day window (quarterly), 7-day hop for venue data.

import type { STFTResult, WindowFunction } from '../types.js';
import { fft, ifftInPlace, applyWindow, detrend, nextPow2 } from './fft.js';

/**
 * Compute the Short-Time Fourier Transform.
 *
 * @param signal Input time series
 * @param fs Sampling frequency
 * @param nperseg Window size (samples per segment)
 * @param noverlap Overlap between adjacent segments
 * @param windowType Window function
 * @param nfft FFT size per segment (default: nperseg rounded to pow2)
 */
export function stft(
  signal: Float64Array,
  fs: number = 1,
  nperseg: number = 90,
  noverlap: number = 83,
  windowType: WindowFunction = 'hann',
  nfft?: number,
): STFTResult {
  const hop = nperseg - noverlap;
  const N = nfft ?? nextPow2(nperseg);
  const nFreqs = Math.floor(N / 2) + 1;
  const detrended = detrend(signal);

  // Count number of time frames
  const nTimes = Math.max(0, Math.floor((detrended.length - nperseg) / hop) + 1);

  const frequencies = new Float64Array(nFreqs);
  for (let k = 0; k < nFreqs; k++) {
    frequencies[k] = (k * fs) / N;
  }

  const times = new Float64Array(nTimes);
  const spectrogram = new Float64Array(nTimes * nFreqs);

  for (let t = 0; t < nTimes; t++) {
    const start = t * hop;
    times[t] = (start + nperseg / 2) / fs; // Center of window

    const segment = detrended.subarray(start, start + nperseg);
    const windowed = applyWindow(segment, windowType);
    const { re, im } = fft(windowed, N);

    for (let k = 0; k < nFreqs; k++) {
      spectrogram[t * nFreqs + k] = Math.sqrt(re[k]! * re[k]! + im[k]! * im[k]!);
    }
  }

  return { frequencies, times, spectrogram, nFreqs, nTimes };
}

/**
 * Inverse STFT: reconstruct time-domain signal from STFT magnitude via
 * Griffin-Lim-like overlap-add (zero phase assumption).
 */
export function istft(
  magnitudes: Float64Array,
  nFreqs: number,
  nTimes: number,
  nperseg: number,
  hop: number,
  windowType: WindowFunction = 'hann',
): Float64Array {
  const N = (nFreqs - 1) * 2;
  const outLen = (nTimes - 1) * hop + nperseg;
  const output = new Float64Array(outLen);
  const windowSum = new Float64Array(outLen);

  // Generate synthesis window
  const winTemplate = new Float64Array(nperseg);
  winTemplate.fill(1);
  const window = applyWindow(winTemplate, windowType);

  for (let t = 0; t < nTimes; t++) {
    // Build symmetric spectrum (zero phase)
    const re = new Float64Array(N);
    const im = new Float64Array(N);
    for (let k = 0; k < nFreqs; k++) {
      re[k] = magnitudes[t * nFreqs + k]!;
    }
    // Mirror for negative frequencies
    for (let k = 1; k < nFreqs - 1; k++) {
      re[N - k] = re[k]!;
    }

    ifftInPlace(re, im);

    const start = t * hop;
    for (let n = 0; n < nperseg && start + n < outLen; n++) {
      output[start + n] = output[start + n]! + re[n]! * window[n]!;
      windowSum[start + n] = windowSum[start + n]! + window[n]! * window[n]!;
    }
  }

  // Normalize by window sum
  for (let i = 0; i < outLen; i++) {
    if (windowSum[i]! > 1e-10) {
      output[i] = output[i]! / windowSum[i]!;
    }
  }

  return output;
}
