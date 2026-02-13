// ---------------------------------------------------------------------------
// SP-1: Frequency-Domain Filtering
// ---------------------------------------------------------------------------
// Apply filters directly in the frequency domain via FFT→mask→IFFT.
// More intuitive for isolating specific periodic components.

import { fft, ifft, nextPow2, detrend } from './fft.js';

/**
 * Low-pass filter in frequency domain.
 * Keeps frequencies below cutoff.
 */
export function lowpassFilter(signal: Float64Array, cutoffFreq: number, fs: number = 1): Float64Array {
  const N = signal.length;
  const Npad = nextPow2(N) * 2;
  const detrended = detrend(signal);
  const { re, im } = fft(detrended, Npad);

  for (let k = 0; k < Npad; k++) {
    const freq = k <= Npad / 2 ? (k * fs) / Npad : ((Npad - k) * fs) / Npad;
    if (freq > cutoffFreq) {
      re[k] = 0;
      im[k] = 0;
    }
  }

  return ifft(re, im).subarray(0, N);
}

/**
 * High-pass filter in frequency domain.
 * Keeps frequencies above cutoff.
 */
export function highpassFilter(signal: Float64Array, cutoffFreq: number, fs: number = 1): Float64Array {
  const N = signal.length;
  const Npad = nextPow2(N) * 2;
  const detrended = detrend(signal);
  const { re, im } = fft(detrended, Npad);

  for (let k = 0; k < Npad; k++) {
    const freq = k <= Npad / 2 ? (k * fs) / Npad : ((Npad - k) * fs) / Npad;
    if (freq < cutoffFreq) {
      re[k] = 0;
      im[k] = 0;
    }
  }

  return ifft(re, im).subarray(0, N);
}

/**
 * Bandpass filter in frequency domain.
 * Keeps frequencies between lowFreq and highFreq.
 */
export function bandpassFilter(
  signal: Float64Array,
  lowFreq: number,
  highFreq: number,
  fs: number = 1,
): Float64Array {
  const N = signal.length;
  const Npad = nextPow2(N) * 2;
  const detrended = detrend(signal);
  const { re, im } = fft(detrended, Npad);

  for (let k = 0; k < Npad; k++) {
    const freq = k <= Npad / 2 ? (k * fs) / Npad : ((Npad - k) * fs) / Npad;
    if (freq < lowFreq || freq > highFreq) {
      re[k] = 0;
      im[k] = 0;
    }
  }

  return ifft(re, im).subarray(0, N);
}

/**
 * Bandstop (notch) filter in frequency domain.
 * Removes frequencies between lowFreq and highFreq.
 */
export function bandstopFilter(
  signal: Float64Array,
  lowFreq: number,
  highFreq: number,
  fs: number = 1,
): Float64Array {
  const N = signal.length;
  const Npad = nextPow2(N) * 2;
  const detrended = detrend(signal);
  const { re, im } = fft(detrended, Npad);

  for (let k = 0; k < Npad; k++) {
    const freq = k <= Npad / 2 ? (k * fs) / Npad : ((Npad - k) * fs) / Npad;
    if (freq >= lowFreq && freq <= highFreq) {
      re[k] = 0;
      im[k] = 0;
    }
  }

  return ifft(re, im).subarray(0, N);
}

/**
 * Extract specific periodic component with Gaussian-shaped frequency window.
 * More gentle than brick-wall filter — reduces Gibbs ringing.
 */
export function extractPeriodic(
  signal: Float64Array,
  targetPeriod: number,
  bandwidthFactor: number = 0.2,
  fs: number = 1,
): Float64Array {
  const N = signal.length;
  const Npad = nextPow2(N) * 2;
  const detrended = detrend(signal);
  const { re, im } = fft(detrended, Npad);

  const targetFreq = 1 / targetPeriod;
  const sigma = targetFreq * bandwidthFactor;

  const filtRe = new Float64Array(Npad);
  const filtIm = new Float64Array(Npad);

  for (let k = 0; k < Npad; k++) {
    const freq = k <= Npad / 2 ? (k * fs) / Npad : ((Npad - k) * fs) / Npad;
    // Gaussian window centered on target frequency
    const weight = Math.exp(-0.5 * ((freq - targetFreq) / sigma) ** 2);
    filtRe[k] = re[k]! * weight;
    filtIm[k] = im[k]! * weight;
  }

  return ifft(filtRe, filtIm).subarray(0, N);
}
