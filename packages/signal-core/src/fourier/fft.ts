// ---------------------------------------------------------------------------
// SP-1: FFT — Cooley-Tukey Radix-2 DIT
// ---------------------------------------------------------------------------
// X[k] = Σ_{n=0}^{N-1} x[n]·e^{-j2πkn/N}
// O(N log N) via recursive butterfly decomposition.
// Zero-pads to next power of 2 for frequency interpolation.

import type { Complex, SpectralResult, SeasonalityResult, WindowFunction } from '../types.js';

/** Check if n is a power of 2. */
function isPow2(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

/** Next power of 2 >= n. */
export function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/** Apply window function to signal. */
export function applyWindow(signal: Float64Array, windowType: WindowFunction): Float64Array {
  const N = signal.length;
  const result = new Float64Array(N);

  for (let n = 0; n < N; n++) {
    let w: number;
    switch (windowType) {
      case 'rectangular':
        w = 1;
        break;
      case 'hann':
        w = 0.5 * (1 - Math.cos((2 * Math.PI * n) / (N - 1)));
        break;
      case 'hamming':
        w = 0.54 - 0.46 * Math.cos((2 * Math.PI * n) / (N - 1));
        break;
      case 'blackman':
        w = 0.42 - 0.5 * Math.cos((2 * Math.PI * n) / (N - 1))
          + 0.08 * Math.cos((4 * Math.PI * n) / (N - 1));
        break;
      case 'blackman-harris':
        w = 0.35875 - 0.48829 * Math.cos((2 * Math.PI * n) / (N - 1))
          + 0.14128 * Math.cos((4 * Math.PI * n) / (N - 1))
          - 0.01168 * Math.cos((6 * Math.PI * n) / (N - 1));
        break;
      case 'kaiser':
        // Kaiser with β=8.6 ≈ Blackman-Harris performance
        w = kaiserWindow(n, N, 8.6);
        break;
      default:
        w = 1;
    }
    result[n] = signal[n]! * w;
  }
  return result;
}

/** Kaiser window: w[n] = I₀(β√(1-(2n/(N-1)-1)²)) / I₀(β) */
function kaiserWindow(n: number, N: number, beta: number): number {
  const x = 2 * n / (N - 1) - 1;
  return besselI0(beta * Math.sqrt(1 - x * x)) / besselI0(beta);
}

/** Modified Bessel function of first kind, order 0 (series expansion). */
function besselI0(x: number): number {
  let sum = 1;
  let term = 1;
  const halfX = x / 2;
  for (let k = 1; k <= 25; k++) {
    term *= (halfX / k) * (halfX / k);
    sum += term;
    if (term < 1e-16 * sum) break;
  }
  return sum;
}

/** Linear detrend: remove best-fit line from signal. */
export function detrend(signal: Float64Array): Float64Array {
  const N = signal.length;
  if (N < 2) return new Float64Array(signal);

  // Fit y = a + b·x via least squares
  let sumX = 0, sumY = 0, sumXX = 0, sumXY = 0;
  for (let i = 0; i < N; i++) {
    sumX += i;
    sumY += signal[i]!;
    sumXX += i * i;
    sumXY += i * signal[i]!;
  }
  const denom = N * sumXX - sumX * sumX;
  const b = denom !== 0 ? (N * sumXY - sumX * sumY) / denom : 0;
  const a = (sumY - b * sumX) / N;

  const result = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    result[i] = signal[i]! - (a + b * i);
  }
  return result;
}

/**
 * In-place Cooley-Tukey radix-2 DIT FFT.
 * Input arrays must have power-of-2 length.
 */
export function fftInPlace(re: Float64Array, im: Float64Array): void {
  const N = re.length;
  if (!isPow2(N)) throw new Error(`FFT length must be power of 2, got ${N}`);

  // Bit-reversal permutation
  for (let i = 1, j = 0; i < N; i++) {
    let bit = N >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      let tmp = re[i]!;
      re[i] = re[j]!;
      re[j] = tmp;
      tmp = im[i]!;
      im[i] = im[j]!;
      im[j] = tmp;
    }
  }

  // Butterfly stages
  for (let len = 2; len <= N; len <<= 1) {
    const halfLen = len >> 1;
    const angleStep = -2 * Math.PI / len;
    const wRe = Math.cos(angleStep);
    const wIm = Math.sin(angleStep);

    for (let i = 0; i < N; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let j = 0; j < halfLen; j++) {
        const evenIdx = i + j;
        const oddIdx = i + j + halfLen;
        const tRe = curRe * re[oddIdx]! - curIm * im[oddIdx]!;
        const tIm = curRe * im[oddIdx]! + curIm * re[oddIdx]!;
        re[oddIdx] = re[evenIdx]! - tRe;
        im[oddIdx] = im[evenIdx]! - tIm;
        re[evenIdx] = re[evenIdx]! + tRe;
        im[evenIdx] = im[evenIdx]! + tIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
}

/**
 * Inverse FFT via conjugate trick: IFFT(X) = conj(FFT(conj(X)))/N
 */
export function ifftInPlace(re: Float64Array, im: Float64Array): void {
  const N = re.length;
  // Conjugate
  for (let i = 0; i < N; i++) im[i] = -im[i]!;
  fftInPlace(re, im);
  // Conjugate and scale
  for (let i = 0; i < N; i++) {
    re[i] = re[i]! / N;
    im[i] = -im[i]! / N;
  }
}

/**
 * Compute FFT of a real signal. Returns complex spectrum.
 * Automatically zero-pads to next power of 2.
 */
export function fft(signal: Float64Array, nfft?: number): { re: Float64Array; im: Float64Array; N: number } {
  const N = nfft ?? nextPow2(signal.length);
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  re.set(signal.subarray(0, Math.min(signal.length, N)));
  fftInPlace(re, im);
  return { re, im, N };
}

/**
 * Inverse FFT: reconstruct real signal from complex spectrum.
 */
export function ifft(re: Float64Array, im: Float64Array): Float64Array {
  const reOut = new Float64Array(re);
  const imOut = new Float64Array(im);
  ifftInPlace(reOut, imOut);
  return reOut;
}

/**
 * Compute magnitude spectrum (positive frequencies only).
 */
export function magnitudeSpectrum(signal: Float64Array, fs: number = 1, windowType: WindowFunction = 'blackman'): SpectralResult {
  const detrended = detrend(signal);
  const windowed = applyWindow(detrended, windowType);
  const N = nextPow2(signal.length) * 2; // Extra padding for interpolation
  const { re, im } = fft(windowed, N);

  const nPos = Math.floor(N / 2);
  const frequencies = new Float64Array(nPos);
  const magnitudes = new Float64Array(nPos);
  const phases = new Float64Array(nPos);

  for (let k = 0; k < nPos; k++) {
    frequencies[k] = (k * fs) / N;
    magnitudes[k] = (2 / signal.length) * Math.sqrt(re[k]! * re[k]! + im[k]! * im[k]!);
    phases[k] = Math.atan2(im[k]!, re[k]!);
  }

  return { frequencies, magnitudes, phases };
}

/**
 * Extract dominant seasonal periods from a booking time series.
 * DFT with zero-padding, peak detection.
 */
export function extractSeasonality(
  bookings: Float64Array,
  fs: number = 1,
  windowType: WindowFunction = 'blackman',
  maxPeaks: number = 10,
): SeasonalityResult {
  const { frequencies, magnitudes, phases } = magnitudeSpectrum(bookings, fs, windowType);

  // Find peaks: local maxima above 5% of max
  const maxMag = magnitudes.reduce((a, b) => Math.max(a, b), 0);
  const threshold = maxMag * 0.05;
  const peakIndices: number[] = [];

  for (let i = 2; i < magnitudes.length - 2; i++) {
    if (
      magnitudes[i]! > threshold &&
      magnitudes[i]! > magnitudes[i - 1]! &&
      magnitudes[i]! > magnitudes[i + 1]! &&
      magnitudes[i]! > magnitudes[i - 2]! &&
      magnitudes[i]! > magnitudes[i + 2]!
    ) {
      peakIndices.push(i);
    }
  }

  // Sort by magnitude descending, take top maxPeaks
  peakIndices.sort((a, b) => magnitudes[b]! - magnitudes[a]!);
  const topPeaks = peakIndices.slice(0, maxPeaks);

  // Compute periods
  const periods = new Float64Array(frequencies.length);
  for (let i = 0; i < frequencies.length; i++) {
    periods[i] = frequencies[i]! > 0 ? 1 / frequencies[i]! : Infinity;
  }

  const dominantPeriods = topPeaks.map(i => ({
    period: periods[i]!,
    magnitude: magnitudes[i]!,
  }));

  return { frequencies, magnitudes, periods, dominantPeriods };
}

/**
 * Frequency-domain filtering: keep only target periodic components
 * and reconstruct via IFFT.
 */
export function reconstructSeasonal(
  bookings: Float64Array,
  targetPeriods: number[],
  fs: number = 1,
  bandwidthBins: number = 3,
): Float64Array {
  const N = bookings.length;
  const detrended = detrend(bookings);
  const Npad = nextPow2(N) * 2;
  const { re, im } = fft(detrended, Npad);

  const filtRe = new Float64Array(Npad);
  const filtIm = new Float64Array(Npad);

  for (const targetP of targetPeriods) {
    const targetFreq = 1 / targetP;
    // Find closest bin
    let bestIdx = 0;
    let bestDist = Infinity;
    for (let k = 1; k < Npad / 2; k++) {
      const freq = (k * fs) / Npad;
      const dist = Math.abs(freq - targetFreq);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = k;
      }
    }

    // Copy bins within bandwidth
    for (let d = -bandwidthBins; d <= bandwidthBins; d++) {
      const idx = bestIdx + d;
      if (idx >= 0 && idx < Npad) {
        filtRe[idx] = re[idx]!;
        filtIm[idx] = im[idx]!;
      }
      // Mirror (negative freq)
      const mirrorIdx = Npad - idx;
      if (mirrorIdx >= 0 && mirrorIdx < Npad) {
        filtRe[mirrorIdx] = re[mirrorIdx]!;
        filtIm[mirrorIdx] = im[mirrorIdx]!;
      }
    }
  }

  const reconstructed = ifft(filtRe, filtIm);
  return reconstructed.subarray(0, N);
}

/**
 * FFT-based convolution: O(N log N) vs O(N²) direct.
 */
export function fftConvolve(signal: Float64Array, kernel: Float64Array): Float64Array {
  const outLen = signal.length + kernel.length - 1;
  const N = nextPow2(outLen);

  const sigResult = fft(signal, N);
  const kerResult = fft(kernel, N);

  // Pointwise complex multiply
  const prodRe = new Float64Array(N);
  const prodIm = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    prodRe[i] = sigResult.re[i]! * kerResult.re[i]! - sigResult.im[i]! * kerResult.im[i]!;
    prodIm[i] = sigResult.re[i]! * kerResult.im[i]! + sigResult.im[i]! * kerResult.re[i]!;
  }

  const result = ifft(prodRe, prodIm);
  return result.subarray(0, outLen);
}
