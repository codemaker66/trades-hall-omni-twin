// ---------------------------------------------------------------------------
// SP-10: Stockwell Transform (S-Transform)
// ---------------------------------------------------------------------------
// Combines STFT absolute phase with CWT multiresolution.
// S(τ,f) = ∫ x(t)·|f|/(√2π)·e^{-(τ-t)²f²/2}·e^{-j2πft} dt
// Gaussian window width scales inversely with frequency.

import type { StockwellResult } from '../types.js';
import { fft, ifft, nextPow2 } from '../fourier/fft.js';

/**
 * Stockwell Transform via FFT.
 * For each frequency f, the Gaussian window adapts: σ = 1/|f|.
 *
 * @param signal Input signal
 * @param fs Sample rate
 * @param fMin Minimum frequency (0 = include DC)
 * @param fMax Maximum frequency (default: Nyquist)
 */
export function stockwellTransform(
  signal: Float64Array,
  fs: number = 1,
  fMin: number = 0,
  fMax?: number,
): StockwellResult {
  const N = signal.length;
  const Nfft = nextPow2(N);
  const nyquist = fs / 2;
  const maxF = fMax ?? nyquist;

  // FFT of input signal
  const { re: sigRe, im: sigIm } = fft(signal, Nfft);

  // Determine frequency bins to compute
  const fMinBin = Math.max(0, Math.floor((fMin / fs) * Nfft));
  const fMaxBin = Math.min(Math.floor(Nfft / 2), Math.ceil((maxF / fs) * Nfft));
  const nFreqs = fMaxBin - fMinBin + 1;
  const nTimes = N;

  const frequencies = new Float64Array(nFreqs);
  const times = new Float64Array(nTimes);
  const stransform = new Float64Array(nFreqs * nTimes);

  for (let t = 0; t < nTimes; t++) {
    times[t] = t / fs;
  }

  for (let fi = 0; fi < nFreqs; fi++) {
    const fBin = fMinBin + fi;
    frequencies[fi] = (fBin * fs) / Nfft;

    if (fBin === 0) {
      // DC component: mean of signal
      let mean = 0;
      for (let t = 0; t < N; t++) mean += signal[t]!;
      mean /= N;
      for (let t = 0; t < nTimes; t++) {
        stransform[fi * nTimes + t] = Math.abs(mean);
      }
      continue;
    }

    // Gaussian window in frequency domain for this f:
    // W(ν,f) = e^{-2π²ν²/f²}
    const gaussRe = new Float64Array(Nfft);
    const gaussIm = new Float64Array(Nfft);

    for (let nu = 0; nu < Nfft; nu++) {
      const nuNorm = nu <= Nfft / 2 ? nu : nu - Nfft;
      const gauss = Math.exp(-2 * Math.PI * Math.PI * nuNorm * nuNorm / (fBin * fBin));

      // Shift: multiply signal FFT by gaussian, then shift by f
      const shiftIdx = ((nu + fBin) % Nfft + Nfft) % Nfft;
      gaussRe[nu] = gauss * sigRe[shiftIdx]!;
      gaussIm[nu] = gauss * sigIm[shiftIdx]!;
    }

    // IFFT to get time-localized transform at this frequency
    const localRe = ifft(gaussRe, gaussIm);

    for (let t = 0; t < nTimes; t++) {
      stransform[fi * nTimes + t] = Math.abs(localRe[t]!);
    }
  }

  return { stransform, frequencies, times, nFreqs, nTimes };
}
