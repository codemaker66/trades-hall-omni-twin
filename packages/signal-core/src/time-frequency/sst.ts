// ---------------------------------------------------------------------------
// SP-10: Synchrosqueezing Transform (SST)
// ---------------------------------------------------------------------------
// Reassigns CWT coefficients for sharper TF representation
// beyond Heisenberg uncertainty limit for AM-FM signals.
// SST: ω(a,b) = -Im(∂_b W_x(a,b)) / W_x(a,b)
// Then reassign: T_x(ω,b) = Σ_a W_x(a,b)·δ(ω - ω(a,b))

import type { SSTResult } from '../types.js';
import { fft, ifft, nextPow2 } from '../fourier/fft.js';

/**
 * Morlet wavelet in frequency domain: Ψ̂(ω) = π^{-1/4}·e^{-(ω-ω₀)²/2}
 */
function morletFreq(omega: number, omega0: number = 6): number {
  const x = omega - omega0;
  return Math.pow(Math.PI, -0.25) * Math.exp(-0.5 * x * x);
}

/**
 * Continuous Wavelet Transform (CWT) via FFT for efficiency.
 * Used as intermediate step for SST.
 */
function cwt(
  signal: Float64Array,
  scales: Float64Array,
  fs: number,
): { coeffRe: Float64Array; coeffIm: Float64Array; nScales: number; nTimes: number } {
  const N = signal.length;
  const Nfft = nextPow2(N) * 2;
  const { re: sigRe, im: sigIm } = fft(signal, Nfft);

  const nScales = scales.length;
  const nTimes = N;
  const coeffRe = new Float64Array(nScales * nTimes);
  const coeffIm = new Float64Array(nScales * nTimes);

  for (let s = 0; s < nScales; s++) {
    const scale = scales[s]!;

    // Wavelet in frequency domain at this scale
    const wavRe = new Float64Array(Nfft);
    const wavIm = new Float64Array(Nfft);

    for (let k = 0; k < Nfft; k++) {
      const omega = (2 * Math.PI * k) / Nfft;
      const scaledOmega = scale * omega * fs;
      wavRe[k] = Math.sqrt(scale) * morletFreq(scaledOmega);
    }

    // Multiply in frequency domain: conj(Ψ̂) · X̂
    const prodRe = new Float64Array(Nfft);
    const prodIm = new Float64Array(Nfft);
    for (let k = 0; k < Nfft; k++) {
      // conj(wavelet) * signal
      prodRe[k] = wavRe[k]! * sigRe[k]! + wavIm[k]! * sigIm[k]!;
      prodIm[k] = wavRe[k]! * sigIm[k]! - wavIm[k]! * sigRe[k]!;
    }

    // IFFT to get CWT coefficients at this scale
    const timeRe = ifft(prodRe, prodIm);
    for (let t = 0; t < nTimes; t++) {
      coeffRe[s * nTimes + t] = timeRe[t]!;
    }

    // For the imaginary part, we'd need the analytic signal
    // Approximation: use Hilbert-like approach
    const prodReH = new Float64Array(Nfft);
    const prodImH = new Float64Array(Nfft);
    for (let k = 0; k < Nfft; k++) {
      // Multiply by -j·sgn(k) for Hilbert
      const sgn = k > 0 && k < Nfft / 2 ? 1 : k > Nfft / 2 ? -1 : 0;
      prodReH[k] = sgn * prodIm[k]!;
      prodImH[k] = -sgn * prodRe[k]!;
    }
    const timeIm = ifft(prodReH, prodImH);
    for (let t = 0; t < nTimes; t++) {
      coeffIm[s * nTimes + t] = timeIm[t]!;
    }
  }

  return { coeffRe, coeffIm, nScales, nTimes };
}

/**
 * Synchrosqueezing Transform.
 *
 * @param signal Input signal
 * @param fs Sample rate
 * @param nVoices Voices per octave (frequency resolution, default 32)
 * @param fMin Minimum frequency (default fs/N)
 * @param fMax Maximum frequency (default fs/2)
 */
export function synchrosqueezingTransform(
  signal: Float64Array,
  fs: number = 1,
  nVoices: number = 32,
  fMin?: number,
  fMax?: number,
): SSTResult {
  const N = signal.length;
  const minF = fMin ?? fs / N;
  const maxF = fMax ?? fs / 2;

  // Generate scales (log-spaced)
  const nScales = nVoices * Math.max(1, Math.ceil(Math.log2(maxF / minF)));
  const scales = new Float64Array(nScales);
  for (let s = 0; s < nScales; s++) {
    const f = minF * Math.pow(maxF / minF, s / (nScales - 1));
    scales[s] = 1 / f; // scale = 1/frequency for Morlet
  }

  // Compute CWT
  const { coeffRe, coeffIm, nTimes } = cwt(signal, scales, fs);

  // Frequency bins for output
  const nFreqs = nScales;
  const frequencies = new Float64Array(nFreqs);
  for (let s = 0; s < nScales; s++) {
    frequencies[s] = 1 / scales[s]!;
  }

  const times = new Float64Array(nTimes);
  for (let t = 0; t < nTimes; t++) {
    times[t] = t / fs;
  }

  // Synchrosqueezing: compute instantaneous frequency and reassign
  const tfr = new Float64Array(nFreqs * nTimes);

  for (let s = 0; s < nScales; s++) {
    for (let t = 1; t < nTimes - 1; t++) {
      const re = coeffRe[s * nTimes + t]!;
      const im = coeffIm[s * nTimes + t]!;
      const mag2 = re * re + im * im;

      if (mag2 < 1e-20) continue;

      // Phase derivative (finite difference)
      const reNext = coeffRe[s * nTimes + t + 1]!;
      const imNext = coeffIm[s * nTimes + t + 1]!;
      const rePrev = coeffRe[s * nTimes + t - 1]!;
      const imPrev = coeffIm[s * nTimes + t - 1]!;

      const phaseNext = Math.atan2(imNext, reNext);
      const phasePrev = Math.atan2(imPrev, rePrev);
      let dPhase = phaseNext - phasePrev;
      // Unwrap
      while (dPhase > Math.PI) dPhase -= 2 * Math.PI;
      while (dPhase < -Math.PI) dPhase += 2 * Math.PI;

      const instFreq = Math.abs(dPhase * fs / (4 * Math.PI));

      // Find nearest frequency bin
      let bestBin = 0;
      let bestDist = Infinity;
      for (let f = 0; f < nFreqs; f++) {
        const dist = Math.abs(frequencies[f]! - instFreq);
        if (dist < bestDist) {
          bestDist = dist;
          bestBin = f;
        }
      }

      // Reassign
      tfr[bestBin * nTimes + t] = tfr[bestBin * nTimes + t]! + Math.sqrt(mag2);
    }
  }

  return { tfr, frequencies, times, nFreqs, nTimes };
}
