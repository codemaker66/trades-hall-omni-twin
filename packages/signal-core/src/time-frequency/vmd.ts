// ---------------------------------------------------------------------------
// SP-10: Variational Mode Decomposition (VMD)
// ---------------------------------------------------------------------------
// Solves variational optimization for K band-limited modes.
// More robust to noise than EMD.
// min Σ_k ||∂_t[δ(t) + jπt) * u_k(t)]·e^{-jω_k·t}||²
// subject to: Σ u_k = f

import type { VMDConfig, VMDResult } from '../types.js';
import { fft, ifft, nextPow2 } from '../fourier/fft.js';

/**
 * Variational Mode Decomposition.
 *
 * @param signal Input signal
 * @param config VMD parameters
 */
export function vmd(signal: Float64Array, config: VMDConfig): VMDResult {
  const { nModes, alpha, tau, dc, maxIter, tolerance } = config;
  const N = signal.length;
  const Nfft = nextPow2(N) * 2;
  const halfN = Math.floor(Nfft / 2);

  // Compute FFT of signal
  const { re: fRe, im: fIm } = fft(signal, Nfft);
  const fHat = new Float64Array(Nfft * 2); // Complex: [re0, im0, re1, im1, ...]
  for (let k = 0; k < Nfft; k++) {
    fHat[k * 2] = fRe[k]!;
    fHat[k * 2 + 1] = fIm[k]!;
  }

  // Frequency array
  const freqs = new Float64Array(Nfft);
  for (let k = 0; k < Nfft; k++) {
    freqs[k] = k <= halfN ? k / Nfft : (k - Nfft) / Nfft;
  }

  // Initialize modes and center frequencies
  const modes = new Array(nModes).fill(null).map(() => new Float64Array(Nfft * 2));
  const centerFreqs = new Float64Array(nModes);

  // Initialize center frequencies evenly spaced
  for (let k = 0; k < nModes; k++) {
    centerFreqs[k] = (0.5 * (k + 1)) / (nModes + 1);
    if (dc && k === 0) centerFreqs[k] = 0;
  }

  // Lagrangian multiplier
  const lambda = new Float64Array(Nfft * 2);

  let nIterations = 0;

  for (let iter = 0; iter < maxIter; iter++) {
    let maxDiff = 0;
    nIterations++;

    for (let k = 0; k < nModes; k++) {
      const prevMode = new Float64Array(modes[k]!);

      // Sum of all modes except k
      const sumOthers = new Float64Array(Nfft * 2);
      for (let j = 0; j < nModes; j++) {
        if (j === k) continue;
        for (let f = 0; f < Nfft * 2; f++) {
          sumOthers[f] = sumOthers[f]! + modes[j]![f]!;
        }
      }

      // Update mode k in frequency domain:
      // û_k = (f̂ - Σ_{j≠k} û_j + λ̂/(2)) / (1 + 2α(ω - ω_k)²)
      for (let f = 0; f < Nfft; f++) {
        const numeratorRe = fHat[f * 2]! - sumOthers[f * 2]! + lambda[f * 2]! / 2;
        const numeratorIm = fHat[f * 2 + 1]! - sumOthers[f * 2 + 1]! + lambda[f * 2 + 1]! / 2;
        const freqDiff = freqs[f]! - centerFreqs[k]!;
        const denominator = 1 + 2 * alpha * freqDiff * freqDiff;
        modes[k]![f * 2] = numeratorRe / denominator;
        modes[k]![f * 2 + 1] = numeratorIm / denominator;
      }

      // Update center frequency:
      // ω_k = ∫ ω|û_k(ω)|² dω / ∫ |û_k(ω)|² dω
      let numSum = 0;
      let denSum = 0;
      for (let f = 0; f < halfN; f++) {
        const mag2 = modes[k]![f * 2]! * modes[k]![f * 2]! +
                     modes[k]![f * 2 + 1]! * modes[k]![f * 2 + 1]!;
        numSum += freqs[f]! * mag2;
        denSum += mag2;
      }
      if (denSum > 1e-20) {
        centerFreqs[k] = numSum / denSum;
      }

      // Track convergence
      let diff = 0;
      for (let f = 0; f < Nfft * 2; f++) {
        const d = modes[k]![f]! - prevMode[f]!;
        diff += d * d;
      }
      maxDiff = Math.max(maxDiff, diff);
    }

    // Update Lagrangian: λ̂ += τ·(f̂ - Σ û_k)
    const sumAll = new Float64Array(Nfft * 2);
    for (let k = 0; k < nModes; k++) {
      for (let f = 0; f < Nfft * 2; f++) {
        sumAll[f] = sumAll[f]! + modes[k]![f]!;
      }
    }
    for (let f = 0; f < Nfft * 2; f++) {
      lambda[f] = lambda[f]! + tau * (fHat[f]! - sumAll[f]!);
    }

    if (maxDiff < tolerance) break;
  }

  // Convert modes back to time domain
  const timeModes: Float64Array[] = [];
  for (let k = 0; k < nModes; k++) {
    const modeRe = new Float64Array(Nfft);
    const modeIm = new Float64Array(Nfft);
    for (let f = 0; f < Nfft; f++) {
      modeRe[f] = modes[k]![f * 2]!;
      modeIm[f] = modes[k]![f * 2 + 1]!;
    }
    const timeMode = ifft(modeRe, modeIm);
    timeModes.push(timeMode.subarray(0, N));
  }

  return {
    modes: timeModes,
    centerFrequencies: centerFreqs,
    nIterations,
  };
}
