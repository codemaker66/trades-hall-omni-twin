// ---------------------------------------------------------------------------
// SP-2: Maximal Overlap DWT (MODWT) — Shift-Invariant Wavelet Transform
// ---------------------------------------------------------------------------
// Advantages over DWT:
// - Shift-invariant (shifting input shifts coefficients by same amount)
// - Works for arbitrary sample sizes (DWT needs powers of 2)
// - Better temporal alignment
// Cost: N coefficients per level (redundant) vs N/2^j for DWT

import type { WaveletFamily, MODWTResult } from '../types.js';
import { getScalingFilter, getWaveletFilter, maxLevel } from './dwt.js';

/**
 * MODWT decomposition via non-decimated filter bank.
 * Filter h_j = h / √2^j (rescaled at each level, no downsampling).
 */
export function modwtDecompose(
  signal: Float64Array,
  wavelet: WaveletFamily = 'db4',
  levels?: number,
): MODWTResult {
  const N = signal.length;
  const maxLev = maxLevel(N, wavelet);
  const nLevels = Math.min(levels ?? maxLev, maxLev);

  const h = getScalingFilter(wavelet);
  const g = getWaveletFilter(wavelet);

  // MODWT uses filters scaled by 1/√2 at each level
  const hMod = new Float64Array(h.length);
  const gMod = new Float64Array(g.length);
  for (let i = 0; i < h.length; i++) {
    hMod[i] = h[i]! / Math.SQRT2;
    gMod[i] = g[i]! / Math.SQRT2;
  }

  const details: Float64Array[] = [];
  let approx = new Float64Array(signal);

  for (let level = 0; level < nLevels; level++) {
    const detail = new Float64Array(N);
    const newApprox = new Float64Array(N);
    const step = 1 << level; // Dyadic upsampling of filter

    for (let n = 0; n < N; n++) {
      let sumLow = 0;
      let sumHigh = 0;
      for (let k = 0; k < hMod.length; k++) {
        const idx = ((n - k * step) % N + N) % N; // Circular convolution
        sumLow += hMod[k]! * approx[idx]!;
        sumHigh += gMod[k]! * approx[idx]!;
      }
      newApprox[n] = sumLow;
      detail[n] = sumHigh;
    }

    details.push(detail);
    approx = newApprox;
  }

  return {
    approximation: approx,
    details,
    wavelet,
    levels: nLevels,
  };
}

/**
 * MODWT reconstruction via inverse filter bank.
 * Reconstructs the original signal from approximation + detail components.
 */
export function modwtReconstruct(result: MODWTResult): Float64Array {
  const N = result.approximation.length;
  const h = getScalingFilter(result.wavelet);
  const g = getWaveletFilter(result.wavelet);

  const hMod = new Float64Array(h.length);
  const gMod = new Float64Array(g.length);
  for (let i = 0; i < h.length; i++) {
    hMod[i] = h[i]! / Math.SQRT2;
    gMod[i] = g[i]! / Math.SQRT2;
  }

  let approx = new Float64Array(result.approximation);

  for (let level = result.levels - 1; level >= 0; level--) {
    const detail = result.details[level]!;
    const step = 1 << level;
    const reconstructed = new Float64Array(N);

    for (let n = 0; n < N; n++) {
      let sumLow = 0;
      let sumHigh = 0;
      for (let k = 0; k < hMod.length; k++) {
        const idx = ((n + k * step) % N + N) % N;
        sumLow += hMod[k]! * approx[idx]!;
        sumHigh += gMod[k]! * detail[idx]!;
      }
      reconstructed[n] = sumLow + sumHigh;
    }

    approx = reconstructed;
  }

  return approx;
}

/**
 * Multi-Resolution Analysis (MRA) via MODWT.
 * Returns additive components that sum to the original signal:
 * x = D1 + D2 + ... + DJ + AJ
 */
export function modwtMRA(
  signal: Float64Array,
  wavelet: WaveletFamily = 'db4',
  levels?: number,
): { details: Float64Array[]; approximation: Float64Array } {
  const decomp = modwtDecompose(signal, wavelet, levels);
  const N = signal.length;

  // Reconstruct each detail component individually
  const mraDetails: Float64Array[] = [];

  for (let level = 0; level < decomp.levels; level++) {
    // Create a result with only this detail level active
    const singleDetail: MODWTResult = {
      approximation: new Float64Array(N), // zeros
      details: [],
      wavelet: decomp.wavelet,
      levels: level + 1,
    };
    for (let l = 0; l <= level; l++) {
      singleDetail.details.push(
        l === level ? decomp.details[l]! : new Float64Array(N),
      );
    }
    const component = modwtReconstruct(singleDetail);
    mraDetails.push(component);
  }

  // Approximation component is what remains
  const approxComponent = new Float64Array(N);
  const detailSum = new Float64Array(N);
  for (const d of mraDetails) {
    for (let i = 0; i < N; i++) {
      detailSum[i] = detailSum[i]! + d[i]!;
    }
  }
  for (let i = 0; i < N; i++) {
    approxComponent[i] = signal[i]! - detailSum[i]!;
  }

  return { details: mraDetails, approximation: approxComponent };
}
