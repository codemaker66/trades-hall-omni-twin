// ---------------------------------------------------------------------------
// SP-2: Discrete Wavelet Transform (DWT) — Mallat's Pyramid Algorithm
// ---------------------------------------------------------------------------
// cA_j[n] = Σ_k h[k-2n]·cA_{j-1}[k]  (low-pass + downsample)
// cD_j[n] = Σ_k g[k-2n]·cA_{j-1}[k]  (high-pass + downsample)
// O(N) per level. Total O(N) for full decomposition.

import type { WaveletFamily, DWTResult } from '../types.js';

/** Low-pass (scaling) filter coefficients for each wavelet family. */
export function getScalingFilter(wavelet: WaveletFamily): Float64Array {
  switch (wavelet) {
    case 'haar':
      return new Float64Array([
        0.7071067811865476,
        0.7071067811865476,
      ]);
    case 'db4':
      return new Float64Array([
        -0.010597401784997278, 0.032883011666982945,
        0.030841381835986965, -0.18703481171888114,
        -0.02798376941698385, 0.6308807679295904,
        0.7148465705525415, 0.23037781330885523,
      ]);
    case 'db8':
      return new Float64Array([
        -0.00011747678400228192, 0.0006754494059985568,
        -0.0003917403729959771, -0.00487035299301066,
        0.008746094047015655, 0.013981027917015516,
        -0.04408825393079038, -0.01736930100202211,
        0.128747426620186, 0.00047248457399797254,
        -0.2840155429624281, -0.015829105256023893,
        0.5853546836548691, 0.6756307362980128,
        0.3128715909144659, 0.05441584224308161,
      ]);
    case 'sym4':
      return new Float64Array([
        -0.07576571478927333, -0.02963552764599851,
        0.49761866763201545, 0.8037387518059161,
        0.29785779560527736, -0.09921954357684722,
        -0.012603967262037833, 0.032223100604042702,
      ]);
    case 'coif2':
      return new Float64Array([
        -0.0007205494453645122, -0.0018232088707029932,
        0.0056114348193944995, 0.023680171946334084,
        -0.0594344186464569, -0.0764885990783064,
        0.41700518442169254, 0.8127236354455423,
        0.3861100668211622, -0.06737255472196302,
        -0.04146493678175915, 0.016387336463522112,
      ]);
  }
}

/** Derive high-pass (wavelet) filter from low-pass via QMF. */
export function getWaveletFilter(wavelet: WaveletFamily): Float64Array {
  const h = getScalingFilter(wavelet);
  const g = new Float64Array(h.length);
  for (let i = 0; i < h.length; i++) {
    g[i] = ((i & 1) === 0 ? 1 : -1) * h[h.length - 1 - i]!;
  }
  return g;
}

/** Polyphase decomposition with periodic extension: cA[n] = Σ_k h[k]·x[(2n+k) mod N]. */
function convolveDownsample(signal: Float64Array, filter: Float64Array): Float64Array {
  const N = signal.length;
  const L = filter.length;
  const outLen = Math.ceil(N / 2);
  const result = new Float64Array(outLen);

  for (let i = 0; i < outLen; i++) {
    let sum = 0;
    for (let k = 0; k < L; k++) {
      const idx = (2 * i + k) % N;
      sum += signal[idx]! * filter[k]!;
    }
    result[i] = sum;
  }
  return result;
}

/** Reconstruction via upsampling + circular convolution: x[n] = Σ_k h[k]·up[(n-k) mod N]. */
function upsampleConvolve(coeffs: Float64Array, filter: Float64Array, originalLen: number): Float64Array {
  const M = coeffs.length;
  const L = filter.length;
  const N = originalLen;
  const result = new Float64Array(N);

  for (let n = 0; n < N; n++) {
    let sum = 0;
    for (let k = 0; k < L; k++) {
      const j = ((n - k) % N + N) % N; // periodic index into upsampled signal
      if (j % 2 === 0) {
        const m = j / 2;
        if (m < M) {
          sum += filter[k]! * coeffs[m]!;
        }
      }
    }
    result[n] = sum;
  }
  return result;
}

/** Maximum decomposition level for given signal length and wavelet. */
export function maxLevel(signalLength: number, wavelet: WaveletFamily): number {
  const filterLen = getScalingFilter(wavelet).length;
  return Math.max(0, Math.floor(Math.log2(signalLength / (filterLen - 1))));
}

/**
 * DWT decomposition via Mallat's pyramid algorithm.
 * Returns approximation coefficients + detail coefficients at each level.
 */
export function dwtDecompose(
  signal: Float64Array,
  wavelet: WaveletFamily = 'db4',
  levels?: number,
): DWTResult {
  const maxLev = maxLevel(signal.length, wavelet);
  const nLevels = levels ?? maxLev;
  const h = getScalingFilter(wavelet);
  const g = getWaveletFilter(wavelet);

  const details: Float64Array[] = [];
  let approx: Float64Array = new Float64Array(signal);

  for (let level = 0; level < nLevels; level++) {
    const detail = convolveDownsample(approx, g);
    approx = convolveDownsample(approx, h);
    details.push(detail);
  }

  return {
    approximation: approx,
    details,
    wavelet,
    levels: nLevels,
  };
}

/**
 * DWT reconstruction from approximation + detail coefficients.
 * Uses the same filters h, g for synthesis (orthogonal wavelets).
 */
export function dwtReconstruct(result: DWTResult): Float64Array {
  const h = getScalingFilter(result.wavelet);
  const g = getWaveletFilter(result.wavelet);

  let approx = result.approximation;

  for (let level = result.levels - 1; level >= 0; level--) {
    const detail = result.details[level]!;
    const targetLen = approx.length * 2;

    const upApprox = upsampleConvolve(approx, h, targetLen);
    const upDetail = upsampleConvolve(detail, g, targetLen);

    approx = new Float64Array(targetLen);
    for (let i = 0; i < targetLen; i++) {
      approx[i] = upApprox[i]! + upDetail[i]!;
    }
  }

  return approx;
}
