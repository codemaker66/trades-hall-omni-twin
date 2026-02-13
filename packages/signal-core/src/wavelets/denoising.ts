// ---------------------------------------------------------------------------
// SP-2: Wavelet Denoising
// ---------------------------------------------------------------------------
// Hard: η_H(w,λ) = w if |w|>λ, else 0
// Soft: η_S(w,λ) = sign(w)·max(|w|-λ, 0)
// Universal threshold: λ = σ̂·√(2 ln n) (Donoho-Johnstone)
// σ̂ = MAD(cD₁)/0.6745  (noise from finest detail level)
// BayesShrink: adapts per level λ_j = σ̂²/σ̂_{x,j}

import type { WaveletFamily, DenoiseConfig } from '../types.js';
import { dwtDecompose, dwtReconstruct } from './dwt.js';

/**
 * Soft thresholding: sign(w)·max(|w|-λ, 0)
 */
function softThreshold(coeffs: Float64Array, threshold: number): Float64Array {
  const result = new Float64Array(coeffs.length);
  for (let i = 0; i < coeffs.length; i++) {
    const val = coeffs[i]!;
    if (Math.abs(val) > threshold) {
      result[i] = Math.sign(val) * (Math.abs(val) - threshold);
    }
  }
  return result;
}

/**
 * Hard thresholding: w if |w|>λ, else 0
 */
function hardThreshold(coeffs: Float64Array, threshold: number): Float64Array {
  const result = new Float64Array(coeffs.length);
  for (let i = 0; i < coeffs.length; i++) {
    const val = coeffs[i]!;
    result[i] = Math.abs(val) > threshold ? val : 0;
  }
  return result;
}

/**
 * Median Absolute Deviation: MAD = median(|x - median(x)|)
 */
function mad(data: Float64Array): number {
  const sorted = Float64Array.from(data).sort();
  const med = median(sorted);
  const deviations = new Float64Array(data.length);
  for (let i = 0; i < data.length; i++) {
    deviations[i] = Math.abs(data[i]! - med);
  }
  return median(Float64Array.from(deviations).sort());
}

/** Median of a sorted array. */
function median(sorted: Float64Array): number {
  const n = sorted.length;
  if (n === 0) return 0;
  if (n % 2 === 1) return sorted[Math.floor(n / 2)]!;
  return (sorted[n / 2 - 1]! + sorted[n / 2]!) / 2;
}

/**
 * Estimate noise standard deviation from finest detail coefficients.
 * σ̂ = MAD(cD₁) / 0.6745
 */
function estimateNoiseSigma(finestDetail: Float64Array): number {
  return mad(finestDetail) / 0.6745;
}

/**
 * Universal threshold (Donoho-Johnstone): λ = σ̂·√(2 ln n)
 */
function universalThreshold(sigma: number, n: number): number {
  return sigma * Math.sqrt(2 * Math.log(n));
}

/**
 * BayesShrink threshold per level: λ_j = σ̂² / σ̂_{x,j}
 * where σ̂_{x,j} = √(max(σ²_j - σ̂², 0)) is signal std at level j
 */
function bayesShrinkThreshold(detailCoeffs: Float64Array, noiseSigma: number): number {
  let energy = 0;
  for (let i = 0; i < detailCoeffs.length; i++) {
    energy += detailCoeffs[i]! * detailCoeffs[i]!;
  }
  const sigmaY = Math.sqrt(energy / detailCoeffs.length);
  const sigmaX = Math.sqrt(Math.max(sigmaY * sigmaY - noiseSigma * noiseSigma, 0));

  if (sigmaX < 1e-10) {
    // Signal component is negligible → kill everything
    return Infinity;
  }
  return (noiseSigma * noiseSigma) / sigmaX;
}

/**
 * Wavelet denoising via thresholding on DWT detail coefficients.
 */
export function waveletDenoise(
  signal: Float64Array,
  config: DenoiseConfig = {
    wavelet: 'db4',
    method: 'soft',
    thresholdRule: 'universal',
  },
): Float64Array {
  const { wavelet, method, thresholdRule, levels } = config;
  const result = dwtDecompose(signal, wavelet, levels);

  if (result.details.length === 0) return new Float64Array(signal);

  // Estimate noise from finest detail level
  const sigma = estimateNoiseSigma(result.details[0]!);
  const thresholdFn = method === 'soft' ? softThreshold : hardThreshold;

  // Threshold each detail level
  const thresholdedDetails: Float64Array[] = [];
  for (let j = 0; j < result.details.length; j++) {
    const detail = result.details[j]!;
    let threshold: number;

    if (thresholdRule === 'bayes-shrink') {
      threshold = bayesShrinkThreshold(detail, sigma);
    } else {
      threshold = universalThreshold(sigma, signal.length);
    }

    thresholdedDetails.push(thresholdFn(detail, threshold));
  }

  return dwtReconstruct({
    approximation: result.approximation,
    details: thresholdedDetails,
    wavelet: result.wavelet,
    levels: result.levels,
  });
}
