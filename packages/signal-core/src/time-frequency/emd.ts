// ---------------------------------------------------------------------------
// SP-10: Empirical Mode Decomposition (EMD)
// ---------------------------------------------------------------------------
// Adaptively decomposes into Intrinsic Mode Functions (IMFs).
// No predefined basis — data-driven.
// CEEMDAN solves mode mixing with exact reconstruction.
// Sifting process: iteratively remove local means until criteria met.

import type { EMDResult, IMF, PRNG } from '../types.js';
import { createPRNG } from '../types.js';

/**
 * Find local extrema (maxima and minima) of a signal.
 */
function findExtrema(signal: Float64Array): { maxIdx: number[]; minIdx: number[] } {
  const maxIdx: number[] = [];
  const minIdx: number[] = [];
  const N = signal.length;

  for (let i = 1; i < N - 1; i++) {
    if (signal[i]! > signal[i - 1]! && signal[i]! > signal[i + 1]!) {
      maxIdx.push(i);
    }
    if (signal[i]! < signal[i - 1]! && signal[i]! < signal[i + 1]!) {
      minIdx.push(i);
    }
  }

  return { maxIdx, minIdx };
}

/**
 * Cubic spline interpolation through given points.
 * Returns interpolated values at all indices 0..N-1.
 */
function cubicSplineInterp(
  knots: number[],
  values: number[],
  N: number,
): Float64Array {
  const n = knots.length;
  if (n === 0) return new Float64Array(N);
  if (n === 1) {
    const result = new Float64Array(N);
    result.fill(values[0]!);
    return result;
  }

  // Natural cubic spline
  const h = new Float64Array(n - 1);
  const alpha = new Float64Array(n - 1);
  for (let i = 0; i < n - 1; i++) {
    h[i] = knots[i + 1]! - knots[i]!;
  }
  for (let i = 1; i < n - 1; i++) {
    alpha[i] = (3 / h[i]!) * (values[i + 1]! - values[i]!) -
               (3 / h[i - 1]!) * (values[i]! - values[i - 1]!);
  }

  const l = new Float64Array(n);
  const mu = new Float64Array(n);
  const z = new Float64Array(n);
  l[0] = 1;

  for (let i = 1; i < n - 1; i++) {
    l[i] = 2 * (knots[i + 1]! - knots[i - 1]!) - h[i - 1]! * mu[i - 1]!;
    mu[i] = l[i]! !== 0 ? h[i]! / l[i]! : 0;
    z[i] = l[i]! !== 0 ? (alpha[i]! - h[i - 1]! * z[i - 1]!) / l[i]! : 0;
  }

  const c = new Float64Array(n);
  const b = new Float64Array(n - 1);
  const d = new Float64Array(n - 1);

  for (let j = n - 2; j >= 0; j--) {
    c[j] = z[j]! - mu[j]! * c[j + 1]!;
    b[j] = (values[j + 1]! - values[j]!) / h[j]! - h[j]! * (c[j + 1]! + 2 * c[j]!) / 3;
    d[j] = (c[j + 1]! - c[j]!) / (3 * h[j]!);
  }

  // Evaluate at all indices
  const result = new Float64Array(N);
  let seg = 0;
  for (let x = 0; x < N; x++) {
    // Find segment
    while (seg < n - 2 && x > knots[seg + 1]!) seg++;
    if (seg >= n - 1) seg = n - 2;

    const dx = x - knots[seg]!;
    result[x] = values[seg]! + b[seg]! * dx + c[seg]! * dx * dx + d[seg]! * dx * dx * dx;
  }

  return result;
}

/**
 * Compute upper and lower envelopes via spline interpolation of extrema.
 */
function computeEnvelopes(signal: Float64Array): { upper: Float64Array; lower: Float64Array } | null {
  const N = signal.length;
  const { maxIdx, minIdx } = findExtrema(signal);

  if (maxIdx.length < 2 || minIdx.length < 2) return null;

  // Add boundary conditions (mirror endpoints)
  const maxKnots = [0, ...maxIdx, N - 1];
  const maxVals = [signal[0]!, ...maxIdx.map(i => signal[i]!), signal[N - 1]!];
  const minKnots = [0, ...minIdx, N - 1];
  const minVals = [signal[0]!, ...minIdx.map(i => signal[i]!), signal[N - 1]!];

  const upper = cubicSplineInterp(maxKnots, maxVals, N);
  const lower = cubicSplineInterp(minKnots, minVals, N);

  return { upper, lower };
}

/**
 * Check if a signal satisfies IMF conditions:
 * 1. Number of extrema and zero crossings differ by at most 1
 * 2. Mean of upper and lower envelopes is approximately zero
 */
function isIMF(signal: Float64Array, tolerance: number = 0.3): boolean {
  const N = signal.length;
  const { maxIdx, minIdx } = findExtrema(signal);
  const nExtrema = maxIdx.length + minIdx.length;

  // Count zero crossings
  let zeroCrossings = 0;
  for (let i = 1; i < N; i++) {
    if (signal[i]! * signal[i - 1]! < 0) zeroCrossings++;
  }

  if (Math.abs(nExtrema - zeroCrossings) > 1) return false;

  // Check envelope mean
  const envs = computeEnvelopes(signal);
  if (!envs) return true; // Can't compute envelopes → accept

  let meanEnv = 0;
  let signalEnergy = 0;
  for (let i = 0; i < N; i++) {
    meanEnv += Math.abs((envs.upper[i]! + envs.lower[i]!) / 2);
    signalEnergy += signal[i]! * signal[i]!;
  }

  return meanEnv / N < tolerance * Math.sqrt(signalEnergy / N);
}

/**
 * Sifting process: extract one IMF from the signal.
 */
function sift(signal: Float64Array, maxSiftings: number = 100, tolerance: number = 1e-6): Float64Array {
  let h = new Float64Array(signal);
  const N = signal.length;

  for (let iter = 0; iter < maxSiftings; iter++) {
    const envs = computeEnvelopes(h);
    if (!envs) break;

    // Local mean of envelopes
    const mean = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      mean[i] = (envs.upper[i]! + envs.lower[i]!) / 2;
    }

    // Subtract mean
    const hNew = new Float64Array(N);
    let diffNorm = 0;
    let hNorm = 0;
    for (let i = 0; i < N; i++) {
      hNew[i] = h[i]! - mean[i]!;
      const diff = hNew[i]! - h[i]!;
      diffNorm += diff * diff;
      hNorm += h[i]! * h[i]!;
    }

    h = hNew;

    // Cauchy convergence criterion
    if (hNorm > 0 && diffNorm / hNorm < tolerance) break;

    if (isIMF(h)) break;
  }

  return h;
}

/**
 * Empirical Mode Decomposition.
 * Decomposes signal into Intrinsic Mode Functions (IMFs) + residue.
 *
 * @param signal Input time series
 * @param maxIMFs Maximum number of IMFs to extract (default 10)
 * @param maxSiftings Maximum sifting iterations per IMF (default 100)
 */
export function emd(
  signal: Float64Array,
  maxIMFs: number = 10,
  maxSiftings: number = 100,
): EMDResult {
  const N = signal.length;
  const imfs: IMF[] = [];
  let residue = new Float64Array(signal);
  let nIterations = 0;

  for (let k = 0; k < maxIMFs; k++) {
    const { maxIdx, minIdx } = findExtrema(residue);
    if (maxIdx.length + minIdx.length < 4) break; // Not enough extrema

    const imf = sift(residue, maxSiftings);
    nIterations++;

    imfs.push({ data: imf });

    // Update residue
    const newResidue = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      newResidue[i] = residue[i]! - imf[i]!;
    }
    residue = newResidue;

    // Check if residue is monotonic (stopping criterion)
    let isMonotonic = true;
    for (let i = 2; i < N; i++) {
      const d1 = residue[i]! - residue[i - 1]!;
      const d2 = residue[i - 1]! - residue[i - 2]!;
      if (d1 * d2 < 0) {
        isMonotonic = false;
        break;
      }
    }
    if (isMonotonic) break;
  }

  return { imfs, residue, nIterations };
}

/**
 * CEEMDAN: Complete Ensemble EMD with Adaptive Noise.
 * Adds white noise ensembles to mitigate mode mixing.
 *
 * @param signal Input signal
 * @param nEnsembles Number of noise-added trials (default 50)
 * @param noiseStd Noise standard deviation as fraction of signal std (default 0.2)
 * @param seed PRNG seed
 */
export function ceemdan(
  signal: Float64Array,
  nEnsembles: number = 50,
  noiseStd: number = 0.2,
  seed: number = 42,
  maxIMFs: number = 10,
): EMDResult {
  const N = signal.length;
  const rng = createPRNG(seed);

  // Estimate signal standard deviation
  let mean = 0;
  for (let i = 0; i < N; i++) mean += signal[i]!;
  mean /= N;
  let variance = 0;
  for (let i = 0; i < N; i++) {
    const diff = signal[i]! - mean;
    variance += diff * diff;
  }
  const sigmaSignal = Math.sqrt(variance / N);
  const noiseAmplitude = noiseStd * sigmaSignal;

  // Box-Muller for Gaussian noise
  function gaussNoise(): number {
    const u1 = Math.max(1e-10, rng());
    const u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  // Run EMD on multiple noisy copies and average IMFs
  const allIMFs: Float64Array[][] = [];

  for (let e = 0; e < nEnsembles; e++) {
    const noisy = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      noisy[i] = signal[i]! + noiseAmplitude * gaussNoise();
    }
    const result = emd(noisy, maxIMFs);
    const imfData = result.imfs.map(imf => imf.data);
    allIMFs.push(imfData);
  }

  // Average across ensembles
  const maxK = Math.max(...allIMFs.map(a => a.length));
  const imfs: IMF[] = [];

  for (let k = 0; k < maxK; k++) {
    const avgIMF = new Float64Array(N);
    let count = 0;
    for (const ensemble of allIMFs) {
      if (k < ensemble.length) {
        for (let i = 0; i < N; i++) {
          avgIMF[i] = avgIMF[i]! + ensemble[k]![i]!;
        }
        count++;
      }
    }
    if (count > 0) {
      for (let i = 0; i < N; i++) avgIMF[i] = avgIMF[i]! / count;
      imfs.push({ data: avgIMF });
    }
  }

  // Compute residue
  const residue = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    residue[i] = signal[i]!;
    for (const imf of imfs) {
      residue[i] = residue[i]! - imf.data[i]!;
    }
  }

  return { imfs, residue, nIterations: nEnsembles };
}
