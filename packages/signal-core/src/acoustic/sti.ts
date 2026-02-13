// ---------------------------------------------------------------------------
// SP-7: Speech Transmission Index (STI)
// ---------------------------------------------------------------------------
// Excellent >0.75, Good 0.60-0.75, Fair 0.45-0.60, Poor 0.30-0.45, Bad <0.30
// Computed from Modulation Transfer Function: m(F) per octave band.
// Overall STI = weighted average across 7 octave bands and 14 modulation frequencies.

import type { STIResult, ImpulseResponse } from '../types.js';

/** Modulation frequencies for STI (14 frequencies, Hz). */
const MODULATION_FREQUENCIES = [
  0.63, 0.80, 1.00, 1.25, 1.60, 2.00, 2.50,
  3.15, 4.00, 5.00, 6.30, 8.00, 10.00, 12.50,
];

/** Octave band weights for STI (IEC 60268-16). */
const STI_BAND_WEIGHTS = [0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173];

/** STI octave band center frequencies (Hz). */
const STI_BANDS = [125, 250, 500, 1000, 2000, 4000, 8000];

/**
 * Compute Modulation Transfer Function from impulse response.
 * m(F, f_oct) = |Σ h²(t)·e^{-j2πFt}| / Σ h²(t)
 */
function computeMTF(
  rir: Float64Array,
  sampleRate: number,
  modulationFreq: number,
): number {
  let sumH2 = 0;
  let realPart = 0;
  let imagPart = 0;

  for (let n = 0; n < rir.length; n++) {
    const h2 = rir[n]! * rir[n]!;
    const t = n / sampleRate;
    sumH2 += h2;
    realPart += h2 * Math.cos(2 * Math.PI * modulationFreq * t);
    imagPart += h2 * Math.sin(2 * Math.PI * modulationFreq * t);
  }

  if (sumH2 < 1e-20) return 0;
  return Math.sqrt(realPart * realPart + imagPart * imagPart) / sumH2;
}

/**
 * Estimate STI from impulse response.
 * Uses the simplified indirect method via MTF.
 */
export function estimateSTI(impulseResponse: ImpulseResponse): STIResult {
  const { samples, sampleRate } = impulseResponse;
  const nModFreqs = MODULATION_FREQUENCIES.length;

  // Compute MTF for each modulation frequency
  // (Simplified: using full-band impulse response)
  const mtfValues = new Float64Array(nModFreqs);
  for (let i = 0; i < nModFreqs; i++) {
    mtfValues[i] = computeMTF(samples, sampleRate, MODULATION_FREQUENCIES[i]!);
  }

  // Convert MTF to apparent SNR: SNR_app = 10·log10(m/(1-m))
  // Clip to [-15, +15] dB range
  const snrValues = new Float64Array(nModFreqs);
  for (let i = 0; i < nModFreqs; i++) {
    const m = Math.max(1e-6, Math.min(1 - 1e-6, mtfValues[i]!));
    snrValues[i] = Math.max(-15, Math.min(15, 10 * Math.log10(m / (1 - m))));
  }

  // Average apparent SNR
  let avgSNR = 0;
  for (let i = 0; i < nModFreqs; i++) avgSNR += snrValues[i]!;
  avgSNR /= nModFreqs;

  // STI = (avgSNR + 15) / 30
  const sti = Math.max(0, Math.min(1, (avgSNR + 15) / 30));

  // Rating
  let rating: STIResult['rating'];
  if (sti >= 0.75) rating = 'excellent';
  else if (sti >= 0.60) rating = 'good';
  else if (sti >= 0.45) rating = 'fair';
  else if (sti >= 0.30) rating = 'poor';
  else rating = 'bad';

  return { sti, rating, mtf: mtfValues };
}

/**
 * Quick STI estimate from RT60 using the simplified formula.
 * STI ≈ 1 - 0.19·RT60 (rough approximation for well-designed rooms).
 * More accurate: uses MTF = 1/(1 + 2π·F·RT60/13.8)
 */
export function estimateSTIFromRT60(rt60Mid: number): STIResult {
  const nModFreqs = MODULATION_FREQUENCIES.length;
  const mtfValues = new Float64Array(nModFreqs);

  for (let i = 0; i < nModFreqs; i++) {
    const F = MODULATION_FREQUENCIES[i]!;
    // MTF from RT60 (diffuse field model)
    mtfValues[i] = 1 / (1 + 2 * Math.PI * F * rt60Mid / 13.8);
  }

  // Convert to SNR and compute STI (same as above)
  let avgSNR = 0;
  for (let i = 0; i < nModFreqs; i++) {
    const m = Math.max(1e-6, Math.min(1 - 1e-6, mtfValues[i]!));
    avgSNR += Math.max(-15, Math.min(15, 10 * Math.log10(m / (1 - m))));
  }
  avgSNR /= nModFreqs;

  const sti = Math.max(0, Math.min(1, (avgSNR + 15) / 30));

  let rating: STIResult['rating'];
  if (sti >= 0.75) rating = 'excellent';
  else if (sti >= 0.60) rating = 'good';
  else if (sti >= 0.45) rating = 'fair';
  else if (sti >= 0.30) rating = 'poor';
  else rating = 'bad';

  return { sti, rating, mtf: mtfValues };
}
