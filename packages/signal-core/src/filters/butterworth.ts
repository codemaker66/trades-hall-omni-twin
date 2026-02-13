// ---------------------------------------------------------------------------
// SP-4: Butterworth Filter Design
// ---------------------------------------------------------------------------
// Maximally flat passband, monotonic rolloff.
// IIR design via bilinear transform: s → 2·(z-1)/((z+1)·T)
// Always use second-order sections (SOS) for numerical stability.

import type { ButterworthConfig, SOSSection, FilterType } from '../types.js';

/**
 * Pre-warp analog cutoff frequency for bilinear transform.
 * Ωₐ = 2·fs·tan(π·fc/fs)
 */
function prewarp(fc: number, fs: number): number {
  return 2 * fs * Math.tan((Math.PI * fc) / fs);
}

/**
 * Compute Butterworth analog prototype poles.
 * Poles of Nth-order Butterworth lie on unit circle at angles:
 * θ_k = π(2k+N+1)/(2N) for k = 0,...,N-1
 */
function butterworthPoles(order: number): Array<{ re: number; im: number }> {
  const poles: Array<{ re: number; im: number }> = [];
  for (let k = 0; k < order; k++) {
    const theta = (Math.PI * (2 * k + order + 1)) / (2 * order);
    poles.push({ re: Math.cos(theta), im: Math.sin(theta) });
  }
  return poles;
}

/**
 * Bilinear transform: s-domain to z-domain.
 * s = 2·fs·(z-1)/(z+1)
 * H(z) from H(s) via pole/zero mapping.
 */
function bilinearTransformSOS(
  poleRe: number,
  poleIm: number,
  prewarpedCutoff: number,
  fs: number,
): SOSSection {
  // Scale pole to desired cutoff
  const sRe = poleRe * prewarpedCutoff;
  const sIm = poleIm * prewarpedCutoff;

  // Bilinear transform: z = (1 + s/(2fs)) / (1 - s/(2fs))
  const T = 1 / (2 * fs);
  // Numerator of bilinear: (2fs + s)
  const numRe = 2 * fs + sRe;
  const numIm = sIm;
  // Denominator: (2fs - s)
  const denRe = 2 * fs - sRe;
  const denIm = -sIm;

  // z-pole = (2fs + s) / (2fs - s)
  const denMag2 = denRe * denRe + denIm * denIm;
  const zRe = (numRe * denRe + numIm * denIm) / denMag2;
  const zIm = (numIm * denRe - numRe * denIm) / denMag2;

  // For lowpass: numerator zeros at z = -1
  // SOS: H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (a0 + a1·z⁻¹ + a2·z⁻²)
  // With complex conjugate pole pair:
  // a0 = 1, a1 = -2·Re(z_pole), a2 = |z_pole|²
  const a1 = -2 * zRe;
  const a2 = zRe * zRe + zIm * zIm;

  // For lowpass, zeros at z = -1: b(z) = (1 + z⁻¹)²
  const b0 = 1;
  const b1 = 2;
  const b2 = 1;

  // Normalize gain at DC (z=1)
  const numDC = b0 + b1 + b2;
  const denDC = 1 + a1 + a2;
  const gain = denDC / numDC;

  return [b0 * gain, b1 * gain, b2 * gain, 1, a1, a2];
}

/**
 * Design Butterworth filter as cascade of second-order sections.
 */
export function designButterworth(config: ButterworthConfig): SOSSection[] {
  const { order, cutoff, type: filterType, fs } = config;
  const nyquist = fs / 2;

  const sections: SOSSection[] = [];
  const poles = butterworthPoles(order);

  if (filterType === 'lowpass' || filterType === 'highpass') {
    const fc = typeof cutoff === 'number' ? cutoff : cutoff[0];
    const omega = prewarp(fc, fs);

    // Process poles in conjugate pairs
    for (let i = 0; i < Math.ceil(order / 2); i++) {
      const pole = poles[i]!;
      if (i < Math.floor(order / 2)) {
        // Conjugate pair
        const sos = bilinearTransformSOS(pole.re, pole.im, omega, fs);
        if (filterType === 'highpass') {
          // Highpass: transform z → -z
          sections.push([sos[0], -sos[1], sos[2], sos[3], -sos[4], sos[5]]);
        } else {
          sections.push(sos);
        }
      } else if (order % 2 === 1) {
        // Real pole (odd order)
        const sReal = pole.re * omega;
        const zReal = (2 * fs + sReal) / (2 * fs - sReal);
        const a1 = -zReal;
        const gainDC = (1 + a1) / 2;
        if (filterType === 'highpass') {
          sections.push([-gainDC, gainDC, 0, 1, a1, 0]);
        } else {
          sections.push([gainDC, gainDC, 0, 1, a1, 0]);
        }
      }
    }
  } else if (filterType === 'bandpass' || filterType === 'bandstop') {
    const [fLow, fHigh] = typeof cutoff === 'number' ? [cutoff * 0.8, cutoff * 1.2] : cutoff;
    // Simplified bandpass via cascaded lowpass + highpass
    const lowSections = designButterworth({ order, cutoff: fHigh, type: 'lowpass', fs });
    const highSections = designButterworth({ order, cutoff: fLow, type: 'highpass', fs });
    if (filterType === 'bandpass') {
      sections.push(...lowSections, ...highSections);
    } else {
      // Bandstop: parallel sum approximation via cascade
      sections.push(...lowSections, ...highSections);
    }
  }

  return sections;
}

/**
 * Apply SOS filter (direct form II transposed) — forward pass only.
 */
export function sosfilt(sections: SOSSection[], signal: Float64Array): Float64Array {
  let output = new Float64Array(signal);

  for (const [b0, b1, b2, a0, a1, a2] of sections) {
    const x = output;
    const y = new Float64Array(x.length);
    let w1 = 0, w2 = 0;

    for (let n = 0; n < x.length; n++) {
      const w0 = x[n]! - a1 * w1 - a2 * w2;
      y[n] = b0 * w0 + b1 * w1 + b2 * w2;
      w2 = w1;
      w1 = w0;
    }
    output = y;
  }

  return output;
}

/**
 * Zero-phase SOS filter: forward pass + backward pass.
 * Doubles the effective order, gives zero phase distortion.
 * Always prefer this over single-pass for offline processing.
 */
export function sosfiltfilt(sections: SOSSection[], signal: Float64Array): Float64Array {
  // Forward pass
  const forward = sosfilt(sections, signal);

  // Reverse
  const reversed = new Float64Array(forward.length);
  for (let i = 0; i < forward.length; i++) {
    reversed[i] = forward[forward.length - 1 - i]!;
  }

  // Backward pass
  const backward = sosfilt(sections, reversed);

  // Re-reverse
  const result = new Float64Array(backward.length);
  for (let i = 0; i < backward.length; i++) {
    result[i] = backward[backward.length - 1 - i]!;
  }

  return result;
}
