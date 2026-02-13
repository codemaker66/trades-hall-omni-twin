// ---------------------------------------------------------------------------
// CV-2: Gaussian Splatting — Compression & Quantization
// ---------------------------------------------------------------------------

import type { SplatCloud, QuantizationConfig } from '../types.js';

/**
 * Quantize an array of float64 positions into fixed-point Int16 values.
 *
 * The positions are first bounded by a global min/max, then linearly
 * mapped into the range [-2^(bits-1), 2^(bits-1)-1].
 *
 * @param positions - Flat Float64Array of coordinates (e.g. [x0,y0,z0, x1,y1,z1, ...]).
 * @param bits      - Number of bits for quantisation (e.g. 16 for Int16).
 * @returns An object containing:
 *          - `quantized`: Int16Array of quantised values (same length as positions).
 *          - `scale`: The scale factor used for de-quantisation.
 *          - `offset`: Float64Array of per-axis offsets (minimum values).
 */
export function quantizePositions(
  positions: Float64Array,
  bits: number,
): { quantized: Int16Array; scale: number; offset: Float64Array } {
  const n = positions.length;
  if (n === 0) {
    return {
      quantized: new Int16Array(0),
      scale: 1,
      offset: new Float64Array(0),
    };
  }

  // Find global min and max across all values
  let globalMin = Infinity;
  let globalMax = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = positions[i]!;
    if (v < globalMin) globalMin = v;
    if (v > globalMax) globalMax = v;
  }

  // Compute per-component min/max for offset (group by xyz triplets)
  const componentCount = Math.min(n, 3);
  const offset = new Float64Array(componentCount);
  for (let c = 0; c < componentCount; c++) {
    let cMin = Infinity;
    for (let i = c; i < n; i += componentCount) {
      const v = positions[i]!;
      if (v < cMin) cMin = v;
    }
    offset[c] = cMin;
  }

  const range = globalMax - globalMin;
  const maxQuantised = (1 << (bits - 1)) - 1; // e.g. 32767 for 16-bit
  const scale = range > 0 ? range / maxQuantised : 1;

  const quantized = new Int16Array(n);
  for (let i = 0; i < n; i++) {
    const normalised = range > 0 ? (positions[i]! - globalMin) / range : 0;
    // Map [0, 1] -> [-maxQuantised, maxQuantised]
    const q = Math.round(normalised * maxQuantised * 2 - maxQuantised);
    // Clamp to Int16 range
    quantized[i] = Math.max(-32768, Math.min(32767, q));
  }

  return { quantized, scale, offset };
}

/**
 * Truncate spherical harmonics coefficients to a lower degree.
 *
 * SH coefficient counts per degree (per colour channel):
 *   degree 0 = 1, degree 1 = 4, degree 2 = 9, degree 3 = 16
 *
 * For RGB (3 channels) the total coefficient counts are:
 *   degree 0 = 3, degree 1 = 12, degree 2 = 27, degree 3 = 48
 *
 * This function retains only the coefficients up to `maxDegree`.
 *
 * @param coeffs         - Full SH coefficient array.
 * @param maxDegree      - Target SH degree (0-3).
 * @param originalDegree - Original SH degree the coefficients were computed for.
 * @returns Truncated Float64Array with the appropriate number of coefficients.
 */
export function truncateSH(
  coeffs: Float64Array,
  maxDegree: number,
  originalDegree: number,
): Float64Array {
  if (maxDegree >= originalDegree) {
    // No truncation needed — return a copy
    return new Float64Array(coeffs);
  }

  const channels = 3; // RGB

  // Number of basis functions per degree: (degree+1)^2
  const targetBasisCount = (maxDegree + 1) * (maxDegree + 1);
  const targetCoeffCount = targetBasisCount * channels;

  const truncated = new Float64Array(targetCoeffCount);

  // Copy only the first targetCoeffCount coefficients
  const copyLen = Math.min(targetCoeffCount, coeffs.length);
  for (let i = 0; i < copyLen; i++) {
    truncated[i] = coeffs[i]!;
  }

  return truncated;
}

/**
 * Estimate the compressed byte size of a {@link SplatCloud} given a
 * quantisation configuration.
 *
 * Layout estimate per Gaussian:
 *   - Position:   3 components * positionBits / 8 bytes
 *   - Covariance: 6 unique elements (symmetric 3x3) * covarianceBits / 8 bytes
 *   - Color:      3 channels * colorBits / 8 bytes
 *   - Opacity:    opacityBits / 8 bytes
 *   - SH:         shCoeffCount * shBits / 8 bytes
 *
 * Plus a fixed header overhead of 64 bytes.
 *
 * @param cloud  - The splat cloud to estimate for.
 * @param config - Quantisation bit-depth configuration.
 * @returns Estimated total byte size.
 */
export function estimateCompressedSize(
  cloud: SplatCloud,
  config: QuantizationConfig,
): number {
  const HEADER_BYTES = 64;

  // SH coefficient count depends on SH degree
  // (degree+1)^2 basis functions * 3 channels
  const shBasisCount = (cloud.shDegree + 1) * (cloud.shDegree + 1);
  const shCoeffCount = shBasisCount * 3;

  const bitsPerGaussian =
    3 * config.positionBits +       // position (x, y, z)
    6 * config.covarianceBits +     // covariance (6 unique elements)
    3 * config.colorBits +          // colour (r, g, b)
    config.opacityBits +            // opacity
    shCoeffCount * config.shBits;   // SH coefficients

  const bytesPerGaussian = Math.ceil(bitsPerGaussian / 8);

  return HEADER_BYTES + cloud.count * bytesPerGaussian;
}
