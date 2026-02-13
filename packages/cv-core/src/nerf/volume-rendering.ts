// ---------------------------------------------------------------------------
// CV-5: NeRF — Volume Rendering
// ---------------------------------------------------------------------------

import type {
  Vector3,
  PRNG,
  VolumeSample,
  VolumeRenderConfig,
  VolumeRenderResult,
} from '../types.js';
import { createPRNG, vec3Scale, vec3Add } from '../types.js';

// ---------------------------------------------------------------------------
// Ray sampling
// ---------------------------------------------------------------------------

/**
 * Sample parametric t-values along a ray using stratified sampling.
 *
 * Divides [near, far] into `nSamples` equal bins and draws one uniform
 * sample per bin, giving low-discrepancy coverage of the interval.
 *
 * @param _origin    - Ray origin (unused, kept for API symmetry).
 * @param _direction - Ray direction (unused, kept for API symmetry).
 * @param near       - Near bound of the sampling interval.
 * @param far        - Far bound of the sampling interval.
 * @param nSamples   - Number of stratified samples.
 * @param rng        - Optional PRNG; defaults to seed 42.
 * @returns Float64Array of length `nSamples` with sorted t-values.
 */
export function sampleRay(
  _origin: Vector3,
  _direction: Vector3,
  near: number,
  far: number,
  nSamples: number,
  rng?: PRNG,
): Float64Array {
  const r = rng ?? createPRNG(42);
  const tValues = new Float64Array(nSamples);
  const binSize = (far - near) / nSamples;

  for (let i = 0; i < nSamples; i++) {
    const lo = near + i * binSize;
    tValues[i] = lo + r() * binSize;
  }

  return tValues;
}

// ---------------------------------------------------------------------------
// Alpha compositing
// ---------------------------------------------------------------------------

/**
 * Front-to-back alpha compositing of volume samples.
 *
 * For each sample i the accumulated transmittance is:
 *   T_i = prod_{j<i} (1 - alpha_j)
 *
 * The final colour is:
 *   C = sum_i T_i * alpha_i * c_i
 *
 * Expected depth is:
 *   depth = sum_i T_i * alpha_i * t_i
 *
 * @param samples - Ordered array of {@link VolumeSample} along a ray.
 * @returns Composited colour, expected depth, and total opacity.
 */
export function alphaComposite(
  samples: VolumeSample[],
): { rgb: Vector3; depth: number; opacity: number } {
  let r = 0;
  let g = 0;
  let b = 0;
  let depth = 0;
  let transmittance = 1.0;

  for (let i = 0; i < samples.length; i++) {
    const s = samples[i]!;

    // Convert density to alpha using the interval width.
    // For uniformly-spaced samples we use delta = 1 as a simplified default;
    // callers should pre-compute alpha from sigma * delta when needed.
    const delta =
      i + 1 < samples.length ? samples[i + 1]!.t - s.t : 1.0;
    const alpha = 1.0 - Math.exp(-s.sigma * delta);

    const weight = transmittance * alpha;

    r += weight * s.rgb.x;
    g += weight * s.rgb.y;
    b += weight * s.rgb.z;
    depth += weight * s.t;

    transmittance *= 1.0 - alpha;

    // Early termination when transmittance is negligible
    if (transmittance < 1e-4) break;
  }

  return {
    rgb: { x: r, y: g, z: b },
    depth,
    opacity: 1.0 - transmittance,
  };
}

// ---------------------------------------------------------------------------
// Transmittance computation
// ---------------------------------------------------------------------------

/**
 * Compute discrete transmittance along a ray.
 *
 *   T_i = exp( -sum_{j<i} sigma_j * delta_j )
 *
 * @param sigmas - Per-sample volume densities.
 * @param deltas - Per-sample interval widths (same length as `sigmas`).
 * @returns Float64Array of transmittance values T_i, length = sigmas.length.
 */
export function computeTransmittance(
  sigmas: Float64Array,
  deltas: Float64Array,
): Float64Array {
  const n = sigmas.length;
  const T = new Float64Array(n);
  let cumulative = 0.0;

  for (let i = 0; i < n; i++) {
    T[i] = Math.exp(-cumulative);
    cumulative += sigmas[i]! * deltas[i]!;
  }

  return T;
}

// ---------------------------------------------------------------------------
// Full ray rendering
// ---------------------------------------------------------------------------

/**
 * Render a single ray through a volume described by the given
 * {@link VolumeRenderConfig}.
 *
 * This is a simplified pipeline that:
 *  1. Stratified-samples `nCoarseSamples` t-values along [near, far].
 *  2. Evaluates a dummy density / colour field (constant for demonstration)
 *     and composites front-to-back.
 *
 * In a real NeRF implementation the density & colour would come from a
 * neural network query; this function provides the compositing framework.
 *
 * @param config    - Volume render configuration.
 * @param origin    - Ray origin.
 * @param direction - Ray direction (should be unit-length).
 * @returns {@link VolumeRenderResult} with accumulated colour, depth, opacity, and weights.
 */
export function renderRay(
  config: VolumeRenderConfig,
  origin: Vector3,
  direction: Vector3,
): VolumeRenderResult {
  const rng = createPRNG(7);
  const nSamples = config.nCoarseSamples;

  // 1. Stratified ray sampling
  const tValues = sampleRay(origin, direction, config.near, config.far, nSamples, rng);

  // 2. Build volume samples — in a real NeRF these would come from the MLP.
  //    Here we use a simple density falloff so the function is self-contained.
  const samples: VolumeSample[] = [];
  for (let i = 0; i < nSamples; i++) {
    const t = tValues[i]!;
    const point = vec3Add(origin, vec3Scale(direction, t));

    // Simple radial density field centred at origin for demonstration
    const dist = Math.sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
    const sigma = Math.max(0, 1.0 - dist);
    const grey = 0.5 + 0.5 * Math.cos(dist * Math.PI);

    samples.push({
      t,
      sigma,
      rgb: { x: grey, y: grey, z: grey },
    });
  }

  // 3. Compute deltas and transmittance
  const sigmas = new Float64Array(nSamples);
  const deltas = new Float64Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    sigmas[i] = samples[i]!.sigma;
    deltas[i] =
      i + 1 < nSamples
        ? tValues[i + 1]! - tValues[i]!
        : (config.far - config.near) / nSamples;
  }

  const T = computeTransmittance(sigmas, deltas);

  // 4. Alpha-composite with explicit weights
  const weights = new Float64Array(nSamples);
  let rAcc = 0;
  let gAcc = 0;
  let bAcc = 0;
  let depthAcc = 0;
  let opacityAcc = 0;

  for (let i = 0; i < nSamples; i++) {
    const alpha = 1.0 - Math.exp(-sigmas[i]! * deltas[i]!);
    const w = T[i]! * alpha;
    weights[i] = w;

    const s = samples[i]!;
    rAcc += w * s.rgb.x;
    gAcc += w * s.rgb.y;
    bAcc += w * s.rgb.z;
    depthAcc += w * s.t;
    opacityAcc += w;
  }

  // Optional white background
  if (config.whiteBackground) {
    const bgWeight = 1.0 - opacityAcc;
    rAcc += bgWeight;
    gAcc += bgWeight;
    bAcc += bgWeight;
  }

  return {
    rgb: { x: rAcc, y: gAcc, z: bAcc },
    opacity: opacityAcc,
    depth: depthAcc,
    weights,
  };
}
