// ---------------------------------------------------------------------------
// SP-7: Impulse Response Generation
// ---------------------------------------------------------------------------
// Simplified Image Source Method for shoebox rooms.
// Generates Room Impulse Response (RIR) for convolution reverb.

import type { ImpulseResponse, RoomGeometry, AcousticMaterial } from '../types.js';
import { weightedAbsorption } from './material-database.js';

/**
 * Generate a synthetic impulse response for a shoebox room
 * using the Image Source Method (Allen & Berkley, 1979).
 *
 * Simplified to first-order reflections (6 images) + direct path.
 *
 * @param dimensions Room [length, width, height] in meters
 * @param source Source position [x, y, z]
 * @param receiver Receiver position [x, y, z]
 * @param surfaces Room surfaces with materials
 * @param sampleRate Sample rate (default 16000 Hz)
 * @param maxOrder Maximum reflection order (default 3)
 */
export function generateShoeboxRIR(
  dimensions: [number, number, number],
  source: [number, number, number],
  receiver: [number, number, number],
  surfaces: RoomGeometry['surfaces'],
  sampleRate: number = 16000,
  maxOrder: number = 3,
): ImpulseResponse {
  const [Lx, Ly, Lz] = dimensions;
  const speedOfSound = 343; // m/s at 20°C

  // Compute average absorption at 1kHz (band index 3)
  const avgAbsorption = weightedAbsorption(surfaces, 3);
  const reflectCoeff = Math.sqrt(1 - avgAbsorption);

  // Maximum propagation time for given order
  const maxDist = maxOrder * Math.sqrt(Lx * Lx + Ly * Ly + Lz * Lz) * 2;
  const maxTime = maxDist / speedOfSound;
  const nSamples = Math.ceil(maxTime * sampleRate);
  const samples = new Float64Array(nSamples);

  // Image source method: iterate over reflection indices
  for (let nx = -maxOrder; nx <= maxOrder; nx++) {
    for (let ny = -maxOrder; ny <= maxOrder; ny++) {
      for (let nz = -maxOrder; nz <= maxOrder; nz++) {
        // Compute image source position for each combination of reflections
        for (let lx = 0; lx <= 1; lx++) {
          for (let ly = 0; ly <= 1; ly++) {
            for (let lz = 0; lz <= 1; lz++) {
              // Image source coordinates
              const imgX = (1 - 2 * lx) * source[0] + 2 * nx * Lx;
              const imgY = (1 - 2 * ly) * source[1] + 2 * ny * Ly;
              const imgZ = (1 - 2 * lz) * source[2] + 2 * nz * Lz;

              // Number of reflections
              const nReflections = Math.abs(2 * nx - lx) + Math.abs(2 * ny - ly) + Math.abs(2 * nz - lz);

              if (nReflections > maxOrder) continue;

              // Distance from image source to receiver
              const dx = imgX - receiver[0];
              const dy = imgY - receiver[1];
              const dz = imgZ - receiver[2];
              const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

              if (dist < 0.01) continue; // Skip coincident sources

              // Arrival time and sample index
              const arrivalTime = dist / speedOfSound;
              const sampleIdx = Math.round(arrivalTime * sampleRate);

              if (sampleIdx >= 0 && sampleIdx < nSamples) {
                // Amplitude: 1/r attenuation × reflection coefficient^nReflections
                const amplitude = Math.pow(reflectCoeff, nReflections) / dist;
                samples[sampleIdx] = (samples[sampleIdx] ?? 0) + amplitude;
              }
            }
          }
        }
      }
    }
  }

  // Normalize peak to 1
  let peak = 0;
  for (let i = 0; i < nSamples; i++) {
    peak = Math.max(peak, Math.abs(samples[i]!));
  }
  if (peak > 0) {
    for (let i = 0; i < nSamples; i++) {
      samples[i] = samples[i]! / peak;
    }
  }

  return {
    samples,
    sampleRate,
    duration: nSamples / sampleRate,
  };
}

/**
 * Apply impulse response to an audio signal via convolution.
 * Uses overlap-add FFT convolution for efficiency.
 */
export function convolveWithRIR(
  signal: Float64Array,
  rir: ImpulseResponse,
): Float64Array {
  // Direct time-domain convolution (simpler for short RIRs)
  const N = signal.length;
  const M = rir.samples.length;
  const outLen = N + M - 1;
  const output = new Float64Array(outLen);

  for (let n = 0; n < outLen; n++) {
    let sum = 0;
    const kMin = Math.max(0, n - M + 1);
    const kMax = Math.min(n, N - 1);
    for (let k = kMin; k <= kMax; k++) {
      sum += signal[k]! * rir.samples[n - k]!;
    }
    output[n] = sum;
  }

  return output;
}
