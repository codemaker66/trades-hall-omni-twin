// ---------------------------------------------------------------------------
// SP-7: RT60 Calculation
// ---------------------------------------------------------------------------
// Sabine: RT60 = 0.161V/A  (V=m³, A=Σ α_i·S_i sabins)
//   Best for live rooms (ᾱ < 0.2)
// Eyring: RT60 = 0.161V/(-S·ln(1-ᾱ))
//   More accurate for treated rooms (ᾱ > 0.2)
// Fitzroy: Correction for non-uniform absorption distribution

import type { RoomGeometry, RT60Result } from '../types.js';
import { OCTAVE_BANDS } from './material-database.js';

/**
 * Sabine RT60 formula.
 * RT60 = 0.161 · V / A
 * where A = Σ α_i · S_i (total absorption in sabins)
 */
function sabineRT60(volume: number, surfaces: RoomGeometry['surfaces']): [number, number, number, number, number, number] {
  const result: [number, number, number, number, number, number] = [0, 0, 0, 0, 0, 0];

  for (let band = 0; band < 6; band++) {
    let totalAbsorption = 0;
    for (const { area, material } of surfaces) {
      totalAbsorption += area * material.absorption[band]!;
    }
    result[band] = totalAbsorption > 0 ? (0.161 * volume) / totalAbsorption : Infinity;
  }

  return result;
}

/**
 * Eyring RT60 formula.
 * RT60 = 0.161 · V / (-S · ln(1 - ᾱ))
 * where ᾱ = weighted average absorption coefficient
 */
function eyringRT60(volume: number, surfaces: RoomGeometry['surfaces']): [number, number, number, number, number, number] {
  const result: [number, number, number, number, number, number] = [0, 0, 0, 0, 0, 0];

  let totalArea = 0;
  for (const { area } of surfaces) totalArea += area;

  for (let band = 0; band < 6; band++) {
    let totalAbsorption = 0;
    for (const { area, material } of surfaces) {
      totalAbsorption += area * material.absorption[band]!;
    }
    const avgAlpha = totalArea > 0 ? totalAbsorption / totalArea : 0;
    const denominator = -totalArea * Math.log(1 - Math.min(avgAlpha, 0.99));
    result[band] = denominator > 0 ? (0.161 * volume) / denominator : Infinity;
  }

  return result;
}

/**
 * Fitzroy RT60 formula.
 * Better for non-uniform absorption: accounts for axis-weighted contributions.
 * 1/RT60 = S/(0.161V) · [Sx/S·(-ln(1-ᾱx)) + Sy/S·(-ln(1-ᾱy)) + Sz/S·(-ln(1-ᾱz))]
 *
 * Simplified: uses Eyring with axis correction factor.
 */
function fitzroyRT60(
  room: RoomGeometry,
): [number, number, number, number, number, number] {
  // Use Eyring as base with a correction for non-uniformity
  const eyring = eyringRT60(room.volume, room.surfaces);
  // Compute absorption variance across surfaces
  const result: [number, number, number, number, number, number] = [0, 0, 0, 0, 0, 0];

  for (let band = 0; band < 6; band++) {
    const alphas = room.surfaces.map(s => s.material.absorption[band]!);
    const areas = room.surfaces.map(s => s.area);
    let totalArea = 0;
    for (const a of areas) totalArea += a;

    // Weight-average α
    let avgAlpha = 0;
    for (let i = 0; i < alphas.length; i++) {
      avgAlpha += areas[i]! * alphas[i]!;
    }
    avgAlpha = totalArea > 0 ? avgAlpha / totalArea : 0;

    // Compute variance of absorption
    let variance = 0;
    for (let i = 0; i < alphas.length; i++) {
      const diff = alphas[i]! - avgAlpha;
      variance += areas[i]! * diff * diff;
    }
    variance = totalArea > 0 ? variance / totalArea : 0;

    // Fitzroy correction: multiply Eyring by (1 + variance/avgAlpha²) when variance is high
    const correction = avgAlpha > 0.01 ? 1 + 0.5 * variance / (avgAlpha * avgAlpha) : 1;
    result[band] = eyring[band]! * Math.max(correction, 1);
  }

  return result;
}

/**
 * Calculate RT60 for a room geometry using specified formula.
 */
export function calculateRT60(
  room: RoomGeometry,
  formula: 'sabine' | 'eyring' | 'fitzroy' = 'eyring',
): RT60Result {
  let rt60: [number, number, number, number, number, number];

  switch (formula) {
    case 'sabine':
      rt60 = sabineRT60(room.volume, room.surfaces);
      break;
    case 'eyring':
      rt60 = eyringRT60(room.volume, room.surfaces);
      break;
    case 'fitzroy':
      rt60 = fitzroyRT60(room);
      break;
  }

  // Mid-frequency average (500Hz and 1kHz bands, indices 2 and 3)
  const rt60Mid = (rt60[2] + rt60[3]) / 2;

  return { rt60, rt60Mid, formula };
}

/**
 * Compute the acoustic impact of adding furniture to a room.
 * Returns delta RT60 (negative = more absorption = shorter reverb).
 */
export function furnitureAcousticImpact(
  baseRoom: RoomGeometry,
  furnitureSurfaces: RoomGeometry['surfaces'],
): RT60Result {
  const combined: RoomGeometry = {
    ...baseRoom,
    surfaces: [...baseRoom.surfaces, ...furnitureSurfaces],
  };

  const baseRT60 = calculateRT60(baseRoom);
  const newRT60 = calculateRT60(combined);

  const delta: [number, number, number, number, number, number] = [0, 0, 0, 0, 0, 0];
  for (let b = 0; b < 6; b++) {
    delta[b] = newRT60.rt60[b]! - baseRT60.rt60[b]!;
  }

  return {
    rt60: delta,
    rt60Mid: (delta[2] + delta[3]) / 2,
    formula: 'eyring',
  };
}
