// ---------------------------------------------------------------------------
// SP-7: Acoustic Material Database
// ---------------------------------------------------------------------------
// Absorption coefficients at standard octave bands [125, 250, 500, 1k, 2k, 4k Hz]
// Sources: Architectural Acoustics (Long), ISO 354 measurements

import type { AcousticMaterial } from '../types.js';

/** Standard octave band center frequencies (Hz). */
export const OCTAVE_BANDS = [125, 250, 500, 1000, 2000, 4000] as const;

/**
 * Built-in acoustic material database.
 * Absorption coefficients at [125, 250, 500, 1k, 2k, 4k Hz].
 */
export const MATERIAL_DB: Record<string, AcousticMaterial> = {
  // Floors
  'concrete': {
    name: 'Concrete (unpainted)',
    absorption: [0.01, 0.01, 0.02, 0.02, 0.02, 0.03],
    scattering: 0.1,
  },
  'wood-floor': {
    name: 'Wood floor (on joists)',
    absorption: [0.15, 0.11, 0.10, 0.07, 0.06, 0.07],
    scattering: 0.1,
  },
  'carpet-thin': {
    name: 'Thin carpet (no underlay)',
    absorption: [0.02, 0.06, 0.14, 0.37, 0.60, 0.65],
    scattering: 0.15,
  },
  'carpet-thick': {
    name: 'Heavy carpet on foam underlay',
    absorption: [0.08, 0.24, 0.57, 0.69, 0.71, 0.73],
    scattering: 0.2,
  },

  // Walls
  'plaster': {
    name: 'Smooth plaster on brick',
    absorption: [0.01, 0.02, 0.02, 0.03, 0.04, 0.05],
    scattering: 0.05,
  },
  'brick': {
    name: 'Unglazed brick',
    absorption: [0.03, 0.03, 0.03, 0.04, 0.05, 0.07],
    scattering: 0.2,
  },
  'glass': {
    name: 'Glass (large pane)',
    absorption: [0.18, 0.06, 0.04, 0.03, 0.02, 0.02],
    scattering: 0.05,
  },
  'curtain-heavy': {
    name: 'Heavy curtains (draped)',
    absorption: [0.07, 0.31, 0.49, 0.75, 0.70, 0.60],
    scattering: 0.4,
  },
  'acoustic-panel': {
    name: 'Acoustic panel (50mm)',
    absorption: [0.28, 0.78, 0.98, 0.93, 0.82, 0.70],
    scattering: 0.3,
  },
  'wood-panel': {
    name: 'Wood paneling (plywood on studs)',
    absorption: [0.28, 0.22, 0.17, 0.09, 0.10, 0.11],
    scattering: 0.15,
  },

  // Ceiling
  'plaster-ceiling': {
    name: 'Plaster ceiling',
    absorption: [0.02, 0.02, 0.03, 0.04, 0.04, 0.05],
    scattering: 0.05,
  },
  'acoustic-tile': {
    name: 'Acoustic ceiling tile',
    absorption: [0.50, 0.70, 0.60, 0.70, 0.70, 0.50],
    scattering: 0.3,
  },

  // Furniture and occupants
  'occupied-fabric-seat': {
    name: 'Occupied upholstered seat',
    absorption: [0.72, 0.82, 0.91, 0.93, 0.94, 0.96],
    scattering: 0.5,
  },
  'empty-fabric-seat': {
    name: 'Empty upholstered seat',
    absorption: [0.49, 0.66, 0.80, 0.88, 0.82, 0.70],
    scattering: 0.4,
  },
  'wooden-chair': {
    name: 'Wooden chair (unoccupied)',
    absorption: [0.02, 0.02, 0.03, 0.04, 0.04, 0.04],
    scattering: 0.15,
  },
  'table-wood': {
    name: 'Wooden table',
    absorption: [0.05, 0.03, 0.03, 0.03, 0.03, 0.02],
    scattering: 0.15,
  },

  // Stage
  'stage-wood': {
    name: 'Stage (wooden floor)',
    absorption: [0.40, 0.30, 0.20, 0.17, 0.15, 0.10],
    scattering: 0.1,
  },

  // People (standing, per person ≈0.5 m²)
  'audience-standing': {
    name: 'Audience (standing)',
    absorption: [0.25, 0.44, 0.60, 0.77, 0.89, 0.82],
    scattering: 0.5,
  },
};

/**
 * Get a material by key, with fallback to concrete.
 */
export function getMaterial(key: string): AcousticMaterial {
  return MATERIAL_DB[key] ?? MATERIAL_DB['concrete']!;
}

/**
 * Compute weighted average absorption coefficient at a specific band index.
 */
export function weightedAbsorption(
  surfaces: Array<{ area: number; material: AcousticMaterial }>,
  bandIndex: number,
): number {
  let totalArea = 0;
  let totalAbsorption = 0;
  for (const { area, material } of surfaces) {
    totalArea += area;
    totalAbsorption += area * material.absorption[bandIndex]!;
  }
  return totalArea > 0 ? totalAbsorption / totalArea : 0;
}
