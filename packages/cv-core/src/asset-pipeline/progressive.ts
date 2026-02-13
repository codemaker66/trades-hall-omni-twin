// ---------------------------------------------------------------------------
// CV-12: Asset Pipeline — Progressive LOD
// LOD chain planning, size estimation, and bandwidth-aware LOD selection.
// ---------------------------------------------------------------------------

import type { Mesh, ProgressiveLODConfig, QEMConfig } from '../types.js';

// ---------------------------------------------------------------------------
// planLODChain
// ---------------------------------------------------------------------------

/**
 * Plan a chain of LOD levels by computing target triangle counts, texture
 * resolutions, and distance thresholds for each level.
 *
 * The base mesh is LOD-0.  Each subsequent level halves (approximately) the
 * triangle count from the previous level and scales the texture resolution
 * down to match.
 *
 * This function produces the *configuration* for LOD generation; it does not
 * perform the actual decimation (which would require QEM or similar).
 *
 * @param baseMesh The highest-detail mesh (LOD-0).
 * @param nLevels  Total number of LOD levels to generate (including the base).
 *                 Must be >= 1.
 * @returns        An array of `ProgressiveLODConfig` descriptors, one per level.
 */
export function planLODChain(
  baseMesh: Mesh,
  nLevels: number,
): ProgressiveLODConfig[] {
  const levels = Math.max(1, Math.round(nLevels));
  const configs: ProgressiveLODConfig[] = [];

  for (let i = 0; i < levels; i++) {
    // Ratio decreases geometrically: 1.0, 0.5, 0.25, ...
    const ratio = 1 / Math.pow(2, i);
    const targetTriangles = Math.max(1, Math.round(baseMesh.triangleCount * ratio));

    // Texture resolution also halves per level, floored to a power of two.
    const baseTexRes = 2048; // Assumed base texture resolution.
    const targetTexRes = Math.max(64, nearestPOT(baseTexRes * ratio));

    // Screen-space error grows with each LOD level.
    const screenSpaceError = i === 0 ? 0 : i * 2;

    // Distance threshold (metres) — heuristic: each level adds 25 m.
    const distance = i * 25;

    // QEM decimation config shared across all levels.
    const decimationConfig: QEMConfig = {
      targetTriangles,
      maxError: 0.01 * (i + 1),
      preserveBoundary: true,
      preserveSharpEdges: i < 2, // Relax for lower LODs.
      sharpEdgeAngle: Math.PI / 6,
    };

    configs.push({
      levels: [
        {
          targetTriangles,
          targetTextureResolution: targetTexRes,
          screenSpaceError,
          distance,
        },
      ],
      decimationConfig,
      generateUVAtlas: i > 0,
      bakeTextures: i > 0,
    });
  }

  return configs;
}

// ---------------------------------------------------------------------------
// estimateLODSizes
// ---------------------------------------------------------------------------

/**
 * Estimate the triangle count and byte size for each LOD level given a set
 * of reduction ratios.
 *
 * @param baseMesh The highest-detail mesh.
 * @param ratios   Array of ratios in (0, 1] representing the fraction of
 *                 triangles to retain at each level.
 * @returns        `triangles` and `estimatedBytes` arrays, one entry per ratio.
 */
export function estimateLODSizes(
  baseMesh: Mesh,
  ratios: number[],
): { triangles: number[]; estimatedBytes: number[] } {
  const triangles: number[] = [];
  const estimatedBytes: number[] = [];

  for (let i = 0; i < ratios.length; i++) {
    const r = ratios[i]!;
    const triCount = Math.max(1, Math.round(baseMesh.triangleCount * r));
    triangles.push(triCount);

    // Estimate bytes: vertices (~32 B each, ~3 unique per tri) + indices (12 B per tri).
    // Plus 10 % container overhead.
    const vertBytes = triCount * 3 * 32;
    const idxBytes = triCount * 3 * 4;
    estimatedBytes.push(Math.ceil((vertBytes + idxBytes) * 1.1));
  }

  return { triangles, estimatedBytes };
}

// ---------------------------------------------------------------------------
// selectLODForBandwidth
// ---------------------------------------------------------------------------

/**
 * Select the highest-quality LOD level that can be loaded within a time
 * budget at the given bandwidth.
 *
 * @param sizes              Array of estimated byte sizes per LOD, ordered from
 *                           highest (index 0) to lowest quality.
 * @param bandwidthBytesPerSec Available network bandwidth in bytes per second.
 * @param targetLoadTime     Maximum acceptable load time in seconds.
 * @returns                  The zero-based index into `sizes` of the selected
 *                           LOD.  Returns the last (smallest) index if no
 *                           level fits the budget.
 */
export function selectLODForBandwidth(
  sizes: number[],
  bandwidthBytesPerSec: number,
  targetLoadTime: number,
): number {
  if (sizes.length === 0) return 0;

  const budget = bandwidthBytesPerSec * targetLoadTime;

  // Iterate from the finest (largest) LOD to the coarsest (smallest).
  for (let i = 0; i < sizes.length; i++) {
    const sz = sizes[i]!;
    if (sz <= budget) return i;
  }

  // None fits — return the coarsest.
  return sizes.length - 1;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Round a value down to the nearest power-of-two. */
function nearestPOT(value: number): number {
  if (value <= 0) return 1;
  let pot = 1;
  while (pot * 2 <= value) {
    pot *= 2;
  }
  return pot;
}
