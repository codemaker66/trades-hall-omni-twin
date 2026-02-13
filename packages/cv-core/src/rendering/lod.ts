// ---------------------------------------------------------------------------
// CV-7: Level-of-Detail — screen-space error computation, LOD selection,
// and LOD chain construction for real-time rendering.
// ---------------------------------------------------------------------------

import type { LODLevel, Mesh } from '../types.js';

// ---------------------------------------------------------------------------
// computeScreenSpaceError
// ---------------------------------------------------------------------------

/**
 * Compute the screen-space error (SSE) for a given geometric error at a
 * particular viewing distance.
 *
 * The formula projects the geometric error (in world-space units) onto the
 * screen using the perspective projection:
 *
 *   SSE = (geometricError * screenHeight) / (2 * distance * tan(fovY / 2))
 *
 * A larger SSE means the error is more visible and a higher LOD should be
 * used; a smaller SSE means the object can be rendered at lower detail.
 *
 * @param distance       Camera distance to the object (metres). Must be > 0.
 * @param geometricError World-space geometric error of the LOD level (metres).
 * @param screenHeight   Viewport height in pixels.
 * @param fovY           Vertical field-of-view in radians.
 * @returns              Screen-space error in pixels.
 */
export function computeScreenSpaceError(
  distance: number,
  geometricError: number,
  screenHeight: number,
  fovY: number,
): number {
  if (distance <= 0) return Infinity;
  const denominator = 2 * distance * Math.tan(fovY / 2);
  if (denominator === 0) return Infinity;
  return (geometricError * screenHeight) / denominator;
}

// ---------------------------------------------------------------------------
// selectLOD
// ---------------------------------------------------------------------------

/**
 * Select the appropriate LOD level for a given camera distance and viewport
 * configuration.
 *
 * Iterates the levels array from finest (index 0) to coarsest and returns the
 * index of the first level whose screen-space error falls below that level's
 * threshold.  If no level qualifies the coarsest (last) level is returned.
 *
 * @param levels         Array of LOD levels ordered finest-to-coarsest.
 * @param cameraDistance Distance from the camera to the object (metres).
 * @param screenHeight   Viewport height in pixels.
 * @param fovY           Vertical field-of-view in radians.
 * @returns              Index into `levels` of the selected LOD.
 */
export function selectLOD(
  levels: LODLevel[],
  cameraDistance: number,
  screenHeight: number,
  fovY: number,
): number {
  if (levels.length === 0) return 0;

  for (let i = 0; i < levels.length; i++) {
    const level = levels[i]!;
    const sse = computeScreenSpaceError(
      cameraDistance,
      level.screenSpaceError,
      screenHeight,
      fovY,
    );
    // If the projected error is small enough, this level is sufficient.
    if (sse <= level.screenSpaceError) {
      return i;
    }
  }

  // Fall back to the coarsest level.
  return levels.length - 1;
}

// ---------------------------------------------------------------------------
// buildLODChain
// ---------------------------------------------------------------------------

/**
 * Build a chain of LOD levels from a base (highest-detail) mesh and a set
 * of reduction ratios.
 *
 * This is a *stub* implementation: it copies the base mesh for every level
 * and assigns estimated geometric error and distance thresholds that increase
 * with each ratio step.  A real implementation would invoke QEM decimation
 * on the mesh geometry.
 *
 * @param baseMesh The highest-detail mesh.
 * @param ratios   Decreasing detail ratios in (0, 1], e.g. [0.5, 0.25, 0.1].
 *                 Each value represents the fraction of triangles to retain.
 * @returns        Array of LOD levels (finest first, with the base mesh at
 *                 index 0 and the coarsest at the last position).
 */
export function buildLODChain(baseMesh: Mesh, ratios: number[]): LODLevel[] {
  const levels: LODLevel[] = [];

  // Level 0 is the original mesh (ratio = 1.0).
  levels.push({
    mesh: baseMesh,
    screenSpaceError: 0,
    distance: 0,
  });

  for (let i = 0; i < ratios.length; i++) {
    const ratio = ratios[i]!;

    // Estimate geometric error: grows as the ratio decreases.
    // Heuristic: error proportional to (1 - ratio).
    const geometricError = (1 - ratio) * 10;

    // Heuristic distance threshold increases with each level.
    const distance = (i + 1) * 50;

    // Stub: clone the base mesh reference.  A real implementation would
    // decimate the mesh to `baseMesh.triangleCount * ratio` triangles.
    const lodMesh: Mesh = {
      vertices: baseMesh.vertices,
      indices: baseMesh.indices,
      normals: baseMesh.normals,
      uvs: baseMesh.uvs,
      colors: baseMesh.colors,
      vertexCount: baseMesh.vertexCount,
      triangleCount: Math.max(1, Math.round(baseMesh.triangleCount * ratio)),
    };

    levels.push({
      mesh: lodMesh,
      screenSpaceError: geometricError,
      distance,
    });
  }

  return levels;
}
