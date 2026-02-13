// ---------------------------------------------------------------------------
// CV-4: Capture Planner — camera placement, coverage analysis, and path
// optimisation for photogrammetric data capture.
// ---------------------------------------------------------------------------

import type { Vector3 } from '../types.js';
import { vec3Sub } from '../types.js';

// ---------------------------------------------------------------------------
// planCapturePositions
// ---------------------------------------------------------------------------

/**
 * Generate a grid of camera positions that cover an axis-aligned bounding
 * box with a specified overlap fraction.
 *
 * The grid spacing is derived from the camera's horizontal field of view and
 * the distance from the camera to the centre of the bounding box, then
 * reduced by the overlap fraction so adjacent images share a strip of pixels.
 *
 * Cameras are placed on a plane above the bounding box (max.y + standoff),
 * looking downward.  For non-nadir use cases, the caller can rotate the
 * output positions.
 *
 * @param bounds          Axis-aligned bounding box of the target region.
 * @param overlapFraction Desired overlap between adjacent images (0-1).
 * @param cameraFOV       Camera horizontal field of view in radians.
 * @returns               Array of camera world-space positions.
 */
export function planCapturePositions(
  bounds: { min: Vector3; max: Vector3 },
  overlapFraction: number,
  cameraFOV: number,
): Vector3[] {
  // Clamp overlap
  const overlap = Math.max(0, Math.min(overlapFraction, 0.99));

  // Target extent
  const extentX = bounds.max.x - bounds.min.x;
  const extentZ = bounds.max.z - bounds.min.z;
  const extentY = bounds.max.y - bounds.min.y;

  // Camera standoff: place cameras above the bounding box by half its Y extent
  // (at minimum 1 unit above)
  const standoff = Math.max(extentY * 0.5, 1);
  const cameraY = bounds.max.y + standoff;

  // Distance from camera to the centre of the top face of the bounding box
  const distance = standoff;

  // Ground footprint width from one camera position
  const footprint = 2 * distance * Math.tan(cameraFOV / 2);

  // Step size accounting for overlap
  const step = footprint * (1 - overlap);

  // Guard against degenerate step
  const safeStep = Math.max(step, 1e-6);

  // Number of positions along each axis
  const nX = Math.max(1, Math.ceil(extentX / safeStep) + 1);
  const nZ = Math.max(1, Math.ceil(extentZ / safeStep) + 1);

  // Centre the grid over the bounding box
  const startX = bounds.min.x + (extentX - (nX - 1) * safeStep) / 2;
  const startZ = bounds.min.z + (extentZ - (nZ - 1) * safeStep) / 2;

  const positions: Vector3[] = [];

  for (let iz = 0; iz < nZ; iz++) {
    for (let ix = 0; ix < nX; ix++) {
      positions.push({
        x: startX + ix * safeStep,
        y: cameraY,
        z: startZ + iz * safeStep,
      });
    }
  }

  return positions;
}

// ---------------------------------------------------------------------------
// computeCoverage
// ---------------------------------------------------------------------------

/**
 * Compute the fraction of target points that are visible from at least one
 * camera position.
 *
 * A target point is considered "visible" from a camera position if:
 *  1. The distance between them is within `maxRange`.
 *  2. The angular offset from the camera's (assumed nadir / -Y) direction is
 *     within half the FOV.
 *
 * This is a simplified visibility model (no occlusion checking).
 *
 * @param positions    Camera positions.
 * @param targetPoints Points to be covered.
 * @param cameraFOV    Camera horizontal field of view in radians.
 * @param maxRange     Maximum visibility range (metres).
 * @returns            Fraction of target points visible from at least one camera (0-1).
 */
export function computeCoverage(
  positions: Vector3[],
  targetPoints: Vector3[],
  cameraFOV: number,
  maxRange: number,
): number {
  if (targetPoints.length === 0) return 1;
  if (positions.length === 0) return 0;

  const halfFOV = cameraFOV / 2;
  const cosHalfFOV = Math.cos(halfFOV);
  const maxRangeSq = maxRange * maxRange;

  let visibleCount = 0;

  for (let ti = 0; ti < targetPoints.length; ti++) {
    const target = targetPoints[ti]!;
    let visible = false;

    for (let ci = 0; ci < positions.length; ci++) {
      const cam = positions[ci]!;
      const diff = vec3Sub(target, cam);
      const distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

      if (distSq > maxRangeSq) continue;

      // Assume camera looks along -Y (nadir).  The angle between the
      // look direction (0,-1,0) and the vector to the target is:
      //   cos(angle) = dot((0,-1,0), normalise(diff))
      const dist = Math.sqrt(distSq);
      if (dist < 1e-15) {
        visible = true;
        break;
      }

      // dot with (0,-1,0) is just -diff.y
      const cosAngle = -diff.y / dist;
      if (cosAngle >= cosHalfFOV) {
        visible = true;
        break;
      }
    }

    if (visible) visibleCount++;
  }

  return visibleCount / targetPoints.length;
}

// ---------------------------------------------------------------------------
// optimizeCapturePath
// ---------------------------------------------------------------------------

/**
 * Reorder camera positions using a greedy nearest-neighbour heuristic for
 * the Travelling Salesman Problem (TSP).
 *
 * Starting from the first position in the input array, the algorithm always
 * moves to the closest unvisited position until all have been visited.
 *
 * @param positions Camera positions to reorder.
 * @returns         Reordered positions forming an approximate shortest path.
 */
export function optimizeCapturePath(positions: Vector3[]): Vector3[] {
  const n = positions.length;
  if (n <= 1) return positions.slice();

  const visited = new Uint8Array(n);
  const result: Vector3[] = new Array(n);

  // Start at position 0
  result[0] = positions[0]!;
  visited[0] = 1;

  for (let step = 1; step < n; step++) {
    const current = result[step - 1]!;
    let bestIdx = -1;
    let bestDistSq = Infinity;

    for (let i = 0; i < n; i++) {
      if (visited[i]) continue;
      const d = vec3Sub(positions[i]!, current);
      const distSq = d.x * d.x + d.y * d.y + d.z * d.z;
      if (distSq < bestDistSq) {
        bestDistSq = distSq;
        bestIdx = i;
      }
    }

    visited[bestIdx] = 1;
    result[step] = positions[bestIdx]!;
  }

  return result;
}
