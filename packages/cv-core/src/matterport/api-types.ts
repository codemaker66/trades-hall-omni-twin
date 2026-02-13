// ---------------------------------------------------------------------------
// CV-1: Matterport API Types & Helpers
// ---------------------------------------------------------------------------

import type { Vector3, Quaternion, Vec2, Room, FloorPlan, Sweep } from '../types.js';
import { vec3Sub, vec3Length } from '../types.js';

// Re-export domain types for convenience
export type { Sweep, FloorPlan, Room } from '../types.js';
export type { MatterportModel, E57Header, CoordinateTransform } from '../types.js';

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

/**
 * Create a Sweep with sensible defaults.
 *
 * @param id       - Unique sweep identifier.
 * @param position - World-space position.
 * @param rotation - Orientation quaternion.
 * @returns A fully populated {@link Sweep}.
 */
export function createSweep(
  id: string,
  position: Vector3,
  rotation: Quaternion,
): Sweep {
  return {
    id,
    position,
    rotation,
    floorIndex: 0,
    timestamp: new Date().toISOString(),
  };
}

/**
 * Create a {@link FloorPlan} from an array of rooms, computing total area
 * and bounding-box dimensions automatically.
 *
 * @param rooms - The rooms that belong to this floor plan.
 * @returns A populated FloorPlan (floorIndex defaults to 0).
 */
export function createFloorPlan(rooms: Room[]): FloorPlan {
  // Compute bounding box across all room polygons
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  const boundary: Vec2[] = [];

  for (let i = 0; i < rooms.length; i++) {
    const room = rooms[i]!;
    for (let j = 0; j < room.polygon.length; j++) {
      const v = room.polygon[j]!;
      if (v.x < minX) minX = v.x;
      if (v.y < minY) minY = v.y;
      if (v.x > maxX) maxX = v.x;
      if (v.y > maxY) maxY = v.y;
    }
  }

  // Build rectangular boundary from the bounding box
  if (rooms.length > 0) {
    boundary.push({ x: minX, y: minY });
    boundary.push({ x: maxX, y: minY });
    boundary.push({ x: maxX, y: maxY });
    boundary.push({ x: minX, y: maxY });
  }

  const width = rooms.length > 0 ? maxX - minX : 0;
  const height = rooms.length > 0 ? maxY - minY : 0;

  return {
    floorIndex: 0,
    rooms,
    boundary,
    width,
    height,
  };
}

/**
 * Compute the Euclidean distance between two sweeps.
 *
 * @param a - First sweep.
 * @param b - Second sweep.
 * @returns Distance in world-space units (metres).
 */
export function sweepDistance(a: Sweep, b: Sweep): number {
  return vec3Length(vec3Sub(a.position, b.position));
}
