// ---------------------------------------------------------------------------
// CV-1: Floor Plan Geometry Utilities
// ---------------------------------------------------------------------------

import type { Vec2, Room } from '../types.js';

/**
 * Compute the area of a simple polygon using the Shoelace formula.
 *
 * The polygon vertices must be ordered (either CW or CCW). The returned
 * area is always non-negative (absolute value of the signed area).
 *
 * @param polygon - Ordered 2D vertices forming a closed polygon.
 * @returns Area in square units.
 */
export function computeRoomArea(polygon: Vec2[]): number {
  const n = polygon.length;
  if (n < 3) return 0;

  let signedArea = 0;
  for (let i = 0; i < n; i++) {
    const curr = polygon[i]!;
    const next = polygon[(i + 1) % n]!;
    signedArea += curr.x * next.y - next.x * curr.y;
  }

  return Math.abs(signedArea) * 0.5;
}

/**
 * Extract the boundary polygons from an array of rooms.
 *
 * @param rooms - Array of rooms, each containing a polygon field.
 * @returns Array of boundary polygons (one per room).
 */
export function extractRoomBoundaries(rooms: Room[]): Vec2[][] {
  const boundaries: Vec2[][] = [];
  for (let i = 0; i < rooms.length; i++) {
    const room = rooms[i]!;
    // Deep-copy the polygon vertices to avoid external mutation
    const boundary: Vec2[] = [];
    for (let j = 0; j < room.polygon.length; j++) {
      const v = room.polygon[j]!;
      boundary.push({ x: v.x, y: v.y });
    }
    boundaries.push(boundary);
  }
  return boundaries;
}

/**
 * Test whether a 2D point lies inside a room's polygon using the
 * ray-casting (even-odd rule) algorithm.
 *
 * A horizontal ray is cast from the point towards +x; the number of
 * polygon edge crossings determines inside/outside.
 *
 * @param point - The 2D query point.
 * @param room  - The room whose polygon is tested.
 * @returns `true` if the point is inside the room polygon.
 */
export function isPointInRoom(point: Vec2, room: Room): boolean {
  const polygon = room.polygon;
  const n = polygon.length;
  if (n < 3) return false;

  let inside = false;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const vi = polygon[i]!;
    const vj = polygon[j]!;

    // Check if the horizontal ray from `point` crosses the edge (vi, vj)
    const yiBelowOrAt = vi.y <= point.y;
    const yjBelowOrAt = vj.y <= point.y;

    if (yiBelowOrAt !== yjBelowOrAt) {
      // Compute the x-coordinate of the intersection with the horizontal ray
      const xIntersect =
        vi.x + ((point.y - vi.y) / (vj.y - vi.y)) * (vj.x - vi.x);
      if (point.x < xIntersect) {
        inside = !inside;
      }
    }
  }

  return inside;
}

/**
 * Compute the axis-aligned bounding box of all rooms' polygon vertices.
 *
 * @param rooms - Array of rooms.
 * @returns An object with `min` and `max` Vec2 corners. If rooms is
 *          empty, returns zeros.
 */
export function computeFloorPlanBounds(
  rooms: Room[],
): { min: Vec2; max: Vec2 } {
  if (rooms.length === 0) {
    return {
      min: { x: 0, y: 0 },
      max: { x: 0, y: 0 },
    };
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

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

  return {
    min: { x: minX, y: minY },
    max: { x: maxX, y: maxY },
  };
}
