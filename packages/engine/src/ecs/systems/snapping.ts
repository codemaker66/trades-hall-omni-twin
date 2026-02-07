/**
 * Snapping system: grid-snap and nearest-neighbor entity snapping.
 */

import { Position } from '../components'
import { SpatialHash } from './spatial-index'

// ─── Grid Snapping ──────────────────────────────────────────────────────────

/**
 * Snap a position to the nearest grid point.
 *
 * @param x - World X coordinate
 * @param z - World Z coordinate
 * @param gridSize - Grid cell size (default 0.5m)
 * @returns Snapped [x, z] coordinates
 */
export function snapToGrid(x: number, z: number, gridSize = 0.5): [number, number] {
  return [
    Math.round(x / gridSize) * gridSize,
    Math.round(z / gridSize) * gridSize,
  ]
}

/**
 * Snap a Y value to a height grid (for stacking).
 *
 * @param y - World Y coordinate
 * @param gridSize - Height step (default 0.1m)
 * @returns Snapped Y
 */
export function snapToHeightGrid(y: number, gridSize = 0.1): number {
  return Math.round(y / gridSize) * gridSize
}

// ─── Nearest-Neighbor Snapping ──────────────────────────────────────────────

/** Result of a nearest-neighbor search. */
export interface NearestResult {
  eid: number
  distance: number
  x: number
  z: number
}

/**
 * Find the nearest entity to a point using the spatial hash.
 * Searches within `radius` and returns the closest one.
 *
 * @param spatialHash - The spatial index
 * @param x - Query X coordinate
 * @param z - Query Z coordinate
 * @param radius - Search radius (default 3.0m)
 * @param exclude - Entity IDs to exclude from results
 * @returns The nearest entity, or undefined if none within radius
 */
export function findNearest(
  spatialHash: SpatialHash,
  x: number,
  z: number,
  radius = 3.0,
  exclude?: Set<number>,
): NearestResult | undefined {
  const candidates = spatialHash.queryRadius(x, z, radius)

  let best: NearestResult | undefined
  let bestDist = radius * radius // squared

  for (const eid of candidates) {
    if (exclude?.has(eid)) continue

    const ex = Position.x[eid]!
    const ez = Position.z[eid]!
    const dx = ex - x
    const dz = ez - z
    const distSq = dx * dx + dz * dz

    if (distSq < bestDist) {
      bestDist = distSq
      best = { eid, distance: Math.sqrt(distSq), x: ex, z: ez }
    }
  }

  return best
}

/**
 * Find the K nearest entities to a point.
 *
 * @param spatialHash - The spatial index
 * @param x - Query X coordinate
 * @param z - Query Z coordinate
 * @param k - Maximum number of results
 * @param radius - Search radius (default 5.0m)
 * @param exclude - Entity IDs to exclude
 * @returns Array of nearest entities sorted by distance (closest first)
 */
export function findKNearest(
  spatialHash: SpatialHash,
  x: number,
  z: number,
  k: number,
  radius = 5.0,
  exclude?: Set<number>,
): NearestResult[] {
  const candidates = spatialHash.queryRadius(x, z, radius)
  const results: NearestResult[] = []

  for (const eid of candidates) {
    if (exclude?.has(eid)) continue

    const ex = Position.x[eid]!
    const ez = Position.z[eid]!
    const dx = ex - x
    const dz = ez - z
    const distSq = dx * dx + dz * dz

    if (distSq <= radius * radius) {
      results.push({ eid, distance: Math.sqrt(distSq), x: ex, z: ez })
    }
  }

  results.sort((a, b) => a.distance - b.distance)
  return results.slice(0, k)
}

/**
 * Snap a position to align with a nearby entity (edge-to-edge alignment).
 * Returns the snapped position if a snap target is found, otherwise the original.
 *
 * @param spatialHash - The spatial index
 * @param x - Position to snap
 * @param z - Position to snap
 * @param snapDistance - Maximum snap distance (default 0.5m)
 * @param exclude - Entities to ignore
 * @returns Snapped [x, z] position
 */
export function snapToNearest(
  spatialHash: SpatialHash,
  x: number,
  z: number,
  snapDistance = 0.5,
  exclude?: Set<number>,
): [number, number] {
  const nearest = findNearest(spatialHash, x, z, snapDistance * 2, exclude)
  if (!nearest || nearest.distance > snapDistance) {
    return [x, z]
  }

  // Snap to the nearest entity's position on the closer axis
  const dx = Math.abs(nearest.x - x)
  const dz = Math.abs(nearest.z - z)

  if (dx < dz && dx < snapDistance) {
    // Align X axis
    return [nearest.x, z]
  } else if (dz < snapDistance) {
    // Align Z axis
    return [x, nearest.z]
  }

  return [x, z]
}
