/**
 * Rectangle-select via spatial query.
 * Finds all selectable entities within a screen-space or world-space rectangle.
 */

import { hasComponent } from 'bitecs'
import { Position, Selectable } from '../components'
import { SpatialHash, type AABB } from './spatial-index'

/**
 * Select all entities whose position falls within the given XZ rectangle.
 * Only returns entities that have the Selectable component.
 *
 * @param spatialHash - The spatial index to query
 * @param world - The ECS world (for component checks)
 * @param rect - The selection rectangle in world-space XZ coordinates
 * @returns Array of entity IDs within the rectangle and marked Selectable
 */
export function rectSelect(
  spatialHash: SpatialHash,
  world: ReturnType<typeof import('bitecs').createWorld>,
  rect: AABB,
): number[] {
  // Broadphase: get candidates from spatial hash
  const candidates = spatialHash.queryAABB(rect)

  // Narrowphase: check position is within rect and entity is selectable
  const selected: number[] = []
  for (const eid of candidates) {
    if (!hasComponent(world, Selectable, eid)) continue

    const px = Position.x[eid]!
    const pz = Position.z[eid]!

    if (px >= rect.minX && px <= rect.maxX && pz >= rect.minZ && pz <= rect.maxZ) {
      selected.push(eid)
    }
  }

  return selected
}

/**
 * Select a single entity at (or very near) a point.
 * Returns the closest selectable entity within `radius`, or undefined.
 */
export function pointSelect(
  spatialHash: SpatialHash,
  world: ReturnType<typeof import('bitecs').createWorld>,
  x: number,
  z: number,
  radius = 0.5,
): number | undefined {
  const candidates = spatialHash.queryRadius(x, z, radius)

  let closestEid: number | undefined
  let closestDist = Infinity

  for (const eid of candidates) {
    if (!hasComponent(world, Selectable, eid)) continue

    const dx = Position.x[eid]! - x
    const dz = Position.z[eid]! - z
    const dist = dx * dx + dz * dz // squared distance

    if (dist < closestDist && dist <= radius * radius) {
      closestDist = dist
      closestEid = eid
    }
  }

  return closestEid
}
