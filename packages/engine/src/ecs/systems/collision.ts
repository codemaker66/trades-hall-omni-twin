/**
 * Broadphase AABB collision detection using the spatial hash.
 * Tests entity bounding boxes for overlap in the XZ plane.
 */

import { Position, BoundingBox, Scale } from '../components'
import { SpatialHash, type AABB } from './spatial-index'

// ─── Types ──────────────────────────────────────────────────────────────────

/** A pair of colliding entity IDs. */
export interface CollisionPair {
  a: number
  b: number
}

// ─── AABB helpers ───────────────────────────────────────────────────────────

/** Get the world-space AABB for an entity (XZ plane, scale-aware). */
export function getEntityAABB(eid: number): AABB {
  const px = Position.x[eid]!
  const pz = Position.z[eid]!
  const hx = (BoundingBox.halfX[eid]!) * (Scale.x[eid]!)
  const hz = (BoundingBox.halfZ[eid]!) * (Scale.z[eid]!)
  return {
    minX: px - hx,
    minZ: pz - hz,
    maxX: px + hx,
    maxZ: pz + hz,
  }
}

/** Test if two AABBs overlap. */
export function aabbOverlap(a: AABB, b: AABB): boolean {
  return a.minX <= b.maxX && a.maxX >= b.minX && a.minZ <= b.maxZ && a.maxZ >= b.minZ
}

// ─── Collision Detection ────────────────────────────────────────────────────

/**
 * Find all colliding entity pairs using the spatial hash for broadphase.
 * Returns unique pairs (no duplicates: if (a,b) is returned, (b,a) is not).
 */
export function detectCollisions(spatialHash: SpatialHash, entities: number[]): CollisionPair[] {
  const pairs: CollisionPair[] = []
  const checked = new Set<number>()

  for (const eid of entities) {
    const aabb = getEntityAABB(eid)

    // Query the spatial hash for nearby entities
    const candidates = spatialHash.queryAABB(aabb)

    for (const other of candidates) {
      if (other === eid) continue
      // Avoid duplicate pairs using a symmetric key
      const pairKey = eid < other ? eid * 100000 + other : other * 100000 + eid
      if (checked.has(pairKey)) continue
      checked.add(pairKey)

      // Narrowphase: AABB overlap test
      const otherAABB = getEntityAABB(other)
      if (aabbOverlap(aabb, otherAABB)) {
        pairs.push({ a: Math.min(eid, other), b: Math.max(eid, other) })
      }
    }
  }

  return pairs
}

/**
 * Check if a single entity collides with any other entity.
 * Useful for drag validation.
 */
export function checkCollision(
  spatialHash: SpatialHash,
  eid: number,
  exclude?: Set<number>,
): number[] {
  const aabb = getEntityAABB(eid)
  const candidates = spatialHash.queryAABB(aabb)
  const colliding: number[] = []

  for (const other of candidates) {
    if (other === eid) continue
    if (exclude?.has(other)) continue
    const otherAABB = getEntityAABB(other)
    if (aabbOverlap(aabb, otherAABB)) {
      colliding.push(other)
    }
  }

  return colliding
}

/**
 * Check if placing an item at (x, z) with given half-extents would collide.
 * Does not require an existing entity — useful for placement preview.
 */
export function checkPlacementCollision(
  spatialHash: SpatialHash,
  x: number,
  z: number,
  halfX: number,
  halfZ: number,
  exclude?: Set<number>,
): number[] {
  const aabb: AABB = {
    minX: x - halfX,
    minZ: z - halfZ,
    maxX: x + halfX,
    maxZ: z + halfZ,
  }

  const candidates = spatialHash.queryAABB(aabb)
  const colliding: number[] = []

  for (const other of candidates) {
    if (exclude?.has(other)) continue
    const otherAABB = getEntityAABB(other)
    if (aabbOverlap(aabb, otherAABB)) {
      colliding.push(other)
    }
  }

  return colliding
}
