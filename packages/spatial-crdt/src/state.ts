/**
 * Per-object state reconstruction from operation sets.
 *
 * Given a set of operations for an object, derives the current snapshot:
 * - Base position/rotation/scale from the latest AddOp (LWW for base)
 * - Accumulated displacements from all MoveOps (additive)
 * - Accumulated rotation deltas from all RotateOps (additive)
 * - Accumulated scale deltas from all ScaleOps (additive)
 * - Alive = latest add HLC >= latest remove HLC
 *
 * The reconstruction function is deterministic: same set of ops → same state.
 * This is the key to CRDT convergence.
 */

import { hlcCompare } from '@omni-twin/wire-protocol'
import type { SpatialOp, ObjectSnapshot, Vec3, AddOp, RemoveOp } from './types'
import { VEC3_ZERO, VEC3_ONE, vec3Add } from './types'
import { opCompare } from './operation'

// ─── Object reconstruction ──────────────────────────────────────────────────

/**
 * Reconstruct an object's snapshot from its complete set of operations.
 * Operations must all target the same objectId.
 *
 * Returns null if no AddOp exists (object was never created).
 */
export function reconstructObject(objectId: string, ops: SpatialOp[]): ObjectSnapshot | null {
  // Separate ops by type
  let latestAdd: AddOp | null = null
  let latestRemove: RemoveOp | null = null
  const moveDeltas: Vec3[] = []
  const rotateDeltas: Vec3[] = []
  const scaleDeltas: Vec3[] = []

  for (const op of ops) {
    switch (op.type) {
      case 'add':
        if (!latestAdd || opCompare(op, latestAdd) > 0) {
          latestAdd = op
        }
        break
      case 'remove':
        if (!latestRemove || opCompare(op, latestRemove) > 0) {
          latestRemove = op
        }
        break
      case 'move':
        moveDeltas.push(op.delta)
        break
      case 'rotate':
        rotateDeltas.push(op.delta)
        break
      case 'scale':
        scaleDeltas.push(op.delta)
        break
    }
  }

  if (!latestAdd) return null // no creation op

  // Alive: add wins if its HLC >= remove's HLC. On tie, add wins.
  const alive = !latestRemove || hlcCompare(latestAdd.hlc, latestRemove.hlc) >= 0

  // Accumulate displacement vectors (commutative addition)
  let position = { ...latestAdd.position }
  for (const d of moveDeltas) {
    position = vec3Add(position, d)
  }

  let rotation = { ...latestAdd.rotation }
  for (const d of rotateDeltas) {
    rotation = vec3Add(rotation, d)
  }

  let scale = { ...latestAdd.scale }
  for (const d of scaleDeltas) {
    scale = vec3Add(scale, d)
  }

  return {
    id: objectId,
    furnitureType: latestAdd.furnitureType,
    position,
    rotation,
    scale,
    alive,
  }
}

/**
 * Reconstruct all objects from a complete operation map.
 * Returns only alive objects by default.
 */
export function reconstructAll(
  ops: Map<string, SpatialOp>,
  includeRemoved = false,
): Map<string, ObjectSnapshot> {
  // Group ops by objectId
  const byObject = new Map<string, SpatialOp[]>()
  for (const op of ops.values()) {
    let list = byObject.get(op.objectId)
    if (!list) {
      list = []
      byObject.set(op.objectId, list)
    }
    list.push(op)
  }

  // Reconstruct each object
  const result = new Map<string, ObjectSnapshot>()
  for (const [objectId, objectOps] of byObject) {
    const snapshot = reconstructObject(objectId, objectOps)
    if (snapshot && (snapshot.alive || includeRemoved)) {
      result.set(objectId, snapshot)
    }
  }

  return result
}
