/**
 * Spatial Intent CRDT types.
 *
 * Operations carry displacement vectors (not absolute positions), so concurrent
 * edits merge additively rather than last-writer-wins. Two users moving the same
 * table in different directions both have their intent preserved.
 *
 * CRDT correctness relies on:
 * - State = union of all received operations
 * - Merge = set union (commutative, associative, idempotent by construction)
 * - Reconstruction = deterministic function from op set to derived state
 */

import type { HlcTimestamp } from '@omni-twin/wire-protocol'

// ─── Vec3 ───────────────────────────────────────────────────────────────────

export interface Vec3 {
  x: number
  y: number
  z: number
}

export const VEC3_ZERO: Readonly<Vec3> = { x: 0, y: 0, z: 0 }
export const VEC3_ONE: Readonly<Vec3> = { x: 1, y: 1, z: 1 }

export function vec3Add(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }
}

export function vec3Eq(a: Vec3, b: Vec3, epsilon = 1e-9): boolean {
  return Math.abs(a.x - b.x) < epsilon
    && Math.abs(a.y - b.y) < epsilon
    && Math.abs(a.z - b.z) < epsilon
}

// ─── Spatial Operation Types ────────────────────────────────────────────────

interface OpBase {
  opId: string        // globally unique (replicaId:counter)
  hlc: HlcTimestamp   // causal timestamp
  replicaId: string   // which replica created this op
  objectId: string    // which spatial object
}

export interface AddOp extends OpBase {
  type: 'add'
  furnitureType: number
  position: Vec3      // initial absolute position
  rotation: Vec3      // initial absolute rotation
  scale: Vec3         // initial absolute scale
}

export interface RemoveOp extends OpBase {
  type: 'remove'
}

export interface MoveOp extends OpBase {
  type: 'move'
  delta: Vec3         // displacement vector
}

export interface RotateOp extends OpBase {
  type: 'rotate'
  delta: Vec3         // rotation delta (Euler angles)
}

export interface ScaleOp extends OpBase {
  type: 'scale'
  delta: Vec3         // scale delta (additive)
}

export type SpatialOp = AddOp | RemoveOp | MoveOp | RotateOp | ScaleOp
export type SpatialOpType = SpatialOp['type']

// ─── Derived State ──────────────────────────────────────────────────────────

export interface ObjectSnapshot {
  id: string
  furnitureType: number
  position: Vec3
  rotation: Vec3
  scale: Vec3
  alive: boolean
}

// ─── State Vector (for delta sync) ──────────────────────────────────────────

/**
 * State vector: per-replica counter of highest seen op.
 * Used for efficient delta sync — only send ops the peer hasn't seen.
 */
export type StateVector = Map<string, number>
