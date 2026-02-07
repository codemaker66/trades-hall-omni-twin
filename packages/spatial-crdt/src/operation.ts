/**
 * Operation constructors and helpers.
 */

import type { HlcTimestamp } from '@omni-twin/wire-protocol'
import type { Vec3, SpatialOp, AddOp, RemoveOp, MoveOp, RotateOp, ScaleOp } from './types'
import { VEC3_ONE } from './types'

// ─── Op ID generation ───────────────────────────────────────────────────────

let _counter = 0

/** Generate a globally unique op ID. Format: "replicaId:counter" */
export function generateOpId(replicaId: string): string {
  return `${replicaId}:${++_counter}`
}

/** Reset the internal counter (for testing). */
export function resetOpIdCounter(): void {
  _counter = 0
}

/** Parse an op ID into its components. */
export function parseOpId(opId: string): { replicaId: string; counter: number } {
  const [replicaId, counter] = opId.split(':')
  return { replicaId: replicaId!, counter: Number(counter) }
}

// ─── Operation Constructors ─────────────────────────────────────────────────

export function createAddOp(
  replicaId: string,
  hlc: HlcTimestamp,
  objectId: string,
  furnitureType: number,
  position: Vec3,
  rotation: Vec3 = { x: 0, y: 0, z: 0 },
  scale: Vec3 = { ...VEC3_ONE },
): AddOp {
  return {
    type: 'add',
    opId: generateOpId(replicaId),
    hlc, replicaId, objectId,
    furnitureType, position, rotation, scale,
  }
}

export function createRemoveOp(
  replicaId: string,
  hlc: HlcTimestamp,
  objectId: string,
): RemoveOp {
  return {
    type: 'remove',
    opId: generateOpId(replicaId),
    hlc, replicaId, objectId,
  }
}

export function createMoveOp(
  replicaId: string,
  hlc: HlcTimestamp,
  objectId: string,
  delta: Vec3,
): MoveOp {
  return {
    type: 'move',
    opId: generateOpId(replicaId),
    hlc, replicaId, objectId, delta,
  }
}

export function createRotateOp(
  replicaId: string,
  hlc: HlcTimestamp,
  objectId: string,
  delta: Vec3,
): RotateOp {
  return {
    type: 'rotate',
    opId: generateOpId(replicaId),
    hlc, replicaId, objectId, delta,
  }
}

export function createScaleOp(
  replicaId: string,
  hlc: HlcTimestamp,
  objectId: string,
  delta: Vec3,
): ScaleOp {
  return {
    type: 'scale',
    opId: generateOpId(replicaId),
    hlc, replicaId, objectId, delta,
  }
}

// ─── Ordering ───────────────────────────────────────────────────────────────

/** Deterministic total order for operations. HLC first, then opId for tie-breaking. */
export function opCompare(a: SpatialOp, b: SpatialOp): number {
  if (a.hlc.wallMs !== b.hlc.wallMs) return a.hlc.wallMs - b.hlc.wallMs
  if (a.hlc.counter !== b.hlc.counter) return a.hlc.counter - b.hlc.counter
  return a.opId < b.opId ? -1 : a.opId > b.opId ? 1 : 0
}
