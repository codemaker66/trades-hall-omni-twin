/**
 * SpatialDocument: the top-level CRDT managing a collection of spatial objects.
 *
 * Internally maintains:
 * - An operation log (Map<opId, SpatialOp>) — the ground truth
 * - A derived state cache (Map<objectId, ObjectSnapshot>) — for fast reads
 *
 * The cache is incrementally updated when new ops arrive, and can be
 * fully rebuilt from the op log at any time for verification.
 */

import type { HlcTimestamp } from '@omni-twin/wire-protocol'
import type { SpatialOp, ObjectSnapshot, Vec3 } from './types'
import { vec3Add } from './types'
import { reconstructAll, reconstructObject } from './state'
import { mergeOpSets, opSetDifference } from './merge'
import { createAddOp, createRemoveOp, createMoveOp, createRotateOp, createScaleOp, opCompare } from './operation'

export class SpatialDocument {
  /** All operations, keyed by opId. This is the canonical CRDT state. */
  private ops = new Map<string, SpatialOp>()

  /** Derived state cache. */
  private cache = new Map<string, ObjectSnapshot>()

  /** Per-object ops for efficient incremental updates. */
  private objectOps = new Map<string, Map<string, SpatialOp>>()

  constructor(readonly replicaId: string) {}

  // ─── Read API ───────────────────────────────────────────────────────────

  /** Get the current snapshot of all alive objects. */
  objects(): ReadonlyMap<string, ObjectSnapshot> {
    return this.cache
  }

  /** Get a single object's snapshot. */
  getObject(objectId: string): ObjectSnapshot | null {
    return this.cache.get(objectId) ?? null
  }

  /** Number of alive objects. */
  get size(): number {
    return this.cache.size
  }

  /** Total number of operations in the log. */
  get opCount(): number {
    return this.ops.size
  }

  /** Get the raw operation log (read-only). */
  getOps(): ReadonlyMap<string, SpatialOp> {
    return this.ops
  }

  // ─── Write API (local mutations) ────────────────────────────────────────

  addObject(
    hlc: HlcTimestamp,
    objectId: string,
    furnitureType: number,
    position: Vec3,
    rotation?: Vec3,
    scale?: Vec3,
  ): SpatialOp {
    const op = createAddOp(this.replicaId, hlc, objectId, furnitureType, position, rotation, scale)
    this.apply(op)
    return op
  }

  removeObject(hlc: HlcTimestamp, objectId: string): SpatialOp {
    const op = createRemoveOp(this.replicaId, hlc, objectId)
    this.apply(op)
    return op
  }

  moveObject(hlc: HlcTimestamp, objectId: string, delta: Vec3): SpatialOp {
    const op = createMoveOp(this.replicaId, hlc, objectId, delta)
    this.apply(op)
    return op
  }

  rotateObject(hlc: HlcTimestamp, objectId: string, delta: Vec3): SpatialOp {
    const op = createRotateOp(this.replicaId, hlc, objectId, delta)
    this.apply(op)
    return op
  }

  scaleObject(hlc: HlcTimestamp, objectId: string, delta: Vec3): SpatialOp {
    const op = createScaleOp(this.replicaId, hlc, objectId, delta)
    this.apply(op)
    return op
  }

  // ─── Apply (local or remote) ────────────────────────────────────────────

  /** Apply a single operation. Idempotent: duplicate opIds are ignored. */
  apply(op: SpatialOp): boolean {
    if (this.ops.has(op.opId)) return false // already applied

    this.ops.set(op.opId, op)

    // Track per-object
    let objOps = this.objectOps.get(op.objectId)
    if (!objOps) {
      objOps = new Map()
      this.objectOps.set(op.objectId, objOps)
    }
    objOps.set(op.opId, op)

    // Incrementally update cache for this object
    this.updateObjectCache(op.objectId)

    return true
  }

  /** Apply multiple operations (e.g., received from a peer). */
  applyBatch(ops: SpatialOp[]): number {
    let applied = 0
    for (const op of ops) {
      if (this.apply(op)) applied++
    }
    return applied
  }

  // ─── Merge ──────────────────────────────────────────────────────────────

  /**
   * Merge another document's state into this one.
   * Returns the number of new operations applied.
   */
  merge(other: SpatialDocument): number {
    const missing = opSetDifference(this.ops, other.ops)
    return this.applyBatch(missing)
  }

  /**
   * Merge a raw operation set into this document.
   */
  mergeOps(otherOps: ReadonlyMap<string, SpatialOp>): number {
    const missing = opSetDifference(this.ops, otherOps)
    return this.applyBatch(missing)
  }

  // ─── Delta Sync ─────────────────────────────────────────────────────────

  /**
   * Get ops that the peer is missing, given their state vector.
   * A state vector maps replicaId → highest counter seen.
   */
  getMissingOps(peerStateVector: ReadonlyMap<string, number>): SpatialOp[] {
    const missing: SpatialOp[] = []
    for (const op of this.ops.values()) {
      const parts = op.opId.split(':')
      const counter = Number(parts[1])
      const peerMax = peerStateVector.get(op.replicaId) ?? 0
      if (counter > peerMax) {
        missing.push(op)
      }
    }
    return missing
  }

  /** Compute this document's state vector (replicaId → max counter). */
  stateVector(): Map<string, number> {
    const sv = new Map<string, number>()
    for (const op of this.ops.values()) {
      const parts = op.opId.split(':')
      const counter = Number(parts[1])
      const current = sv.get(op.replicaId) ?? 0
      if (counter > current) {
        sv.set(op.replicaId, counter)
      }
    }
    return sv
  }

  // ─── Full Rebuild (for verification) ────────────────────────────────────

  /**
   * Rebuild the entire cache from the op log. Used for verification
   * that incremental updates match a full rebuild.
   */
  rebuild(): Map<string, ObjectSnapshot> {
    return reconstructAll(this.ops)
  }

  // ─── Internal ───────────────────────────────────────────────────────────

  private updateObjectCache(objectId: string): void {
    const objOps = this.objectOps.get(objectId)
    if (!objOps) return

    const snapshot = reconstructObject(objectId, [...objOps.values()])
    if (snapshot && snapshot.alive) {
      this.cache.set(objectId, snapshot)
    } else {
      this.cache.delete(objectId)
    }
  }
}
