/**
 * Standalone spatial hash for the constraint solver.
 *
 * Modeled on the ECS SpatialHash (packages/engine/src/ecs/systems/spatial-index.ts)
 * but operates on placement indices instead of entity IDs, and uses AABB-based
 * multi-cell insertion (large furniture spans multiple cells).
 *
 * No bitECS dependency — the solver works standalone.
 */

import type { Placement } from './types'

// ─── SolverSpatialHash ─────────────────────────────────────────────────────

export class SolverSpatialHash {
  readonly cellSize: number
  private readonly invCellSize: number

  /** cellKey → set of placement indices in that cell. */
  private cells = new Map<number, Set<number>>()

  /** placementIndex → array of cell keys it occupies (multi-cell for large items). */
  private entityCells = new Map<number, number[]>()

  constructor(cellSize = 2) {
    this.cellSize = cellSize
    this.invCellSize = 1 / cellSize
  }

  /** Cantor pairing cell key, shifted by 16384 for negative coordinates. */
  private cellKey(x: number, z: number): number {
    const cx = Math.floor(x * this.invCellSize) + 16384
    const cz = Math.floor(z * this.invCellSize) + 16384
    return cx * 32768 + cz
  }

  /**
   * Insert a placement by index, occupying all cells its AABB covers.
   */
  insert(index: number, cx: number, cz: number, halfW: number, halfD: number): void {
    this.remove(index) // idempotent

    const minCX = Math.floor((cx - halfW) * this.invCellSize)
    const maxCX = Math.floor((cx + halfW) * this.invCellSize)
    const minCZ = Math.floor((cz - halfD) * this.invCellSize)
    const maxCZ = Math.floor((cz + halfD) * this.invCellSize)

    const keys: number[] = []
    for (let gx = minCX; gx <= maxCX; gx++) {
      for (let gz = minCZ; gz <= maxCZ; gz++) {
        const key = (gx + 16384) * 32768 + (gz + 16384)
        keys.push(key)
        let set = this.cells.get(key)
        if (!set) {
          set = new Set()
          this.cells.set(key, set)
        }
        set.add(index)
      }
    }
    this.entityCells.set(index, keys)
  }

  /** Insert from a Placement object. */
  insertPlacement(index: number, p: Placement): void {
    this.insert(index, p.x, p.z, p.effectiveWidth / 2, p.effectiveDepth / 2)
  }

  /** Remove a placement by index. */
  remove(index: number): void {
    const keys = this.entityCells.get(index)
    if (!keys) return
    for (const key of keys) {
      const set = this.cells.get(key)
      if (set) {
        set.delete(index)
        if (set.size === 0) this.cells.delete(key)
      }
    }
    this.entityCells.delete(index)
  }

  /** Update: remove then re-insert. */
  update(index: number, cx: number, cz: number, halfW: number, halfD: number): void {
    this.remove(index)
    this.insert(index, cx, cz, halfW, halfD)
  }

  /** Query all placement indices whose cells overlap the given AABB. */
  queryAABB(minX: number, minZ: number, maxX: number, maxZ: number): number[] {
    const minCX = Math.floor(minX * this.invCellSize)
    const maxCX = Math.floor(maxX * this.invCellSize)
    const minCZ = Math.floor(minZ * this.invCellSize)
    const maxCZ = Math.floor(maxZ * this.invCellSize)

    const result = new Set<number>()
    for (let gx = minCX; gx <= maxCX; gx++) {
      for (let gz = minCZ; gz <= maxCZ; gz++) {
        const key = (gx + 16384) * 32768 + (gz + 16384)
        const set = this.cells.get(key)
        if (set) {
          for (const idx of set) result.add(idx)
        }
      }
    }
    return Array.from(result)
  }

  /** Query all placement indices within radius of (x, z). */
  queryRadius(x: number, z: number, radius: number): number[] {
    return this.queryAABB(x - radius, z - radius, x + radius, z + radius)
  }

  /** Build from an array of placements. */
  buildFromPlacements(placements: Placement[]): void {
    this.clear()
    for (let i = 0; i < placements.length; i++) {
      this.insertPlacement(i, placements[i]!)
    }
  }

  /** Remove all entries. */
  clear(): void {
    this.cells.clear()
    this.entityCells.clear()
  }

  /** Number of tracked placements. */
  get size(): number {
    return this.entityCells.size
  }
}
