/**
 * Grid-based spatial hash for fast spatial queries.
 * Divides the XZ plane into uniform cells and maps entity IDs to cells.
 */

import { defineQuery } from 'bitecs'
import { Position, BoundingBox } from '../components'
import type { EcsWorld } from '../world'

// ─── Types ──────────────────────────────────────────────────────────────────

/** Axis-aligned bounding box in world space. */
export interface AABB {
  minX: number
  minZ: number
  maxX: number
  maxZ: number
}

// ─── Spatial Hash ───────────────────────────────────────────────────────────

const positionedQuery = defineQuery([Position])

/**
 * A grid-based spatial hash index.
 * Entities are bucketed by their XZ position into uniform grid cells.
 * Supports AABB range queries and nearest-neighbor lookups.
 */
export class SpatialHash {
  /** Cell size in world units. */
  readonly cellSize: number

  /** Inverse cell size for fast division. */
  private readonly invCellSize: number

  /** Map from cell key to set of entity IDs. */
  private cells = new Map<number, Set<number>>()

  /** Reverse map: entity → cell key (for fast removal). */
  private entityCell = new Map<number, number>()

  constructor(cellSize = 2) {
    this.cellSize = cellSize
    this.invCellSize = 1 / cellSize
  }

  /** Compute the cell key for an (x, z) world position. */
  private cellKey(x: number, z: number): number {
    // Cantor pairing on grid coordinates (shifted to avoid negative issues)
    const cx = Math.floor(x * this.invCellSize) + 16384
    const cz = Math.floor(z * this.invCellSize) + 16384
    return cx * 32768 + cz
  }

  /** Insert an entity at (x, z). Removes from old cell if present. */
  insert(eid: number, x: number, z: number): void {
    const key = this.cellKey(x, z)
    const oldKey = this.entityCell.get(eid)

    if (oldKey === key) return // already in correct cell

    // Remove from old cell
    if (oldKey !== undefined) {
      const oldCell = this.cells.get(oldKey)
      if (oldCell) {
        oldCell.delete(eid)
        if (oldCell.size === 0) this.cells.delete(oldKey)
      }
    }

    // Insert into new cell
    let cell = this.cells.get(key)
    if (!cell) {
      cell = new Set()
      this.cells.set(key, cell)
    }
    cell.add(eid)
    this.entityCell.set(eid, key)
  }

  /** Remove an entity from the index. */
  remove(eid: number): void {
    const key = this.entityCell.get(eid)
    if (key === undefined) return

    const cell = this.cells.get(key)
    if (cell) {
      cell.delete(eid)
      if (cell.size === 0) this.cells.delete(key)
    }
    this.entityCell.delete(eid)
  }

  /** Clear all entries. */
  clear(): void {
    this.cells.clear()
    this.entityCell.clear()
  }

  /**
   * Query all entity IDs whose cell overlaps the given AABB.
   * This is a broadphase — results may include entities outside the AABB.
   */
  queryAABB(aabb: AABB): number[] {
    const minCX = Math.floor(aabb.minX * this.invCellSize) + 16384
    const minCZ = Math.floor(aabb.minZ * this.invCellSize) + 16384
    const maxCX = Math.floor(aabb.maxX * this.invCellSize) + 16384
    const maxCZ = Math.floor(aabb.maxZ * this.invCellSize) + 16384

    const result: number[] = []
    for (let cx = minCX; cx <= maxCX; cx++) {
      for (let cz = minCZ; cz <= maxCZ; cz++) {
        const key = cx * 32768 + cz
        const cell = this.cells.get(key)
        if (cell) {
          for (const eid of cell) {
            result.push(eid)
          }
        }
      }
    }
    return result
  }

  /**
   * Query all entity IDs within `radius` of point (x, z).
   * Broadphase — uses AABB around the circle.
   */
  queryRadius(x: number, z: number, radius: number): number[] {
    return this.queryAABB({
      minX: x - radius,
      minZ: z - radius,
      maxX: x + radius,
      maxZ: z + radius,
    })
  }

  /** Number of tracked entities. */
  get size(): number {
    return this.entityCell.size
  }

  /**
   * Rebuild the index from the current ECS world state.
   * Call after bulk changes or on initialization.
   */
  rebuild(world: EcsWorld): void {
    this.clear()
    const entities = positionedQuery(world)
    for (const eid of entities) {
      this.insert(eid, Position.x[eid]!, Position.z[eid]!)
    }
  }

  /**
   * Update a single entity's position in the index.
   * Call when an entity moves.
   */
  update(eid: number): void {
    this.insert(eid, Position.x[eid]!, Position.z[eid]!)
  }
}
