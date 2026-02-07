/**
 * Discretized room grid for constraint solving.
 *
 * Divides the room into uniform cells (~6-inch resolution by default).
 * Each cell tracks occupancy state for fast placement validation.
 */

import type { RoomConfig, Rect, Exit, Point2D } from './types'

// ─── Cell States ────────────────────────────────────────────────────────────

export const CELL_EMPTY = 0
export const CELL_WALL = 1
export const CELL_OBSTACLE = 2
export const CELL_OCCUPIED = 3
export const CELL_EXIT_ZONE = 4

export type CellState = typeof CELL_EMPTY | typeof CELL_WALL | typeof CELL_OBSTACLE | typeof CELL_OCCUPIED | typeof CELL_EXIT_ZONE

// ─── Grid ───────────────────────────────────────────────────────────────────

export class LayoutGrid {
  readonly cellSize: number
  readonly cols: number
  readonly rows: number
  readonly roomWidth: number
  readonly roomDepth: number

  /** Flat array: cells[row * cols + col]. */
  private cells: Uint8Array

  constructor(room: RoomConfig, cellSize = 0.15) {
    this.cellSize = cellSize
    this.roomWidth = room.width
    this.roomDepth = room.depth
    this.cols = Math.ceil(room.width / cellSize)
    this.rows = Math.ceil(room.depth / cellSize)
    this.cells = new Uint8Array(this.cols * this.rows)

    // Mark obstacles
    for (const obs of room.obstacles) {
      this.markRect(obs, CELL_OBSTACLE)
    }

    // Mark exit clearance zones
    for (const exit of room.exits) {
      this.markExitZone(exit)
    }
  }

  // ─── Coordinate conversion ──────────────────────────────────────────────

  /** World position → grid column. */
  toCol(x: number): number {
    return Math.floor(x / this.cellSize)
  }

  /** World position → grid row. */
  toRow(z: number): number {
    return Math.floor(z / this.cellSize)
  }

  /** Grid column → world X (cell center). */
  toX(col: number): number {
    return (col + 0.5) * this.cellSize
  }

  /** Grid row → world Z (cell center). */
  toZ(row: number): number {
    return (row + 0.5) * this.cellSize
  }

  // ─── Cell access ────────────────────────────────────────────────────────

  /** Get cell state. Returns CELL_WALL for out-of-bounds. */
  get(col: number, row: number): CellState {
    if (col < 0 || col >= this.cols || row < 0 || row >= this.rows) return CELL_WALL
    return this.cells[row * this.cols + col] as CellState
  }

  /** Set cell state. */
  private set(col: number, row: number, state: CellState): void {
    if (col < 0 || col >= this.cols || row < 0 || row >= this.rows) return
    this.cells[row * this.cols + col] = state
  }

  /** Check if a cell is free for placement. */
  isFree(col: number, row: number): boolean {
    return this.get(col, row) === CELL_EMPTY
  }

  // ─── Rect operations ───────────────────────────────────────────────────

  /** Mark a world-space rectangle with a cell state. */
  private markRect(rect: Rect, state: CellState): void {
    const minCol = this.toCol(rect.x)
    const minRow = this.toRow(rect.z)
    const maxCol = this.toCol(rect.x + rect.width)
    const maxRow = this.toRow(rect.z + rect.depth)
    for (let r = minRow; r <= maxRow; r++) {
      for (let c = minCol; c <= maxCol; c++) {
        this.set(c, r, state)
      }
    }
  }

  /** Mark an exit's clearance zone (fire code: area in front of exit must stay clear). */
  private markExitZone(exit: Exit, clearance = 1.12): void {
    // Mark a rectangle in front of the exit
    const hw = exit.width / 2
    const cos = Math.cos(exit.facing)
    const sin = Math.sin(exit.facing)

    // Project clearance zone outward from exit
    const cx = exit.position.x + cos * clearance / 2
    const cz = exit.position.z + sin * clearance / 2

    const rectX = cx - hw - clearance / 2
    const rectZ = cz - hw - clearance / 2
    const rectW = exit.width + clearance
    const rectD = exit.width + clearance

    this.markRect(
      { x: Math.max(0, rectX), z: Math.max(0, rectZ), width: rectW, depth: rectD },
      CELL_EXIT_ZONE,
    )
  }

  /**
   * Check if a furniture footprint can be placed at (cx, cz) with given half-extents.
   * Returns true if all cells in the footprint are free.
   */
  canPlace(cx: number, cz: number, halfW: number, halfD: number): boolean {
    const minCol = this.toCol(cx - halfW)
    const minRow = this.toRow(cz - halfD)
    const maxCol = this.toCol(cx + halfW)
    const maxRow = this.toRow(cz + halfD)

    for (let r = minRow; r <= maxRow; r++) {
      for (let c = minCol; c <= maxCol; c++) {
        if (!this.isFree(c, r)) return false
      }
    }
    return true
  }

  /**
   * Mark a furniture footprint as occupied.
   */
  occupy(cx: number, cz: number, halfW: number, halfD: number): void {
    const minCol = this.toCol(cx - halfW)
    const minRow = this.toRow(cz - halfD)
    const maxCol = this.toCol(cx + halfW)
    const maxRow = this.toRow(cz + halfD)

    for (let r = minRow; r <= maxRow; r++) {
      for (let c = minCol; c <= maxCol; c++) {
        this.set(c, r, CELL_OCCUPIED)
      }
    }
  }

  /**
   * Unmark a furniture footprint (set back to empty).
   */
  vacate(cx: number, cz: number, halfW: number, halfD: number): void {
    const minCol = this.toCol(cx - halfW)
    const minRow = this.toRow(cz - halfD)
    const maxCol = this.toCol(cx + halfW)
    const maxRow = this.toRow(cz + halfD)

    for (let r = minRow; r <= maxRow; r++) {
      for (let c = minCol; c <= maxCol; c++) {
        if (this.get(c, r) === CELL_OCCUPIED) {
          this.set(c, r, CELL_EMPTY)
        }
      }
    }
  }

  /**
   * Check minimum aisle width around a placement.
   * Returns true if there is at least `minAisle` meters of clearance
   * on at least two opposite sides.
   */
  hasAisleClearance(cx: number, cz: number, halfW: number, halfD: number, minAisle: number): boolean {
    const aisleCells = Math.ceil(minAisle / this.cellSize)

    // Check clearance on left/right (X axis)
    const leftCol = this.toCol(cx - halfW) - aisleCells
    const rightCol = this.toCol(cx + halfW) + aisleCells
    const topRow = this.toRow(cz - halfD)
    const bottomRow = this.toRow(cz + halfD)

    let leftClear = true
    let rightClear = true

    for (let r = topRow; r <= bottomRow; r++) {
      for (let c = this.toCol(cx - halfW) - aisleCells; c < this.toCol(cx - halfW); c++) {
        if (this.get(c, r) === CELL_OCCUPIED) { leftClear = false; break }
      }
      if (!leftClear) break
    }

    for (let r = topRow; r <= bottomRow; r++) {
      for (let c = this.toCol(cx + halfW) + 1; c <= rightCol; c++) {
        if (this.get(c, r) === CELL_OCCUPIED) { rightClear = false; break }
      }
      if (!rightClear) break
    }

    if (leftClear || rightClear) return true

    // Check clearance on top/bottom (Z axis)
    let topClear = true
    let bottomClear2 = true
    const leftC = this.toCol(cx - halfW)
    const rightC = this.toCol(cx + halfW)

    for (let c = leftC; c <= rightC; c++) {
      for (let r = this.toRow(cz - halfD) - aisleCells; r < this.toRow(cz - halfD); r++) {
        if (this.get(c, r) === CELL_OCCUPIED) { topClear = false; break }
      }
      if (!topClear) break
    }

    for (let c = leftC; c <= rightC; c++) {
      for (let r = this.toRow(cz + halfD) + 1; r <= this.toRow(cz + halfD) + aisleCells; r++) {
        if (this.get(c, r) === CELL_OCCUPIED) { bottomClear2 = false; break }
      }
      if (!bottomClear2) break
    }

    return topClear || bottomClear2
  }

  /** Count total free cells. */
  freeCellCount(): number {
    let count = 0
    for (let i = 0; i < this.cells.length; i++) {
      if (this.cells[i] === CELL_EMPTY) count++
    }
    return count
  }

  /** Total cell count. */
  totalCells(): number {
    return this.cols * this.rows
  }

  /** Create a snapshot for backtracking. */
  snapshot(): Uint8Array {
    return new Uint8Array(this.cells)
  }

  /** Restore from snapshot. */
  restore(snapshot: Uint8Array): void {
    this.cells.set(snapshot)
  }
}
