/**
 * Adapter between the 2D editor's FloorPlanItem and the physics-solver's FurnitureItem.
 *
 * Handles:
 *   - FurnitureCategory → ItemType mapping (with round/rect table detection)
 *   - Degrees ↔ radians rotation conversion
 *   - Seat estimation from item dimensions
 *   - RoomBoundary construction from plan dimensions
 */
import type { FloorPlanItem, FurnitureCategory } from './store'
import {
  ItemType,
  type FurnitureItem as SolverItem,
  type RoomBoundary,
  type LayoutWeights,
  DEFAULT_WEIGHTS,
} from '@omni-twin/physics-solvers'

// ── Category → ItemType mapping ──────────────────────────────────────────────

const CATEGORY_TO_ITEM_TYPE: Record<FurnitureCategory, ItemType> = {
  table: ItemType.RoundTable,       // refined below based on aspect ratio
  chair: ItemType.Chair,
  stage: ItemType.Stage,
  decor: ItemType.Chair,            // treat as fixed obstacle
  equipment: ItemType.ServiceStation,
}

const DEG_TO_RAD = Math.PI / 180
const RAD_TO_DEG = 180 / Math.PI

// ── Seat estimation ──────────────────────────────────────────────────────────

function estimateSeats(item: FloorPlanItem): number {
  switch (item.category) {
    case 'chair':
      return 1
    case 'table':
      // Round tables: ~8 seats for 5ft+, ~6 for smaller
      // Rect tables: ~6 per 6ft length
      return item.widthFt >= 5 ? 8 : 6
    default:
      return 0
  }
}

// ── Conversion functions ─────────────────────────────────────────────────────

/** Convert a single editor item to solver format. */
export function editorToSolverItem(item: FloorPlanItem): SolverItem {
  // Distinguish round vs rect tables by aspect ratio
  let itemType: ItemType
  if (item.category === 'table') {
    itemType = Math.abs(item.widthFt - item.depthFt) < 0.5
      ? ItemType.RoundTable
      : ItemType.RectTable
  } else {
    itemType = CATEGORY_TO_ITEM_TYPE[item.category]
  }

  return {
    x: item.x,
    y: item.y,
    width: item.widthFt,
    depth: item.depthFt,
    rotation: item.rotation * DEG_TO_RAD,
    itemType,
    seats: estimateSeats(item),
  }
}

/** Convert editor items to solver items (excluding locked items). */
export function editorToSolverItems(items: FloorPlanItem[]): {
  solverItems: SolverItem[]
  participatingIndices: number[]
} {
  const solverItems: SolverItem[] = []
  const participatingIndices: number[] = []

  for (let i = 0; i < items.length; i++) {
    const item = items[i]!
    if (!item.locked) {
      solverItems.push(editorToSolverItem(item))
      participatingIndices.push(i)
    }
  }

  return { solverItems, participatingIndices }
}

/** Apply solver results back to editor items. Only position & rotation change. */
export function solverToEditorItems(
  solved: SolverItem[],
  originalItems: FloorPlanItem[],
  participatingIndices: number[],
): FloorPlanItem[] {
  const result = originalItems.map((item) => ({ ...item }))

  for (let si = 0; si < solved.length; si++) {
    const editorIdx = participatingIndices[si]!
    const solverItem = solved[si]!
    result[editorIdx] = {
      ...result[editorIdx]!,
      x: solverItem.x,
      y: solverItem.y,
      rotation: solverItem.rotation * RAD_TO_DEG,
    }
  }

  return result
}

// ── Room boundary ────────────────────────────────────────────────────────────

/** Create a rectangular room boundary with two default 6ft exits (top + bottom center). */
export function createRoomBoundary(widthFt: number, heightFt: number): RoomBoundary {
  return {
    vertices: new Float64Array([
      0, 0,
      widthFt, 0,
      widthFt, heightFt,
      0, heightFt,
    ]),
    // Two 6ft exits at center of top and bottom walls
    exits: new Float64Array([
      widthFt / 2, 0, 6,
      widthFt / 2, heightFt, 6,
    ]),
    width: widthFt,
    height: heightFt,
  }
}

// ── Default solver config ────────────────────────────────────────────────────

export { DEFAULT_WEIGHTS }
export type { LayoutWeights, SolverItem, RoomBoundary }
