/**
 * 2D ↔ 3D Coordinate Bridge
 *
 * Converts between the 2D floor plan coordinate system (feet, top-left origin)
 * and the 3D scene coordinate system (meters, centered origin).
 *
 * 2D: x = horizontal (feet from left), y = vertical (feet from top)
 * 3D: x = horizontal (meters), y = up, z = depth (meters)
 */
import type { FloorPlanItem, FurnitureCategory } from './store'
import type { FurnitureType } from '../../../store/types'

// 1 foot = 0.3048 meters
export const FT_TO_M = 0.3048
export const M_TO_FT = 1 / FT_TO_M

export interface Scene3DItem {
  id: string
  type: FurnitureType | null // null = unsupported category, render as box
  category: FurnitureCategory
  name: string
  position: [number, number, number]
  rotation: [number, number, number]
  /** Width and depth in meters (for fallback box rendering) */
  widthM: number
  depthM: number
}

/**
 * Map 2D furniture category + dimensions to a 3D FurnitureType.
 * Returns null for categories that don't have a 3D model.
 */
export function categoryTo3DType(item: FloorPlanItem): FurnitureType | null {
  switch (item.category) {
    case 'chair':
      return 'chair'
    case 'table':
      // Round tables have equal width/depth
      if (Math.abs(item.widthFt - item.depthFt) < 0.5) return 'round-table'
      return 'trestle-table'
    case 'stage':
      return 'platform'
    default:
      return null // decor, equipment → rendered as colored box
  }
}

/**
 * Convert a single 2D floor plan item to 3D scene coordinates.
 * Origin is centered on the floor plan.
 */
export function floorPlanItemTo3D(
  item: FloorPlanItem,
  planWidthFt: number,
  planHeightFt: number,
): Scene3DItem {
  // Center the origin: (0,0) in 2D becomes center of 3D scene
  const x3d = (item.x - planWidthFt / 2) * FT_TO_M
  const z3d = (item.y - planHeightFt / 2) * FT_TO_M

  // 2D rotation (degrees CW) → 3D y-rotation (radians CCW)
  const yRotation = -(item.rotation * Math.PI) / 180

  return {
    id: item.id,
    type: categoryTo3DType(item),
    category: item.category,
    name: item.name,
    position: [x3d, 0, z3d],
    rotation: [0, yRotation, 0],
    widthM: item.widthFt * FT_TO_M,
    depthM: item.depthFt * FT_TO_M,
  }
}

/**
 * Convert all 2D floor plan items to 3D.
 */
export function floorPlanTo3D(
  items: FloorPlanItem[],
  planWidthFt: number,
  planHeightFt: number,
): Scene3DItem[] {
  return items.map((item) => floorPlanItemTo3D(item, planWidthFt, planHeightFt))
}

/**
 * Get the 3D floor plane dimensions in meters.
 */
export function floorDimensions3D(planWidthFt: number, planHeightFt: number) {
  return {
    widthM: planWidthFt * FT_TO_M,
    depthM: planHeightFt * FT_TO_M,
  }
}
