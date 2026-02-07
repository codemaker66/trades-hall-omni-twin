/**
 * Chair-table grouping: places chairs around tables based on chairsPerUnit.
 *
 * Implements the previously unused `chairsPerUnit` field in FurnitureSpec.
 * Round tables get chairs in a circle; rectangular/trestle tables get chairs
 * along the two long edges.
 */

import type { Placement, FurnitureSpec, RoomConfig, TableGrouping, Point2D } from './types'
import type { LayoutGrid } from './grid'
import type { SolverSpatialHash } from './spatial-hash'

/** Table types that can have chairs grouped around them. */
const TABLE_TYPES = new Set(['round-table', 'rect-table', 'trestle-table'])

/** Chair setback from table edge (meters). */
const CHAIR_SETBACK = 0.35

/** Default chair dimensions (matching FurnitureSpec standard). */
const DEFAULT_CHAIR_WIDTH = 0.44
const DEFAULT_CHAIR_DEPTH = 0.44

/**
 * Generate ideal chair positions around a table.
 *
 * Round tables: chairs evenly spaced in a circle at radius = tableRadius + setback.
 * Rectangular/trestle: chairs along the two long edges, evenly spaced.
 */
export function generateChairPositions(
  table: Placement,
  tableSpec: FurnitureSpec,
  count: number,
): Point2D[] {
  if (count <= 0) return []

  const positions: Point2D[] = []

  if (tableSpec.type === 'round-table') {
    // Circle layout: chairs evenly spaced around the table
    const radius = Math.max(tableSpec.width, tableSpec.depth) / 2 + CHAIR_SETBACK
    for (let i = 0; i < count; i++) {
      const angle = (2 * Math.PI * i) / count + table.rotation
      positions.push({
        x: table.x + Math.cos(angle) * radius,
        z: table.z + Math.sin(angle) * radius,
      })
    }
  } else {
    // Rectangular/trestle: chairs along the two long edges
    const isRotated = table.effectiveWidth !== tableSpec.width
    const longSide = isRotated ? tableSpec.depth : tableSpec.width
    const shortSide = isRotated ? tableSpec.width : tableSpec.depth

    const offset = shortSide / 2 + CHAIR_SETBACK
    const perSide = Math.ceil(count / 2)
    const spacing = longSide / (perSide + 1)

    // Start from one end of the long side
    const startX = table.x - table.effectiveWidth / 2
    const startZ = table.z - table.effectiveDepth / 2

    let placed = 0
    // Side 1 (positive offset from center on short axis)
    for (let i = 0; i < perSide && placed < count; i++) {
      if (isRotated) {
        positions.push({
          x: table.x + offset,
          z: startZ + spacing * (i + 1),
        })
      } else {
        positions.push({
          x: startX + spacing * (i + 1),
          z: table.z + offset,
        })
      }
      placed++
    }

    // Side 2 (negative offset)
    for (let i = 0; i < perSide && placed < count; i++) {
      if (isRotated) {
        positions.push({
          x: table.x - offset,
          z: startZ + spacing * (i + 1),
        })
      } else {
        positions.push({
          x: startX + spacing * (i + 1),
          z: table.z - offset,
        })
      }
      placed++
    }
  }

  return positions
}

/**
 * Compute chair rotation to face toward the table center.
 */
function chairRotationFacingTable(chairX: number, chairZ: number, tableX: number, tableZ: number): number {
  return Math.atan2(tableZ - chairZ, tableX - chairX)
}

/**
 * Place chairs around tables according to chairsPerUnit.
 *
 * For each table spec with chairsPerUnit > 0, generates ideal chair positions
 * around each placed table, validates against the grid and spatial hash,
 * and places valid chairs.
 */
export function placeChairGroups(
  room: RoomConfig,
  specs: FurnitureSpec[],
  placements: Placement[],
  grid: LayoutGrid,
  spatialHash: SolverSpatialHash,
  opts: { minAisle: number; gridCellSize: number },
): { chairs: Placement[]; groupings: TableGrouping[] } {
  const chairs: Placement[] = []
  const groupings: TableGrouping[] = []

  // Find chair spec index (first 'chair' spec)
  const chairSpecIndex = specs.findIndex((s) => s.type === 'chair')
  if (chairSpecIndex < 0) return { chairs, groupings }

  const chairSpec = specs[chairSpecIndex]!
  const chairHW = (chairSpec.width || DEFAULT_CHAIR_WIDTH) / 2
  const chairHD = (chairSpec.depth || DEFAULT_CHAIR_DEPTH) / 2

  // Track how many grouped chairs we've placed (for instanceIndex)
  let groupedChairCount = 0

  for (let pi = 0; pi < placements.length; pi++) {
    const placement = placements[pi]!
    const spec = specs[placement.specIndex]!

    if (!TABLE_TYPES.has(spec.type) || spec.chairsPerUnit <= 0) continue

    const positions = generateChairPositions(placement, spec, spec.chairsPerUnit)
    const chairIndices: number[] = []

    for (const pos of positions) {
      // Bounds check
      if (pos.x - chairHW < 0 || pos.x + chairHW > room.width ||
          pos.z - chairHD < 0 || pos.z + chairHD > room.depth) {
        continue
      }

      // Grid check
      if (!grid.canPlace(pos.x, pos.z, chairHW, chairHD)) continue

      // Snap to grid
      const sx = Math.round(pos.x / opts.gridCellSize) * opts.gridCellSize
      const sz = Math.round(pos.z / opts.gridCellSize) * opts.gridCellSize

      // Check snapped position too
      if (!grid.canPlace(sx, sz, chairHW, chairHD)) continue

      const rotation = chairRotationFacingTable(sx, sz, placement.x, placement.z)

      const chairPlacement: Placement = {
        specIndex: chairSpecIndex,
        instanceIndex: groupedChairCount,
        x: sx,
        z: sz,
        rotation,
        type: 'chair',
        effectiveWidth: chairSpec.width || DEFAULT_CHAIR_WIDTH,
        effectiveDepth: chairSpec.depth || DEFAULT_CHAIR_DEPTH,
      }

      // Record the index in the combined placements array
      const combinedIndex = placements.length + chairs.length
      chairIndices.push(combinedIndex)
      chairs.push(chairPlacement)
      groupedChairCount++

      // Occupy grid and spatial hash
      grid.occupy(sx, sz, chairHW, chairHD)
      spatialHash.insert(combinedIndex, sx, sz, chairHW, chairHD)
    }

    if (chairIndices.length > 0) {
      groupings.push({
        tableIndex: pi,
        chairIndices,
        chairsPerUnit: spec.chairsPerUnit,
      })
    }
  }

  return { chairs, groupings }
}
