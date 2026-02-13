/**
 * PS-5: Venue Layout Energy Function
 *
 * Computes total layout energy as a weighted sum of 8 code-compliance,
 * aesthetic, and operational terms. Used by SA/PT/MCMC solvers to
 * optimise furniture placement in event venues.
 *
 * Energy terms:
 *   E_overlap   — No furniture overlaps (SAT on oriented bounding boxes)
 *   E_aisle     — Minimum aisle widths (IBC §1029.9.2)
 *   E_egress    — Fire code egress paths (IBC §1017)
 *   E_sightline — Visibility to focal point (stage/podium)
 *   E_capacity  — Target seat count
 *   E_ADA       — Accessibility requirements (ADA / IBC)
 *   E_aesthetic  — Visual quality (alignment, symmetry, balance, spacing)
 *   E_service   — Catering/AV service paths
 */

import type { FurnitureItem, RoomBoundary, LayoutWeights, PRNG } from '../types.js'
import { ItemType } from '../types.js'

// ---------------------------------------------------------------------------
// Constants — IBC / ADA dimensional requirements (all in feet)
// ---------------------------------------------------------------------------

/** Main aisle minimum width: 54" = 4.5 ft (IBC §1029.9.2) */
const MAIN_AISLE_MIN = 4.5
/** Service aisle minimum width: 60" = 5 ft */
const SERVICE_AISLE_MIN = 5.0
/** Between table rows minimum: 36" = 3 ft */
const TABLE_ROW_GAP_MIN = 3.0
/** Chair back to table edge minimum: 18" = 1.5 ft */
const CHAIR_TABLE_GAP_MIN = 1.5
/** Assembly occupancy egress path width: 44" ~ 3.67 ft (IBC §1017) */
const EGRESS_PATH_WIDTH = 3.67
/** Maximum travel distance to exit: 200 ft (IBC §1017) */
const EGRESS_MAX_DISTANCE = 200
/** Wheelchair space width: 36" = 3 ft */
const WHEELCHAIR_WIDTH = 3.0
/** Wheelchair space depth: 48" = 4 ft */
const WHEELCHAIR_DEPTH = 4.0
/** Accessible path width: 36" = 3 ft */
const ACCESSIBLE_PATH_WIDTH = 3.0
/** BFS grid cell size in feet */
const GRID_STEP = 1.0
/** Penalty for unreachable seats in egress calculation */
const UNREACHABLE_PENALTY = 1e8
/** Penalty per missing wheelchair space */
const ADA_MISSING_SPACE_PENALTY = 1e6
/** Penalty per blocked service path */
const SERVICE_BLOCKED_PENALTY = 1e4
/** Service perimeter clearance: 60" = 5 ft */
const SERVICE_PATH_WIDTH = 5.0

// ---------------------------------------------------------------------------
// Vec2 helpers
// ---------------------------------------------------------------------------

type Vec2 = [number, number]

function vec2Sub(a: Vec2, b: Vec2): Vec2 {
  return [a[0] - b[0], a[1] - b[1]]
}

function vec2Dot(a: Vec2, b: Vec2): number {
  return a[0] * b[0] + a[1] * b[1]
}

function vec2Length(v: Vec2): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1])
}

function vec2Normalize(v: Vec2): Vec2 {
  const len = vec2Length(v)
  if (len < 1e-12) return [0, 0]
  return [v[0] / len, v[1] / len]
}

function vec2Perp(v: Vec2): Vec2 {
  return [-v[1], v[0]]
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/**
 * Get the 4 corners of a furniture item's oriented bounding box.
 * Item position (x, y) is the center. Rotation in radians.
 */
export function getOBBCorners(item: FurnitureItem): [Vec2, Vec2, Vec2, Vec2] {
  const cos = Math.cos(item.rotation)
  const sin = Math.sin(item.rotation)
  const hw = item.width / 2
  const hd = item.depth / 2

  // Local-space corners, then rotate and translate
  const localCorners: [Vec2, Vec2, Vec2, Vec2] = [
    [-hw, -hd],
    [hw, -hd],
    [hw, hd],
    [-hw, hd],
  ]

  return localCorners.map(([lx, ly]): Vec2 => [
    item.x + lx * cos - ly * sin,
    item.y + lx * sin + ly * cos,
  ]) as [Vec2, Vec2, Vec2, Vec2]
}

/**
 * Project polygon corners onto an axis, returning min and max scalar values.
 */
export function projectOntoAxis(
  corners: readonly Vec2[],
  axis: Vec2,
): { min: number; max: number } {
  let min = Infinity
  let max = -Infinity
  for (const corner of corners) {
    const proj = vec2Dot(corner, axis)
    if (proj < min) min = proj
    if (proj > max) max = proj
  }
  return { min, max }
}

/**
 * Test if a point lies inside a polygon using ray casting.
 * Vertices is a flat Float64Array [x1,y1, x2,y2, ...] or an array of Vec2.
 */
export function pointInPolygon(
  px: number,
  py: number,
  vertices: Float64Array | readonly Vec2[],
): boolean {
  let inside = false

  const isFlat = vertices instanceof Float64Array
  const n = isFlat
    ? (vertices as Float64Array).length / 2
    : (vertices as readonly Vec2[]).length

  for (let i = 0, j = n - 1; i < n; j = i++) {
    let xi: number, yi: number, xj: number, yj: number
    if (isFlat) {
      const flat = vertices as Float64Array
      xi = flat[i * 2]!
      yi = flat[i * 2 + 1]!
      xj = flat[j * 2]!
      yj = flat[j * 2 + 1]!
    } else {
      const arr = vertices as readonly Vec2[]
      const pi = arr[i]!
      const pj = arr[j]!
      xi = pi[0]
      yi = pi[1]
      xj = pj[0]
      yj = pj[1]
    }

    if (
      (yi > py) !== (yj > py) &&
      px < ((xj - xi) * (py - yi)) / (yj - yi) + xi
    ) {
      inside = !inside
    }
  }

  return inside
}

/**
 * Check if a 2D line segment from (x1,y1) to (x2,y2) intersects an
 * axis-aligned bounding box defined by (minX, minY, maxX, maxY).
 * Returns the parametric t of the first intersection, or -1 if none.
 */
export function lineSegmentIntersectsAABB(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
): number {
  const dx = x2 - x1
  const dy = y2 - y1

  let tMin = 0
  let tMax = 1

  // X slab
  if (Math.abs(dx) < 1e-12) {
    if (x1 < minX || x1 > maxX) return -1
  } else {
    const invDx = 1 / dx
    let t0 = (minX - x1) * invDx
    let t1 = (maxX - x1) * invDx
    if (t0 > t1) {
      const tmp = t0
      t0 = t1
      t1 = tmp
    }
    tMin = Math.max(tMin, t0)
    tMax = Math.min(tMax, t1)
    if (tMin > tMax) return -1
  }

  // Y slab
  if (Math.abs(dy) < 1e-12) {
    if (y1 < minY || y1 > maxY) return -1
  } else {
    const invDy = 1 / dy
    let t0 = (minY - y1) * invDy
    let t1 = (maxY - y1) * invDy
    if (t0 > t1) {
      const tmp = t0
      t0 = t1
      t1 = tmp
    }
    tMin = Math.max(tMin, t0)
    tMax = Math.min(tMax, t1)
    if (tMin > tMax) return -1
  }

  return tMin
}

// ---------------------------------------------------------------------------
// Separating Axis Theorem — OBB overlap
// ---------------------------------------------------------------------------

/**
 * Get the two unique edge normals (axes) for an OBB.
 * A rectangle has only 2 unique normals (the other 2 are parallel).
 */
function getOBBAxes(corners: readonly Vec2[]): [Vec2, Vec2] {
  const c0 = corners[0]!
  const c1 = corners[1]!
  const c3 = corners[3]!

  const edge1 = vec2Sub(c1, c0)
  const edge2 = vec2Sub(c3, c0)

  return [vec2Normalize(vec2Perp(edge1)), vec2Normalize(vec2Perp(edge2))]
}

/**
 * Compute the overlap area between two oriented bounding boxes.
 *
 * Uses the Separating Axis Theorem (SAT) for rotated rectangles.
 * If separated on any axis, returns 0. Otherwise, estimates the
 * overlap area from the minimum overlap depth on the separating axes
 * multiplied by the perpendicular overlap extent.
 */
export function computeOBBOverlap(a: FurnitureItem, b: FurnitureItem): number {
  const cornersA = getOBBCorners(a)
  const cornersB = getOBBCorners(b)

  const [axisA1, axisA2] = getOBBAxes(cornersA)
  const [axisB1, axisB2] = getOBBAxes(cornersB)

  const axes: Vec2[] = [axisA1, axisA2, axisB1, axisB2]

  let minOverlap = Infinity
  let minOverlapPerp = 0

  for (let i = 0; i < axes.length; i++) {
    const axis = axes[i]!
    const projA = projectOntoAxis(cornersA, axis)
    const projB = projectOntoAxis(cornersB, axis)

    const overlap = Math.min(projA.max, projB.max) - Math.max(projA.min, projB.min)
    if (overlap <= 0) return 0 // Separating axis found — no overlap

    if (overlap < minOverlap) {
      minOverlap = overlap

      // Get perpendicular extent for area estimation
      const perpAxis = vec2Perp(axis)
      const perpA = projectOntoAxis(cornersA, perpAxis)
      const perpB = projectOntoAxis(cornersB, perpAxis)
      const perpOverlap =
        Math.min(perpA.max, perpB.max) - Math.max(perpA.min, perpB.min)
      minOverlapPerp = Math.max(0, perpOverlap)
    }
  }

  // Approximate overlap area = min penetration depth x perpendicular overlap extent
  return minOverlap * minOverlapPerp
}

/**
 * Compute the minimum gap between two OBBs.
 * Returns 0 if they overlap, otherwise the minimum distance between edges.
 */
function computeOBBGap(a: FurnitureItem, b: FurnitureItem): number {
  const cornersA = getOBBCorners(a)
  const cornersB = getOBBCorners(b)

  const [axisA1, axisA2] = getOBBAxes(cornersA)
  const [axisB1, axisB2] = getOBBAxes(cornersB)

  const axes: Vec2[] = [axisA1, axisA2, axisB1, axisB2]

  let maxSeparation = -Infinity

  for (const axis of axes) {
    const projA = projectOntoAxis(cornersA, axis)
    const projB = projectOntoAxis(cornersB, axis)

    // Separation = gap between projections (negative means overlap)
    const separation = Math.max(projA.min - projB.max, projB.min - projA.max)
    if (separation > maxSeparation) {
      maxSeparation = separation
    }
  }

  // If maxSeparation > 0, that is the minimum gap distance.
  // If <= 0, the OBBs overlap and gap is 0.
  return Math.max(0, maxSeparation)
}

// ---------------------------------------------------------------------------
// BFS distance grid for egress calculation
// ---------------------------------------------------------------------------

/** Cardinal + diagonal directions for BFS (8-connected grid) */
const BFS_DIRS: readonly [number, number, number][] = [
  [1, 0, 1.0],
  [-1, 0, 1.0],
  [0, 1, 1.0],
  [0, -1, 1.0],
  [1, 1, 1.414],
  [1, -1, 1.414],
  [-1, 1, 1.414],
  [-1, -1, 1.414],
]

/**
 * Check if a point is inside an OBB with expanded half-extents.
 * Used for clearance-buffered obstacle marking.
 */
function pointInOBB(
  px: number,
  py: number,
  expandedHalfW: number,
  expandedHalfD: number,
  item: FurnitureItem,
): boolean {
  // Transform point to OBB local space
  const cos = Math.cos(-item.rotation)
  const sin = Math.sin(-item.rotation)
  const dx = px - item.x
  const dy = py - item.y
  const localX = dx * cos - dy * sin
  const localY = dx * sin + dy * cos

  return Math.abs(localX) <= expandedHalfW && Math.abs(localY) <= expandedHalfD
}

/** Binary-search insert into a sorted array of [distance, row, col] */
function insertSorted(
  arr: [number, number, number][],
  item: [number, number, number],
): void {
  let lo = 0
  let hi = arr.length
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if (arr[mid]![0] < item[0]) {
      lo = mid + 1
    } else {
      hi = mid
    }
  }
  arr.splice(lo, 0, item)
}

/**
 * Build a BFS distance grid from room exits.
 *
 * Discretises the room into a grid of cells. Furniture items act as
 * obstacles (cells covered by furniture are impassable). BFS flood-fills
 * from exit positions outward, producing shortest distance (in feet)
 * from each cell to the nearest exit.
 *
 * @param room      Room boundary with exits
 * @param items     Furniture items acting as obstacles
 * @param gridStep  Cell size in feet (default 1 ft)
 * @param pathWidth Minimum clear path width — cells covered by furniture
 *                  expanded by this half-width are marked blocked.
 * @returns Object with distances (flat array), cols, and rows
 */
export function bfsGrid(
  room: RoomBoundary,
  items: FurnitureItem[],
  gridStep: number = GRID_STEP,
  pathWidth: number = EGRESS_PATH_WIDTH,
): { distances: Float64Array; cols: number; rows: number } {
  const cols = Math.ceil(room.width / gridStep)
  const rows = Math.ceil(room.height / gridStep)
  const totalCells = rows * cols

  // Build obstacle map — 1 means cell is blocked
  const blocked = new Uint8Array(totalCells)

  // Mark cells occupied by furniture (with clearance buffer for path width)
  for (const item of items) {
    // Skip chairs — they can be moved aside during egress
    if (item.itemType === ItemType.Chair) continue

    const expandedHalfW = item.width / 2 + pathWidth / 2
    const expandedHalfD = item.depth / 2 + pathWidth / 2

    // Compute AABB of the expanded OBB for fast grid iteration
    const corners = getOBBCorners(item)
    let aabbMinX = Infinity
    let aabbMinY = Infinity
    let aabbMaxX = -Infinity
    let aabbMaxY = -Infinity
    for (const c of corners) {
      if (c[0] < aabbMinX) aabbMinX = c[0]
      if (c[1] < aabbMinY) aabbMinY = c[1]
      if (c[0] > aabbMaxX) aabbMaxX = c[0]
      if (c[1] > aabbMaxY) aabbMaxY = c[1]
    }

    // Expand AABB by path clearance
    const rMin = Math.max(0, Math.floor((aabbMinY - pathWidth / 2) / gridStep))
    const rMax = Math.min(
      rows - 1,
      Math.ceil((aabbMaxY + pathWidth / 2) / gridStep),
    )
    const cMin = Math.max(0, Math.floor((aabbMinX - pathWidth / 2) / gridStep))
    const cMax = Math.min(
      cols - 1,
      Math.ceil((aabbMaxX + pathWidth / 2) / gridStep),
    )

    for (let r = rMin; r <= rMax; r++) {
      for (let c = cMin; c <= cMax; c++) {
        const cellX = c * gridStep + gridStep / 2
        const cellY = r * gridStep + gridStep / 2
        if (pointInOBB(cellX, cellY, expandedHalfW, expandedHalfD, item)) {
          blocked[r * cols + c] = 1
        }
      }
    }
  }

  // Also block cells outside the room polygon
  if (room.vertices.length >= 6) {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cellX = c * gridStep + gridStep / 2
        const cellY = r * gridStep + gridStep / 2
        if (!pointInPolygon(cellX, cellY, room.vertices)) {
          blocked[r * cols + c] = 1
        }
      }
    }
  }

  // Initialize distances to Infinity
  const distances = new Float64Array(totalCells).fill(Infinity)

  // Dijkstra-style BFS on 8-connected grid
  const queue: [number, number, number][] = []

  // Seed from exit positions
  const nExits = Math.floor(room.exits.length / 3)
  for (let e = 0; e < nExits; e++) {
    const ex = room.exits[e * 3]!
    const ey = room.exits[e * 3 + 1]!
    const ew = room.exits[e * 3 + 2]!

    // Mark all grid cells within the exit opening as distance 0
    const exitHalfW = ew / 2
    const cStart = Math.max(0, Math.floor((ex - exitHalfW) / gridStep))
    const cEnd = Math.min(cols - 1, Math.ceil((ex + exitHalfW) / gridStep))
    const rStart = Math.max(0, Math.floor((ey - 1) / gridStep))
    const rEnd = Math.min(rows - 1, Math.ceil((ey + 1) / gridStep))

    for (let r = rStart; r <= rEnd; r++) {
      for (let c = cStart; c <= cEnd; c++) {
        const idx = r * cols + c
        if (!blocked[idx]) {
          distances[idx] = 0
          queue.push([0, r, c])
        }
      }
    }
  }

  // Sort initial queue
  queue.sort((a, b) => a[0] - b[0])

  // Process priority queue
  while (queue.length > 0) {
    const current = queue.shift()!
    const [dist, cr, cc] = current

    const currentIdx = cr * cols + cc
    if (dist > distances[currentIdx]!) continue

    for (const [dr, dc, stepCost] of BFS_DIRS) {
      const nr = cr + dr
      const nc = cc + dc
      if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue

      const nIdx = nr * cols + nc
      if (blocked[nIdx]) continue

      const newDist = dist + stepCost * gridStep
      if (newDist < distances[nIdx]!) {
        distances[nIdx] = newDist
        insertSorted(queue, [newDist, nr, nc])
      }
    }
  }

  return { distances, cols, rows }
}

// ---------------------------------------------------------------------------
// Individual energy terms
// ---------------------------------------------------------------------------

/**
 * E_overlap: Penalise overlapping furniture.
 *
 * For each pair of items, computes OBB overlap area using SAT.
 * Energy = sum of (overlapArea)^2.
 */
export function eOverlap(items: FurnitureItem[]): number {
  let energy = 0
  for (let i = 0; i < items.length; i++) {
    for (let j = i + 1; j < items.length; j++) {
      const overlap = computeOBBOverlap(items[i]!, items[j]!)
      energy += overlap * overlap
    }
  }
  return energy
}

/**
 * E_aisle: Penalise insufficient aisle widths per IBC §1029.9.2.
 *
 * Checks minimum gaps between adjacent furniture pairs and applies
 * context-dependent minimum width requirements.
 */
export function eAisle(items: FurnitureItem[]): number {
  let energy = 0

  for (let i = 0; i < items.length; i++) {
    const a = items[i]!
    for (let j = i + 1; j < items.length; j++) {
      const b = items[j]!
      const gap = computeOBBGap(a, b)

      // Skip pairs that are far apart (not adjacent)
      const maxExtent = Math.max(a.width, a.depth, b.width, b.depth)
      if (gap > maxExtent * 2) continue

      // Determine required minimum gap based on item types
      const minGap = getRequiredGap(a, b)
      const deficit = minGap - gap
      if (deficit > 0) {
        energy += deficit * deficit
      }
    }
  }

  return energy
}

/**
 * Determine the minimum required gap between two items based on their types.
 */
function getRequiredGap(a: FurnitureItem, b: FurnitureItem): number {
  const aIsChair = a.itemType === ItemType.Chair
  const bIsChair = b.itemType === ItemType.Chair
  const aIsTable =
    a.itemType === ItemType.RoundTable || a.itemType === ItemType.RectTable
  const bIsTable =
    b.itemType === ItemType.RoundTable || b.itemType === ItemType.RectTable
  const aIsService =
    a.itemType === ItemType.ServiceStation || a.itemType === ItemType.Bar
  const bIsService =
    b.itemType === ItemType.ServiceStation || b.itemType === ItemType.Bar

  // Chair back to table edge
  if ((aIsChair && bIsTable) || (bIsChair && aIsTable)) {
    return CHAIR_TABLE_GAP_MIN
  }

  // Between table rows
  if (aIsTable && bIsTable) {
    return TABLE_ROW_GAP_MIN
  }

  // Service aisles
  if (aIsService || bIsService) {
    return SERVICE_AISLE_MIN
  }

  // Default: main aisle requirement
  return MAIN_AISLE_MIN
}

/**
 * E_egress: Penalise blocked or overly long egress paths per IBC §1017.
 *
 * Discretises the room into a 1-ft grid, marks furniture as obstacles,
 * BFS flood-fills from exits, and penalises seats exceeding 200 ft
 * travel distance or that are completely unreachable.
 */
export function eEgress(items: FurnitureItem[], room: RoomBoundary): number {
  if (room.exits.length < 3) {
    // No exits defined — every seat is unreachable
    const totalSeats = items.reduce((sum, it) => sum + it.seats, 0)
    return totalSeats * UNREACHABLE_PENALTY
  }

  const { distances, cols } = bfsGrid(room, items, GRID_STEP, EGRESS_PATH_WIDTH)

  let energy = 0

  for (const item of items) {
    if (item.itemType !== ItemType.Chair && item.seats <= 0) continue

    // Grid cell of this seat
    const col = Math.floor(item.x / GRID_STEP)
    const row = Math.floor(item.y / GRID_STEP)
    const idx = row * cols + col
    const dist = distances[idx]

    if (dist === undefined || dist === Infinity) {
      energy += UNREACHABLE_PENALTY
    } else if (dist > EGRESS_MAX_DISTANCE) {
      const excess = dist - EGRESS_MAX_DISTANCE
      energy += excess * excess
    }
  }

  return energy
}

/**
 * E_sightline: Penalise obstructed views to focal point.
 *
 * Determines the focal point (center of stage area, or room center).
 * For each seating item, casts a 2D ray to the focal point and checks
 * for intersection with other furniture AABBs.
 */
export function eSightline(
  items: FurnitureItem[],
  room: RoomBoundary,
): number {
  const focal = getFocalPoint(room)
  let energy = 0

  const seatingItems: FurnitureItem[] = []
  const obstacleItems: FurnitureItem[] = []

  for (const item of items) {
    if (item.itemType === ItemType.Chair || item.seats > 0) {
      seatingItems.push(item)
    }
    // All non-chair items can occlude (tables, stages, bars, etc.)
    if (item.itemType !== ItemType.Chair) {
      obstacleItems.push(item)
    }
  }

  for (const seat of seatingItems) {
    let occlusionCount = 0

    for (const obstacle of obstacleItems) {
      // Get AABB of obstacle
      const corners = getOBBCorners(obstacle)
      let minX = Infinity
      let minY = Infinity
      let maxX = -Infinity
      let maxY = -Infinity
      for (const c of corners) {
        if (c[0] < minX) minX = c[0]
        if (c[1] < minY) minY = c[1]
        if (c[0] > maxX) maxX = c[0]
        if (c[1] > maxY) maxY = c[1]
      }

      const t = lineSegmentIntersectsAABB(
        seat.x,
        seat.y,
        focal[0],
        focal[1],
        minX,
        minY,
        maxX,
        maxY,
      )

      if (t >= 0 && t <= 1) {
        occlusionCount++
      }
    }

    // Occlusion ratio = proportion of obstacles blocking sightline (capped at 1)
    const ratio =
      obstacleItems.length > 0
        ? Math.min(1, occlusionCount / Math.max(1, obstacleItems.length))
        : 0
    energy += ratio * ratio
  }

  return energy
}

/**
 * E_capacity: Penalise deviation from target seat count.
 */
function eCapacity(items: FurnitureItem[], targetCapacity: number): number {
  const totalSeats = items.reduce((sum, it) => sum + it.seats, 0)
  const diff = totalSeats - targetCapacity
  return diff * diff
}

/**
 * E_ADA: Penalise lack of wheelchair accessibility.
 *
 * Requirements:
 *  - At least 1 wheelchair space per 25 seats (min 36x48 in clear)
 *  - Accessible path (36" wide) from entrance to wheelchair spaces
 */
export function eAda(items: FurnitureItem[], room: RoomBoundary): number {
  const totalSeats = items.reduce((sum, it) => sum + it.seats, 0)
  const requiredSpaces = Math.max(1, Math.ceil(totalSeats / 25))

  // Find wheelchair-compatible spaces:
  // Look for clear areas of at least WHEELCHAIR_WIDTH x WHEELCHAIR_DEPTH
  // adjacent to seating but not occupied by furniture
  const wheelchairSpaces = findWheelchairSpaces(items, room)
  const actualSpaces = wheelchairSpaces.length

  let energy = 0

  // Penalty for missing spaces
  if (actualSpaces < requiredSpaces) {
    const deficit = requiredSpaces - actualSpaces
    energy += deficit * deficit * ADA_MISSING_SPACE_PENALTY
  }

  // Check accessible path exists (36" wide) from entrance to each space
  if (room.exits.length >= 3 && wheelchairSpaces.length > 0) {
    const { distances, cols } = bfsGrid(
      room,
      items,
      GRID_STEP,
      ACCESSIBLE_PATH_WIDTH,
    )

    for (const space of wheelchairSpaces) {
      const col = Math.floor(space[0] / GRID_STEP)
      const row = Math.floor(space[1] / GRID_STEP)
      const idx = row * cols + col
      const dist = distances[idx]

      if (dist === undefined || dist === Infinity) {
        energy += ADA_MISSING_SPACE_PENALTY
      }
    }
  }

  return energy
}

/**
 * Find potential wheelchair spaces adjacent to seating areas.
 * Looks for clear rectangular areas of sufficient size next to tables.
 */
function findWheelchairSpaces(
  items: FurnitureItem[],
  room: RoomBoundary,
): Vec2[] {
  const spaces: Vec2[] = []

  // Candidate positions: beside each table, check if clear space exists
  const tables = items.filter(
    it =>
      it.itemType === ItemType.RoundTable || it.itemType === ItemType.RectTable,
  )

  for (const table of tables) {
    const cos = Math.cos(table.rotation)
    const sin = Math.sin(table.rotation)

    // Check 4 sides of each table for clear space
    const offsets: Vec2[] = [
      [table.width / 2 + WHEELCHAIR_WIDTH / 2 + 0.5, 0],
      [-(table.width / 2 + WHEELCHAIR_WIDTH / 2 + 0.5), 0],
      [0, table.depth / 2 + WHEELCHAIR_DEPTH / 2 + 0.5],
      [0, -(table.depth / 2 + WHEELCHAIR_DEPTH / 2 + 0.5)],
    ]

    for (const [ox, oy] of offsets) {
      const wx = table.x + ox * cos - oy * sin
      const wy = table.y + ox * sin + oy * cos

      // Check this space is inside the room
      if (
        room.vertices.length >= 6 &&
        !pointInPolygon(wx, wy, room.vertices)
      ) {
        continue
      }

      // Check no other furniture overlaps this space
      const testItem: FurnitureItem = {
        x: wx,
        y: wy,
        width: WHEELCHAIR_WIDTH,
        depth: WHEELCHAIR_DEPTH,
        rotation: table.rotation,
        itemType: ItemType.Chair, // dummy type for overlap check
        seats: 0,
      }

      let isBlocked = false
      for (const other of items) {
        if (computeOBBOverlap(testItem, other) > 0.01) {
          isBlocked = true
          break
        }
      }

      if (!isBlocked) {
        spaces.push([wx, wy])
      }
    }
  }

  return spaces
}

/**
 * E_aesthetic: Penalise visual disorder.
 *
 * Sub-terms:
 *  - Alignment: items not parallel to nearest wall (mod 90 deg)
 *  - Symmetry: deviation from room center axis
 *  - Balance: centroid of items far from room center
 *  - Spacing uniformity: variance of nearest-neighbor table distances
 */
export function eAesthetic(
  items: FurnitureItem[],
  room: RoomBoundary,
): number {
  if (items.length === 0) return 0

  const roomCenterX = room.width / 2
  const roomCenterY = room.height / 2

  // --- Alignment: penalise items not axis-aligned (mod 90 deg) ---
  let alignmentPenalty = 0
  const wallAngles = getRoomWallAngles(room)
  for (const item of items) {
    const itemAngle = normaliseAngle(item.rotation)
    let minAngleDiff = Infinity
    for (const wallAngle of wallAngles) {
      const diff = angleDiffMod90(itemAngle, wallAngle)
      if (diff < minAngleDiff) minAngleDiff = diff
    }
    alignmentPenalty += minAngleDiff * minAngleDiff
  }

  // --- Symmetry: deviation from room center Y-axis ---
  let symmetryPenalty = 0
  for (const item of items) {
    const mirrorX = 2 * roomCenterX - item.x
    let minDist = Infinity
    for (const other of items) {
      if (other === item) continue
      const dx = other.x - mirrorX
      const dy = other.y - item.y
      const dist = Math.sqrt(dx * dx + dy * dy)
      if (dist < minDist) minDist = dist
    }
    if (minDist !== Infinity) {
      symmetryPenalty += minDist * minDist
    }
  }
  // Normalise by item count to avoid scaling with layout size
  if (items.length > 1) {
    symmetryPenalty /= items.length
  }

  // --- Balance: centroid distance from room center ---
  let centroidX = 0
  let centroidY = 0
  for (const item of items) {
    centroidX += item.x
    centroidY += item.y
  }
  centroidX /= items.length
  centroidY /= items.length
  const centroidDx = centroidX - roomCenterX
  const centroidDy = centroidY - roomCenterY
  const balancePenalty = centroidDx * centroidDx + centroidDy * centroidDy

  // --- Spacing uniformity: variance of inter-table distances ---
  let spacingVariance = 0
  const tables = items.filter(
    it =>
      it.itemType === ItemType.RoundTable || it.itemType === ItemType.RectTable,
  )
  if (tables.length >= 2) {
    const nearestDistances: number[] = []
    for (let i = 0; i < tables.length; i++) {
      let nearestDist = Infinity
      for (let j = 0; j < tables.length; j++) {
        if (i === j) continue
        const dx = tables[i]!.x - tables[j]!.x
        const dy = tables[i]!.y - tables[j]!.y
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < nearestDist) nearestDist = dist
      }
      if (nearestDist !== Infinity) {
        nearestDistances.push(nearestDist)
      }
    }

    if (nearestDistances.length >= 2) {
      const mean =
        nearestDistances.reduce((s, d) => s + d, 0) / nearestDistances.length
      spacingVariance =
        nearestDistances.reduce((s, d) => s + (d - mean) * (d - mean), 0) /
        nearestDistances.length
    }
  }

  return alignmentPenalty + symmetryPenalty + balancePenalty + spacingVariance
}

/**
 * E_service: Penalise blocked catering/AV service paths.
 *
 * Checks:
 *  - Clear path (60" wide) from room edges to all tables
 *  - Perimeter path exists for staff circulation
 */
export function eService(items: FurnitureItem[], room: RoomBoundary): number {
  let energy = 0

  const tables = items.filter(
    it =>
      it.itemType === ItemType.RoundTable || it.itemType === ItemType.RectTable,
  )

  if (tables.length === 0) return 0

  // Check perimeter clearance — items too close to walls block service
  for (const item of items) {
    const perimeterDist = distanceToRoomEdge(item.x, item.y, room)
    if (perimeterDist < SERVICE_PATH_WIDTH) {
      const deficit = SERVICE_PATH_WIDTH - perimeterDist
      energy += deficit * deficit
    }
  }

  // Check service accessibility to each table via BFS with service path width
  if (room.exits.length >= 3) {
    const { distances, cols } = bfsGrid(
      room,
      items,
      GRID_STEP,
      SERVICE_PATH_WIDTH,
    )

    for (const table of tables) {
      const col = Math.floor(table.x / GRID_STEP)
      const row = Math.floor(table.y / GRID_STEP)
      const idx = row * cols + col
      const dist = distances[idx]

      if (dist === undefined || dist === Infinity) {
        energy += SERVICE_BLOCKED_PENALTY
      }
    }
  } else {
    // No exits: check if any table is hemmed in by other furniture
    for (const table of tables) {
      let blockedSides = 0
      const checkDist =
        SERVICE_PATH_WIDTH + Math.max(table.width, table.depth) / 2

      for (const other of items) {
        if (other === table) continue
        const dx = other.x - table.x
        const dy = other.y - table.y
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < checkDist) {
          blockedSides++
        }
      }

      // Penalise if surrounded on 3+ sides
      if (blockedSides >= 3) {
        energy += SERVICE_BLOCKED_PENALTY * (blockedSides - 2)
      }
    }
  }

  return energy
}

// ---------------------------------------------------------------------------
// Aesthetic sub-helpers
// ---------------------------------------------------------------------------

/** Get wall angles from room polygon vertices (in radians) */
function getRoomWallAngles(room: RoomBoundary): number[] {
  const angles: number[] = []
  const n = Math.floor(room.vertices.length / 2)
  if (n < 2) return [0] // Default: axis-aligned

  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n
    const dx = room.vertices[j * 2]! - room.vertices[i * 2]!
    const dy = room.vertices[j * 2 + 1]! - room.vertices[i * 2 + 1]!
    angles.push(Math.atan2(dy, dx))
  }

  return angles
}

/** Normalise angle to [0, 2*PI) */
function normaliseAngle(a: number): number {
  const TWO_PI = 2 * Math.PI
  let result = a % TWO_PI
  if (result < 0) result += TWO_PI
  return result
}

/** Minimum angular difference modulo 90 degrees (in radians) */
function angleDiffMod90(a: number, b: number): number {
  const HALF_PI = Math.PI / 2
  let diff = Math.abs(a - b) % HALF_PI
  if (diff > HALF_PI / 2) diff = HALF_PI - diff
  return diff
}

/** Minimum distance from a point to the room boundary edges */
function distanceToRoomEdge(
  px: number,
  py: number,
  room: RoomBoundary,
): number {
  const n = Math.floor(room.vertices.length / 2)
  if (n < 2) {
    // Fallback: rectangular room — use min distance to 4 edges
    return Math.min(px, py, room.width - px, room.height - py)
  }

  let minDist = Infinity
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n
    const ax = room.vertices[i * 2]!
    const ay = room.vertices[i * 2 + 1]!
    const bx = room.vertices[j * 2]!
    const by = room.vertices[j * 2 + 1]!

    const dist = pointToSegmentDistance(px, py, ax, ay, bx, by)
    if (dist < minDist) minDist = dist
  }

  return minDist
}

/** Distance from point (px,py) to line segment (ax,ay)-(bx,by) */
function pointToSegmentDistance(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): number {
  const abx = bx - ax
  const aby = by - ay
  const apx = px - ax
  const apy = py - ay

  const ab2 = abx * abx + aby * aby
  if (ab2 < 1e-12) {
    // Degenerate segment (point)
    return Math.sqrt(apx * apx + apy * apy)
  }

  let t = (apx * abx + apy * aby) / ab2
  t = Math.max(0, Math.min(1, t))

  const closestX = ax + t * abx
  const closestY = ay + t * aby
  const dx = px - closestX
  const dy = py - closestY
  return Math.sqrt(dx * dx + dy * dy)
}

/** Get the focal point of the room (stage center, or room center) */
function getFocalPoint(room: RoomBoundary): Vec2 {
  if (room.stageArea && room.stageArea.length >= 6) {
    // Compute centroid of stage polygon
    const n = Math.floor(room.stageArea.length / 2)
    let cx = 0
    let cy = 0
    for (let i = 0; i < n; i++) {
      cx += room.stageArea[i * 2]!
      cy += room.stageArea[i * 2 + 1]!
    }
    return [cx / n, cy / n]
  }

  // Default: room center
  return [room.width / 2, room.height / 2]
}

// ---------------------------------------------------------------------------
// Main energy function
// ---------------------------------------------------------------------------

/**
 * Compute the total layout energy as a weighted sum of 8 sub-terms.
 *
 * Lower energy = better layout. A layout at energy 0 would perfectly
 * satisfy all code, accessibility, aesthetic, and operational constraints.
 *
 * @param items          Furniture items in the layout
 * @param room           Room boundary with exits, dimensions, optional stage
 * @param weights        Per-term weight multipliers
 * @param targetCapacity Desired number of seats
 * @returns Total weighted energy (non-negative)
 */
export function computeLayoutEnergy(
  items: FurnitureItem[],
  room: RoomBoundary,
  weights: LayoutWeights,
  targetCapacity: number,
): number {
  const overlap = weights.overlap * eOverlap(items)
  const aisle = weights.aisle * eAisle(items)
  const egress = weights.egress * eEgress(items, room)
  const sightline = weights.sightline * eSightline(items, room)
  const capacity = weights.capacity * eCapacity(items, targetCapacity)
  const ada = weights.ada * eAda(items, room)
  const aesthetic = weights.aesthetic * eAesthetic(items, room)
  const service = weights.service * eService(items, room)

  return (
    overlap + aisle + egress + sightline + capacity + ada + aesthetic + service
  )
}

// ---------------------------------------------------------------------------
// Neighborhood operator for SA/MCMC
// ---------------------------------------------------------------------------

/**
 * Generate a neighboring layout by applying one of 6 random operators:
 *
 *   1. moveItem      — translate a random item by a small delta
 *   2. rotateItem    — rotate a random item by a small angle
 *   3. swapItems     — swap positions of two random items
 *   4. shiftRow      — shift an entire row of aligned items
 *   5. adjustSpacing — move two adjacent items closer/further apart
 *   6. mirrorItem    — mirror an item about the room center axis
 *
 * Returns a new array (the original is not mutated).
 */
export function generateLayoutNeighbor(
  items: FurnitureItem[],
  rng: PRNG,
): FurnitureItem[] {
  if (items.length === 0) return []

  // Deep copy items
  const result: FurnitureItem[] = items.map(it => ({ ...it }))
  const n = result.length

  // Pick a random operator (0-5)
  const op = Math.floor(rng.random() * 6)

  switch (op) {
    case 0: {
      // moveItem: translate a random item by a small delta
      const idx = Math.floor(rng.random() * n)
      const item = result[idx]!
      const moveScale = Math.max(item.width, item.depth) * 0.5
      item.x += (rng.random() - 0.5) * 2 * moveScale
      item.y += (rng.random() - 0.5) * 2 * moveScale
      break
    }

    case 1: {
      // rotateItem: rotate a random item by +/-15 deg or snap to 90 deg
      const idx = Math.floor(rng.random() * n)
      const item = result[idx]!
      if (rng.random() < 0.3) {
        // Snap to nearest 90 deg increment
        item.rotation =
          Math.round(item.rotation / (Math.PI / 2)) * (Math.PI / 2)
      } else {
        // Small random rotation (+/-15 deg)
        const maxAngle = Math.PI / 12
        item.rotation += (rng.random() - 0.5) * 2 * maxAngle
      }
      break
    }

    case 2: {
      // swapItems: swap positions of two random items
      if (n < 2) break
      const i = Math.floor(rng.random() * n)
      let j = Math.floor(rng.random() * (n - 1))
      if (j >= i) j++

      const itemI = result[i]!
      const itemJ = result[j]!
      const tmpX = itemI.x
      const tmpY = itemI.y
      itemI.x = itemJ.x
      itemI.y = itemJ.y
      itemJ.x = tmpX
      itemJ.y = tmpY
      break
    }

    case 3: {
      // shiftRow: find items approximately on the same Y, shift all by dx
      const pivot = result[Math.floor(rng.random() * n)]!
      const rowThreshold = Math.max(pivot.depth, 2.0)
      const shiftDx = (rng.random() - 0.5) * 4 // shift up to +/-2 ft

      for (const item of result) {
        if (Math.abs(item.y - pivot.y) < rowThreshold) {
          item.x += shiftDx
        }
      }
      break
    }

    case 4: {
      // adjustSpacing: pick two nearest items and push/pull them
      if (n < 2) break
      const baseIdx = Math.floor(rng.random() * n)
      const base = result[baseIdx]!

      let nearestIdx = -1
      let nearestDist = Infinity
      for (let k = 0; k < n; k++) {
        if (k === baseIdx) continue
        const other = result[k]!
        const dx = other.x - base.x
        const dy = other.y - base.y
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < nearestDist) {
          nearestDist = dist
          nearestIdx = k
        }
      }

      if (nearestIdx >= 0 && nearestDist > 1e-6) {
        const nearest = result[nearestIdx]!
        const dx = nearest.x - base.x
        const dy = nearest.y - base.y
        const dirX = dx / nearestDist
        const dirY = dy / nearestDist
        // Adjust by +/-1 ft along the axis between them
        const adjustment = (rng.random() - 0.5) * 2
        nearest.x += dirX * adjustment
        nearest.y += dirY * adjustment
      }
      break
    }

    case 5: {
      // mirrorItem: mirror a random item about the layout centroid X-axis
      const idx = Math.floor(rng.random() * n)
      const item = result[idx]!
      // Use centroid of all items as the mirror axis (no room info needed)
      let centroidX = 0
      for (const it of result) {
        centroidX += it.x
      }
      centroidX /= n
      item.x = 2 * centroidX - item.x
      item.rotation = Math.PI - item.rotation
      break
    }

    default:
      break
  }

  return result
}
