/**
 * Hard constraint validators for the layout solver.
 *
 * Each checker takes the current layout and returns violations.
 * All hard constraints must be satisfied for a layout to be feasible.
 *
 * Spatial-accelerated variants (checkNoOverlapSpatial, checkAisleWidthSpatial,
 * validateSinglePlacement) reduce O(n²) to O(n·k) using the SolverSpatialHash.
 */

import type { RoomConfig, Placement, Violation, Rect, Point2D, ViolationType } from './types'
import { assertNeverViolation } from './types'
import type { SolverSpatialHash } from './spatial-hash'

// ─── Geometry Helpers ───────────────────────────────────────────────────────

/** AABB for a placement. */
interface AABB {
  minX: number
  minZ: number
  maxX: number
  maxZ: number
}

function placementAABB(p: Placement): AABB {
  const hw = p.effectiveWidth / 2
  const hd = p.effectiveDepth / 2
  return { minX: p.x - hw, minZ: p.z - hd, maxX: p.x + hw, maxZ: p.z + hd }
}

function aabbOverlap(a: AABB, b: AABB): boolean {
  return a.minX < b.maxX && a.maxX > b.minX && a.minZ < b.maxZ && a.maxZ > b.minZ
}

function rectToAABB(r: Rect): AABB {
  return { minX: r.x, minZ: r.z, maxX: r.x + r.width, maxZ: r.z + r.depth }
}

/** Compute effective gap between two non-overlapping AABBs. */
function aabbGap(a: AABB, b: AABB): number {
  // Gap on X axis
  const gapX = Math.max(0, Math.max(b.minX - a.maxX, a.minX - b.maxX))
  // Gap on Z axis
  const gapZ = Math.max(0, Math.max(b.minZ - a.maxZ, a.minZ - b.maxZ))

  // If they overlap on one axis, the gap is on the other axis
  const overlapX = a.maxX > b.minX && b.maxX > a.minX
  const overlapZ = a.maxZ > b.minZ && b.maxZ > a.minZ

  if (overlapX && !overlapZ) return gapZ
  if (overlapZ && !overlapX) return gapX
  // Diagonal separation — use maximum of axis gaps
  return Math.max(gapX, gapZ)
}

// ─── Hard Constraints (O(n²) brute-force — backward-compatible) ────────────

/** Check that no two placements overlap. O(n²). */
export function checkNoOverlap(placements: Placement[]): Violation[] {
  const violations: Violation[] = []
  for (let i = 0; i < placements.length; i++) {
    const a = placementAABB(placements[i]!)
    for (let j = i + 1; j < placements.length; j++) {
      const b = placementAABB(placements[j]!)
      if (aabbOverlap(a, b)) {
        violations.push({
          type: 'overlap',
          message: `Items ${i} and ${j} overlap`,
          placements: [i, j],
        })
      }
    }
  }
  return violations
}

/** Check that all placements are within room boundaries. */
export function checkBounds(room: RoomConfig, placements: Placement[]): Violation[] {
  const violations: Violation[] = []
  for (let i = 0; i < placements.length; i++) {
    const p = placements[i]!
    const hw = p.effectiveWidth / 2
    const hd = p.effectiveDepth / 2
    if (p.x - hw < 0 || p.x + hw > room.width || p.z - hd < 0 || p.z + hd > room.depth) {
      violations.push({
        type: 'out-of-bounds',
        message: `Item ${i} extends outside room boundaries`,
        placements: [i],
      })
    }
  }
  return violations
}

/** Check that no placement overlaps an obstacle. */
export function checkObstacles(room: RoomConfig, placements: Placement[]): Violation[] {
  const violations: Violation[] = []
  for (let i = 0; i < placements.length; i++) {
    const a = placementAABB(placements[i]!)
    for (const obs of room.obstacles) {
      const b = rectToAABB(obs)
      if (aabbOverlap(a, b)) {
        violations.push({
          type: 'obstacle-overlap',
          message: `Item ${i} overlaps an obstacle`,
          placements: [i],
        })
        break
      }
    }
  }
  return violations
}

/** Check that exits are not blocked by furniture. */
export function checkExitClearance(room: RoomConfig, placements: Placement[], clearance: number): Violation[] {
  const violations: Violation[] = []
  for (const exit of room.exits) {
    const exitAABB: AABB = {
      minX: exit.position.x - exit.width / 2 - clearance,
      minZ: exit.position.z - exit.width / 2 - clearance,
      maxX: exit.position.x + exit.width / 2 + clearance,
      maxZ: exit.position.z + exit.width / 2 + clearance,
    }
    for (let i = 0; i < placements.length; i++) {
      const pAABB = placementAABB(placements[i]!)
      if (aabbOverlap(pAABB, exitAABB)) {
        violations.push({
          type: 'exit-blocked',
          message: `Item ${i} blocks an exit (${clearance}m clearance required)`,
          placements: [i],
        })
      }
    }
  }
  return violations
}

/**
 * Check minimum aisle width between adjacent placements. O(n²).
 * Tests that there's at least `minWidth` gap between closest edges
 * of any two nearby placements.
 */
export function checkAisleWidth(placements: Placement[], minWidth: number): Violation[] {
  const violations: Violation[] = []
  for (let i = 0; i < placements.length; i++) {
    const a = placementAABB(placements[i]!)
    for (let j = i + 1; j < placements.length; j++) {
      const b = placementAABB(placements[j]!)

      // Compute minimum gap between edges (only if not overlapping)
      if (aabbOverlap(a, b)) continue // overlap is a separate violation

      const effectiveGap = aabbGap(a, b)

      if (effectiveGap > 0 && effectiveGap < minWidth) {
        violations.push({
          type: 'aisle-too-narrow',
          message: `Gap between items ${i} and ${j} is ${effectiveGap.toFixed(2)}m (min: ${minWidth}m)`,
          placements: [i, j],
        })
      }
    }
  }
  return violations
}

// ─── Spatial-Accelerated Constraints (O(n·k)) ─────────────────────────────

/**
 * Check that no two placements overlap using spatial hash. O(n·k).
 * Equivalent to checkNoOverlap but only checks neighbors.
 */
export function checkNoOverlapSpatial(
  placements: Placement[],
  spatialHash: SolverSpatialHash,
): Violation[] {
  const violations: Violation[] = []
  const seen = new Set<string>()

  for (let i = 0; i < placements.length; i++) {
    const p = placements[i]!
    const a = placementAABB(p)
    const neighbors = spatialHash.queryAABB(a.minX, a.minZ, a.maxX, a.maxZ)

    for (const j of neighbors) {
      if (j <= i) continue // avoid duplicates
      const key = `${i}:${j}`
      if (seen.has(key)) continue
      seen.add(key)

      const b = placementAABB(placements[j]!)
      if (aabbOverlap(a, b)) {
        violations.push({
          type: 'overlap',
          message: `Items ${i} and ${j} overlap`,
          placements: [i, j],
        })
      }
    }
  }
  return violations
}

/**
 * Check minimum aisle width using spatial hash. O(n·k).
 * Expands each AABB by minWidth to find potentially-too-close neighbors.
 */
export function checkAisleWidthSpatial(
  placements: Placement[],
  spatialHash: SolverSpatialHash,
  minWidth: number,
): Violation[] {
  const violations: Violation[] = []
  const seen = new Set<string>()

  for (let i = 0; i < placements.length; i++) {
    const p = placements[i]!
    const a = placementAABB(p)

    // Expand AABB by minWidth to catch neighbors within aisle distance
    const neighbors = spatialHash.queryAABB(
      a.minX - minWidth,
      a.minZ - minWidth,
      a.maxX + minWidth,
      a.maxZ + minWidth,
    )

    for (const j of neighbors) {
      if (j <= i) continue
      const key = `${i}:${j}`
      if (seen.has(key)) continue
      seen.add(key)

      const b = placementAABB(placements[j]!)
      if (aabbOverlap(a, b)) continue // overlap handled separately

      const effectiveGap = aabbGap(a, b)
      if (effectiveGap > 0 && effectiveGap < minWidth) {
        violations.push({
          type: 'aisle-too-narrow',
          message: `Gap between items ${i} and ${j} is ${effectiveGap.toFixed(2)}m (min: ${minWidth}m)`,
          placements: [i, j],
        })
      }
    }
  }
  return violations
}

/**
 * Validate a SINGLE placement against its spatial neighbors + room constraints.
 * Returns all violations involving placement at `index`.
 *
 * This is the Jane Street incremental computation primitive — instead of
 * revalidating the entire layout O(n²), we only check the neighbors O(k)
 * of the moved item. The IncrementalConstraintGraph calls this per-node.
 */
export function validateSinglePlacement(
  room: RoomConfig,
  placements: Placement[],
  index: number,
  spatialHash: SolverSpatialHash,
  minAisle: number,
  exitClearance: number,
): Violation[] {
  const violations: Violation[] = []
  const p = placements[index]!
  const a = placementAABB(p)
  const hw = p.effectiveWidth / 2
  const hd = p.effectiveDepth / 2

  // 1. Bounds check
  if (p.x - hw < 0 || p.x + hw > room.width || p.z - hd < 0 || p.z + hd > room.depth) {
    violations.push({
      type: 'out-of-bounds',
      message: `Item ${index} extends outside room boundaries`,
      placements: [index],
    })
  }

  // 2. Obstacle check
  for (const obs of room.obstacles) {
    const b = rectToAABB(obs)
    if (aabbOverlap(a, b)) {
      violations.push({
        type: 'obstacle-overlap',
        message: `Item ${index} overlaps an obstacle`,
        placements: [index],
      })
      break
    }
  }

  // 3. Exit clearance check
  for (const exit of room.exits) {
    const exitAABB: AABB = {
      minX: exit.position.x - exit.width / 2 - exitClearance,
      minZ: exit.position.z - exit.width / 2 - exitClearance,
      maxX: exit.position.x + exit.width / 2 + exitClearance,
      maxZ: exit.position.z + exit.width / 2 + exitClearance,
    }
    if (aabbOverlap(a, exitAABB)) {
      violations.push({
        type: 'exit-blocked',
        message: `Item ${index} blocks an exit (${exitClearance}m clearance required)`,
        placements: [index],
      })
    }
  }

  // 4. Spatial-hash-accelerated overlap check against neighbors
  const expand = minAisle + 1 // expand query to catch both overlap and aisle violations
  const neighbors = spatialHash.queryAABB(
    a.minX - expand, a.minZ - expand,
    a.maxX + expand, a.maxZ + expand,
  )

  for (const j of neighbors) {
    if (j === index || j >= placements.length) continue
    const b = placementAABB(placements[j]!)

    // Overlap check
    if (aabbOverlap(a, b)) {
      violations.push({
        type: 'overlap',
        message: `Items ${index} and ${j} overlap`,
        placements: [index, j],
      })
      continue
    }

    // Aisle width check
    const effectiveGap = aabbGap(a, b)
    if (effectiveGap > 0 && effectiveGap < minAisle) {
      violations.push({
        type: 'aisle-too-narrow',
        message: `Gap between items ${index} and ${j} is ${effectiveGap.toFixed(2)}m (min: ${minAisle}m)`,
        placements: [index, j],
      })
    }
  }

  return violations
}

// ─── Violation Severity ────────────────────────────────────────────────────

/**
 * Severity score for each violation type (higher = more severe).
 * Uses exhaustive `never` check — adding a new ViolationType without
 * handling it here is a compile error.
 */
export function violationSeverity(type: ViolationType): number {
  switch (type) {
    case 'overlap': return 10
    case 'out-of-bounds': return 8
    case 'exit-blocked': return 9
    case 'obstacle-overlap': return 7
    case 'aisle-too-narrow': return 5
    default: return assertNeverViolation(type)
  }
}

// ─── Validate All ───────────────────────────────────────────────────────────

/** Run all hard constraint checks (brute-force O(n²)). */
export function validateLayout(
  room: RoomConfig,
  placements: Placement[],
  minAisle = 0.914,
  exitClearance = 1.12,
): Violation[] {
  return [
    ...checkNoOverlap(placements),
    ...checkBounds(room, placements),
    ...checkObstacles(room, placements),
    ...checkExitClearance(room, placements, exitClearance),
    ...checkAisleWidth(placements, minAisle),
  ]
}

/**
 * Run all hard constraint checks using spatial acceleration. O(n·k).
 * Equivalent to validateLayout but faster for large layouts.
 */
export function validateLayoutSpatial(
  room: RoomConfig,
  placements: Placement[],
  spatialHash: SolverSpatialHash,
  minAisle = 0.914,
  exitClearance = 1.12,
): Violation[] {
  return [
    ...checkNoOverlapSpatial(placements, spatialHash),
    ...checkBounds(room, placements),
    ...checkObstacles(room, placements),
    ...checkExitClearance(room, placements, exitClearance),
    ...checkAisleWidthSpatial(placements, spatialHash, minAisle),
  ]
}
