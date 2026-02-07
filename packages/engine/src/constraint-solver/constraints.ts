/**
 * Hard constraint validators for the layout solver.
 *
 * Each checker takes the current layout and returns violations.
 * All hard constraints must be satisfied for a layout to be feasible.
 */

import type { RoomConfig, Placement, Violation, Rect, Exit, Point2D } from './types'

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

function distToPoint(px: number, pz: number, target: Point2D): number {
  const dx = px - target.x
  const dz = pz - target.z
  return Math.sqrt(dx * dx + dz * dz)
}

// ─── Hard Constraints ───────────────────────────────────────────────────────

/** Check that no two placements overlap. */
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
 * Check minimum aisle width between adjacent placements.
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

      // Gap on X axis
      const gapX = Math.max(0, Math.max(b.minX - a.maxX, a.minX - b.maxX))
      // Gap on Z axis
      const gapZ = Math.max(0, Math.max(b.minZ - a.maxZ, a.minZ - b.maxZ))

      // If they overlap on one axis, the gap is on the other axis
      const overlapX = a.maxX > b.minX && b.maxX > a.minX
      const overlapZ = a.maxZ > b.minZ && b.maxZ > a.minZ

      let effectiveGap: number
      if (overlapX && !overlapZ) {
        effectiveGap = gapZ
      } else if (overlapZ && !overlapX) {
        effectiveGap = gapX
      } else {
        // Diagonal separation — use minimum of axis gaps
        effectiveGap = Math.max(gapX, gapZ)
      }

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

// ─── Validate All ───────────────────────────────────────────────────────────

/** Run all hard constraint checks. */
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
