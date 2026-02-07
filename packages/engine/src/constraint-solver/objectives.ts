/**
 * Soft objective scoring functions for the layout solver.
 *
 * Each function returns a 0-1 score (higher is better).
 * The solver uses weighted combination of these to guide simulated annealing.
 */

import type { RoomConfig, Placement, FurnitureSpec, LayoutScores, ObjectiveWeights, Point2D } from './types'

// ─── Default Weights ────────────────────────────────────────────────────────

export const DEFAULT_WEIGHTS: ObjectiveWeights = {
  spaceUtilization: 0.3,
  sightlineCoverage: 0.3,
  symmetry: 0.2,
  exitAccess: 0.2,
}

// ─── Individual Objectives ──────────────────────────────────────────────────

/**
 * Capacity utilization: fraction of requested items that were placed.
 */
export function scoreCapacity(specs: FurnitureSpec[], placements: Placement[]): number {
  const requested = specs.reduce((sum, s) => sum + s.count, 0)
  if (requested === 0) return 1
  return Math.min(1, placements.length / requested)
}

/**
 * Space utilization: fraction of room floor area covered by furniture.
 * Normalized to a target of ~40-60% coverage (too sparse or too dense is bad).
 */
export function scoreSpaceUtilization(room: RoomConfig, placements: Placement[]): number {
  const roomArea = room.width * room.depth
  if (roomArea === 0) return 0

  let furnitureArea = 0
  for (const p of placements) {
    furnitureArea += p.effectiveWidth * p.effectiveDepth
  }

  const coverage = furnitureArea / roomArea
  // Ideal coverage: 30-50%. Score peaks at 40%.
  const ideal = 0.4
  const score = 1 - Math.min(1, Math.abs(coverage - ideal) / ideal)
  return Math.max(0, score)
}

/**
 * Sightline coverage: fraction of seats/chairs with unobstructed line-of-sight
 * to the focal point. Uses simplified 2D raycasting.
 */
export function scoreSightlines(room: RoomConfig, placements: Placement[]): number {
  if (!room.focalPoint) return 1 // no focal point = full score

  const focal = room.focalPoint
  const chairs = placements.filter((p) => p.type === 'chair')
  if (chairs.length === 0) return 1

  // Build AABB list for non-chair items (obstacles for sightlines)
  const obstacles: Array<{ minX: number; minZ: number; maxX: number; maxZ: number }> = []
  for (const p of placements) {
    if (p.type === 'chair') continue
    const hw = p.effectiveWidth / 2
    const hd = p.effectiveDepth / 2
    obstacles.push({ minX: p.x - hw, minZ: p.z - hd, maxX: p.x + hw, maxZ: p.z + hd })
  }

  let clearCount = 0
  for (const chair of chairs) {
    if (hasLineOfSight(chair.x, chair.z, focal.x, focal.z, obstacles)) {
      clearCount++
    }
  }

  return clearCount / chairs.length
}

/** 2D ray-AABB intersection test. */
function hasLineOfSight(
  x1: number, z1: number,
  x2: number, z2: number,
  obstacles: Array<{ minX: number; minZ: number; maxX: number; maxZ: number }>,
): boolean {
  for (const obs of obstacles) {
    if (rayIntersectsAABB(x1, z1, x2, z2, obs)) return false
  }
  return true
}

/** Check if a 2D ray segment from (x1,z1) to (x2,z2) intersects an AABB. */
function rayIntersectsAABB(
  x1: number, z1: number, x2: number, z2: number,
  aabb: { minX: number; minZ: number; maxX: number; maxZ: number },
): boolean {
  const dx = x2 - x1
  const dz = z2 - z1

  let tmin = 0
  let tmax = 1

  if (Math.abs(dx) > 1e-10) {
    const tx1 = (aabb.minX - x1) / dx
    const tx2 = (aabb.maxX - x1) / dx
    tmin = Math.max(tmin, Math.min(tx1, tx2))
    tmax = Math.min(tmax, Math.max(tx1, tx2))
  } else {
    if (x1 < aabb.minX || x1 > aabb.maxX) return false
  }

  if (Math.abs(dz) > 1e-10) {
    const tz1 = (aabb.minZ - z1) / dz
    const tz2 = (aabb.maxZ - z1) / dz
    tmin = Math.max(tmin, Math.min(tz1, tz2))
    tmax = Math.min(tmax, Math.max(tz1, tz2))
  } else {
    if (z1 < aabb.minZ || z1 > aabb.maxZ) return false
  }

  return tmin <= tmax
}

/**
 * Symmetry score: measures mirror symmetry along the room's primary axis.
 * Higher score = more symmetric layout.
 */
export function scoreSymmetry(room: RoomConfig, placements: Placement[]): number {
  if (placements.length <= 1) return 1

  const centerX = room.width / 2

  let totalDeviation = 0
  let matchCount = 0

  for (const p of placements) {
    // For each item on one side, find the closest mirror match
    const mirrorX = 2 * centerX - p.x
    let bestDist = Infinity

    for (const other of placements) {
      if (other === p) continue
      const dx = other.x - mirrorX
      const dz = other.z - p.z
      const dist = Math.sqrt(dx * dx + dz * dz)
      if (other.type === p.type && dist < bestDist) {
        bestDist = dist
      }
    }

    if (bestDist < room.width * 0.1) {
      // Found a reasonable mirror match
      matchCount++
      totalDeviation += bestDist
    }
  }

  if (matchCount === 0) return 0

  const matchRatio = matchCount / placements.length
  const avgDeviation = totalDeviation / matchCount
  const deviationScore = Math.max(0, 1 - avgDeviation / (room.width * 0.1))

  return matchRatio * deviationScore
}

/**
 * Exit accessibility: inverse of max distance from any seat to nearest exit.
 * Lower max distance = higher score.
 */
export function scoreExitAccess(room: RoomConfig, placements: Placement[]): number {
  if (room.exits.length === 0 || placements.length === 0) return 0

  const chairs = placements.filter((p) => p.type === 'chair')
  const items = chairs.length > 0 ? chairs : placements

  let maxMinDist = 0
  for (const p of items) {
    let minDist = Infinity
    for (const exit of room.exits) {
      const dx = p.x - exit.position.x
      const dz = p.z - exit.position.z
      const dist = Math.sqrt(dx * dx + dz * dz)
      if (dist < minDist) minDist = dist
    }
    if (minDist > maxMinDist) maxMinDist = minDist
  }

  // Normalize: assume max reasonable distance is the room diagonal
  const roomDiag = Math.sqrt(room.width * room.width + room.depth * room.depth)
  if (roomDiag === 0) return 0

  return Math.max(0, 1 - maxMinDist / roomDiag)
}

// ─── Combined Scoring ───────────────────────────────────────────────────────

/**
 * Compute all scores for a layout.
 */
export function scoreLayout(
  room: RoomConfig,
  specs: FurnitureSpec[],
  placements: Placement[],
  weights: ObjectiveWeights = DEFAULT_WEIGHTS,
): LayoutScores {
  const capacity = scoreCapacity(specs, placements)
  const space = scoreSpaceUtilization(room, placements)
  const sightlines = scoreSightlines(room, placements)
  const symmetry = scoreSymmetry(room, placements)
  const exitAccess = scoreExitAccess(room, placements)

  const total =
    weights.spaceUtilization * space +
    weights.sightlineCoverage * sightlines +
    weights.symmetry * symmetry +
    weights.exitAccess * exitAccess

  return {
    capacityUtilization: capacity,
    spaceUtilization: space,
    sightlineCoverage: sightlines,
    symmetry,
    exitAccess,
    total,
  }
}
