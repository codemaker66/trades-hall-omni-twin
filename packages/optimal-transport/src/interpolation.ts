/**
 * Displacement Interpolation (OT-5).
 *
 * McCann's displacement interpolation creates the "optimal path" between two
 * distributions. For venue layouts: given Layout A and Layout B, compute the
 * smoothest possible transition — furniture slides along optimal paths rather
 * than teleporting.
 */

import type { FurniturePosition, SinkhornConfig } from './types'
import { sinkhorn } from './sinkhorn'

/**
 * Extract the dominant assignment from a transport plan.
 * Each source item maps to the target item with highest transport mass.
 *
 * @returns Array of [sourceIndex, targetIndex] pairs
 */
export function extractAssignment(
  plan: Float64Array,
  N: number,
  M: number,
): Array<[number, number]> {
  const assignments: Array<[number, number]> = []
  for (let i = 0; i < N; i++) {
    let bestJ = 0
    let bestVal = -1
    for (let j = 0; j < M; j++) {
      const val = plan[i * M + j]!
      if (val > bestVal) {
        bestVal = val
        bestJ = j
      }
    }
    assignments.push([i, bestJ])
  }
  return assignments
}

/**
 * Linear interpolation between two angles, taking the shortest path.
 */
function lerpAngle(a: number, b: number, t: number): number {
  let diff = b - a
  // Normalize to [-π, π]
  while (diff > Math.PI) diff -= 2 * Math.PI
  while (diff < -Math.PI) diff += 2 * Math.PI
  return a + t * diff
}

/**
 * Displacement interpolation between two layouts.
 *
 * Given transport plan T from layout A to layout B:
 * At time t ∈ [0, 1], each furniture piece is at:
 *   pos(t) = (1-t) · posA + t · posB
 *   rot(t) = slerp(rotA, rotB, t)
 *
 * Unmatched pieces (from partial OT) fade in/out with opacity.
 *
 * @param layoutA - Source layout positions
 * @param layoutB - Target layout positions
 * @param transportPlan - OT plan mapping A → B (N×M)
 * @param t - Interpolation parameter [0, 1]
 */
export function displacementInterpolation(
  layoutA: FurniturePosition[],
  layoutB: FurniturePosition[],
  transportPlan: Float64Array,
  t: number,
): FurniturePosition[] {
  const N = layoutA.length
  const M = layoutB.length
  const tClamped = Math.max(0, Math.min(1, t))

  // Extract dominant assignment
  const assignments = extractAssignment(transportPlan, N, M)

  // Track which B items are matched
  const matchedB = new Set<number>()

  const result: FurniturePosition[] = []

  // Interpolate matched pairs
  for (const [iA, iB] of assignments) {
    const a = layoutA[iA]!
    const b = layoutB[iB]!
    matchedB.add(iB)

    // Check if this is a meaningful match (transport mass > threshold)
    const mass = transportPlan[iA * M + iB]!
    if (mass < 1e-6) {
      // Fade out from A
      result.push({
        id: a.id,
        x: a.x,
        z: a.z,
        rotation: a.rotation,
        type: a.type,
        opacity: 1 - tClamped,
      })
      continue
    }

    result.push({
      id: a.id,
      x: (1 - tClamped) * a.x + tClamped * b.x,
      z: (1 - tClamped) * a.z + tClamped * b.z,
      rotation: lerpAngle(a.rotation, b.rotation, tClamped),
      type: tClamped < 0.5 ? a.type : b.type,
      opacity: 1,
    })
  }

  // Fade in unmatched B items
  for (let j = 0; j < M; j++) {
    if (!matchedB.has(j)) {
      const b = layoutB[j]!
      result.push({
        id: b.id,
        x: b.x,
        z: b.z,
        rotation: b.rotation,
        type: b.type,
        opacity: tClamped,
      })
    }
  }

  return result
}

/**
 * Build a positional cost matrix from two layouts.
 * Cost = squared Euclidean distance between positions.
 */
export function buildPositionCostMatrix(
  layoutA: FurniturePosition[],
  layoutB: FurniturePosition[],
): Float64Array {
  const N = layoutA.length
  const M = layoutB.length
  const C = new Float64Array(N * M)
  for (let i = 0; i < N; i++) {
    const a = layoutA[i]!
    for (let j = 0; j < M; j++) {
      const b = layoutB[j]!
      const dx = a.x - b.x
      const dz = a.z - b.z
      C[i * M + j] = dx * dx + dz * dz
    }
  }
  return C
}

/**
 * Generate keyframes for a smooth layout transition animation.
 * Returns positions at N evenly spaced time steps.
 *
 * @param layoutA - Source layout
 * @param layoutB - Target layout
 * @param steps - Number of frames (default 60 = 1 second at 60fps)
 * @param sinkhornConfig - Optional Sinkhorn configuration
 */
export function generateTransitionKeyframes(
  layoutA: FurniturePosition[],
  layoutB: FurniturePosition[],
  steps: number = 60,
  sinkhornConfig: Partial<SinkhornConfig> = {},
): FurniturePosition[][] {
  const N = layoutA.length
  const M = layoutB.length

  if (N === 0 || M === 0) {
    return Array.from({ length: steps + 1 }, () => [])
  }

  // Build cost matrix from positions
  const C = buildPositionCostMatrix(layoutA, layoutB)

  // Build uniform distributions (each piece has equal weight)
  const a = new Float64Array(N).fill(1 / N)
  const b = new Float64Array(M).fill(1 / M)

  // Solve OT for optimal assignment
  const result = sinkhorn(a, b, C, { epsilon: 0.01, ...sinkhornConfig })

  // Generate interpolated layouts at t = 0, 1/steps, ..., 1
  const keyframes: FurniturePosition[][] = []
  for (let step = 0; step <= steps; step++) {
    const t = step / steps
    keyframes.push(displacementInterpolation(layoutA, layoutB, result.plan, t))
  }

  return keyframes
}
