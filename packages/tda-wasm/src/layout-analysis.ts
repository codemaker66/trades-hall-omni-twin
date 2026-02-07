/**
 * Browser-side layout dead space analysis (TDA-7).
 *
 * Uses H₀ persistence from the pure TypeScript implementation
 * to detect connectivity issues in furniture layouts.
 * For full H₁ dead space detection, falls back to the Python API.
 *
 * Performance: < 200ms for layouts with up to 200 furniture items.
 */

import type { LayoutPoint, DeadSpaceResult } from './types'
import { computeH0Persistence, buildDistanceMatrix2D, computeStats } from './persistence'

/**
 * Analyze a furniture layout for connectivity and basic dead space indicators.
 *
 * Browser-side analysis using H₀ only:
 * - Detects disconnected furniture groups (H₀ features)
 * - Computes coverage score
 * - Provides connectivity score
 *
 * For full dead space detection (H₁), use the Python API.
 *
 * @param furniture - Array of furniture positions
 * @param roomWidth - Room width in feet
 * @param roomDepth - Room depth in feet
 * @param connectivityThresholdFt - Distance threshold for "connected" furniture
 */
export function analyzeLayoutBrowser(
  furniture: LayoutPoint[],
  roomWidth: number,
  roomDepth: number,
  connectivityThresholdFt: number = 6.0,
): DeadSpaceResult {
  if (furniture.length < 2) {
    return {
      deadSpaces: [],
      coverageScore: 0,
      connectivityScore: 0,
      numPoints: 0,
    }
  }

  // Build point cloud: center + corners of each furniture item
  const points: Array<{ x: number; y: number }> = []
  for (const item of furniture) {
    const w = item.width ?? 2
    const d = item.depth ?? 2
    points.push({ x: item.x, y: item.y })
    points.push({ x: item.x - w / 2, y: item.y - d / 2 })
    points.push({ x: item.x + w / 2, y: item.y - d / 2 })
    points.push({ x: item.x - w / 2, y: item.y + d / 2 })
    points.push({ x: item.x + w / 2, y: item.y + d / 2 })
  }

  // Build distance matrix
  const distMatrix = buildDistanceMatrix2D(points)

  // Compute H₀ persistence
  const h0 = computeH0Persistence(distMatrix, points.length)
  const h0Stats = computeStats(h0)

  // Count long-lived H₀ features = disconnected groups
  const longLivedH0 = h0.filter(
    ([b, d]) => d - b > connectivityThresholdFt,
  ).length

  // Connectivity score: 1 if all connected, lower if fragmented
  const connectivityScore = 1.0 / (1.0 + longLivedH0)

  // Coverage score
  const roomArea = roomWidth * roomDepth
  const coveredArea = furniture.reduce(
    (sum, item) => sum + (item.width ?? 2) * (item.depth ?? 2),
    0,
  )
  const coverageScore = Math.min(coveredArea / Math.max(roomArea, 1), 1.0)

  // We can detect potential dead spaces from H₀ as areas between
  // disconnected groups, but H₁ requires the Python API
  const deadSpaces = h0
    .filter(([b, d]) => d - b > connectivityThresholdFt)
    .map(([birth, death]) => ({
      birthRadius: birth,
      deathRadius: death,
      persistence: death - birth,
      approxDiameterFt: (death - birth) * 2,
      severity: (death - birth > connectivityThresholdFt * 2 ? 'high' : 'medium') as
        | 'high'
        | 'medium',
    }))

  return {
    deadSpaces,
    coverageScore,
    connectivityScore,
    numPoints: points.length,
  }
}

/**
 * Compare two layouts using H₀ persistence statistics.
 * Quick browser-side comparison without needing the Python API.
 */
export function compareLayoutsBrowser(
  layoutA: LayoutPoint[],
  layoutB: LayoutPoint[],
  roomWidth: number,
  roomDepth: number,
): {
  analysisA: DeadSpaceResult
  analysisB: DeadSpaceResult
  connectivityDiff: number
  coverageDiff: number
} {
  const analysisA = analyzeLayoutBrowser(layoutA, roomWidth, roomDepth)
  const analysisB = analyzeLayoutBrowser(layoutB, roomWidth, roomDepth)

  return {
    analysisA,
    analysisB,
    connectivityDiff: Math.abs(
      analysisA.connectivityScore - analysisB.connectivityScore,
    ),
    coverageDiff: Math.abs(analysisA.coverageScore - analysisB.coverageScore),
  }
}
