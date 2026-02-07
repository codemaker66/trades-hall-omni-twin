/**
 * Pure TypeScript persistent homology computation (TDA-7).
 *
 * Provides a fallback implementation for small datasets when WASM
 * Ripser is not available. Uses Vietoris-Rips filtration with a
 * simplified Union-Find approach for H₀.
 *
 * For H₁ and above, delegates to the WASM module or falls back to
 * the Python backend API.
 *
 * Performance expectations (pure TS, no WASM):
 *   50 points, H₀: < 10ms
 *   100 points, H₀: < 50ms
 *   200 points, H₀: < 200ms
 *   H₁: Only supported via WASM or API
 */

import type { PersistenceResult, DimensionStats } from './types'

// ─── Union-Find for H₀ computation ───────────────────────────────────────

class UnionFind {
  private parent: Int32Array
  private rank: Int32Array
  private birthTime: Float64Array
  numComponents: number

  constructor(n: number) {
    this.parent = new Int32Array(n)
    this.rank = new Int32Array(n)
    this.birthTime = new Float64Array(n)
    this.numComponents = n
    for (let i = 0; i < n; i++) {
      this.parent[i] = i
    }
  }

  find(x: number): number {
    let root = x
    while (this.parent[root]! !== root) {
      root = this.parent[root]!
    }
    // Path compression
    while (this.parent[x]! !== root) {
      const next = this.parent[x]!
      this.parent[x] = root
      x = next
    }
    return root
  }

  union(x: number, y: number, time: number): [number, number] | null {
    const rx = this.find(x)
    const ry = this.find(y)
    if (rx === ry) return null

    this.numComponents--

    // Younger component dies (higher birth time = younger)
    const bx = this.birthTime[rx]!
    const by = this.birthTime[ry]!

    let dying: number
    let surviving: number

    if (bx >= by) {
      dying = rx
      surviving = ry
    } else {
      dying = ry
      surviving = rx
    }

    // Union by rank
    if (this.rank[surviving]! < this.rank[dying]!) {
      this.parent[surviving] = dying
      surviving = dying
    } else if (this.rank[surviving]! === this.rank[dying]!) {
      this.parent[dying] = surviving
      this.rank[surviving]!++
    } else {
      this.parent[dying] = surviving
    }

    return [this.birthTime[dying]!, time]
  }
}

// ─── H₀ Persistence via Kruskal's Algorithm ──────────────────────────────

/**
 * Compute H₀ persistence diagram from a distance matrix.
 *
 * Uses Kruskal's minimum spanning tree approach:
 * 1. Sort all edges by distance
 * 2. Process edges in order; each merge kills a connected component
 * 3. The death time = distance at which the merge occurs
 *
 * Returns birth-death pairs for H₀ features.
 */
export function computeH0Persistence(
  distanceMatrix: Float64Array,
  n: number,
  threshold?: number,
): [number, number][] {
  // Build edge list
  const edges: Array<{ i: number; j: number; d: number }> = []
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = distanceMatrix[i * n + j]!
      if (threshold !== undefined && d > threshold) continue
      edges.push({ i, j, d })
    }
  }

  // Sort by distance
  edges.sort((a, b) => a.d - b.d)

  const uf = new UnionFind(n)
  const diagram: [number, number][] = []

  for (const edge of edges) {
    const result = uf.union(edge.i, edge.j, edge.d)
    if (result) {
      diagram.push(result)
    }
  }

  return diagram
}

// ─── Euclidean Distance Matrix ───────────────────────────────────────────

/**
 * Build a Euclidean distance matrix from 2D points.
 */
export function buildDistanceMatrix2D(
  points: Array<{ x: number; y: number }>,
): Float64Array {
  const n = points.length
  const matrix = new Float64Array(n * n)

  for (let i = 0; i < n; i++) {
    const pi = points[i]!
    for (let j = i + 1; j < n; j++) {
      const pj = points[j]!
      const dx = pi.x - pj.x
      const dy = pi.y - pj.y
      const d = Math.sqrt(dx * dx + dy * dy)
      matrix[i * n + j] = d
      matrix[j * n + i] = d
    }
  }

  return matrix
}

// ─── Persistence Statistics ──────────────────────────────────────────────

/**
 * Compute statistics from a persistence diagram.
 */
export function computeStats(diagram: [number, number][]): DimensionStats {
  const lifespans = diagram
    .filter(([b, d]) => Number.isFinite(d))
    .map(([b, d]) => d - b)

  if (lifespans.length === 0) {
    return {
      count: 0,
      meanLifespan: 0,
      stdLifespan: 0,
      medianLifespan: 0,
      maxLifespan: 0,
      iqrLifespan: 0,
      entropy: 0,
    }
  }

  lifespans.sort((a, b) => a - b)

  const n = lifespans.length
  const sum = lifespans.reduce((s, v) => s + v, 0)
  const mean = sum / n
  const variance =
    lifespans.reduce((s, v) => s + (v - mean) * (v - mean), 0) / n
  const std = Math.sqrt(variance)
  const median =
    n % 2 === 0
      ? (lifespans[n / 2 - 1]! + lifespans[n / 2]!) / 2
      : lifespans[Math.floor(n / 2)]!
  const max = lifespans[n - 1]!
  const q25 = lifespans[Math.floor(n * 0.25)]!
  const q75 = lifespans[Math.floor(n * 0.75)]!
  const iqr = q75 - q25

  // Shannon entropy
  let entropy = 0
  for (const l of lifespans) {
    const p = l / (sum + 1e-30)
    if (p > 0) entropy -= p * Math.log(p)
  }

  return {
    count: n,
    meanLifespan: mean,
    stdLifespan: std,
    medianLifespan: median,
    maxLifespan: max,
    iqrLifespan: iqr,
    entropy,
  }
}
