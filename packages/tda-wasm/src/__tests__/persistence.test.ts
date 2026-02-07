/**
 * Tests for browser-side TDA persistence computation.
 */

import { describe, test, expect } from 'vitest'
import {
  computeH0Persistence,
  buildDistanceMatrix2D,
  computeStats,
} from '../persistence'
import { analyzeLayoutBrowser, compareLayoutsBrowser } from '../layout-analysis'
import type { LayoutPoint } from '../types'

// ─── H₀ Persistence ──────────────────────────────────────────────────────

describe('H₀ Persistence', () => {
  test('single point has no persistence features', () => {
    const dist = new Float64Array([0])
    const h0 = computeH0Persistence(dist, 1)
    expect(h0).toHaveLength(0)
  })

  test('two points produce one H₀ feature', () => {
    // Two points at distance 5
    const dist = new Float64Array([0, 5, 5, 0])
    const h0 = computeH0Persistence(dist, 2)
    expect(h0).toHaveLength(1)
    expect(h0[0]![0]).toBe(0) // birth
    expect(h0[0]![1]).toBe(5) // death
  })

  test('three equidistant points produce two H₀ features', () => {
    const d = 3
    const dist = new Float64Array([0, d, d, d, 0, d, d, d, 0])
    const h0 = computeH0Persistence(dist, 3)
    expect(h0).toHaveLength(2) // 3 components merge into 1 = 2 deaths
  })

  test('two well-separated clusters have one long-lived H₀ feature', () => {
    // Cluster A: points 0,1 close together (dist=1)
    // Cluster B: points 2,3 close together (dist=1)
    // Inter-cluster distance: 10
    const dist = new Float64Array([
      0, 1, 10, 10,
      1, 0, 10, 10,
      10, 10, 0, 1,
      10, 10, 1, 0,
    ])
    const h0 = computeH0Persistence(dist, 4)

    // Should have 3 features (4 components → 1)
    expect(h0).toHaveLength(3)

    // At least one feature should have death ≈ 10 (inter-cluster merge)
    const longLived = h0.filter(([b, d]) => d > 5)
    expect(longLived.length).toBeGreaterThanOrEqual(1)
  })

  test('threshold filters out long-range connections', () => {
    const dist = new Float64Array([
      0, 1, 100,
      1, 0, 100,
      100, 100, 0,
    ])
    const h0 = computeH0Persistence(dist, 3, 50) // threshold=50
    // Only the edge at distance 1 is included
    expect(h0).toHaveLength(1)
    expect(h0[0]![1]).toBe(1)
  })
})

// ─── Distance Matrix ──────────────────────────────────────────────────────

describe('Distance Matrix', () => {
  test('builds correct Euclidean distances', () => {
    const points = [
      { x: 0, y: 0 },
      { x: 3, y: 4 },
      { x: 1, y: 0 },
    ]
    const dist = buildDistanceMatrix2D(points)

    expect(dist[0 * 3 + 1]).toBeCloseTo(5, 10) // (0,0) to (3,4)
    expect(dist[0 * 3 + 2]).toBeCloseTo(1, 10) // (0,0) to (1,0)
    expect(dist[1 * 3 + 0]).toBeCloseTo(5, 10) // symmetric
  })

  test('diagonal is zero', () => {
    const points = [{ x: 0, y: 0 }, { x: 1, y: 1 }]
    const dist = buildDistanceMatrix2D(points)
    expect(dist[0]).toBe(0)
    expect(dist[3]).toBe(0)
  })

  test('matrix is symmetric', () => {
    const points = [
      { x: 0, y: 0 },
      { x: 3, y: 4 },
      { x: 1, y: 2 },
    ]
    const dist = buildDistanceMatrix2D(points)
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(dist[i * 3 + j]).toBeCloseTo(dist[j * 3 + i]!, 10)
      }
    }
  })
})

// ─── Statistics ───────────────────────────────────────────────────────────

describe('Persistence Statistics', () => {
  test('empty diagram returns zero stats', () => {
    const stats = computeStats([])
    expect(stats.count).toBe(0)
    expect(stats.meanLifespan).toBe(0)
  })

  test('single feature has correct stats', () => {
    const stats = computeStats([[0, 5]])
    expect(stats.count).toBe(1)
    expect(stats.meanLifespan).toBe(5)
    expect(stats.maxLifespan).toBe(5)
    expect(stats.stdLifespan).toBe(0)
  })

  test('multiple features have correct mean', () => {
    const stats = computeStats([[0, 2], [0, 4], [0, 6]])
    expect(stats.count).toBe(3)
    expect(stats.meanLifespan).toBe(4) // (2+4+6)/3
    expect(stats.maxLifespan).toBe(6)
    expect(stats.medianLifespan).toBe(4)
  })

  test('infinite features are excluded', () => {
    const stats = computeStats([[0, 3], [0, Infinity]])
    expect(stats.count).toBe(1)
    expect(stats.meanLifespan).toBe(3)
  })

  test('entropy is non-negative', () => {
    const stats = computeStats([[0, 1], [0, 2], [0, 3]])
    expect(stats.entropy).toBeGreaterThanOrEqual(0)
  })
})

// ─── Layout Analysis ──────────────────────────────────────────────────────

describe('Layout Analysis (Browser)', () => {
  test('empty layout has zero scores', () => {
    const result = analyzeLayoutBrowser([], 20, 20)
    expect(result.coverageScore).toBe(0)
    expect(result.connectivityScore).toBe(0)
    expect(result.deadSpaces).toHaveLength(0)
  })

  test('two-item layout has valid scores', () => {
    const items: LayoutPoint[] = [
      { x: 10, y: 10, width: 2, depth: 2 },
      { x: 12, y: 10, width: 2, depth: 2 },
    ]
    const result = analyzeLayoutBrowser(items, 20, 20)
    expect(result.coverageScore).toBeGreaterThan(0)
    expect(result.coverageScore).toBeLessThanOrEqual(1)
    expect(result.numPoints).toBe(10) // 2 × (center + 4 corners)
  })

  test('tightly packed layout has high connectivity', () => {
    const items: LayoutPoint[] = [
      { x: 5, y: 5, width: 3, depth: 3 },
      { x: 8, y: 5, width: 3, depth: 3 },
      { x: 5, y: 8, width: 3, depth: 3 },
      { x: 8, y: 8, width: 3, depth: 3 },
    ]
    const result = analyzeLayoutBrowser(items, 20, 20)
    expect(result.connectivityScore).toBeGreaterThan(0.5)
  })

  test('widely separated items have low connectivity', () => {
    const items: LayoutPoint[] = [
      { x: 1, y: 1, width: 1, depth: 1 },
      { x: 50, y: 50, width: 1, depth: 1 },
    ]
    const result = analyzeLayoutBrowser(items, 60, 60)
    expect(result.connectivityScore).toBeLessThanOrEqual(0.5)
  })

  test('compareLayoutsBrowser returns valid diff', () => {
    const layoutA: LayoutPoint[] = [
      { x: 5, y: 5 },
      { x: 6, y: 5 },
    ]
    const layoutB: LayoutPoint[] = [
      { x: 5, y: 5 },
      { x: 15, y: 15 },
    ]
    const result = compareLayoutsBrowser(layoutA, layoutB, 20, 20)
    expect(result.connectivityDiff).toBeGreaterThanOrEqual(0)
    expect(result.coverageDiff).toBeGreaterThanOrEqual(0)
  })
})
