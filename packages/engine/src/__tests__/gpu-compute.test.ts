import { describe, it, expect } from 'vitest'
import fc from 'fast-check'

import type { AnalysisItem, RoomGeometry, Point2D } from '../gpu-compute/types'
import { detectCollisionsCPU } from '../gpu-compute/collision'
import { analyzeSightlinesCPU } from '../gpu-compute/sightlines'
import { simulateCrowdFlowCPU } from '../gpu-compute/crowd-flow'
import { COLLISION_SHADER, SIGHTLINE_SHADER, CROWD_FLOW_SHADER } from '../gpu-compute'

// ─── Helpers ────────────────────────────────────────────────────────────────

function makeItem(
  id: string, x: number, z: number,
  halfWidth: number, halfDepth: number,
  isChair = false,
): AnalysisItem {
  return { id, x, z, halfWidth, halfDepth, isChair }
}

function makeRoom(width = 20, depth = 15, exits?: Point2D[]): RoomGeometry {
  return {
    width, depth,
    exits: exits ?? [{ x: width / 2, z: 0 }],
    walls: [
      { a: { x: 0, z: 0 }, b: { x: width, z: 0 } },
      { a: { x: width, z: 0 }, b: { x: width, z: depth } },
      { a: { x: width, z: depth }, b: { x: 0, z: depth } },
      { a: { x: 0, z: depth }, b: { x: 0, z: 0 } },
    ],
  }
}

// ─── Collision Detection ────────────────────────────────────────────────────

describe('Parallel collision detection (CPU)', () => {
  it('detects overlapping items', () => {
    const items = [
      makeItem('a', 5, 5, 1, 1),
      makeItem('b', 5.5, 5.5, 1, 1), // overlaps with a
      makeItem('c', 20, 20, 1, 1),    // far away
    ]
    const result = detectCollisionsCPU(items)
    expect(result.count).toBe(1)
    expect(result.pairs).toEqual([[0, 1]])
    expect(result.colliding[0]).toBe(true)
    expect(result.colliding[1]).toBe(true)
    expect(result.colliding[2]).toBe(false)
  })

  it('returns no collisions for separated items', () => {
    const items = [
      makeItem('a', 0, 0, 1, 1),
      makeItem('b', 10, 10, 1, 1),
      makeItem('c', 20, 20, 1, 1),
    ]
    const result = detectCollisionsCPU(items)
    expect(result.count).toBe(0)
    expect(result.pairs.length).toBe(0)
  })

  it('handles empty list', () => {
    const result = detectCollisionsCPU([])
    expect(result.count).toBe(0)
  })

  it('detects all pairs in a cluster', () => {
    // 3 items all overlapping each other
    const items = [
      makeItem('a', 5, 5, 2, 2),
      makeItem('b', 5.5, 5, 2, 2),
      makeItem('c', 5, 5.5, 2, 2),
    ]
    const result = detectCollisionsCPU(items)
    expect(result.count).toBe(3)
    expect(result.colliding.every(c => c)).toBe(true)
  })

  it('handles touching edges (no overlap)', () => {
    const items = [
      makeItem('a', 5, 5, 1, 1),   // occupies [4,6] x [4,6]
      makeItem('b', 7, 5, 1, 1),   // occupies [6,8] x [4,6] — touching edge
    ]
    const result = detectCollisionsCPU(items)
    expect(result.count).toBe(0)
  })
})

// ─── Sightline Analysis ────────────────────────────────────────────────────

describe('Sightline analysis (CPU)', () => {
  it('all chairs have clear sightlines with no obstacles', () => {
    const items = [
      makeItem('c1', 5, 10, 0.3, 0.3, true),
      makeItem('c2', 10, 10, 0.3, 0.3, true),
      makeItem('c3', 15, 10, 0.3, 0.3, true),
    ]
    const room = makeRoom(20, 15)
    const focal: Point2D = { x: 10, z: 2 }

    const result = analyzeSightlinesCPU(items, focal, room)
    expect(result.coverage).toBe(1)
    expect(result.perChair.every(s => s === 1)).toBe(true)
  })

  it('chair behind obstacle has blocked sightline', () => {
    const items = [
      makeItem('chair', 5, 10, 0.3, 0.3, true),  // chair at back
      makeItem('table', 5, 6, 2, 1, false),        // table blocking
    ]
    const room = makeRoom(20, 15)
    const focal: Point2D = { x: 5, z: 2 }

    const result = analyzeSightlinesCPU(items, focal, room)
    expect(result.coverage).toBe(0)
    expect(result.perChair[0]).toBe(0)
  })

  it('partial coverage with some chairs blocked', () => {
    const items = [
      makeItem('c1', 3, 10, 0.3, 0.3, true),  // clear
      makeItem('c2', 10, 10, 0.3, 0.3, true),  // blocked by table
      makeItem('table', 10, 6, 2, 1, false),
    ]
    const room = makeRoom(20, 15)
    const focal: Point2D = { x: 10, z: 2 }

    const result = analyzeSightlinesCPU(items, focal, room)
    expect(result.coverage).toBe(0.5)
  })

  it('generates heatmap with correct dimensions', () => {
    const items: AnalysisItem[] = []
    const room = makeRoom(10, 8)
    const focal: Point2D = { x: 5, z: 1 }

    const result = analyzeSightlinesCPU(items, focal, room, 1.0)
    expect(result.heatmapWidth).toBe(10)
    expect(result.heatmapHeight).toBe(8)
    expect(result.heatmap.length).toBe(80)
    // No obstacles → all cells should be 1
    expect(Array.from(result.heatmap).every(v => v === 1)).toBe(true)
  })

  it('heatmap shows shadow behind obstacle', () => {
    const items = [
      makeItem('table', 5, 5, 2, 1, false),
    ]
    const room = makeRoom(10, 10)
    const focal: Point2D = { x: 5, z: 1 }

    const result = analyzeSightlinesCPU(items, focal, room, 1.0)
    // Cells behind the table (higher z) and aligned with focal should be blocked
    const behindTableRow = 7 // z = 7.5 is behind table at z=5
    const behindTableCol = 5 // x = 5.5 is in line
    const idx = behindTableRow * result.heatmapWidth + behindTableCol
    expect(result.heatmap[idx]).toBe(0)
  })

  it('returns full coverage when no chairs', () => {
    const items = [makeItem('table', 5, 5, 2, 1, false)]
    const room = makeRoom(10, 10)
    const result = analyzeSightlinesCPU(items, { x: 5, z: 1 }, room)
    expect(result.coverage).toBe(1)
  })
})

// ─── Crowd Flow Simulation ──────────────────────────────────────────────────

describe('Crowd flow simulation (CPU)', () => {
  it('agents evacuate to exit', () => {
    const items = [
      makeItem('c1', 5, 5, 0.3, 0.3, true),
      makeItem('c2', 10, 5, 0.3, 0.3, true),
    ]
    const room = makeRoom(20, 10, [{ x: 10, z: 0 }])

    const result = simulateCrowdFlowCPU(items, room, {
      maxTime: 30,
      dt: 0.1,
      speed: 2.0,
    })

    expect(result.evacuationTimes.length).toBe(2)
    expect(result.maxTime).toBeLessThan(30) // should evacuate before timeout
    expect(result.avgTime).toBeGreaterThan(0)
  })

  it('produces density heatmap', () => {
    const items = [
      makeItem('c1', 5, 5, 0.3, 0.3, true),
    ]
    const room = makeRoom(10, 10, [{ x: 5, z: 0 }])

    const result = simulateCrowdFlowCPU(items, room, {
      maxTime: 20,
      cellSize: 1.0,
    })

    expect(result.heatmapWidth).toBe(10)
    expect(result.heatmapHeight).toBe(10)
    expect(result.densityHeatmap.length).toBe(100)
    // At least one cell should have been occupied
    const maxDensity = Math.max(...Array.from(result.densityHeatmap))
    expect(maxDensity).toBeGreaterThan(0)
  })

  it('obstacle avoidance increases evacuation time', () => {
    const room = makeRoom(10, 10, [{ x: 5, z: 0 }])

    // Without obstacle
    const noObs = simulateCrowdFlowCPU(
      [makeItem('c', 5, 8, 0.3, 0.3, true)],
      room,
      { maxTime: 30, dt: 0.1 },
    )

    // With obstacle between chair and exit
    const withObs = simulateCrowdFlowCPU(
      [
        makeItem('c', 5, 8, 0.3, 0.3, true),
        makeItem('wall', 5, 4, 3, 0.5, false),
      ],
      room,
      { maxTime: 30, dt: 0.1 },
    )

    // Obstacle should slow evacuation
    expect(withObs.maxTime).toBeGreaterThanOrEqual(noObs.maxTime)
  })

  it('handles empty input', () => {
    const room = makeRoom(10, 10)
    const result = simulateCrowdFlowCPU([], room)
    expect(result.evacuationTimes.length).toBe(0)
    expect(result.maxTime).toBe(0)
  })

  it('handles no exits', () => {
    const items = [makeItem('c', 5, 5, 0.3, 0.3, true)]
    const room = makeRoom(10, 10, [])
    const result = simulateCrowdFlowCPU(items, room)
    expect(result.evacuationTimes.length).toBe(0) // no chairs processed
  })

  it('multiple exits: agents use nearest', () => {
    // Exit on left and right
    const room = makeRoom(20, 10, [
      { x: 0, z: 5 },   // left exit
      { x: 20, z: 5 },  // right exit
    ])
    const items = [
      makeItem('c1', 3, 5, 0.3, 0.3, true),   // near left exit
      makeItem('c2', 17, 5, 0.3, 0.3, true),  // near right exit
    ]

    const result = simulateCrowdFlowCPU(items, room, {
      maxTime: 20,
      dt: 0.1,
      speed: 2.0,
    })

    expect(result.evacuationTimes.length).toBe(2)
    // Both should evacuate quickly since each is near an exit
    expect(result.maxTime).toBeLessThan(10)
  })
})

// ─── WGSL Shader Source Validation ──────────────────────────────────────────

describe('WGSL shader sources', () => {
  it('collision shader contains required entry point', () => {
    expect(COLLISION_SHADER).toContain('@compute')
    expect(COLLISION_SHADER).toContain('@workgroup_size(64)')
    expect(COLLISION_SHADER).toContain('fn main')
    expect(COLLISION_SHADER).toContain('aabbOverlap')
  })

  it('sightline shader contains required entry points', () => {
    expect(SIGHTLINE_SHADER).toContain('@compute')
    expect(SIGHTLINE_SHADER).toContain('fn chairSightlines')
    expect(SIGHTLINE_SHADER).toContain('fn heatmapSightlines')
    expect(SIGHTLINE_SHADER).toContain('rayIntersectsAABB')
  })

  it('crowd flow shader contains required entry point', () => {
    expect(CROWD_FLOW_SHADER).toContain('@compute')
    expect(CROWD_FLOW_SHADER).toContain('fn updateAgents')
    expect(CROWD_FLOW_SHADER).toContain('struct Agent')
  })
})

// ─── Property-Based Tests ───────────────────────────────────────────────────

describe('Property-based tests', () => {
  const arbItem = fc.record({
    id: fc.string({ minLength: 1, maxLength: 5 }),
    x: fc.float({ noNaN: true, noDefaultInfinity: true, min: 0, max: 50 }),
    z: fc.float({ noNaN: true, noDefaultInfinity: true, min: 0, max: 50 }),
    halfWidth: fc.float({ noNaN: true, noDefaultInfinity: true, min: Math.fround(0.1), max: 3 }),
    halfDepth: fc.float({ noNaN: true, noDefaultInfinity: true, min: Math.fround(0.1), max: 3 }),
    isChair: fc.boolean(),
  })

  it('collision detection is symmetric: (i,j) in pairs implies (j,i) not separate', () => {
    fc.assert(fc.property(
      fc.array(arbItem, { minLength: 2, maxLength: 20 }),
      (items) => {
        const result = detectCollisionsCPU(items)
        // For each pair (i,j), both should be marked as colliding
        for (const [i, j] of result.pairs) {
          expect(result.colliding[i]).toBe(true)
          expect(result.colliding[j]).toBe(true)
        }
      },
    ), { numRuns: 100 })
  })

  it('collision count matches pairs array length', () => {
    fc.assert(fc.property(
      fc.array(arbItem, { minLength: 0, maxLength: 30 }),
      (items) => {
        const result = detectCollisionsCPU(items)
        expect(result.count).toBe(result.pairs.length)
      },
    ), { numRuns: 100 })
  })

  it('sightline coverage is in [0, 1]', () => {
    fc.assert(fc.property(
      fc.array(arbItem, { minLength: 1, maxLength: 15 }),
      (items) => {
        const room = makeRoom(60, 60)
        const focal: Point2D = { x: 30, z: 5 }
        const result = analyzeSightlinesCPU(items, focal, room)
        expect(result.coverage).toBeGreaterThanOrEqual(0)
        expect(result.coverage).toBeLessThanOrEqual(1)
      },
    ), { numRuns: 50 })
  })

  it('sightline heatmap values are 0 or 1', () => {
    fc.assert(fc.property(
      fc.array(arbItem, { minLength: 0, maxLength: 10 }),
      (items) => {
        const room = makeRoom(10, 10)
        const focal: Point2D = { x: 5, z: 1 }
        const result = analyzeSightlinesCPU(items, focal, room, 2.0)
        for (const v of result.heatmap) {
          expect(v === 0 || v === 1).toBe(true)
        }
      },
    ), { numRuns: 50 })
  })

  it('crowd flow: all evacuation times are positive', () => {
    fc.assert(fc.property(
      fc.array(arbItem.filter(i => i.isChair), { minLength: 1, maxLength: 5 }),
      (chairs) => {
        const room = makeRoom(60, 60, [{ x: 30, z: 0 }])
        const result = simulateCrowdFlowCPU(chairs, room, {
          maxTime: 60,
          dt: 0.2,
          speed: 2.0,
        })
        for (const t of result.evacuationTimes) {
          expect(t).toBeGreaterThan(0)
        }
      },
    ), { numRuns: 30 })
  })
})
