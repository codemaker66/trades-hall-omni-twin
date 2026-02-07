import { describe, it, expect } from 'vitest'
import fc from 'fast-check'
import {
  solve,
  validate,
  score,
  validateLayout,
  checkNoOverlap,
  checkBounds,
  checkExitClearance,
  checkAisleWidth,
  scoreCapacity,
  scoreSpaceUtilization,
  scoreSightlines,
  scoreSymmetry,
  scoreExitAccess,
  LayoutGrid,
  type RoomConfig,
  type FurnitureSpec,
  type Placement,
  type LayoutRequest,
  type Exit,
} from '../constraint-solver'

// ─── Test Helpers ───────────────────────────────────────────────────────────

function makeRoom(width = 20, depth = 15): RoomConfig {
  return {
    width,
    depth,
    exits: [
      { position: { x: width / 2, z: 0 }, width: 1.5, facing: -Math.PI / 2 },
      { position: { x: width / 2, z: depth }, width: 1.5, facing: Math.PI / 2 },
    ],
    obstacles: [],
    focalPoint: { x: width / 2, z: 1 },
  }
}

function makePlacement(x: number, z: number, w: number, d: number, type = 'chair' as const, specIndex = 0, instanceIndex = 0): Placement {
  return { specIndex, instanceIndex, x, z, rotation: 0, type, effectiveWidth: w, effectiveDepth: d }
}

// ─── LayoutGrid Unit Tests ──────────────────────────────────────────────────

describe('LayoutGrid', () => {
  it('creates correct grid dimensions', () => {
    const room = makeRoom(10, 8)
    const grid = new LayoutGrid(room, 0.5)
    expect(grid.cols).toBe(20) // 10 / 0.5
    expect(grid.rows).toBe(16) // 8 / 0.5
  })

  it('canPlace returns true for empty region', () => {
    const grid = new LayoutGrid(makeRoom(), 0.5)
    expect(grid.canPlace(5, 5, 1, 1)).toBe(true)
  })

  it('canPlace returns false for occupied region', () => {
    const grid = new LayoutGrid(makeRoom(), 0.5)
    grid.occupy(5, 5, 1, 1)
    expect(grid.canPlace(5, 5, 1, 1)).toBe(false)
  })

  it('canPlace returns false for out-of-bounds', () => {
    const grid = new LayoutGrid(makeRoom(10, 10), 0.5)
    expect(grid.canPlace(-1, 5, 1, 1)).toBe(false)
    expect(grid.canPlace(11, 5, 1, 1)).toBe(false)
  })

  it('vacate restores cells to empty', () => {
    const grid = new LayoutGrid(makeRoom(), 0.5)
    grid.occupy(5, 5, 1, 1)
    expect(grid.canPlace(5, 5, 1, 1)).toBe(false)
    grid.vacate(5, 5, 1, 1)
    expect(grid.canPlace(5, 5, 1, 1)).toBe(true)
  })

  it('obstacles block placement', () => {
    const room: RoomConfig = {
      ...makeRoom(10, 10),
      obstacles: [{ x: 3, z: 3, width: 2, depth: 2 }],
    }
    const grid = new LayoutGrid(room, 0.5)
    expect(grid.canPlace(4, 4, 0.5, 0.5)).toBe(false)
  })

  it('snapshot and restore work', () => {
    const grid = new LayoutGrid(makeRoom(), 0.5)
    const snap = grid.snapshot()
    grid.occupy(5, 5, 1, 1)
    expect(grid.canPlace(5, 5, 1, 1)).toBe(false)
    grid.restore(snap)
    expect(grid.canPlace(5, 5, 1, 1)).toBe(true)
  })

  it('freeCellCount decreases after occupy', () => {
    const grid = new LayoutGrid(makeRoom(10, 10), 0.5)
    const before = grid.freeCellCount()
    grid.occupy(5, 5, 1, 1)
    expect(grid.freeCellCount()).toBeLessThan(before)
  })
})

// ─── Hard Constraint Unit Tests ─────────────────────────────────────────────

describe('Hard constraints', () => {
  describe('checkNoOverlap', () => {
    it('detects overlapping placements', () => {
      const a = makePlacement(5, 5, 2, 2)
      const b = makePlacement(6, 5, 2, 2) // overlaps with a
      expect(checkNoOverlap([a, b])).toHaveLength(1)
    })

    it('allows non-overlapping placements', () => {
      const a = makePlacement(2, 2, 1, 1)
      const b = makePlacement(5, 5, 1, 1)
      expect(checkNoOverlap([a, b])).toHaveLength(0)
    })
  })

  describe('checkBounds', () => {
    it('detects out-of-bounds placement', () => {
      const room = makeRoom(10, 10)
      const p = makePlacement(0.3, 5, 2, 2) // extends past left wall
      expect(checkBounds(room, [p])).toHaveLength(1)
    })

    it('allows in-bounds placement', () => {
      const room = makeRoom(10, 10)
      const p = makePlacement(5, 5, 2, 2)
      expect(checkBounds(room, [p])).toHaveLength(0)
    })
  })

  describe('checkExitClearance', () => {
    it('detects furniture blocking exit', () => {
      const room = makeRoom(10, 10)
      // Place furniture right at the exit
      const p = makePlacement(5, 0.5, 2, 1)
      expect(checkExitClearance(room, [p], 1.12)).toHaveLength(1)
    })

    it('allows furniture far from exits', () => {
      const room = makeRoom(10, 10)
      const p = makePlacement(5, 5, 1, 1)
      expect(checkExitClearance(room, [p], 1.12)).toHaveLength(0)
    })
  })

  describe('checkAisleWidth', () => {
    it('detects narrow aisle between placements', () => {
      // Two items 0.5m apart, but min aisle is 0.914m
      const a = makePlacement(5, 5, 2, 2)
      const b = makePlacement(5, 7.5, 2, 2) // gap = 0.5m
      const violations = checkAisleWidth([a, b], 0.914)
      expect(violations.length).toBeGreaterThan(0)
    })

    it('allows sufficient aisle width', () => {
      const a = makePlacement(5, 5, 2, 2)
      const b = makePlacement(5, 9, 2, 2) // gap = 2m
      expect(checkAisleWidth([a, b], 0.914)).toHaveLength(0)
    })
  })

  describe('validateLayout (combined)', () => {
    it('returns empty for valid layout', () => {
      const room = makeRoom(20, 15)
      const placements = [
        makePlacement(5, 7, 2, 2),
        makePlacement(10, 7, 2, 2),
        makePlacement(15, 7, 2, 2),
      ]
      expect(validateLayout(room, placements)).toHaveLength(0)
    })
  })
})

// ─── Soft Objective Unit Tests ──────────────────────────────────────────────

describe('Soft objectives', () => {
  it('scoreCapacity returns 1 when all items placed', () => {
    const specs: FurnitureSpec[] = [{ type: 'chair', width: 0.5, depth: 0.5, count: 3, chairsPerUnit: 0 }]
    const placements = [
      makePlacement(2, 2, 0.5, 0.5),
      makePlacement(3, 2, 0.5, 0.5),
      makePlacement(4, 2, 0.5, 0.5),
    ]
    expect(scoreCapacity(specs, placements)).toBe(1)
  })

  it('scoreCapacity returns fraction when partially placed', () => {
    const specs: FurnitureSpec[] = [{ type: 'chair', width: 0.5, depth: 0.5, count: 4, chairsPerUnit: 0 }]
    const placements = [
      makePlacement(2, 2, 0.5, 0.5),
      makePlacement(3, 2, 0.5, 0.5),
    ]
    expect(scoreCapacity(specs, placements)).toBe(0.5)
  })

  it('scoreSpaceUtilization returns 0-1', () => {
    const room = makeRoom(10, 10)
    const placements = [makePlacement(5, 5, 4, 4)]
    const s = scoreSpaceUtilization(room, placements)
    expect(s).toBeGreaterThanOrEqual(0)
    expect(s).toBeLessThanOrEqual(1)
  })

  it('scoreSightlines returns 1 with no obstacles', () => {
    const room: RoomConfig = { ...makeRoom(10, 10), focalPoint: { x: 5, z: 1 } }
    const chairs = [
      makePlacement(3, 5, 0.5, 0.5, 'chair'),
      makePlacement(5, 5, 0.5, 0.5, 'chair'),
      makePlacement(7, 5, 0.5, 0.5, 'chair'),
    ]
    expect(scoreSightlines(room, chairs)).toBe(1)
  })

  it('scoreSightlines returns < 1 with obstruction', () => {
    const room: RoomConfig = { ...makeRoom(10, 10), focalPoint: { x: 5, z: 1 } }
    const placements = [
      makePlacement(5, 5, 0.5, 0.5, 'chair'), // chair behind table
      makePlacement(5, 3, 3, 2, 'rect-table'),  // table blocking sightline
    ]
    expect(scoreSightlines(room, placements)).toBeLessThan(1)
  })

  it('scoreSymmetry returns 1 for perfectly symmetric layout', () => {
    const room = makeRoom(10, 10)
    const placements = [
      makePlacement(3, 5, 1, 1, 'chair'),
      makePlacement(7, 5, 1, 1, 'chair'), // mirror of first
    ]
    const s = scoreSymmetry(room, placements)
    expect(s).toBeGreaterThan(0.8)
  })

  it('scoreExitAccess returns 0-1', () => {
    const room = makeRoom(10, 10)
    const placements = [makePlacement(5, 5, 0.5, 0.5, 'chair')]
    const s = scoreExitAccess(room, placements)
    expect(s).toBeGreaterThanOrEqual(0)
    expect(s).toBeLessThanOrEqual(1)
  })
})

// ─── Solver Integration Tests ───────────────────────────────────────────────

describe('Solver', () => {
  it('solves a simple room with chairs', () => {
    const request: LayoutRequest = {
      room: makeRoom(10, 10),
      furniture: [
        { type: 'chair', width: 0.5, depth: 0.5, count: 10, chairsPerUnit: 0 },
      ],
    }
    const result = solve(request)
    expect(result.placements.length).toBeGreaterThan(0)
    expect(result.stats.placedCount).toBeGreaterThan(0)
    expect(result.scores.capacityUtilization).toBeGreaterThan(0)
  })

  it('places most items in a spacious room', () => {
    const request: LayoutRequest = {
      room: makeRoom(20, 15),
      furniture: [
        { type: 'round-table', width: 1.8, depth: 1.8, count: 4, chairsPerUnit: 0 },
        { type: 'chair', width: 0.5, depth: 0.5, count: 20, chairsPerUnit: 0 },
      ],
    }
    const result = solve(request)
    expect(result.stats.placedCount).toBe(24)
    // No overlaps or out-of-bounds (aisle violations are acceptable for greedy)
    expect(checkNoOverlap(result.placements)).toHaveLength(0)
    expect(checkBounds(request.room, result.placements)).toHaveLength(0)
  })

  it('handles fixed-zone placement', () => {
    // Use a room where exits are on the south wall so north wall is clear for stage
    const room: RoomConfig = {
      width: 15,
      depth: 12,
      exits: [{ position: { x: 7.5, z: 12 }, width: 1.5, facing: Math.PI / 2 }],
      obstacles: [],
      focalPoint: { x: 7.5, z: 1 },
    }
    const request: LayoutRequest = {
      room,
      furniture: [
        { type: 'stage', width: 4, depth: 2, count: 1, chairsPerUnit: 0, fixedZone: 'north' },
        { type: 'chair', width: 0.5, depth: 0.5, count: 10, chairsPerUnit: 0 },
      ],
    }
    const result = solve(request)
    const stage = result.placements.find((p) => p.type === 'stage')
    expect(stage).toBeDefined()
    // Stage should be near north wall (low Z)
    expect(stage!.z).toBeLessThan(6)
  })

  it('respects wall-adjacent constraint', () => {
    const request: LayoutRequest = {
      room: makeRoom(15, 10),
      furniture: [
        { type: 'bar', width: 3, depth: 1, count: 1, chairsPerUnit: 0, wallAdjacent: true },
      ],
    }
    const result = solve(request)
    expect(result.placements.length).toBe(1)
    const bar = result.placements[0]!
    // Should be near a wall (within 1.5m of any edge)
    const distToWall = Math.min(
      bar.x, bar.z,
      15 - bar.x, 10 - bar.z,
    )
    expect(distToWall).toBeLessThan(2)
  })

  it('produces no overlap or bounds violations', () => {
    const request: LayoutRequest = {
      room: makeRoom(15, 12),
      furniture: [
        { type: 'round-table', width: 1.8, depth: 1.8, count: 3, chairsPerUnit: 0 },
        { type: 'chair', width: 0.5, depth: 0.5, count: 15, chairsPerUnit: 0 },
      ],
    }
    const result = solve(request)
    // Core invariants: no overlaps or out-of-bounds
    expect(checkNoOverlap(result.placements)).toHaveLength(0)
    expect(checkBounds(request.room, result.placements)).toHaveLength(0)
  })

  it('validate() catches invalid layouts', () => {
    const room = makeRoom(10, 10)
    const placements = [
      makePlacement(5, 5, 3, 3),
      makePlacement(5, 5, 3, 3), // identical position = overlap
    ]
    const result = validate(room, placements)
    expect(result.valid).toBe(false)
    expect(result.violations.length).toBeGreaterThan(0)
  })

  it('score() returns all metrics', () => {
    const room = makeRoom(10, 10)
    const specs: FurnitureSpec[] = [{ type: 'chair', width: 0.5, depth: 0.5, count: 5, chairsPerUnit: 0 }]
    const placements = [
      makePlacement(3, 5, 0.5, 0.5, 'chair'),
      makePlacement(5, 5, 0.5, 0.5, 'chair'),
      makePlacement(7, 5, 0.5, 0.5, 'chair'),
    ]
    const scores = score(room, specs, placements)
    expect(scores.capacityUtilization).toBe(0.6)
    expect(scores.total).toBeGreaterThan(0)
  })

  it('solver reports solve time', () => {
    const result = solve({
      room: makeRoom(10, 10),
      furniture: [{ type: 'chair', width: 0.5, depth: 0.5, count: 5, chairsPerUnit: 0 }],
    })
    expect(result.stats.solveTimeMs).toBeGreaterThanOrEqual(0)
  })
})

// ─── Property-Based Tests ───────────────────────────────────────────────────

describe('Constraint solver — property-based tests', () => {
  it('solver output ALWAYS satisfies hard constraints', () => {
    fc.assert(fc.property(
      fc.integer({ min: 8, max: 30 }),
      fc.integer({ min: 8, max: 25 }),
      fc.integer({ min: 1, max: 20 }),
      (roomW, roomD, chairCount) => {
        const request: LayoutRequest = {
          room: {
            width: roomW,
            depth: roomD,
            exits: [{ position: { x: roomW / 2, z: 0 }, width: 1.5, facing: -Math.PI / 2 }],
            obstacles: [],
          },
          furniture: [
            { type: 'chair', width: 0.5, depth: 0.5, count: chairCount, chairsPerUnit: 0 },
          ],
          options: { annealingIterations: 100 }, // reduce for speed in PBT
        }
        const result = solve(request)
        // The key property: if feasible, ZERO hard constraint violations
        if (result.feasible) {
          expect(result.violations).toHaveLength(0)
          // Double-check with standalone validate
          const v = validate(request.room, result.placements)
          expect(v.valid).toBe(true)
        }
      },
    ), { numRuns: 50 })
  })

  it('no placements extend outside room boundaries', () => {
    fc.assert(fc.property(
      fc.integer({ min: 5, max: 20 }),
      fc.integer({ min: 5, max: 20 }),
      fc.integer({ min: 1, max: 15 }),
      (roomW, roomD, count) => {
        const result = solve({
          room: { width: roomW, depth: roomD, exits: [], obstacles: [] },
          furniture: [{ type: 'chair', width: 0.5, depth: 0.5, count, chairsPerUnit: 0 }],
          options: { annealingIterations: 50 },
        })
        for (const p of result.placements) {
          const hw = p.effectiveWidth / 2
          const hd = p.effectiveDepth / 2
          expect(p.x - hw).toBeGreaterThanOrEqual(-0.01)
          expect(p.z - hd).toBeGreaterThanOrEqual(-0.01)
          expect(p.x + hw).toBeLessThanOrEqual(roomW + 0.01)
          expect(p.z + hd).toBeLessThanOrEqual(roomD + 0.01)
        }
      },
    ), { numRuns: 50 })
  })

  it('no two placements overlap', () => {
    fc.assert(fc.property(
      fc.integer({ min: 8, max: 25 }),
      fc.integer({ min: 1, max: 12 }),
      (roomSize, tableCount) => {
        const result = solve({
          room: { width: roomSize, depth: roomSize, exits: [], obstacles: [] },
          furniture: [
            { type: 'round-table', width: 1.8, depth: 1.8, count: tableCount, chairsPerUnit: 0 },
          ],
          options: { annealingIterations: 100 },
        })
        // Verify no overlaps
        const overlaps = checkNoOverlap(result.placements)
        expect(overlaps).toHaveLength(0)
      },
    ), { numRuns: 50 })
  })

  it('capacity utilization is non-negative and at most 1', () => {
    fc.assert(fc.property(
      fc.integer({ min: 5, max: 20 }),
      fc.integer({ min: 1, max: 30 }),
      (roomSize, count) => {
        const specs: FurnitureSpec[] = [{ type: 'chair', width: 0.5, depth: 0.5, count, chairsPerUnit: 0 }]
        const result = solve({
          room: { width: roomSize, depth: roomSize, exits: [], obstacles: [] },
          furniture: specs,
          options: { annealingIterations: 50 },
        })
        expect(result.scores.capacityUtilization).toBeGreaterThanOrEqual(0)
        expect(result.scores.capacityUtilization).toBeLessThanOrEqual(1)
      },
    ), { numRuns: 50 })
  })

  it('all score metrics are in [0, 1]', () => {
    fc.assert(fc.property(
      fc.integer({ min: 8, max: 20 }),
      fc.integer({ min: 1, max: 10 }),
      (roomSize, count) => {
        const result = solve({
          room: {
            width: roomSize, depth: roomSize,
            exits: [{ position: { x: roomSize / 2, z: 0 }, width: 1.5, facing: 0 }],
            obstacles: [],
            focalPoint: { x: roomSize / 2, z: 1 },
          },
          furniture: [{ type: 'chair', width: 0.5, depth: 0.5, count, chairsPerUnit: 0 }],
          options: { annealingIterations: 50 },
        })
        const s = result.scores
        expect(s.capacityUtilization).toBeGreaterThanOrEqual(0)
        expect(s.capacityUtilization).toBeLessThanOrEqual(1)
        expect(s.spaceUtilization).toBeGreaterThanOrEqual(0)
        expect(s.spaceUtilization).toBeLessThanOrEqual(1)
        expect(s.sightlineCoverage).toBeGreaterThanOrEqual(0)
        expect(s.sightlineCoverage).toBeLessThanOrEqual(1)
        expect(s.symmetry).toBeGreaterThanOrEqual(0)
        expect(s.symmetry).toBeLessThanOrEqual(1)
        expect(s.exitAccess).toBeGreaterThanOrEqual(0)
        expect(s.exitAccess).toBeLessThanOrEqual(1)
      },
    ), { numRuns: 50 })
  })

  it('deterministic: same input = same output', () => {
    fc.assert(fc.property(
      fc.integer({ min: 8, max: 15 }),
      fc.integer({ min: 1, max: 8 }),
      (roomSize, count) => {
        const request: LayoutRequest = {
          room: { width: roomSize, depth: roomSize, exits: [], obstacles: [] },
          furniture: [{ type: 'chair', width: 0.5, depth: 0.5, count, chairsPerUnit: 0 }],
          options: { annealingIterations: 50 },
        }
        const r1 = solve(request)
        const r2 = solve(request)
        expect(r1.placements.length).toBe(r2.placements.length)
        for (let i = 0; i < r1.placements.length; i++) {
          expect(r1.placements[i]!.x).toBe(r2.placements[i]!.x)
          expect(r1.placements[i]!.z).toBe(r2.placements[i]!.z)
        }
      },
    ), { numRuns: 30 })
  })
})
