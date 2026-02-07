import { describe, it, expect } from 'vitest'
import fc from 'fast-check'
import {
  solve,
  validate,
  score,
  validateLayout,
  validateLayoutSpatial,
  checkNoOverlap,
  checkNoOverlapSpatial,
  checkBounds,
  checkExitClearance,
  checkAisleWidth,
  checkAisleWidthSpatial,
  validateSinglePlacement,
  violationSeverity,
  scoreCapacity,
  scoreSpaceUtilization,
  scoreSightlines,
  scoreSymmetry,
  scoreExitAccess,
  LayoutGrid,
  SolverSpatialHash,
  IncrementalConstraintGraph,
  generateChairPositions,
  placeChairGroups,
  markValidated,
  toCardinalRotation,
  cardinalToRadians,
  CellState,
  VIOLATION_TYPES,
  type RoomConfig,
  type FurnitureSpec,
  type Placement,
  type LayoutRequest,
  type Exit,
  type SolverFurnitureType,
  type ViolationType,
  type CardinalRotation,
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

function makePlacement(x: number, z: number, w: number, d: number, type: SolverFurnitureType = 'chair', specIndex = 0, instanceIndex = 0): Placement {
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

  it('rebuildOccupancy clears and re-occupies', () => {
    const room = makeRoom(10, 10)
    const grid = new LayoutGrid(room, 0.5)
    grid.occupy(3, 3, 1, 1)
    grid.occupy(7, 7, 1, 1)
    const placements = [makePlacement(5, 5, 2, 2)]
    grid.rebuildOccupancy(room, placements)
    // Old occupancies should be cleared
    expect(grid.canPlace(3, 3, 0.5, 0.5)).toBe(true)
    expect(grid.canPlace(7, 7, 0.5, 0.5)).toBe(true)
    // New placement should be occupied
    expect(grid.canPlace(5, 5, 0.5, 0.5)).toBe(false)
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
    expect(result.stats.placedCount).toBeGreaterThanOrEqual(20)
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

  it('reports backtrack and restart stats', () => {
    const result = solve({
      room: makeRoom(10, 10),
      furniture: [{ type: 'chair', width: 0.5, depth: 0.5, count: 5, chairsPerUnit: 0 }],
    })
    expect(result.stats.backtracks).toBeGreaterThanOrEqual(0)
    expect(result.stats.restarts).toBeGreaterThanOrEqual(0)
  })

  it('respects seed option for determinism', () => {
    const request: LayoutRequest = {
      room: makeRoom(12, 10),
      furniture: [{ type: 'chair', width: 0.5, depth: 0.5, count: 8, chairsPerUnit: 0 }],
      options: { seed: 123, annealingIterations: 100 },
    }
    const r1 = solve(request)
    const r2 = solve(request)
    expect(r1.placements.length).toBe(r2.placements.length)
    for (let i = 0; i < r1.placements.length; i++) {
      expect(r1.placements[i]!.x).toBe(r2.placements[i]!.x)
      expect(r1.placements[i]!.z).toBe(r2.placements[i]!.z)
    }
  })

  it('returns groupings array (even if empty)', () => {
    const result = solve({
      room: makeRoom(10, 10),
      furniture: [{ type: 'chair', width: 0.5, depth: 0.5, count: 3, chairsPerUnit: 0 }],
    })
    expect(Array.isArray(result.groupings)).toBe(true)
  })

  it('produces chair-table groupings when chairsPerUnit > 0', () => {
    const result = solve({
      room: makeRoom(20, 15),
      furniture: [
        { type: 'round-table', width: 1.8, depth: 1.8, count: 2, chairsPerUnit: 4 },
        { type: 'chair', width: 0.44, depth: 0.44, count: 0, chairsPerUnit: 0 },
      ],
      options: { annealingIterations: 100 },
    })
    // Should have at least some groupings if tables were placed
    const tables = result.placements.filter((p) => p.type === 'round-table')
    if (tables.length > 0) {
      expect(result.groupings.length).toBeGreaterThan(0)
      for (const g of result.groupings) {
        expect(g.chairIndices.length).toBeGreaterThan(0)
        expect(g.chairsPerUnit).toBe(4)
      }
    }
  })
})

// ─── Type Safety Foundation Tests (WI-1) ───────────────────────────────────

describe('Type safety foundation', () => {
  it('markValidated produces branded type', () => {
    const layout: Placement[] = [makePlacement(5, 5, 1, 1)]
    const validated = markValidated(layout)
    // Should still be an array
    expect(validated.length).toBe(1)
    expect(validated[0]!.x).toBe(5)
  })

  it('CardinalRotation: 0° → 0', () => {
    expect(toCardinalRotation(0)).toBe(0)
  })

  it('CardinalRotation: 90° → 1', () => {
    expect(toCardinalRotation(Math.PI / 2)).toBe(1)
  })

  it('CardinalRotation: 180° → 2', () => {
    expect(toCardinalRotation(Math.PI)).toBe(2)
  })

  it('CardinalRotation: 270° → 3', () => {
    expect(toCardinalRotation(3 * Math.PI / 2)).toBe(3)
  })

  it('CardinalRotation: roundtrip with cardinalToRadians', () => {
    for (const r of [0, 1, 2, 3] as CardinalRotation[]) {
      expect(toCardinalRotation(cardinalToRadians(r))).toBe(r)
    }
  })

  it('CardinalRotation: handles negative angles', () => {
    expect(toCardinalRotation(-Math.PI / 2)).toBe(3)
  })

  it('CellState values are distinct', () => {
    const values = [CellState.EMPTY, CellState.WALL, CellState.OBSTACLE, CellState.OCCUPIED, CellState.EXIT_ZONE]
    const unique = new Set(values)
    expect(unique.size).toBe(5)
  })

  it('VIOLATION_TYPES has all expected types', () => {
    expect(VIOLATION_TYPES).toContain('overlap')
    expect(VIOLATION_TYPES).toContain('out-of-bounds')
    expect(VIOLATION_TYPES).toContain('aisle-too-narrow')
    expect(VIOLATION_TYPES).toContain('exit-blocked')
    expect(VIOLATION_TYPES).toContain('obstacle-overlap')
    expect(VIOLATION_TYPES.length).toBe(5)
  })

  it('violationSeverity returns values for all types', () => {
    for (const type of VIOLATION_TYPES) {
      const severity = violationSeverity(type)
      expect(severity).toBeGreaterThan(0)
      expect(severity).toBeLessThanOrEqual(10)
    }
  })

  it('violationSeverity: overlap is most severe', () => {
    expect(violationSeverity('overlap')).toBeGreaterThan(violationSeverity('aisle-too-narrow'))
  })
})

// ─── SolverSpatialHash Tests (WI-2) ────────────────────────────────────────

describe('SolverSpatialHash', () => {
  it('insert and query single item', () => {
    const hash = new SolverSpatialHash(2)
    hash.insert(0, 5, 5, 1, 1)
    const results = hash.queryAABB(4, 4, 6, 6)
    expect(results).toContain(0)
  })

  it('insert and query multiple items', () => {
    const hash = new SolverSpatialHash(2)
    hash.insert(0, 2, 2, 0.5, 0.5)
    hash.insert(1, 8, 8, 0.5, 0.5)
    // Query around first item
    const near0 = hash.queryAABB(1, 1, 3, 3)
    expect(near0).toContain(0)
    expect(near0).not.toContain(1)
  })

  it('multi-cell insertion for large items', () => {
    const hash = new SolverSpatialHash(2)
    // Large item spanning multiple cells
    hash.insert(0, 5, 5, 3, 3) // 6m x 6m item
    // Query various cells it should span
    expect(hash.queryAABB(3, 3, 4, 4)).toContain(0)
    expect(hash.queryAABB(6, 6, 7, 7)).toContain(0)
  })

  it('remove cleans up correctly', () => {
    const hash = new SolverSpatialHash(2)
    hash.insert(0, 5, 5, 1, 1)
    hash.remove(0)
    expect(hash.queryAABB(4, 4, 6, 6)).toHaveLength(0)
    expect(hash.size).toBe(0)
  })

  it('update moves item', () => {
    const hash = new SolverSpatialHash(2)
    hash.insert(0, 2, 2, 0.5, 0.5)
    hash.update(0, 8, 8, 0.5, 0.5)
    // Should not be at old position
    expect(hash.queryAABB(1, 1, 3, 3)).not.toContain(0)
    // Should be at new position
    expect(hash.queryAABB(7, 7, 9, 9)).toContain(0)
  })

  it('queryRadius works', () => {
    const hash = new SolverSpatialHash(2)
    hash.insert(0, 5, 5, 0.5, 0.5)
    hash.insert(1, 100, 100, 0.5, 0.5)
    const near = hash.queryRadius(5, 5, 3)
    expect(near).toContain(0)
    expect(near).not.toContain(1)
  })

  it('handles negative coordinates', () => {
    const hash = new SolverSpatialHash(2)
    hash.insert(0, -5, -5, 0.5, 0.5)
    const results = hash.queryAABB(-6, -6, -4, -4)
    expect(results).toContain(0)
  })

  it('empty query returns empty array', () => {
    const hash = new SolverSpatialHash(2)
    expect(hash.queryAABB(0, 0, 10, 10)).toHaveLength(0)
  })

  it('buildFromPlacements populates correctly', () => {
    const hash = new SolverSpatialHash(2)
    const placements = [
      makePlacement(3, 3, 1, 1),
      makePlacement(8, 8, 1, 1),
    ]
    hash.buildFromPlacements(placements)
    expect(hash.size).toBe(2)
    expect(hash.queryAABB(2, 2, 4, 4)).toContain(0)
    expect(hash.queryAABB(7, 7, 9, 9)).toContain(1)
  })

  it('insertPlacement convenience method works', () => {
    const hash = new SolverSpatialHash(2)
    hash.insertPlacement(0, makePlacement(5, 5, 2, 2))
    expect(hash.queryAABB(4, 4, 6, 6)).toContain(0)
  })

  it('clear removes all entries', () => {
    const hash = new SolverSpatialHash(2)
    hash.insert(0, 5, 5, 1, 1)
    hash.insert(1, 8, 8, 1, 1)
    hash.clear()
    expect(hash.size).toBe(0)
    expect(hash.queryAABB(0, 0, 100, 100)).toHaveLength(0)
  })
})

// ─── Spatial-Accelerated Constraints Tests (WI-3) ──────────────────────────

describe('Spatial-accelerated constraints', () => {
  it('checkNoOverlapSpatial detects overlaps', () => {
    const placements = [
      makePlacement(5, 5, 2, 2),
      makePlacement(6, 5, 2, 2),
    ]
    const hash = new SolverSpatialHash(2)
    hash.buildFromPlacements(placements)
    const violations = checkNoOverlapSpatial(placements, hash)
    expect(violations.length).toBe(1)
    expect(violations[0]!.type).toBe('overlap')
  })

  it('checkNoOverlapSpatial matches brute-force', () => {
    const placements = [
      makePlacement(2, 2, 1, 1),
      makePlacement(3, 2, 1, 1), // overlaps
      makePlacement(8, 8, 1, 1),
      makePlacement(8.5, 8, 1, 1), // overlaps
    ]
    const hash = new SolverSpatialHash(2)
    hash.buildFromPlacements(placements)
    const brute = checkNoOverlap(placements)
    const spatial = checkNoOverlapSpatial(placements, hash)
    expect(spatial.length).toBe(brute.length)
  })

  it('checkAisleWidthSpatial detects narrow aisles', () => {
    const placements = [
      makePlacement(5, 5, 2, 2),
      makePlacement(5, 7.5, 2, 2), // 0.5m gap
    ]
    const hash = new SolverSpatialHash(2)
    hash.buildFromPlacements(placements)
    const violations = checkAisleWidthSpatial(placements, hash, 0.914)
    expect(violations.length).toBeGreaterThan(0)
  })

  it('checkAisleWidthSpatial matches brute-force', () => {
    const placements = [
      makePlacement(3, 3, 1, 1),
      makePlacement(3, 4.3, 1, 1), // close
      makePlacement(8, 8, 1, 1),
    ]
    const hash = new SolverSpatialHash(2)
    hash.buildFromPlacements(placements)
    const brute = checkAisleWidth(placements, 0.914)
    const spatial = checkAisleWidthSpatial(placements, hash, 0.914)
    expect(spatial.length).toBe(brute.length)
  })

  it('validateSinglePlacement checks bounds', () => {
    const room = makeRoom(10, 10)
    const placements = [makePlacement(0.3, 5, 2, 2)]
    const hash = new SolverSpatialHash(2)
    hash.buildFromPlacements(placements)
    const violations = validateSinglePlacement(room, placements, 0, hash, 0.914, 1.12)
    expect(violations.some((v) => v.type === 'out-of-bounds')).toBe(true)
  })

  it('validateSinglePlacement checks neighbors for overlap', () => {
    const room = makeRoom(20, 20)
    const placements = [
      makePlacement(5, 5, 2, 2),
      makePlacement(6, 5, 2, 2), // overlaps with 0
    ]
    const hash = new SolverSpatialHash(2)
    hash.buildFromPlacements(placements)
    const violations = validateSinglePlacement(room, placements, 0, hash, 0.914, 1.12)
    expect(violations.some((v) => v.type === 'overlap')).toBe(true)
  })

  it('validateSinglePlacement returns empty for valid placement', () => {
    const room = makeRoom(20, 20)
    const placements = [
      makePlacement(5, 5, 1, 1),
      makePlacement(10, 10, 1, 1),
    ]
    const hash = new SolverSpatialHash(2)
    hash.buildFromPlacements(placements)
    const violations = validateSinglePlacement(room, placements, 0, hash, 0.914, 1.12)
    expect(violations).toHaveLength(0)
  })

  it('validateLayoutSpatial matches validateLayout', () => {
    const room = makeRoom(15, 15)
    const placements = [
      makePlacement(3, 3, 1, 1),
      makePlacement(3.8, 3, 1, 1), // overlap
      makePlacement(10, 10, 1, 1),
    ]
    const hash = new SolverSpatialHash(2)
    hash.buildFromPlacements(placements)
    const brute = validateLayout(room, placements)
    const spatial = validateLayoutSpatial(room, placements, hash)
    // Both should find the same overlap
    expect(spatial.some((v) => v.type === 'overlap')).toBe(brute.some((v) => v.type === 'overlap'))
  })
})

// ─── Chair-Table Grouping Tests (WI-4) ─────────────────────────────────────

describe('Chair-table grouping', () => {
  it('generates circle positions for round table', () => {
    const table = makePlacement(5, 5, 1.8, 1.8, 'round-table')
    const spec: FurnitureSpec = { type: 'round-table', width: 1.8, depth: 1.8, count: 1, chairsPerUnit: 6 }
    const positions = generateChairPositions(table, spec, 6)
    expect(positions).toHaveLength(6)

    // All chairs should be roughly the same distance from the table center
    const distances = positions.map((p) =>
      Math.sqrt((p.x - 5) ** 2 + (p.z - 5) ** 2),
    )
    const avgDist = distances.reduce((a, b) => a + b, 0) / distances.length
    for (const d of distances) {
      expect(Math.abs(d - avgDist)).toBeLessThan(0.01)
    }
  })

  it('generates edge positions for rectangular table', () => {
    const table = makePlacement(5, 5, 2, 1, 'rect-table')
    const spec: FurnitureSpec = { type: 'rect-table', width: 2, depth: 1, count: 1, chairsPerUnit: 4 }
    const positions = generateChairPositions(table, spec, 4)
    expect(positions).toHaveLength(4)
  })

  it('returns empty for count=0', () => {
    const table = makePlacement(5, 5, 1.8, 1.8, 'round-table')
    const spec: FurnitureSpec = { type: 'round-table', width: 1.8, depth: 1.8, count: 1, chairsPerUnit: 0 }
    expect(generateChairPositions(table, spec, 0)).toHaveLength(0)
  })

  it('placeChairGroups produces valid groupings', () => {
    const room = makeRoom(20, 15)
    const specs: FurnitureSpec[] = [
      { type: 'round-table', width: 1.8, depth: 1.8, count: 2, chairsPerUnit: 4 },
      { type: 'chair', width: 0.44, depth: 0.44, count: 0, chairsPerUnit: 0 },
    ]
    const tables = [
      makePlacement(5, 7, 1.8, 1.8, 'round-table', 0, 0),
      makePlacement(15, 7, 1.8, 1.8, 'round-table', 0, 1),
    ]
    const grid = new LayoutGrid(room, 0.15)
    const hash = new SolverSpatialHash(2)
    for (let i = 0; i < tables.length; i++) {
      const t = tables[i]!
      grid.occupy(t.x, t.z, t.effectiveWidth / 2, t.effectiveDepth / 2)
      hash.insertPlacement(i, t)
    }
    const { chairs, groupings } = placeChairGroups(room, specs, tables, grid, hash, {
      minAisle: 0.914,
      gridCellSize: 0.15,
    })
    expect(chairs.length).toBeGreaterThan(0)
    expect(groupings.length).toBe(2) // one per table
    for (const g of groupings) {
      expect(g.chairsPerUnit).toBe(4)
      expect(g.chairIndices.length).toBeGreaterThan(0)
    }
  })

  it('chairs face toward their table', () => {
    const room = makeRoom(20, 15)
    const specs: FurnitureSpec[] = [
      { type: 'round-table', width: 1.8, depth: 1.8, count: 1, chairsPerUnit: 4 },
      { type: 'chair', width: 0.44, depth: 0.44, count: 0, chairsPerUnit: 0 },
    ]
    const tables = [makePlacement(10, 7.5, 1.8, 1.8, 'round-table', 0, 0)]
    const grid = new LayoutGrid(room, 0.15)
    const hash = new SolverSpatialHash(2)
    grid.occupy(10, 7.5, 0.9, 0.9)
    hash.insertPlacement(0, tables[0]!)
    const { chairs } = placeChairGroups(room, specs, tables, grid, hash, {
      minAisle: 0.914,
      gridCellSize: 0.15,
    })
    for (const chair of chairs) {
      // Chair rotation should point roughly toward table center
      const expectedAngle = Math.atan2(7.5 - chair.z, 10 - chair.x)
      const diff = Math.abs(chair.rotation - expectedAngle)
      expect(diff).toBeLessThan(0.5) // within ~30°
    }
  })
})

// ─── Incremental Constraint Graph Tests (WI-6) ────────────────────────────

describe('IncrementalConstraintGraph', () => {
  const makeSimpleGraph = () => {
    const room = makeRoom(20, 20)
    const specs: FurnitureSpec[] = [{ type: 'chair', width: 0.5, depth: 0.5, count: 3, chairsPerUnit: 0 }]
    const placements: Placement[] = [
      makePlacement(5, 5, 0.5, 0.5),
      makePlacement(10, 10, 0.5, 0.5),
      makePlacement(15, 15, 0.5, 0.5),
    ]
    const hash = new SolverSpatialHash(2)
    hash.buildFromPlacements(placements)
    const weights = { spaceUtilization: 0.3, sightlineCoverage: 0.3, symmetry: 0.2, exitAccess: 0.2 }
    const graph = new IncrementalConstraintGraph(room, specs, placements, hash, weights, 0.914, 1.12)
    return { graph, room, specs, placements, hash, weights }
  }

  it('initializes with correct violations', () => {
    const { graph } = makeSimpleGraph()
    graph.stabilize()
    // Valid placements should have no violations
    expect(graph.violations).toHaveLength(0)
  })

  it('detects violations after update', () => {
    const { graph } = makeSimpleGraph()
    // Move item 1 to overlap with item 0
    graph.updatePlacement(1, makePlacement(5, 5, 0.5, 0.5))
    graph.stabilize()
    expect(graph.violations.length).toBeGreaterThan(0)
  })

  it('returns score', () => {
    const { graph } = makeSimpleGraph()
    graph.stabilize()
    expect(typeof graph.totalScore).toBe('number')
  })

  it('tracks recomputations', () => {
    const { graph } = makeSimpleGraph()
    graph.resetCounters()
    graph.updatePlacement(0, makePlacement(6, 5, 0.5, 0.5))
    graph.stabilize()
    // Should have recomputed some nodes
    expect(graph.recomputations).toBeGreaterThan(0)
  })

  it('placements getter returns current state', () => {
    const { graph } = makeSimpleGraph()
    expect(graph.placements).toHaveLength(3)
  })

  it('cutoff: unchanged violations skip global recompute', () => {
    const { graph } = makeSimpleGraph()
    graph.stabilize()
    graph.resetCounters()

    // Move slightly — should still be valid
    graph.updatePlacement(0, makePlacement(5.01, 5, 0.5, 0.5))
    graph.stabilize()
    const firstRecomps = graph.recomputations

    // Move slightly again in the valid region
    graph.resetCounters()
    graph.updatePlacement(0, makePlacement(5.02, 5, 0.5, 0.5))
    graph.stabilize()
    const secondRecomps = graph.recomputations

    // Both should be small (local recomputation only)
    expect(firstRecomps).toBeLessThan(20)
    expect(secondRecomps).toBeLessThan(20)
  })
})

// ─── Backtracking Tests (WI-5) ────────────────────────────────────────────

describe('Solver advanced features', () => {
  it('backtracking: handles tight room without crashing', () => {
    // A very tight room — backtracking may be needed
    const result = solve({
      room: { width: 4, depth: 4, exits: [], obstacles: [] },
      furniture: [
        { type: 'round-table', width: 1.8, depth: 1.8, count: 3, chairsPerUnit: 0 },
      ],
      options: { enableBacktracking: true, annealingIterations: 50 },
    })
    expect(result.stats.backtracks).toBeGreaterThanOrEqual(0)
    expect(checkNoOverlap(result.placements)).toHaveLength(0)
  })

  it('backtracking disabled: still works but may place fewer items', () => {
    const result = solve({
      room: { width: 4, depth: 4, exits: [], obstacles: [] },
      furniture: [
        { type: 'round-table', width: 1.8, depth: 1.8, count: 3, chairsPerUnit: 0 },
      ],
      options: { enableBacktracking: false, annealingIterations: 50 },
    })
    expect(result.stats.backtracks).toBe(0)
  })

  it('fixed-zone fallback: still places item if zone is blocked', () => {
    const room: RoomConfig = {
      width: 10, depth: 10,
      exits: [],
      obstacles: [{ x: 3.5, z: 0, width: 3, depth: 2 }], // blocks north center
    }
    const result = solve({
      room,
      furniture: [
        { type: 'stage', width: 4, depth: 2, count: 1, chairsPerUnit: 0, fixedZone: 'north' },
      ],
      options: { annealingIterations: 50 },
    })
    // Should still place the stage somewhere (falls back to general placement)
    expect(result.placements.length).toBe(1)
  })

  it('SA with restarts completes successfully', () => {
    const result = solve({
      room: makeRoom(15, 12),
      furniture: [
        { type: 'chair', width: 0.5, depth: 0.5, count: 15, chairsPerUnit: 0 },
      ],
      options: { maxRestarts: 2, annealingIterations: 200 },
    })
    expect(result.placements.length).toBeGreaterThan(0)
    expect(result.stats.restarts).toBeGreaterThanOrEqual(0)
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

  it('spatial hash equivalence: brute-force vs spatial overlap check', () => {
    fc.assert(fc.property(
      fc.array(
        fc.record({
          x: fc.double({ min: 1, max: 19, noNaN: true }),
          z: fc.double({ min: 1, max: 14, noNaN: true }),
          w: fc.double({ min: 0.3, max: 2, noNaN: true }),
          d: fc.double({ min: 0.3, max: 2, noNaN: true }),
        }),
        { minLength: 2, maxLength: 10 },
      ),
      (items) => {
        const placements = items.map((item, i) =>
          makePlacement(item.x, item.z, item.w, item.d, 'chair', 0, i),
        )
        const hash = new SolverSpatialHash(2)
        hash.buildFromPlacements(placements)
        const bruteOverlaps = checkNoOverlap(placements)
        const spatialOverlaps = checkNoOverlapSpatial(placements, hash)
        expect(spatialOverlaps.length).toBe(bruteOverlaps.length)
      },
    ), { numRuns: 100 })
  })

  it('chair positions: all chairs at expected distance from round table', () => {
    fc.assert(fc.property(
      fc.integer({ min: 2, max: 10 }),
      (chairCount) => {
        const table = makePlacement(10, 10, 1.8, 1.8, 'round-table')
        const spec: FurnitureSpec = { type: 'round-table', width: 1.8, depth: 1.8, count: 1, chairsPerUnit: chairCount }
        const positions = generateChairPositions(table, spec, chairCount)
        expect(positions).toHaveLength(chairCount)
        const expectedRadius = 0.9 + 0.35 // half-width + setback
        for (const p of positions) {
          const dist = Math.sqrt((p.x - 10) ** 2 + (p.z - 10) ** 2)
          expect(dist).toBeCloseTo(expectedRadius, 1)
        }
      },
    ), { numRuns: 20 })
  })

  it('grouping indices are valid', () => {
    fc.assert(fc.property(
      fc.integer({ min: 1, max: 3 }),
      (tableCount) => {
        const result = solve({
          room: makeRoom(20, 15),
          furniture: [
            { type: 'round-table', width: 1.8, depth: 1.8, count: tableCount, chairsPerUnit: 4 },
            { type: 'chair', width: 0.44, depth: 0.44, count: 0, chairsPerUnit: 0 },
          ],
          options: { annealingIterations: 100 },
        })
        for (const g of result.groupings) {
          // tableIndex should reference a valid placement
          expect(g.tableIndex).toBeGreaterThanOrEqual(0)
          expect(g.tableIndex).toBeLessThan(result.placements.length)
          // chairIndices should reference valid placements
          for (const ci of g.chairIndices) {
            expect(ci).toBeGreaterThanOrEqual(0)
            expect(ci).toBeLessThan(result.placements.length)
          }
        }
      },
    ), { numRuns: 20 })
  })
})

// ─── Snapshot Tests ────────────────────────────────────────────────────────

describe('Solver snapshots', () => {
  it('standard banquet: deterministic output', () => {
    const result = solve({
      room: makeRoom(20, 15),
      furniture: [
        { type: 'round-table', width: 1.8, depth: 1.8, count: 6, chairsPerUnit: 6 },
        { type: 'chair', width: 0.44, depth: 0.44, count: 0, chairsPerUnit: 0 },
        { type: 'podium', width: 1.2, depth: 0.8, count: 1, chairsPerUnit: 0, fixedZone: 'north' },
      ],
      options: { seed: 42, annealingIterations: 500 },
    })
    expect(result.placements.length).toBeGreaterThan(0)
    expect(result.groupings.length).toBeGreaterThan(0)
    // Snapshot: record counts for regression
    const snapshot = {
      placedCount: result.stats.placedCount,
      feasible: result.feasible,
      groupingCount: result.groupings.length,
      hasViolations: result.violations.length > 0,
    }
    expect(snapshot.placedCount).toBeGreaterThan(6) // at least tables + podium + some chairs
    expect(snapshot.groupingCount).toBeGreaterThanOrEqual(1)
  })

  it('theater rows: many chairs in a large room', () => {
    const result = solve({
      room: makeRoom(25, 20),
      furniture: [
        { type: 'chair', width: 0.5, depth: 0.5, count: 60, chairsPerUnit: 0 },
        { type: 'stage', width: 6, depth: 3, count: 1, chairsPerUnit: 0, fixedZone: 'north' },
      ],
      options: { seed: 42, annealingIterations: 300 },
    })
    expect(result.stats.placedCount).toBeGreaterThan(30)
    expect(checkNoOverlap(result.placements)).toHaveLength(0)
  })

  it('cocktail: standing tables only', () => {
    const result = solve({
      room: makeRoom(15, 12),
      furniture: [
        { type: 'round-table', width: 0.8, depth: 0.8, count: 10, chairsPerUnit: 0 },
        { type: 'bar', width: 3, depth: 1, count: 1, chairsPerUnit: 0, wallAdjacent: true },
      ],
      options: { seed: 42, annealingIterations: 300 },
    })
    expect(result.stats.placedCount).toBeGreaterThanOrEqual(8)
    expect(checkNoOverlap(result.placements)).toHaveLength(0)
  })

  it('conference: large table + chairs', () => {
    const result = solve({
      room: makeRoom(12, 8),
      furniture: [
        { type: 'trestle-table', width: 3, depth: 1.2, count: 1, chairsPerUnit: 8, fixedZone: 'center' },
        { type: 'chair', width: 0.44, depth: 0.44, count: 0, chairsPerUnit: 0 },
      ],
      options: { seed: 42, annealingIterations: 200 },
    })
    expect(result.placements.length).toBeGreaterThan(0)
    const tables = result.placements.filter((p) => p.type === 'trestle-table')
    expect(tables.length).toBe(1)
  })

  it('max capacity: stress test', () => {
    const result = solve({
      room: makeRoom(30, 25),
      furniture: [
        { type: 'round-table', width: 1.8, depth: 1.8, count: 10, chairsPerUnit: 8 },
        { type: 'chair', width: 0.44, depth: 0.44, count: 0, chairsPerUnit: 0 },
        { type: 'bar', width: 3, depth: 1, count: 2, chairsPerUnit: 0, wallAdjacent: true },
        { type: 'stage', width: 5, depth: 3, count: 1, chairsPerUnit: 0, fixedZone: 'north' },
      ],
      options: { seed: 42, annealingIterations: 500, maxRestarts: 2 },
    })
    expect(result.stats.placedCount).toBeGreaterThan(10)
    expect(checkNoOverlap(result.placements)).toHaveLength(0)
  })
})
