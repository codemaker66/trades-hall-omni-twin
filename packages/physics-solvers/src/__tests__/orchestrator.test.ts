/**
 * Comprehensive tests for PS-12 (Orchestrator) and PS-11 (Layout Generation).
 *
 * Tests layout generation templates, LLM/diffusion fallback stubs,
 * perturbation utility, state <-> items conversion, and the full
 * VenueSolverPipeline orchestrator.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import type {
  RoomBoundary,
  FurnitureItem,
  LayoutWeights,
  Layout,
  EventSpec,
  RoomSpec,
  TimeslotSpec,
  ParetoSolution,
  ScheduleResult,
} from '../types.js'
import { ItemType, DEFAULT_WEIGHTS, createPRNG } from '../types.js'

// ---------------------------------------------------------------------------
// Mocks — isolate orchestrator from heavy solver dependencies
// ---------------------------------------------------------------------------

vi.mock('../sa.js', () => ({
  simulatedAnnealing: vi.fn(),
}))

vi.mock('../parallel-tempering.js', () => ({
  parallelTempering: vi.fn(
    (initialState: Float64Array, _energyFn: unknown, _neighborFn: unknown, _config: unknown) => ({
      bestEnergy: 50.0,
      bestState: new Float64Array(initialState),
      replicaEnergies: new Float64Array(8),
      swapAcceptanceRates: new Float64Array(7),
      energyTraces: [],
    }),
  ),
}))

vi.mock('../energy/layout-energy.js', () => ({
  computeLayoutEnergy: vi.fn(
    (_items: FurnitureItem[], _room: RoomBoundary, _weights: LayoutWeights, _cap: number) => 42.0,
  ),
  generateLayoutNeighbor: vi.fn((items: FurnitureItem[], _rng: unknown) =>
    items.map((item) => ({ ...item, x: item.x + 0.1 })),
  ),
}))

vi.mock('../mip-scheduler.js', () => ({
  solveScheduleMIP: vi.fn(
    async (_events: EventSpec[], _rooms: RoomSpec[], _timeslots: TimeslotSpec[]): Promise<ScheduleResult> => ({
      assignments: [{ eventId: 'e1', roomId: 'r1', timeslotId: 't1' }],
      objectiveValue: 5.0,
      feasible: true,
      solveDurationMs: 10,
    }),
  ),
}))

vi.mock('../mcmc.js', () => ({
  sampleLayoutsMH: vi.fn(
    (initialState: Float64Array, _config: unknown, _energyFn: unknown, _neighborFn: unknown) => ({
      samples: [new Float64Array(initialState), new Float64Array(initialState)],
      energies: new Float64Array([40.0, 41.0]),
      acceptanceRate: 0.35,
    }),
  ),
}))

vi.mock('../nsga2.js', () => ({
  nsga2: vi.fn(
    (_pop: Float64Array[], _objectiveFn: unknown, _config: unknown): ParetoSolution[] => [
      {
        state: new Float64Array([1, 2, 0]),
        objectives: new Float64Array([10, 20, 30]),
        frontRank: 0,
        crowdingDistance: Infinity,
      },
    ],
  ),
}))

vi.mock('../cmaes.js', () => ({
  cmaes: vi.fn(
    (initialMean: Float64Array, _config: unknown, _energyFn: unknown) => ({
      bestState: new Float64Array(initialMean),
      bestEnergy: 25.0,
      evaluations: 500,
    }),
  ),
}))

// ---------------------------------------------------------------------------
// Imports (AFTER mocks are registered)
// ---------------------------------------------------------------------------

import {
  generateTemplateLayout,
  generateLayoutLLM,
  generateLayoutDiffusion,
  perturbLayout,
} from '../layout-generation.js'
import type { LayoutStyle, LLMLayoutOptions, DiffusionLayoutOptions } from '../layout-generation.js'

import {
  VenueSolverPipeline,
  itemsToState,
  stateToItems,
} from '../orchestrator.js'

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

const testRoom: RoomBoundary = {
  vertices: new Float64Array([0, 0, 60, 0, 60, 40, 0, 40]),
  exits: new Float64Array([30, 0, 6]),
  width: 60,
  height: 40,
}

/** A small room to stress edge cases */
const smallRoom: RoomBoundary = {
  vertices: new Float64Array([0, 0, 20, 0, 20, 15, 0, 15]),
  exits: new Float64Array([10, 0, 4]),
  width: 20,
  height: 15,
}

/** Helper: count items of a specific type */
function countByType(items: FurnitureItem[], type: ItemType): number {
  return items.filter((i) => i.itemType === type).length
}

/** Helper: total seat count */
function totalSeats(items: FurnitureItem[]): number {
  return items.reduce((sum, i) => sum + i.seats, 0)
}

/** Helper: verify all items are within room bounds (with some margin tolerance) */
function allWithinBounds(items: FurnitureItem[], room: RoomBoundary, margin = 0): boolean {
  return items.every(
    (i) =>
      i.x - i.width / 2 >= -margin &&
      i.y - i.depth / 2 >= -margin &&
      i.x + i.width / 2 <= room.width + margin &&
      i.y + i.depth / 2 <= room.height + margin,
  )
}

// =========================================================================
// Layout Generation Tests (PS-11)
// =========================================================================

describe('PS-11: Layout Generation', () => {
  // -----------------------------------------------------------------------
  // Template: Theater
  // -----------------------------------------------------------------------
  describe('generateTemplateLayout("theater")', () => {
    it('returns items with chairs and a stage', () => {
      const items = generateTemplateLayout(testRoom, 'theater', 100)

      expect(items.length).toBeGreaterThan(0)

      const stages = items.filter((i) => i.itemType === ItemType.Stage)
      expect(stages.length).toBe(1)

      const chairs = items.filter((i) => i.itemType === ItemType.Chair)
      expect(chairs.length).toBeGreaterThan(0)
    })

    it('stage is positioned at the front of the room', () => {
      const items = generateTemplateLayout(testRoom, 'theater', 100)
      const stage = items.find((i) => i.itemType === ItemType.Stage)!
      // Stage should be in the first half of the room (y-wise)
      expect(stage.y).toBeLessThan(testRoom.height / 2)
    })

    it('chairs have 1 seat each', () => {
      const items = generateTemplateLayout(testRoom, 'theater', 50)
      const chairs = items.filter((i) => i.itemType === ItemType.Chair)
      for (const chair of chairs) {
        expect(chair.seats).toBe(1)
      }
    })

    it('respects capacity target (does not exceed it)', () => {
      const items = generateTemplateLayout(testRoom, 'theater', 80)
      const seated = totalSeats(items)
      // Should produce at most the requested capacity worth of chairs
      expect(seated).toBeLessThanOrEqual(80)
      // But should produce a reasonable amount (at least 50% of target for a 60x40 room)
      expect(seated).toBeGreaterThanOrEqual(40)
    })

    it('all items are within room bounds', () => {
      const items = generateTemplateLayout(testRoom, 'theater', 60)
      expect(allWithinBounds(items, testRoom, 1)).toBe(true)
    })
  })

  // -----------------------------------------------------------------------
  // Template: Banquet
  // -----------------------------------------------------------------------
  describe('generateTemplateLayout("banquet")', () => {
    it('returns items with round tables and a bar', () => {
      const items = generateTemplateLayout(testRoom, 'banquet', 80)

      expect(items.length).toBeGreaterThan(0)

      const roundTables = countByType(items, ItemType.RoundTable)
      expect(roundTables).toBeGreaterThan(0)

      const bars = countByType(items, ItemType.Bar)
      expect(bars).toBeGreaterThanOrEqual(1)
    })

    it('includes a dance floor for capacity > 50', () => {
      const items = generateTemplateLayout(testRoom, 'banquet', 80)
      const danceFloors = countByType(items, ItemType.DanceFloor)
      expect(danceFloors).toBe(1)
    })

    it('does not include a dance floor for capacity <= 50', () => {
      const items = generateTemplateLayout(testRoom, 'banquet', 40)
      const danceFloors = countByType(items, ItemType.DanceFloor)
      expect(danceFloors).toBe(0)
    })

    it('round tables seat up to 8 each', () => {
      const items = generateTemplateLayout(testRoom, 'banquet', 80)
      const tables = items.filter((i) => i.itemType === ItemType.RoundTable)
      for (const table of tables) {
        expect(table.seats).toBeLessThanOrEqual(8)
        expect(table.seats).toBeGreaterThan(0)
      }
    })
  })

  // -----------------------------------------------------------------------
  // Template: Classroom
  // -----------------------------------------------------------------------
  describe('generateTemplateLayout("classroom")', () => {
    it('returns items with rectangular tables and a podium', () => {
      const items = generateTemplateLayout(testRoom, 'classroom', 30)

      expect(items.length).toBeGreaterThan(0)

      const rectTables = countByType(items, ItemType.RectTable)
      expect(rectTables).toBeGreaterThan(0)

      const podiums = countByType(items, ItemType.Podium)
      expect(podiums).toBe(1)
    })

    it('podium is at the front of the room', () => {
      const items = generateTemplateLayout(testRoom, 'classroom', 30)
      const podium = items.find((i) => i.itemType === ItemType.Podium)!
      expect(podium.y).toBeLessThan(testRoom.height / 3)
    })

    it('tables have 2 seats each (classroom style)', () => {
      const items = generateTemplateLayout(testRoom, 'classroom', 30)
      const tables = items.filter((i) => i.itemType === ItemType.RectTable)
      for (const table of tables) {
        expect(table.seats).toBe(2)
      }
    })
  })

  // -----------------------------------------------------------------------
  // Template: Cocktail
  // -----------------------------------------------------------------------
  describe('generateTemplateLayout("cocktail")', () => {
    it('returns items with round tables (high-tops) and bars', () => {
      const items = generateTemplateLayout(testRoom, 'cocktail', 60)

      expect(items.length).toBeGreaterThan(0)

      const bars = countByType(items, ItemType.Bar)
      expect(bars).toBe(2) // Two bars along walls

      const tables = countByType(items, ItemType.RoundTable)
      expect(tables).toBeGreaterThan(0)
    })

    it('includes a service station', () => {
      const items = generateTemplateLayout(testRoom, 'cocktail', 60)
      const stations = countByType(items, ItemType.ServiceStation)
      expect(stations).toBe(1)
    })

    it('high-top tables seat 4 each', () => {
      const items = generateTemplateLayout(testRoom, 'cocktail', 40)
      const tables = items.filter((i) => i.itemType === ItemType.RoundTable)
      for (const table of tables) {
        expect(table.seats).toBe(4)
      }
    })

    it('produces roughly ceil(capacity/4) high-top tables', () => {
      const capacity = 48
      const items = generateTemplateLayout(testRoom, 'cocktail', capacity)
      const tables = items.filter((i) => i.itemType === ItemType.RoundTable)
      expect(tables.length).toBe(Math.ceil(capacity / 4))
    })
  })

  // -----------------------------------------------------------------------
  // Template: Ceremony
  // -----------------------------------------------------------------------
  describe('generateTemplateLayout("ceremony")', () => {
    it('returns items with chairs in rows and a stage/altar', () => {
      const items = generateTemplateLayout(testRoom, 'ceremony', 80)

      expect(items.length).toBeGreaterThan(0)

      const stages = countByType(items, ItemType.Stage)
      expect(stages).toBe(1)

      const chairs = countByType(items, ItemType.Chair)
      expect(chairs).toBeGreaterThan(0)
    })

    it('altar/stage is at the front', () => {
      const items = generateTemplateLayout(testRoom, 'ceremony', 80)
      const stage = items.find((i) => i.itemType === ItemType.Stage)!
      expect(stage.y).toBeLessThan(testRoom.height / 3)
    })

    it('chairs have 1 seat each', () => {
      const items = generateTemplateLayout(testRoom, 'ceremony', 60)
      const chairs = items.filter((i) => i.itemType === ItemType.Chair)
      for (const chair of chairs) {
        expect(chair.seats).toBe(1)
      }
    })

    it('does not exceed capacity', () => {
      const items = generateTemplateLayout(testRoom, 'ceremony', 50)
      const seated = totalSeats(items)
      expect(seated).toBeLessThanOrEqual(50)
    })
  })

  // -----------------------------------------------------------------------
  // Template: Conference
  // -----------------------------------------------------------------------
  describe('generateTemplateLayout("conference")', () => {
    it('returns items with rectangular tables for small capacity (<= 16)', () => {
      const items = generateTemplateLayout(testRoom, 'conference', 12)

      expect(items.length).toBeGreaterThan(0)

      const rectTables = items.filter((i) => i.itemType === ItemType.RectTable)
      // Single large conference table
      expect(rectTables.length).toBe(1)
    })

    it('returns U-shape tables for larger capacity (> 16)', () => {
      const items = generateTemplateLayout(testRoom, 'conference', 30)

      const rectTables = items.filter((i) => i.itemType === ItemType.RectTable)
      // U-shape: head + left side + right side = 3 tables
      expect(rectTables.length).toBe(3)
    })

    it('includes a podium and AV booth', () => {
      const items = generateTemplateLayout(testRoom, 'conference', 20)

      const podiums = countByType(items, ItemType.Podium)
      expect(podiums).toBe(1)

      const avBooths = countByType(items, ItemType.AVBooth)
      expect(avBooths).toBe(1)
    })

    it('podium is at front, AV booth at back', () => {
      const items = generateTemplateLayout(testRoom, 'conference', 20)
      const podium = items.find((i) => i.itemType === ItemType.Podium)!
      const avBooth = items.find((i) => i.itemType === ItemType.AVBooth)!

      expect(podium.y).toBeLessThan(testRoom.height / 2)
      expect(avBooth.y).toBeGreaterThan(testRoom.height / 2)
    })
  })

  // -----------------------------------------------------------------------
  // Determinism with seed
  // -----------------------------------------------------------------------
  describe('template determinism', () => {
    const styles: LayoutStyle[] = ['theater', 'banquet', 'classroom', 'cocktail', 'ceremony', 'conference']

    it.each(styles)('generateTemplateLayout("%s") is deterministic with same seed', (style) => {
      const a = generateTemplateLayout(testRoom, style, 50, 123)
      const b = generateTemplateLayout(testRoom, style, 50, 123)
      expect(a).toEqual(b)
    })
  })

  // -----------------------------------------------------------------------
  // All items have valid properties
  // -----------------------------------------------------------------------
  describe('item property validation', () => {
    const styles: LayoutStyle[] = ['theater', 'banquet', 'classroom', 'cocktail', 'ceremony', 'conference']

    it.each(styles)('all items from "%s" have positive width and depth', (style) => {
      const items = generateTemplateLayout(testRoom, style, 50)
      for (const item of items) {
        expect(item.width).toBeGreaterThan(0)
        expect(item.depth).toBeGreaterThan(0)
      }
    })

    it.each(styles)('all items from "%s" have valid ItemType', (style) => {
      const items = generateTemplateLayout(testRoom, style, 50)
      const validTypes = Object.values(ItemType)
      for (const item of items) {
        expect(validTypes).toContain(item.itemType)
      }
    })

    it.each(styles)('all items from "%s" have non-negative seats', (style) => {
      const items = generateTemplateLayout(testRoom, style, 50)
      for (const item of items) {
        expect(item.seats).toBeGreaterThanOrEqual(0)
      }
    })
  })

  // -----------------------------------------------------------------------
  // perturbLayout
  // -----------------------------------------------------------------------
  describe('perturbLayout', () => {
    it('returns modified items with same length', () => {
      const items = generateTemplateLayout(testRoom, 'banquet', 40)
      const rng = createPRNG(99)
      const perturbed = perturbLayout(items, testRoom, 1.0, rng)

      expect(perturbed.length).toBe(items.length)
    })

    it('preserves itemType, width, depth, and seats', () => {
      const items = generateTemplateLayout(testRoom, 'theater', 50)
      const rng = createPRNG(77)
      const perturbed = perturbLayout(items, testRoom, 2.0, rng)

      for (let i = 0; i < items.length; i++) {
        expect(perturbed[i]!.itemType).toBe(items[i]!.itemType)
        expect(perturbed[i]!.width).toBe(items[i]!.width)
        expect(perturbed[i]!.depth).toBe(items[i]!.depth)
        expect(perturbed[i]!.seats).toBe(items[i]!.seats)
      }
    })

    it('modifies x, y, or rotation for at least some items', () => {
      const items = generateTemplateLayout(testRoom, 'classroom', 30)
      const rng = createPRNG(55)
      const perturbed = perturbLayout(items, testRoom, 2.0, rng)

      let anyDifference = false
      for (let i = 0; i < items.length; i++) {
        if (
          perturbed[i]!.x !== items[i]!.x ||
          perturbed[i]!.y !== items[i]!.y ||
          perturbed[i]!.rotation !== items[i]!.rotation
        ) {
          anyDifference = true
          break
        }
      }
      expect(anyDifference).toBe(true)
    })

    it('clamps items within room bounds', () => {
      const items = generateTemplateLayout(smallRoom, 'cocktail', 20)
      const rng = createPRNG(11)
      // Use a very large magnitude to try to push items out of bounds
      const perturbed = perturbLayout(items, smallRoom, 50.0, rng)

      for (const item of perturbed) {
        expect(item.x).toBeGreaterThanOrEqual(item.width / 2 + 1)
        expect(item.y).toBeGreaterThanOrEqual(item.depth / 2 + 1)
        expect(item.x).toBeLessThanOrEqual(smallRoom.width - item.width / 2 - 1)
        expect(item.y).toBeLessThanOrEqual(smallRoom.height - item.depth / 2 - 1)
      }
    })

    it('magnitude=0 returns items at original positions', () => {
      const items = generateTemplateLayout(testRoom, 'theater', 30)
      const rng = createPRNG(42)
      const perturbed = perturbLayout(items, testRoom, 0, rng)

      for (let i = 0; i < items.length; i++) {
        // With magnitude 0, perturbation is 0 for position (rotation might still be 0)
        expect(perturbed[i]!.x).toBeCloseTo(items[i]!.x, 5)
        expect(perturbed[i]!.y).toBeCloseTo(items[i]!.y, 5)
        expect(perturbed[i]!.rotation).toBeCloseTo(items[i]!.rotation, 5)
      }
    })

    it('returns a new array (does not mutate original)', () => {
      const items = generateTemplateLayout(testRoom, 'banquet', 40)
      const originalX = items.map((i) => i.x)
      const rng = createPRNG(10)
      perturbLayout(items, testRoom, 3.0, rng)

      for (let i = 0; i < items.length; i++) {
        expect(items[i]!.x).toBe(originalX[i])
      }
    })
  })

  // -----------------------------------------------------------------------
  // generateLayoutLLM (fallback path, no API)
  // -----------------------------------------------------------------------
  describe('generateLayoutLLM', () => {
    it('falls back to template parsing from description (theater)', async () => {
      const items = await generateLayoutLLM({
        description: 'A theater setup for 80 people with a presentation area',
        room: testRoom,
      })

      expect(items.length).toBeGreaterThan(0)
      // Should parse "theater" from "theater setup" keyword
      const stages = countByType(items, ItemType.Stage)
      expect(stages).toBe(1)
      const chairs = countByType(items, ItemType.Chair)
      expect(chairs).toBeGreaterThan(0)
    })

    it('falls back to template parsing from description (banquet)', async () => {
      const items = await generateLayoutLLM({
        description: 'A banquet dinner for 60 guests',
        room: testRoom,
      })

      expect(items.length).toBeGreaterThan(0)
      const tables = countByType(items, ItemType.RoundTable)
      expect(tables).toBeGreaterThan(0)
    })

    it('falls back to template parsing from description (classroom)', async () => {
      const items = await generateLayoutLLM({
        description: 'A training workshop for 24 attendees',
        room: testRoom,
      })

      expect(items.length).toBeGreaterThan(0)
      const rectTables = countByType(items, ItemType.RectTable)
      expect(rectTables).toBeGreaterThan(0)
    })

    it('falls back to template parsing from description (cocktail)', async () => {
      const items = await generateLayoutLLM({
        description: 'A cocktail reception for 100 guests',
        room: testRoom,
      })

      expect(items.length).toBeGreaterThan(0)
      const bars = countByType(items, ItemType.Bar)
      expect(bars).toBeGreaterThan(0)
    })

    it('falls back to template parsing from description (ceremony)', async () => {
      const items = await generateLayoutLLM({
        description: 'A wedding ceremony for 120 guests',
        room: testRoom,
      })

      expect(items.length).toBeGreaterThan(0)
      const stages = countByType(items, ItemType.Stage)
      expect(stages).toBe(1)
      const chairs = countByType(items, ItemType.Chair)
      expect(chairs).toBeGreaterThan(0)
    })

    it('falls back to template parsing from description (conference)', async () => {
      const items = await generateLayoutLLM({
        description: 'A boardroom meeting for 10 people',
        room: testRoom,
      })

      expect(items.length).toBeGreaterThan(0)
      const rectTables = countByType(items, ItemType.RectTable)
      expect(rectTables).toBeGreaterThan(0)
    })

    it('defaults to banquet with 100 capacity when no keywords match', async () => {
      const items = await generateLayoutLLM({
        description: 'Something completely different',
        room: testRoom,
      })

      expect(items.length).toBeGreaterThan(0)
      // Default is banquet, which uses round tables
      const roundTables = countByType(items, ItemType.RoundTable)
      expect(roundTables).toBeGreaterThan(0)
    })

    it('parses capacity from description', async () => {
      const items = await generateLayoutLLM({
        description: 'A theater presentation for 30 people',
        room: testRoom,
      })

      // Should have approximately 30 chairs (may be a bit less due to room constraints)
      const chairs = countByType(items, ItemType.Chair)
      expect(chairs).toBeGreaterThan(0)
      expect(chairs).toBeLessThanOrEqual(30)
    })
  })

  // -----------------------------------------------------------------------
  // generateLayoutDiffusion (fallback path, no API)
  // -----------------------------------------------------------------------
  describe('generateLayoutDiffusion', () => {
    it('falls back to perturbation (returns base + perturbed layouts)', async () => {
      const options: DiffusionLayoutOptions = {
        room: testRoom,
        itemSpecs: [
          { type: ItemType.Chair, count: 20 },
          { type: ItemType.RoundTable, count: 5 },
        ],
        style: 'banquet',
        nLayouts: 4,
      }

      const layouts = await generateLayoutDiffusion(options)

      expect(layouts.length).toBe(4)
      // Each layout should have the same number of items
      const firstLength = layouts[0]!.length
      expect(firstLength).toBeGreaterThan(0)
    })

    it('first layout is the unperturbed template', async () => {
      const options: DiffusionLayoutOptions = {
        room: testRoom,
        itemSpecs: [{ type: ItemType.Chair, count: 25 }],
        style: 'theater',
        nLayouts: 3,
      }

      const layouts = await generateLayoutDiffusion(options)
      const totalItems = options.itemSpecs.reduce((s, spec) => s + spec.count, 0)
      const base = generateTemplateLayout(testRoom, 'theater', totalItems)

      // First layout should be the base template
      expect(layouts[0]).toEqual(base)
    })

    it('subsequent layouts differ from the first', async () => {
      const options: DiffusionLayoutOptions = {
        room: testRoom,
        itemSpecs: [
          { type: ItemType.Chair, count: 10 },
          { type: ItemType.RoundTable, count: 3 },
        ],
        style: 'banquet',
        nLayouts: 3,
      }

      const layouts = await generateLayoutDiffusion(options)

      // At least one item should differ between first and second layouts
      let anyDiff = false
      for (let i = 0; i < layouts[0]!.length; i++) {
        if (
          layouts[0]![i]!.x !== layouts[1]![i]!.x ||
          layouts[0]![i]!.y !== layouts[1]![i]!.y ||
          layouts[0]![i]!.rotation !== layouts[1]![i]!.rotation
        ) {
          anyDiff = true
          break
        }
      }
      expect(anyDiff).toBe(true)
    })

    it('nLayouts=1 returns only the base template', async () => {
      const options: DiffusionLayoutOptions = {
        room: testRoom,
        itemSpecs: [{ type: ItemType.Chair, count: 15 }],
        style: 'ceremony',
        nLayouts: 1,
      }

      const layouts = await generateLayoutDiffusion(options)
      expect(layouts.length).toBe(1)
    })
  })
})

// =========================================================================
// Orchestrator Tests (PS-12)
// =========================================================================

describe('PS-12: Orchestrator (VenueSolverPipeline)', () => {
  let pipeline: VenueSolverPipeline

  beforeEach(() => {
    vi.clearAllMocks()
    pipeline = new VenueSolverPipeline()
  })

  // -----------------------------------------------------------------------
  // Instantiation
  // -----------------------------------------------------------------------
  describe('constructor', () => {
    it('can be instantiated', () => {
      expect(pipeline).toBeInstanceOf(VenueSolverPipeline)
    })

    it('implements all SolverPipeline methods', () => {
      expect(typeof pipeline.generateInitial).toBe('function')
      expect(typeof pipeline.scheduleEvents).toBe('function')
      expect(typeof pipeline.optimizeLayout).toBe('function')
      expect(typeof pipeline.sampleAlternatives).toBe('function')
      expect(typeof pipeline.computeParetoFront).toBe('function')
      expect(typeof pipeline.runFullPipeline).toBe('function')
    })
  })

  // -----------------------------------------------------------------------
  // itemsToState / stateToItems
  // -----------------------------------------------------------------------
  describe('itemsToState and stateToItems', () => {
    const sampleItems: FurnitureItem[] = [
      { x: 10, y: 20, width: 2, depth: 2, rotation: 0.5, itemType: ItemType.Chair, seats: 1 },
      { x: 30, y: 15, width: 5, depth: 5, rotation: 1.2, itemType: ItemType.RoundTable, seats: 8 },
      { x: 5, y: 35, width: 6, depth: 3, rotation: 0, itemType: ItemType.Bar, seats: 0 },
    ]

    it('itemsToState packs x,y,rotation triplets into Float64Array', () => {
      const state = itemsToState(sampleItems)

      expect(state).toBeInstanceOf(Float64Array)
      expect(state.length).toBe(9) // 3 items * 3 values each

      // Item 0
      expect(state[0]).toBe(10)
      expect(state[1]).toBe(20)
      expect(state[2]).toBe(0.5)

      // Item 1
      expect(state[3]).toBe(30)
      expect(state[4]).toBe(15)
      expect(state[5]).toBe(1.2)

      // Item 2
      expect(state[6]).toBe(5)
      expect(state[7]).toBe(35)
      expect(state[8]).toBe(0)
    })

    it('stateToItems unpacks state using template for metadata', () => {
      const state = new Float64Array([11, 21, 0.6, 31, 16, 1.3, 6, 36, 0.1])
      const result = stateToItems(state, sampleItems)

      expect(result.length).toBe(3)

      // Check x,y,rotation were updated from state
      expect(result[0]!.x).toBe(11)
      expect(result[0]!.y).toBe(21)
      expect(result[0]!.rotation).toBe(0.6)

      // Check metadata preserved from template
      expect(result[0]!.width).toBe(2)
      expect(result[0]!.depth).toBe(2)
      expect(result[0]!.itemType).toBe(ItemType.Chair)
      expect(result[0]!.seats).toBe(1)

      expect(result[1]!.itemType).toBe(ItemType.RoundTable)
      expect(result[1]!.width).toBe(5)
      expect(result[1]!.seats).toBe(8)
    })

    it('are inverse operations (round-trip)', () => {
      const state = itemsToState(sampleItems)
      const roundTripped = stateToItems(state, sampleItems)

      for (let i = 0; i < sampleItems.length; i++) {
        expect(roundTripped[i]!.x).toBeCloseTo(sampleItems[i]!.x)
        expect(roundTripped[i]!.y).toBeCloseTo(sampleItems[i]!.y)
        expect(roundTripped[i]!.rotation).toBeCloseTo(sampleItems[i]!.rotation)
        expect(roundTripped[i]!.width).toBe(sampleItems[i]!.width)
        expect(roundTripped[i]!.depth).toBe(sampleItems[i]!.depth)
        expect(roundTripped[i]!.itemType).toBe(sampleItems[i]!.itemType)
        expect(roundTripped[i]!.seats).toBe(sampleItems[i]!.seats)
      }
    })

    it('handles empty items array', () => {
      const state = itemsToState([])
      expect(state.length).toBe(0)

      const items = stateToItems(new Float64Array(0), [])
      expect(items.length).toBe(0)
    })

    it('handles single item', () => {
      const single = [sampleItems[0]!]
      const state = itemsToState(single)
      expect(state.length).toBe(3)

      const result = stateToItems(state, single)
      expect(result.length).toBe(1)
      expect(result[0]!.x).toBe(single[0]!.x)
      expect(result[0]!.y).toBe(single[0]!.y)
      expect(result[0]!.rotation).toBe(single[0]!.rotation)
    })

    it('double round-trip preserves exact values', () => {
      const state1 = itemsToState(sampleItems)
      const items1 = stateToItems(state1, sampleItems)
      const state2 = itemsToState(items1)
      const items2 = stateToItems(state2, sampleItems)

      for (let i = 0; i < sampleItems.length; i++) {
        expect(items2[i]!.x).toBe(items1[i]!.x)
        expect(items2[i]!.y).toBe(items1[i]!.y)
        expect(items2[i]!.rotation).toBe(items1[i]!.rotation)
      }
    })
  })

  // -----------------------------------------------------------------------
  // Layer 1: generateInitial
  // -----------------------------------------------------------------------
  describe('generateInitial', () => {
    it('returns a Layout object', async () => {
      const layout = await pipeline.generateInitial('a theater for 50 guests', testRoom)

      expect(layout).toBeDefined()
      expect(layout.items).toBeInstanceOf(Array)
      expect(layout.items.length).toBeGreaterThan(0)
      expect(layout.room).toBe(testRoom)
      expect(typeof layout.energy).toBe('number')
    })

    it('energy is computed via computeLayoutEnergy mock', async () => {
      const layout = await pipeline.generateInitial('a banquet for 100 guests', testRoom)
      // Our mock always returns 42.0
      expect(layout.energy).toBe(42.0)
    })

    it('returns items from the parsed description', async () => {
      const layout = await pipeline.generateInitial('a ceremony for 60 people', testRoom)
      // The LLM fallback parses "ceremony" and generates ceremony template
      const stages = countByType(layout.items, ItemType.Stage)
      expect(stages).toBe(1)
    })
  })

  // -----------------------------------------------------------------------
  // Layer 2: scheduleEvents
  // -----------------------------------------------------------------------
  describe('scheduleEvents', () => {
    const events: EventSpec[] = [
      { id: 'e1', guests: 50, duration: 2, preferences: { r1: 1.5 } },
      { id: 'e2', guests: 30, duration: 1, preferences: {} },
    ]
    const rooms: RoomSpec[] = [
      { id: 'r1', capacity: 60, amenities: ['projector'] },
      { id: 'r2', capacity: 40, amenities: [] },
    ]
    const timeslots: TimeslotSpec[] = [
      { id: 't1', start: 9, end: 11, day: 1 },
      { id: 't2', start: 11, end: 13, day: 1 },
    ]

    it('returns a ScheduleResult', async () => {
      const result = await pipeline.scheduleEvents(events, rooms, timeslots)

      expect(result).toBeDefined()
      expect(result.assignments).toBeInstanceOf(Array)
      expect(result.assignments.length).toBeGreaterThan(0)
      expect(typeof result.objectiveValue).toBe('number')
      expect(typeof result.feasible).toBe('boolean')
      expect(typeof result.solveDurationMs).toBe('number')
    })

    it('returns feasible schedule from mock', async () => {
      const result = await pipeline.scheduleEvents(events, rooms, timeslots)
      expect(result.feasible).toBe(true)
    })

    it('assignments have event, room, and timeslot IDs', async () => {
      const result = await pipeline.scheduleEvents(events, rooms, timeslots)
      for (const assignment of result.assignments) {
        expect(typeof assignment.eventId).toBe('string')
        expect(typeof assignment.roomId).toBe('string')
        expect(typeof assignment.timeslotId).toBe('string')
      }
    })
  })

  // -----------------------------------------------------------------------
  // Layer 3: optimizeLayout
  // -----------------------------------------------------------------------
  describe('optimizeLayout', () => {
    it('returns a Layout with energy from the pipeline', async () => {
      const initial: Layout = {
        items: generateTemplateLayout(testRoom, 'theater', 50),
        room: testRoom,
        energy: 100.0,
      }

      const optimized = await pipeline.optimizeLayout(initial, DEFAULT_WEIGHTS, 50)

      expect(optimized).toBeDefined()
      expect(optimized.items).toBeInstanceOf(Array)
      expect(optimized.items.length).toBe(initial.items.length)
      expect(optimized.room).toBe(testRoom)
      expect(typeof optimized.energy).toBe('number')
    })

    it('returns energy from CMA-ES refinement (25.0 from mock)', async () => {
      const initial: Layout = {
        items: generateTemplateLayout(testRoom, 'banquet', 40),
        room: testRoom,
        energy: 200.0,
      }

      const optimized = await pipeline.optimizeLayout(initial, DEFAULT_WEIGHTS, 40)
      // CMA-ES mock returns bestEnergy: 25.0
      expect(optimized.energy).toBe(25.0)
    })

    it('preserves item count through optimization', async () => {
      const initial: Layout = {
        items: generateTemplateLayout(testRoom, 'classroom', 30),
        room: testRoom,
        energy: 150.0,
      }

      const optimized = await pipeline.optimizeLayout(initial, DEFAULT_WEIGHTS, 30)
      expect(optimized.items.length).toBe(initial.items.length)
    })

    it('preserves item types through optimization', async () => {
      const initial: Layout = {
        items: generateTemplateLayout(testRoom, 'cocktail', 40),
        room: testRoom,
        energy: 300.0,
      }

      const optimized = await pipeline.optimizeLayout(initial, DEFAULT_WEIGHTS, 40)
      for (let i = 0; i < initial.items.length; i++) {
        expect(optimized.items[i]!.itemType).toBe(initial.items[i]!.itemType)
        expect(optimized.items[i]!.width).toBe(initial.items[i]!.width)
        expect(optimized.items[i]!.depth).toBe(initial.items[i]!.depth)
      }
    })
  })

  // -----------------------------------------------------------------------
  // Layer 4: sampleAlternatives
  // -----------------------------------------------------------------------
  describe('sampleAlternatives', () => {
    it('returns an array of Layout objects', async () => {
      const layout: Layout = {
        items: generateTemplateLayout(testRoom, 'theater', 50),
        room: testRoom,
        energy: 42.0,
      }

      const alts = await pipeline.sampleAlternatives(layout, DEFAULT_WEIGHTS, 5)

      expect(alts).toBeInstanceOf(Array)
      // Mock returns 2 samples (we limited the mock)
      expect(alts.length).toBe(2)
      for (const alt of alts) {
        expect(alt.items).toBeInstanceOf(Array)
        expect(alt.room).toBe(testRoom)
        expect(typeof alt.energy).toBe('number')
      }
    })
  })

  // -----------------------------------------------------------------------
  // Layer 5: computeParetoFront
  // -----------------------------------------------------------------------
  describe('computeParetoFront', () => {
    it('returns an array of ParetoSolution objects', async () => {
      const layout: Layout = {
        items: generateTemplateLayout(testRoom, 'banquet', 40),
        room: testRoom,
        energy: 50.0,
      }

      const pareto = await pipeline.computeParetoFront(layout, testRoom)

      expect(pareto).toBeInstanceOf(Array)
      expect(pareto.length).toBeGreaterThan(0)
      for (const sol of pareto) {
        expect(sol.state).toBeInstanceOf(Float64Array)
        expect(sol.objectives).toBeInstanceOf(Float64Array)
        expect(typeof sol.frontRank).toBe('number')
        expect(typeof sol.crowdingDistance).toBe('number')
      }
    })
  })

  // -----------------------------------------------------------------------
  // runFullPipeline
  // -----------------------------------------------------------------------
  describe('runFullPipeline', () => {
    const request = {
      description: 'A theater presentation for 60 people',
      room: testRoom,
      events: [
        { id: 'e1', guests: 60, duration: 2, preferences: { r1: 2.0 } },
      ] as EventSpec[],
      rooms: [
        { id: 'r1', capacity: 100, amenities: ['projector', 'sound'] },
      ] as RoomSpec[],
      timeslots: [
        { id: 't1', start: 9, end: 12, day: 1 },
      ] as TimeslotSpec[],
      weights: DEFAULT_WEIGHTS,
      targetCapacity: 60,
    }

    it('returns a PlanningResult with all layers', async () => {
      const result = await pipeline.runFullPipeline(request)

      expect(result).toBeDefined()

      // Layer 2: schedule
      expect(result.schedule).toBeDefined()
      expect(result.schedule.feasible).toBe(true)
      expect(result.schedule.assignments.length).toBeGreaterThan(0)

      // Layer 3: optimized layout
      expect(result.optimized).toBeDefined()
      expect(result.optimized.items).toBeInstanceOf(Array)
      expect(result.optimized.items.length).toBeGreaterThan(0)
      expect(result.optimized.room).toBe(testRoom)
      expect(typeof result.optimized.energy).toBe('number')

      // Layer 4: alternatives
      expect(result.alternatives).toBeInstanceOf(Array)

      // Layer 5: pareto
      expect(result.pareto).toBeInstanceOf(Array)
      expect(result.pareto.length).toBeGreaterThan(0)
    })

    it('optimized layout has lower or equal energy compared to initial (via mock)', async () => {
      const result = await pipeline.runFullPipeline(request)
      // CMA-ES mock returns 25.0, which is lower than initial mock energy of 42.0
      expect(result.optimized.energy).toBeLessThanOrEqual(42.0)
    })

    it('orchestrates all five layers in sequence', async () => {
      // Import mocked modules to verify call counts
      const { solveScheduleMIP } = await import('../mip-scheduler.js')
      const { parallelTempering } = await import('../parallel-tempering.js')
      const { cmaes } = await import('../cmaes.js')
      const { sampleLayoutsMH } = await import('../mcmc.js')
      const { nsga2 } = await import('../nsga2.js')

      await pipeline.runFullPipeline(request)

      expect(solveScheduleMIP).toHaveBeenCalledOnce()
      expect(parallelTempering).toHaveBeenCalledOnce()
      expect(cmaes).toHaveBeenCalledOnce()
      expect(sampleLayoutsMH).toHaveBeenCalledOnce()
      expect(nsga2).toHaveBeenCalledOnce()
    })
  })
})
