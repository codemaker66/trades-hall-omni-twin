import { describe, it, expect } from 'vitest'
import fc from 'fast-check'
import type { DomainEvent, FurnitureType, Position3D, Rotation3D } from '@omni-twin/types'
import { applyEvent, projectState, emptyVenueState } from '../projector'
import { SpatialHash, type AABB } from '../ecs/systems/spatial-index'
import { aabbOverlap } from '../ecs/systems/collision'
import { snapToGrid, snapToHeightGrid, findKNearest } from '../ecs/systems/snapping'
import { Position } from '../ecs/components'

// ─── Shared Arbitraries ─────────────────────────────────────────────────────

const FURNITURE_TYPES: FurnitureType[] = [
  'chair', 'round-table', 'rect-table', 'trestle-table', 'podium', 'stage', 'bar',
]

const arbFurnitureType = fc.constantFrom(...FURNITURE_TYPES)

const arbPosition3D: fc.Arbitrary<Position3D> = fc.tuple(
  fc.double({ min: -50, max: 50, noNaN: true, noDefaultInfinity: true }),
  fc.double({ min: 0, max: 10, noNaN: true, noDefaultInfinity: true }),
  fc.double({ min: -50, max: 50, noNaN: true, noDefaultInfinity: true }),
)

const arbRotation3D: fc.Arbitrary<Rotation3D> = fc.tuple(
  fc.double({ min: -Math.PI, max: Math.PI, noNaN: true, noDefaultInfinity: true }),
  fc.double({ min: -Math.PI, max: Math.PI, noNaN: true, noDefaultInfinity: true }),
  fc.double({ min: -Math.PI, max: Math.PI, noNaN: true, noDefaultInfinity: true }),
)

let eventSeq = 0
function makeEvent(version: number, partial: Partial<DomainEvent> & { type: string; payload: unknown }): DomainEvent {
  return {
    id: `evt-${++eventSeq}`,
    timestamp: new Date().toISOString(),
    userId: 'user-1',
    venueId: 'venue-1',
    version,
    ...partial,
  } as DomainEvent
}

// ─── Projector Property-Based Tests ─────────────────────────────────────────

describe('Projector — property-based tests', () => {
  it('immutability: applyEvent never mutates the input state', () => {
    fc.assert(fc.property(
      arbFurnitureType,
      arbPosition3D,
      arbRotation3D,
      (ftype, pos, rot) => {
        const state = emptyVenueState('venue-1')
        const frozenItems = new Map(state.items)
        const frozenGroups = new Set(state.groups)

        const event = makeEvent(1, {
          type: 'ItemPlaced',
          payload: { itemId: 'item-1', furnitureType: ftype, position: pos, rotation: rot },
        })

        const newState = applyEvent(state, event)

        // Original state must be unchanged
        expect(state.items).toEqual(frozenItems)
        expect(state.groups).toEqual(frozenGroups)
        expect(state.items.size).toBe(0)
        // New state must have the item
        expect(newState.items.size).toBe(1)
      },
    ), { numRuns: 100 })
  })

  it('version monotonicity: each event advances the version', () => {
    fc.assert(fc.property(
      fc.integer({ min: 1, max: 50 }),
      (count) => {
        let state = emptyVenueState('venue-1')
        for (let i = 1; i <= count; i++) {
          const event = makeEvent(i, {
            type: 'ItemPlaced',
            payload: {
              itemId: `item-${i}`,
              furnitureType: 'chair',
              position: [i, 0, 0] as Position3D,
              rotation: [0, 0, 0] as Rotation3D,
            },
          })
          const prev = state.version
          state = applyEvent(state, event)
          expect(state.version).toBeGreaterThan(prev)
        }
      },
    ), { numRuns: 50 })
  })

  it('place-remove roundtrip: placing then removing an item restores item count', () => {
    fc.assert(fc.property(
      fc.array(fc.record({
        id: fc.stringMatching(/^item-[0-9a-z]{4}$/),
        type: arbFurnitureType,
        pos: arbPosition3D,
        rot: arbRotation3D,
      }), { minLength: 1, maxLength: 10 }),
      (items) => {
        // Deduplicate IDs
        const seen = new Set<string>()
        const unique = items.filter((i) => {
          if (seen.has(i.id)) return false
          seen.add(i.id)
          return true
        })

        let state = emptyVenueState('venue-1')
        let v = 0

        // Place all items
        for (const item of unique) {
          state = applyEvent(state, makeEvent(++v, {
            type: 'ItemPlaced',
            payload: { itemId: item.id, furnitureType: item.type, position: item.pos, rotation: item.rot },
          }))
        }
        expect(state.items.size).toBe(unique.length)

        // Remove all items
        for (const item of unique) {
          state = applyEvent(state, makeEvent(++v, {
            type: 'ItemRemoved',
            payload: { itemId: item.id },
          }))
        }
        expect(state.items.size).toBe(0)
      },
    ), { numRuns: 100 })
  })

  it('move idempotency: moving to same position is stable', () => {
    fc.assert(fc.property(
      arbPosition3D,
      arbPosition3D,
      (startPos, movePos) => {
        let state = emptyVenueState('venue-1')
        state = applyEvent(state, makeEvent(1, {
          type: 'ItemPlaced',
          payload: { itemId: 'a', furnitureType: 'chair', position: startPos, rotation: [0, 0, 0] as Rotation3D },
        }))
        state = applyEvent(state, makeEvent(2, {
          type: 'ItemMoved',
          payload: { itemId: 'a', position: movePos },
        }))
        const stateAfterFirst = state
        state = applyEvent(state, makeEvent(3, {
          type: 'ItemMoved',
          payload: { itemId: 'a', position: movePos },
        }))

        // Position identical after both moves
        const item1 = stateAfterFirst.items.get('a')!
        const item2 = state.items.get('a')!
        expect(item2.position).toEqual(item1.position)
      },
    ), { numRuns: 100 })
  })

  it('group-dissolve roundtrip: dissolving clears all groupId references', () => {
    fc.assert(fc.property(
      fc.integer({ min: 2, max: 8 }),
      (count) => {
        let state = emptyVenueState('venue-1')
        let v = 0
        const itemIds: string[] = []

        for (let i = 0; i < count; i++) {
          const id = `item-${i}`
          itemIds.push(id)
          state = applyEvent(state, makeEvent(++v, {
            type: 'ItemPlaced',
            payload: { itemId: id, furnitureType: 'chair', position: [i, 0, 0] as Position3D, rotation: [0, 0, 0] as Rotation3D },
          }))
        }

        // Group them
        state = applyEvent(state, makeEvent(++v, {
          type: 'GroupCreated',
          payload: { groupId: 'g1', itemIds },
        }))
        expect(state.groups.has('g1')).toBe(true)
        for (const id of itemIds) {
          expect(state.items.get(id)!.groupId).toBe('g1')
        }

        // Dissolve
        state = applyEvent(state, makeEvent(++v, {
          type: 'GroupDissolved',
          payload: { groupId: 'g1' },
        }))
        expect(state.groups.has('g1')).toBe(false)
        for (const id of itemIds) {
          expect(state.items.get(id)!.groupId).toBeUndefined()
        }
      },
    ), { numRuns: 50 })
  })

  it('projectState = sequential applyEvent', () => {
    fc.assert(fc.property(
      fc.integer({ min: 1, max: 20 }),
      (count) => {
        const events: DomainEvent[] = []
        for (let i = 1; i <= count; i++) {
          events.push(makeEvent(i, {
            type: 'ItemPlaced',
            payload: {
              itemId: `item-${i}`,
              furnitureType: 'chair',
              position: [i, 0, i] as Position3D,
              rotation: [0, 0, 0] as Rotation3D,
            },
          }))
        }

        // Build via projectState
        const projected = projectState('venue-1', events)

        // Build via sequential applyEvent
        let sequential = emptyVenueState('venue-1')
        for (const e of events) {
          sequential = applyEvent(sequential, e)
        }

        expect(projected.items.size).toBe(sequential.items.size)
        expect(projected.version).toBe(sequential.version)
        for (const [id, item] of projected.items) {
          expect(sequential.items.get(id)).toEqual(item)
        }
      },
    ), { numRuns: 50 })
  })
})

// ─── SpatialHash Property-Based Tests ───────────────────────────────────────

describe('SpatialHash — property-based tests', () => {
  it('insert-query consistency: inserted entity always found by queryAABB', () => {
    fc.assert(fc.property(
      fc.double({ min: -100, max: 100, noNaN: true, noDefaultInfinity: true }),
      fc.double({ min: -100, max: 100, noNaN: true, noDefaultInfinity: true }),
      fc.double({ min: 0.5, max: 5, noNaN: true, noDefaultInfinity: true }),
      (x, z, cellSize) => {
        const hash = new SpatialHash(cellSize)
        hash.insert(42, x, z)

        // Query an AABB centered on the entity's position
        const results = hash.queryAABB({
          minX: x - 0.01,
          minZ: z - 0.01,
          maxX: x + 0.01,
          maxZ: z + 0.01,
        })
        expect(results).toContain(42)
      },
    ), { numRuns: 200 })
  })

  it('negative coordinate handling: entities at negative positions are found', () => {
    fc.assert(fc.property(
      fc.double({ min: -500, max: -0.1, noNaN: true, noDefaultInfinity: true }),
      fc.double({ min: -500, max: -0.1, noNaN: true, noDefaultInfinity: true }),
      (x, z) => {
        const hash = new SpatialHash(2)
        hash.insert(1, x, z)
        expect(hash.size).toBe(1)

        const results = hash.queryAABB({
          minX: x - 1,
          minZ: z - 1,
          maxX: x + 1,
          maxZ: z + 1,
        })
        expect(results).toContain(1)
      },
    ), { numRuns: 200 })
  })

  it('removal completeness: removed entity never in queries', () => {
    fc.assert(fc.property(
      fc.array(fc.record({
        eid: fc.integer({ min: 0, max: 999 }),
        x: fc.double({ min: -50, max: 50, noNaN: true, noDefaultInfinity: true }),
        z: fc.double({ min: -50, max: 50, noNaN: true, noDefaultInfinity: true }),
      }), { minLength: 1, maxLength: 20 }),
      (entities) => {
        const hash = new SpatialHash(2)
        const deduped = new Map<number, { x: number; z: number }>()
        for (const e of entities) {
          deduped.set(e.eid, { x: e.x, z: e.z })
        }

        // Insert all
        for (const [eid, pos] of deduped) {
          hash.insert(eid, pos.x, pos.z)
        }

        // Remove first entity
        const firstEid = deduped.keys().next().value as number
        hash.remove(firstEid)

        // Query entire space
        const results = hash.queryAABB({ minX: -51, minZ: -51, maxX: 51, maxZ: 51 })
        expect(results).not.toContain(firstEid)
        expect(hash.size).toBe(deduped.size - 1)
      },
    ), { numRuns: 100 })
  })

  it('size = inserted - removed', () => {
    fc.assert(fc.property(
      fc.integer({ min: 1, max: 50 }),
      fc.integer({ min: 0, max: 25 }),
      (insertCount, removeCount) => {
        const actualRemove = Math.min(removeCount, insertCount)
        const hash = new SpatialHash(2)

        for (let i = 0; i < insertCount; i++) {
          hash.insert(i, i * 2, i * 3)
        }
        expect(hash.size).toBe(insertCount)

        for (let i = 0; i < actualRemove; i++) {
          hash.remove(i)
        }
        expect(hash.size).toBe(insertCount - actualRemove)
      },
    ), { numRuns: 100 })
  })

  it('cell boundary: entity on exact grid boundary is still found', () => {
    fc.assert(fc.property(
      fc.integer({ min: -50, max: 50 }),
      fc.integer({ min: -50, max: 50 }),
      fc.constantFrom(0.5, 1, 2, 5),
      (cx, cz, cellSize) => {
        const hash = new SpatialHash(cellSize)
        // Place entity exactly on a cell boundary
        const x = cx * cellSize
        const z = cz * cellSize
        hash.insert(0, x, z)

        const results = hash.queryAABB({
          minX: x - cellSize,
          minZ: z - cellSize,
          maxX: x + cellSize,
          maxZ: z + cellSize,
        })
        expect(results).toContain(0)
      },
    ), { numRuns: 100 })
  })
})

// ─── AABB Overlap Property-Based Tests ──────────────────────────────────────

describe('AABB overlap — property-based tests', () => {
  const arbAABB: fc.Arbitrary<AABB> = fc.tuple(
    fc.double({ min: -50, max: 50, noNaN: true, noDefaultInfinity: true }),
    fc.double({ min: -50, max: 50, noNaN: true, noDefaultInfinity: true }),
    fc.double({ min: 0.1, max: 10, noNaN: true, noDefaultInfinity: true }),
    fc.double({ min: 0.1, max: 10, noNaN: true, noDefaultInfinity: true }),
  ).map(([cx, cz, hw, hd]) => ({
    minX: cx - hw,
    minZ: cz - hd,
    maxX: cx + hw,
    maxZ: cz + hd,
  }))

  it('symmetry: overlap(a, b) === overlap(b, a)', () => {
    fc.assert(fc.property(
      arbAABB, arbAABB,
      (a, b) => {
        expect(aabbOverlap(a, b)).toBe(aabbOverlap(b, a))
      },
    ), { numRuns: 200 })
  })

  it('self-overlap: every AABB overlaps itself', () => {
    fc.assert(fc.property(
      arbAABB,
      (a) => {
        expect(aabbOverlap(a, a)).toBe(true)
      },
    ), { numRuns: 100 })
  })

  it('separation: non-overlapping AABBs correctly detected', () => {
    fc.assert(fc.property(
      fc.double({ min: -20, max: 20, noNaN: true, noDefaultInfinity: true }),
      fc.double({ min: -20, max: 20, noNaN: true, noDefaultInfinity: true }),
      fc.double({ min: 0.5, max: 5, noNaN: true, noDefaultInfinity: true }),
      fc.double({ min: 1, max: 20, noNaN: true, noDefaultInfinity: true }),
      (cx, cz, halfSize, gap) => {
        const a: AABB = { minX: cx - halfSize, minZ: cz - halfSize, maxX: cx + halfSize, maxZ: cz + halfSize }
        const b: AABB = {
          minX: cx + halfSize + gap,
          minZ: cz - halfSize,
          maxX: cx + halfSize + gap + halfSize * 2,
          maxZ: cz + halfSize,
        }
        // b is to the right of a with a gap — no overlap
        expect(aabbOverlap(a, b)).toBe(false)
      },
    ), { numRuns: 100 })
  })

  it('containment: if a contains b, they overlap', () => {
    fc.assert(fc.property(
      fc.double({ min: -20, max: 20, noNaN: true, noDefaultInfinity: true }),
      fc.double({ min: -20, max: 20, noNaN: true, noDefaultInfinity: true }),
      fc.double({ min: 1, max: 10, noNaN: true, noDefaultInfinity: true }),
      fc.double({ min: 0.1, max: 0.9, noNaN: true, noDefaultInfinity: true }),
      (cx, cz, outerSize, innerRatio) => {
        const outer: AABB = { minX: cx - outerSize, minZ: cz - outerSize, maxX: cx + outerSize, maxZ: cz + outerSize }
        const inner: AABB = {
          minX: cx - outerSize * innerRatio,
          minZ: cz - outerSize * innerRatio,
          maxX: cx + outerSize * innerRatio,
          maxZ: cz + outerSize * innerRatio,
        }
        expect(aabbOverlap(outer, inner)).toBe(true)
        expect(aabbOverlap(inner, outer)).toBe(true)
      },
    ), { numRuns: 100 })
  })
})

// ─── Snapping Property-Based Tests ──────────────────────────────────────────

describe('Snapping — property-based tests', () => {
  it('grid snap idempotency: snapToGrid(snapToGrid(x, z)) === snapToGrid(x, z)', () => {
    fc.assert(fc.property(
      fc.double({ min: -100, max: 100, noNaN: true, noDefaultInfinity: true }),
      fc.double({ min: -100, max: 100, noNaN: true, noDefaultInfinity: true }),
      fc.constantFrom(0.25, 0.5, 1, 2),
      (x, z, gridSize) => {
        const [sx, sz] = snapToGrid(x, z, gridSize)
        const [sx2, sz2] = snapToGrid(sx, sz, gridSize)
        // Use tolerance for floating-point
        expect(Math.abs(sx - sx2)).toBeLessThan(1e-10)
        expect(Math.abs(sz - sz2)).toBeLessThan(1e-10)
      },
    ), { numRuns: 200 })
  })

  it('grid snap proximity: snapped value is within 0.5 * gridSize of original', () => {
    fc.assert(fc.property(
      fc.double({ min: -100, max: 100, noNaN: true, noDefaultInfinity: true }),
      fc.double({ min: -100, max: 100, noNaN: true, noDefaultInfinity: true }),
      fc.constantFrom(0.25, 0.5, 1, 2),
      (x, z, gridSize) => {
        const [sx, sz] = snapToGrid(x, z, gridSize)
        const halfGrid = gridSize / 2 + 1e-10 // small tolerance
        expect(Math.abs(sx - x)).toBeLessThanOrEqual(halfGrid)
        expect(Math.abs(sz - z)).toBeLessThanOrEqual(halfGrid)
      },
    ), { numRuns: 200 })
  })

  it('grid snap alignment: snapped value is a multiple of gridSize', () => {
    fc.assert(fc.property(
      fc.double({ min: -50, max: 50, noNaN: true, noDefaultInfinity: true }),
      fc.double({ min: -50, max: 50, noNaN: true, noDefaultInfinity: true }),
      fc.constantFrom(0.25, 0.5, 1, 2),
      (x, z, gridSize) => {
        const [sx, sz] = snapToGrid(x, z, gridSize)
        // sx / gridSize should be very close to an integer
        const xRatio = sx / gridSize
        const zRatio = sz / gridSize
        expect(Math.abs(xRatio - Math.round(xRatio))).toBeLessThan(1e-10)
        expect(Math.abs(zRatio - Math.round(zRatio))).toBeLessThan(1e-10)
      },
    ), { numRuns: 200 })
  })

  it('height snap idempotency: snapToHeightGrid(snapToHeightGrid(y)) === snapToHeightGrid(y)', () => {
    fc.assert(fc.property(
      fc.double({ min: -10, max: 30, noNaN: true, noDefaultInfinity: true }),
      fc.constantFrom(0.05, 0.1, 0.25, 0.5),
      (y, gridSize) => {
        const s1 = snapToHeightGrid(y, gridSize)
        const s2 = snapToHeightGrid(s1, gridSize)
        expect(Math.abs(s1 - s2)).toBeLessThan(1e-10)
      },
    ), { numRuns: 200 })
  })

  it('findKNearest ordering: results are sorted by distance ascending', () => {
    fc.assert(fc.property(
      fc.integer({ min: 3, max: 20 }),
      fc.integer({ min: 1, max: 10 }),
      fc.double({ min: -20, max: 20, noNaN: true, noDefaultInfinity: true }),
      fc.double({ min: -20, max: 20, noNaN: true, noDefaultInfinity: true }),
      (entityCount, k, qx, qz) => {
        const hash = new SpatialHash(2)

        for (let i = 0; i < entityCount; i++) {
          const ex = qx + (i - entityCount / 2) * 0.5
          const ez = qz + (i % 3) * 0.5
          Position.x[i] = ex
          Position.z[i] = ez
          hash.insert(i, ex, ez)
        }

        const results = findKNearest(hash, qx, qz, k, 50)

        // Results should be sorted by distance
        for (let i = 1; i < results.length; i++) {
          expect(results[i]!.distance).toBeGreaterThanOrEqual(results[i - 1]!.distance)
        }
        // Should not exceed k
        expect(results.length).toBeLessThanOrEqual(k)
      },
    ), { numRuns: 100 })
  })
})
