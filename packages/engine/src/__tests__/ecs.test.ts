import { describe, it, expect, beforeEach } from 'vitest'
import { defineQuery, hasComponent, removeEntity } from 'bitecs'
import {
  createEcsWorld,
  resetEcsWorld,
  createFurnitureEntity,
  removeFurnitureEntity,
  isFurnitureEntity,
  getAllEntityIds,
  EntityIdMap,
  Position,
  Rotation,
  Scale,
  BoundingBox,
  FurnitureTag,
  GroupMember,
  Selectable,
  Draggable,
  furnitureTypeToIndex,
  indexToFurnitureType,
  type EcsWorld,
  type FurnitureEntityInput,
} from '../ecs'

// ─── Queries for testing ────────────────────────────────────────────────────

const allFurniture = defineQuery([Position, FurnitureTag])
const selectables = defineQuery([Selectable, Position])
const draggables = defineQuery([Draggable, Position])
const grouped = defineQuery([GroupMember])

// ─── Helpers ────────────────────────────────────────────────────────────────

function makeChair(overrides?: Partial<FurnitureEntityInput>): FurnitureEntityInput {
  return {
    type: 'chair',
    position: [1, 0, 2],
    ...overrides,
  }
}

function makeTable(overrides?: Partial<FurnitureEntityInput>): FurnitureEntityInput {
  return {
    type: 'round-table',
    position: [5, 0, 5],
    rotation: [0, Math.PI / 4, 0],
    scale: [1.5, 1.5, 1.5],
    ...overrides,
  }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

describe('ECS: components', () => {
  it('furnitureTypeToIndex maps all types correctly', () => {
    expect(furnitureTypeToIndex('chair')).toBe(0)
    expect(furnitureTypeToIndex('round-table')).toBe(1)
    expect(furnitureTypeToIndex('rect-table')).toBe(2)
    expect(furnitureTypeToIndex('trestle-table')).toBe(3)
    expect(furnitureTypeToIndex('podium')).toBe(4)
    expect(furnitureTypeToIndex('stage')).toBe(5)
    expect(furnitureTypeToIndex('bar')).toBe(6)
  })

  it('indexToFurnitureType round-trips correctly', () => {
    for (let i = 0; i <= 6; i++) {
      const type = indexToFurnitureType(i)
      expect(type).toBeDefined()
      expect(furnitureTypeToIndex(type!)).toBe(i)
    }
  })

  it('indexToFurnitureType returns undefined for invalid index', () => {
    expect(indexToFurnitureType(99)).toBeUndefined()
  })
})

describe('ECS: world and entity factory', () => {
  let world: EcsWorld

  beforeEach(() => {
    world = createEcsWorld()
  })

  it('creates a world with no entities', () => {
    const entities = getAllEntityIds(world)
    expect(entities).toHaveLength(0)
  })

  it('creates a furniture entity with correct Position', () => {
    const eid = createFurnitureEntity(world, makeChair({ position: [3, 0.5, 7] }))
    expect(Position.x[eid]).toBeCloseTo(3)
    expect(Position.y[eid]).toBeCloseTo(0.5)
    expect(Position.z[eid]).toBeCloseTo(7)
  })

  it('creates a furniture entity with correct Rotation', () => {
    const eid = createFurnitureEntity(world, makeTable({ rotation: [0.1, 0.2, 0.3] }))
    expect(Rotation.x[eid]).toBeCloseTo(0.1)
    expect(Rotation.y[eid]).toBeCloseTo(0.2)
    expect(Rotation.z[eid]).toBeCloseTo(0.3)
  })

  it('defaults Rotation to zero', () => {
    const eid = createFurnitureEntity(world, makeChair())
    expect(Rotation.x[eid]).toBe(0)
    expect(Rotation.y[eid]).toBe(0)
    expect(Rotation.z[eid]).toBe(0)
  })

  it('creates a furniture entity with specified Scale', () => {
    const eid = createFurnitureEntity(world, makeTable({ scale: [2, 2, 2] }))
    expect(Scale.x[eid]).toBeCloseTo(2)
    expect(Scale.y[eid]).toBeCloseTo(2)
    expect(Scale.z[eid]).toBeCloseTo(2)
  })

  it('defaults Scale to 1,1,1', () => {
    const eid = createFurnitureEntity(world, makeChair())
    expect(Scale.x[eid]).toBeCloseTo(1)
    expect(Scale.y[eid]).toBeCloseTo(1)
    expect(Scale.z[eid]).toBeCloseTo(1)
  })

  it('sets BoundingBox from furniture type defaults', () => {
    const eid = createFurnitureEntity(world, makeChair())
    // Chair bounds: [0.22, 0.45, 0.22]
    expect(BoundingBox.halfX[eid]).toBeCloseTo(0.22)
    expect(BoundingBox.halfY[eid]).toBeCloseTo(0.45)
    expect(BoundingBox.halfZ[eid]).toBeCloseTo(0.22)
  })

  it('sets BoundingBox for stage', () => {
    const eid = createFurnitureEntity(world, {
      type: 'stage',
      position: [0, 0, 0],
    })
    // Stage bounds: [2.0, 0.3, 1.5]
    expect(BoundingBox.halfX[eid]).toBeCloseTo(2.0)
    expect(BoundingBox.halfY[eid]).toBeCloseTo(0.3)
    expect(BoundingBox.halfZ[eid]).toBeCloseTo(1.5)
  })

  it('sets FurnitureTag type index', () => {
    const eid = createFurnitureEntity(world, makeTable())
    expect(FurnitureTag.type[eid]).toBe(furnitureTypeToIndex('round-table'))
  })

  it('sets GroupMember to 0 when no group', () => {
    const eid = createFurnitureEntity(world, makeChair())
    expect(GroupMember.groupId[eid]).toBe(0)
  })

  it('sets GroupMember to specified groupId', () => {
    const eid = createFurnitureEntity(world, makeChair({ groupId: 42 }))
    expect(GroupMember.groupId[eid]).toBe(42)
  })

  it('adds Selectable and Draggable tags', () => {
    const eid = createFurnitureEntity(world, makeChair())
    expect(hasComponent(world, Selectable, eid)).toBe(true)
    expect(hasComponent(world, Draggable, eid)).toBe(true)
  })

  it('isFurnitureEntity returns true for furniture', () => {
    const eid = createFurnitureEntity(world, makeChair())
    expect(isFurnitureEntity(world, eid)).toBe(true)
  })

  it('removes a furniture entity', () => {
    const eid = createFurnitureEntity(world, makeChair())
    removeFurnitureEntity(world, eid)
    expect(isFurnitureEntity(world, eid)).toBe(false)
  })

  it('resetEcsWorld clears all entities', () => {
    createFurnitureEntity(world, makeChair())
    createFurnitureEntity(world, makeTable())
    expect(getAllEntityIds(world).length).toBe(2)
    resetEcsWorld(world)
    expect(getAllEntityIds(world).length).toBe(0)
  })
})

describe('ECS: queries', () => {
  let world: EcsWorld

  beforeEach(() => {
    world = createEcsWorld()
  })

  it('allFurniture query returns all furniture entities', () => {
    const e1 = createFurnitureEntity(world, makeChair())
    const e2 = createFurnitureEntity(world, makeTable())
    const results = allFurniture(world)
    expect(results).toContain(e1)
    expect(results).toContain(e2)
    expect(results).toHaveLength(2)
  })

  it('selectables query returns selectable entities', () => {
    const e1 = createFurnitureEntity(world, makeChair())
    const results = selectables(world)
    expect(results).toContain(e1)
  })

  it('draggables query returns draggable entities', () => {
    const e1 = createFurnitureEntity(world, makeChair())
    const results = draggables(world)
    expect(results).toContain(e1)
  })

  it('grouped query includes all entities (GroupMember always added)', () => {
    const e1 = createFurnitureEntity(world, makeChair({ groupId: 10 }))
    const e2 = createFurnitureEntity(world, makeChair())
    const results = grouped(world)
    expect(results).toContain(e1)
    expect(results).toContain(e2)
  })

  it('query excludes removed entities', () => {
    const e1 = createFurnitureEntity(world, makeChair())
    const e2 = createFurnitureEntity(world, makeTable())
    removeFurnitureEntity(world, e1)
    const results = allFurniture(world)
    expect(results).not.toContain(e1)
    expect(results).toContain(e2)
    expect(results).toHaveLength(1)
  })
})

describe('ECS: 100 entities bulk test', () => {
  it('creates and queries 100 entities efficiently', () => {
    const world = createEcsWorld()
    const eids: number[] = []

    for (let i = 0; i < 100; i++) {
      const type = i % 2 === 0 ? 'chair' : 'round-table'
      const eid = createFurnitureEntity(world, {
        type,
        position: [i * 0.5, 0, i * 0.3],
        rotation: [0, (i * Math.PI) / 50, 0],
      })
      eids.push(eid)
    }

    // All 100 entities exist
    expect(getAllEntityIds(world)).toHaveLength(100)

    // All 100 show up in furniture query
    const results = allFurniture(world)
    expect(results).toHaveLength(100)

    // Verify first entity data
    const first = eids[0]!
    expect(Position.x[first]).toBeCloseTo(0)
    expect(Position.z[first]).toBeCloseTo(0)
    expect(FurnitureTag.type[first]).toBe(furnitureTypeToIndex('chair'))

    // Verify last entity data
    const last = eids[99]!
    expect(Position.x[last]).toBeCloseTo(49.5)
    expect(Position.z[last]).toBeCloseTo(29.7)
    expect(FurnitureTag.type[last]).toBe(furnitureTypeToIndex('round-table'))
  })

  it('handles mixed furniture types correctly', () => {
    const world = createEcsWorld()
    const types = ['chair', 'round-table', 'rect-table', 'trestle-table', 'podium', 'stage', 'bar'] as const

    for (let i = 0; i < 100; i++) {
      const furnitureType = types[i % types.length]!
      createFurnitureEntity(world, {
        type: furnitureType,
        position: [i, 0, 0],
      })
    }

    const results = allFurniture(world)
    expect(results).toHaveLength(100)

    // Count each type
    const counts = new Map<number, number>()
    for (const eid of results) {
      const t = FurnitureTag.type[eid] ?? 0
      counts.set(t, (counts.get(t) ?? 0) + 1)
    }

    // 100 items across 7 types: 15 of the first 2 types, 14 of the rest
    // Actually: 100/7 = 14r2, so types 0,1 get 15, rest get 14
    expect(counts.get(0)).toBe(15) // chair (indices 0,7,14,...,98)
    expect(counts.get(1)).toBe(15) // round-table (indices 1,8,15,...,99)
    expect(counts.get(2)).toBe(14) // rect-table
    expect(counts.get(3)).toBe(14) // trestle-table
    expect(counts.get(4)).toBe(14) // podium
    expect(counts.get(5)).toBe(14) // stage
    expect(counts.get(6)).toBe(14) // bar
  })
})

describe('ECS: EntityIdMap', () => {
  it('maps domain ID to ECS ID and back', () => {
    const map = new EntityIdMap()
    map.set('item-abc', 42)
    expect(map.getEcs('item-abc')).toBe(42)
    expect(map.getDomain(42)).toBe('item-abc')
  })

  it('returns undefined for unknown IDs', () => {
    const map = new EntityIdMap()
    expect(map.getEcs('nope')).toBeUndefined()
    expect(map.getDomain(999)).toBeUndefined()
  })

  it('delete removes both directions', () => {
    const map = new EntityIdMap()
    map.set('item-1', 10)
    map.delete('item-1')
    expect(map.getEcs('item-1')).toBeUndefined()
    expect(map.getDomain(10)).toBeUndefined()
  })

  it('deleteByEid removes both directions', () => {
    const map = new EntityIdMap()
    map.set('item-2', 20)
    map.deleteByEid(20)
    expect(map.getEcs('item-2')).toBeUndefined()
    expect(map.getDomain(20)).toBeUndefined()
  })

  it('tracks size correctly', () => {
    const map = new EntityIdMap()
    expect(map.size).toBe(0)
    map.set('a', 1)
    map.set('b', 2)
    expect(map.size).toBe(2)
    map.delete('a')
    expect(map.size).toBe(1)
  })

  it('clear removes all entries', () => {
    const map = new EntityIdMap()
    map.set('a', 1)
    map.set('b', 2)
    map.set('c', 3)
    map.clear()
    expect(map.size).toBe(0)
    expect(map.getEcs('a')).toBeUndefined()
  })
})
