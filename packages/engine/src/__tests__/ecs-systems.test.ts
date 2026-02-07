import { describe, it, expect, beforeEach } from 'vitest'
import {
  createEcsWorld,
  createFurnitureEntity,
  removeFurnitureEntity,
  Position,
  BoundingBox,
  Scale,
  type EcsWorld,
} from '../ecs'
import {
  SpatialHash,
  getEntityAABB,
  aabbOverlap,
  detectCollisions,
  checkCollision,
  checkPlacementCollision,
  rectSelect,
  pointSelect,
  snapToGrid,
  snapToHeightGrid,
  findNearest,
  findKNearest,
  snapToNearest,
} from '../ecs/systems'

// ─── Helpers ────────────────────────────────────────────────────────────────

function placeChair(world: EcsWorld, x: number, z: number, hash?: SpatialHash): number {
  const eid = createFurnitureEntity(world, { type: 'chair', position: [x, 0, z] })
  if (hash) hash.insert(eid, x, z)
  return eid
}

function placeTable(world: EcsWorld, x: number, z: number, hash?: SpatialHash): number {
  const eid = createFurnitureEntity(world, { type: 'round-table', position: [x, 0, z] })
  if (hash) hash.insert(eid, x, z)
  return eid
}

// ═════════════════════════════════════════════════════════════════════════════
// Spatial Hash
// ═════════════════════════════════════════════════════════════════════════════

describe('SpatialHash', () => {
  let world: EcsWorld
  let hash: SpatialHash

  beforeEach(() => {
    world = createEcsWorld()
    hash = new SpatialHash(2) // 2m cells
  })

  it('starts empty', () => {
    expect(hash.size).toBe(0)
  })

  it('inserts and queries a single entity', () => {
    const eid = placeChair(world, 5, 5, hash)
    const results = hash.queryAABB({ minX: 4, minZ: 4, maxX: 6, maxZ: 6 })
    expect(results).toContain(eid)
  })

  it('does not return entities outside query AABB', () => {
    placeChair(world, 5, 5, hash)
    const results = hash.queryAABB({ minX: 10, minZ: 10, maxX: 12, maxZ: 12 })
    expect(results).toHaveLength(0)
  })

  it('removes an entity', () => {
    const eid = placeChair(world, 5, 5, hash)
    hash.remove(eid)
    expect(hash.size).toBe(0)
    const results = hash.queryAABB({ minX: 4, minZ: 4, maxX: 6, maxZ: 6 })
    expect(results).toHaveLength(0)
  })

  it('updates entity position (moves between cells)', () => {
    const eid = placeChair(world, 1, 1, hash)

    // Move to (10, 10) — different cell
    hash.insert(eid, 10, 10)

    // Old position should be empty
    const oldResults = hash.queryAABB({ minX: 0, minZ: 0, maxX: 2, maxZ: 2 })
    expect(oldResults).not.toContain(eid)

    // New position should contain entity
    const newResults = hash.queryAABB({ minX: 9, minZ: 9, maxX: 11, maxZ: 11 })
    expect(newResults).toContain(eid)
  })

  it('queryRadius returns entities within radius', () => {
    const e1 = placeChair(world, 0, 0, hash)
    const e2 = placeChair(world, 1, 0, hash)
    placeChair(world, 100, 100, hash) // far away

    const results = hash.queryRadius(0, 0, 2)
    expect(results).toContain(e1)
    expect(results).toContain(e2)
    expect(results).toHaveLength(2)
  })

  it('rebuild populates from world state', () => {
    placeChair(world, 3, 3)
    placeChair(world, 7, 7)

    hash.rebuild(world)
    expect(hash.size).toBe(2)
  })

  it('clear empties the hash', () => {
    placeChair(world, 1, 1, hash)
    placeChair(world, 2, 2, hash)
    hash.clear()
    expect(hash.size).toBe(0)
  })

  it('handles negative coordinates', () => {
    const eid = placeChair(world, -5, -3, hash)
    const results = hash.queryAABB({ minX: -6, minZ: -4, maxX: -4, maxZ: -2 })
    expect(results).toContain(eid)
  })
})

// ═════════════════════════════════════════════════════════════════════════════
// Collision
// ═════════════════════════════════════════════════════════════════════════════

describe('Collision', () => {
  let world: EcsWorld
  let hash: SpatialHash

  beforeEach(() => {
    world = createEcsWorld()
    hash = new SpatialHash(2)
  })

  it('getEntityAABB returns correct bounds', () => {
    const eid = placeChair(world, 5, 3)
    const aabb = getEntityAABB(eid)
    // Chair half-extents: [0.22, 0.45, 0.22], scale 1
    expect(aabb.minX).toBeCloseTo(5 - 0.22)
    expect(aabb.maxX).toBeCloseTo(5 + 0.22)
    expect(aabb.minZ).toBeCloseTo(3 - 0.22)
    expect(aabb.maxZ).toBeCloseTo(3 + 0.22)
  })

  it('getEntityAABB accounts for scale', () => {
    const eid = createFurnitureEntity(world, {
      type: 'chair',
      position: [0, 0, 0],
      scale: [2, 2, 2],
    })
    const aabb = getEntityAABB(eid)
    expect(aabb.minX).toBeCloseTo(-0.44) // 0.22 * 2
    expect(aabb.maxX).toBeCloseTo(0.44)
  })

  it('aabbOverlap detects overlap', () => {
    expect(aabbOverlap(
      { minX: 0, minZ: 0, maxX: 2, maxZ: 2 },
      { minX: 1, minZ: 1, maxX: 3, maxZ: 3 },
    )).toBe(true)
  })

  it('aabbOverlap detects no overlap', () => {
    expect(aabbOverlap(
      { minX: 0, minZ: 0, maxX: 1, maxZ: 1 },
      { minX: 5, minZ: 5, maxX: 6, maxZ: 6 },
    )).toBe(false)
  })

  it('aabbOverlap detects touching edges', () => {
    expect(aabbOverlap(
      { minX: 0, minZ: 0, maxX: 1, maxZ: 1 },
      { minX: 1, minZ: 0, maxX: 2, maxZ: 1 },
    )).toBe(true) // touching = overlapping
  })

  it('detectCollisions finds overlapping entities', () => {
    // Two chairs at the same position
    const e1 = placeChair(world, 5, 5, hash)
    const e2 = placeChair(world, 5.1, 5, hash) // very close, AABBs overlap

    const pairs = detectCollisions(hash, [e1, e2])
    expect(pairs).toHaveLength(1)
    expect(pairs[0]!.a).toBe(Math.min(e1, e2))
    expect(pairs[0]!.b).toBe(Math.max(e1, e2))
  })

  it('detectCollisions returns empty for distant entities', () => {
    const e1 = placeChair(world, 0, 0, hash)
    const e2 = placeChair(world, 50, 50, hash)

    const pairs = detectCollisions(hash, [e1, e2])
    expect(pairs).toHaveLength(0)
  })

  it('checkCollision returns colliders for one entity', () => {
    const e1 = placeChair(world, 5, 5, hash)
    const e2 = placeChair(world, 5.1, 5, hash)
    placeChair(world, 50, 50, hash) // far away

    const colliders = checkCollision(hash, e1)
    expect(colliders).toContain(e2)
    expect(colliders).toHaveLength(1)
  })

  it('checkCollision respects exclude set', () => {
    const e1 = placeChair(world, 5, 5, hash)
    const e2 = placeChair(world, 5.1, 5, hash)

    const colliders = checkCollision(hash, e1, new Set([e2]))
    expect(colliders).toHaveLength(0)
  })

  it('checkPlacementCollision checks without existing entity', () => {
    placeChair(world, 5, 5, hash)

    const colliders = checkPlacementCollision(hash, 5.1, 5, 0.22, 0.22)
    expect(colliders.length).toBeGreaterThan(0)
  })

  it('checkPlacementCollision returns empty for clear area', () => {
    placeChair(world, 5, 5, hash)

    const colliders = checkPlacementCollision(hash, 50, 50, 0.22, 0.22)
    expect(colliders).toHaveLength(0)
  })
})

// ═════════════════════════════════════════════════════════════════════════════
// Selection
// ═════════════════════════════════════════════════════════════════════════════

describe('Selection', () => {
  let world: EcsWorld
  let hash: SpatialHash

  beforeEach(() => {
    world = createEcsWorld()
    hash = new SpatialHash(2)
  })

  it('rectSelect finds entities within rectangle', () => {
    const e1 = placeChair(world, 2, 2, hash)
    const e2 = placeChair(world, 3, 3, hash)
    placeChair(world, 50, 50, hash)

    const selected = rectSelect(hash, world, { minX: 0, minZ: 0, maxX: 5, maxZ: 5 })
    expect(selected).toContain(e1)
    expect(selected).toContain(e2)
    expect(selected).toHaveLength(2)
  })

  it('rectSelect returns empty when no entities in rect', () => {
    placeChair(world, 50, 50, hash)
    const selected = rectSelect(hash, world, { minX: 0, minZ: 0, maxX: 5, maxZ: 5 })
    expect(selected).toHaveLength(0)
  })

  it('pointSelect finds closest entity', () => {
    const e1 = placeChair(world, 5, 5, hash)
    placeChair(world, 6, 6, hash)

    const result = pointSelect(hash, world, 5.1, 5.1, 1.0)
    expect(result).toBe(e1)
  })

  it('pointSelect returns undefined when nothing nearby', () => {
    placeChair(world, 50, 50, hash)
    const result = pointSelect(hash, world, 0, 0, 1.0)
    expect(result).toBeUndefined()
  })
})

// ═════════════════════════════════════════════════════════════════════════════
// Snapping
// ═════════════════════════════════════════════════════════════════════════════

describe('Snapping', () => {
  let world: EcsWorld
  let hash: SpatialHash

  beforeEach(() => {
    world = createEcsWorld()
    hash = new SpatialHash(2)
  })

  it('snapToGrid snaps to nearest 0.5m grid point', () => {
    expect(snapToGrid(1.3, 2.7)).toEqual([1.5, 2.5])
    expect(snapToGrid(1.0, 2.0)).toEqual([1.0, 2.0])
    expect(snapToGrid(0.24, 0.26)).toEqual([0.0, 0.5])
  })

  it('snapToGrid with custom grid size', () => {
    expect(snapToGrid(1.3, 2.7, 1.0)).toEqual([1.0, 3.0])
  })

  it('snapToHeightGrid snaps Y to 0.1m increments', () => {
    expect(snapToHeightGrid(0.55)).toBeCloseTo(0.6)
    expect(snapToHeightGrid(0.44)).toBeCloseTo(0.4)
  })

  it('findNearest returns closest entity', () => {
    placeChair(world, 3, 3, hash)
    const e2 = placeChair(world, 1, 1, hash)
    placeChair(world, 10, 10, hash)

    const result = findNearest(hash, 0.5, 0.5, 5.0)
    expect(result).toBeDefined()
    expect(result!.eid).toBe(e2)
    expect(result!.distance).toBeCloseTo(Math.sqrt(0.25 + 0.25))
  })

  it('findNearest returns undefined when no entities in radius', () => {
    placeChair(world, 50, 50, hash)
    const result = findNearest(hash, 0, 0, 1.0)
    expect(result).toBeUndefined()
  })

  it('findNearest respects exclude set', () => {
    const e1 = placeChair(world, 1, 1, hash)
    const e2 = placeChair(world, 3, 3, hash)

    const result = findNearest(hash, 0, 0, 10, new Set([e1]))
    expect(result).toBeDefined()
    expect(result!.eid).toBe(e2)
  })

  it('findKNearest returns K closest sorted by distance', () => {
    placeChair(world, 1, 0, hash)
    placeChair(world, 2, 0, hash)
    placeChair(world, 3, 0, hash)
    placeChair(world, 100, 0, hash)

    const results = findKNearest(hash, 0, 0, 2, 10)
    expect(results).toHaveLength(2)
    expect(results[0]!.distance).toBeCloseTo(1)
    expect(results[1]!.distance).toBeCloseTo(2)
  })

  it('snapToNearest aligns to nearby entity axis', () => {
    placeChair(world, 5, 5, hash)

    // Very close in X, slightly off in Z — within snap distance
    const [sx, sz] = snapToNearest(hash, 5.1, 5.3, 0.5)
    expect(sx).toBeCloseTo(5.0) // snaps X to match entity (dx=0.1 < dz=0.3)
    expect(sz).toBeCloseTo(5.3) // Z unchanged
  })

  it('snapToNearest returns original when no nearby entity', () => {
    const [sx, sz] = snapToNearest(hash, 5, 5, 0.5)
    expect(sx).toBeCloseTo(5)
    expect(sz).toBeCloseTo(5)
  })
})

// ═════════════════════════════════════════════════════════════════════════════
// Performance: 1000 items
// ═════════════════════════════════════════════════════════════════════════════

describe('ECS Systems: 1000-item benchmark', () => {
  it('builds spatial hash for 1000 items in < 50ms', () => {
    const world = createEcsWorld()
    const hash = new SpatialHash(2)

    // Create 1000 items in a 100x10 grid
    for (let i = 0; i < 1000; i++) {
      const x = (i % 100) * 0.6
      const z = Math.floor(i / 100) * 0.6
      createFurnitureEntity(world, { type: 'chair', position: [x, 0, z] })
    }

    const start = performance.now()
    hash.rebuild(world)
    const elapsed = performance.now() - start

    expect(hash.size).toBe(1000)
    expect(elapsed).toBeLessThan(50)
  })

  it('spatial query on 1000 items completes in < 1ms', () => {
    const world = createEcsWorld()
    const hash = new SpatialHash(2)

    for (let i = 0; i < 1000; i++) {
      const x = (i % 100) * 0.6
      const z = Math.floor(i / 100) * 0.6
      const eid = createFurnitureEntity(world, { type: 'chair', position: [x, 0, z] })
      hash.insert(eid, x, z)
    }

    // Query a 5x5 area in the middle
    const start = performance.now()
    const results = hash.queryAABB({ minX: 25, minZ: 2, maxX: 30, maxZ: 4 })
    const elapsed = performance.now() - start

    expect(results.length).toBeGreaterThan(0)
    expect(elapsed).toBeLessThan(1)
  })

  it('collision detection on 1000 items completes in < 50ms', () => {
    const world = createEcsWorld()
    const hash = new SpatialHash(2)
    const eids: number[] = []

    // Place chairs at 0.3m spacing — close enough for AABBs to overlap (half-extent 0.22)
    for (let i = 0; i < 1000; i++) {
      const x = (i % 100) * 0.3
      const z = Math.floor(i / 100) * 0.3
      const eid = createFurnitureEntity(world, { type: 'chair', position: [x, 0, z] })
      hash.insert(eid, x, z)
      eids.push(eid)
    }

    const start = performance.now()
    const pairs = detectCollisions(hash, eids)
    const elapsed = performance.now() - start

    // Chairs at 0.3m spacing with 0.44m width (2*0.22) definitely overlap
    expect(pairs.length).toBeGreaterThan(0)
    expect(elapsed).toBeLessThan(50)
  })

  it('rect-select on 1000 items completes in < 1ms', () => {
    const world = createEcsWorld()
    const hash = new SpatialHash(2)

    for (let i = 0; i < 1000; i++) {
      const x = (i % 100) * 0.6
      const z = Math.floor(i / 100) * 0.6
      const eid = createFurnitureEntity(world, { type: 'chair', position: [x, 0, z] })
      hash.insert(eid, x, z)
    }

    const start = performance.now()
    const selected = rectSelect(hash, world, { minX: 10, minZ: 0, maxX: 20, maxZ: 3 })
    const elapsed = performance.now() - start

    expect(selected.length).toBeGreaterThan(0)
    expect(elapsed).toBeLessThan(1)
  })
})
