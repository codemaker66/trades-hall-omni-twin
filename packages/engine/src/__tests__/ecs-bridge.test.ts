import { describe, it, expect, beforeEach } from 'vitest'
import { EcsBridge, type BridgeItem } from '../ecs/bridge'
import { Position, Rotation, Scale, FurnitureTag, GroupMember, furnitureTypeToIndex } from '../ecs/components'
import { rectSelect } from '../ecs/systems/selection'

// ─── Helpers ────────────────────────────────────────────────────────────────

function makeChairItem(id: string, x: number, z: number): BridgeItem {
  return {
    id,
    type: 'chair',
    position: [x, 0, z],
    rotation: [0, -Math.PI / 2, 0],
  }
}

function makeTableItem(id: string, x: number, z: number): BridgeItem {
  return {
    id,
    type: 'round-table',
    position: [x, 0, z],
    rotation: [0, 0, 0],
    scale: [1.5, 1.5, 1.5],
  }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

describe('EcsBridge', () => {
  let bridge: EcsBridge

  beforeEach(() => {
    bridge = new EcsBridge()
  })

  it('starts empty', () => {
    expect(bridge.size).toBe(0)
    expect(bridge.readAll()).toHaveLength(0)
  })

  // ─── addItem ────────────────────────────────────────────────────────────

  it('adds an item and tracks in ID map', () => {
    const eid = bridge.addItem(makeChairItem('chair-1', 5, 3))
    expect(eid).toBeGreaterThanOrEqual(0)
    expect(bridge.size).toBe(1)
    expect(bridge.getEid('chair-1')).toBe(eid)
    expect(bridge.getDomainId(eid)).toBe('chair-1')
  })

  it('sets correct Position on ECS entity', () => {
    const eid = bridge.addItem(makeChairItem('c1', 7, 9))
    expect(Position.x[eid]).toBeCloseTo(7)
    expect(Position.y[eid]).toBeCloseTo(0)
    expect(Position.z[eid]).toBeCloseTo(9)
  })

  it('sets correct Rotation on ECS entity', () => {
    const eid = bridge.addItem(makeChairItem('c1', 0, 0))
    expect(Rotation.x[eid]).toBeCloseTo(0)
    expect(Rotation.y[eid]).toBeCloseTo(-Math.PI / 2)
    expect(Rotation.z[eid]).toBeCloseTo(0)
  })

  it('sets Scale from item', () => {
    const eid = bridge.addItem(makeTableItem('t1', 0, 0))
    expect(Scale.x[eid]).toBeCloseTo(1.5)
    expect(Scale.y[eid]).toBeCloseTo(1.5)
    expect(Scale.z[eid]).toBeCloseTo(1.5)
  })

  it('defaults Scale to 1,1,1 when not provided', () => {
    const eid = bridge.addItem(makeChairItem('c1', 0, 0))
    expect(Scale.x[eid]).toBeCloseTo(1)
    expect(Scale.y[eid]).toBeCloseTo(1)
    expect(Scale.z[eid]).toBeCloseTo(1)
  })

  it('sets FurnitureTag type', () => {
    const eid = bridge.addItem(makeTableItem('t1', 0, 0))
    expect(FurnitureTag.type[eid]).toBe(furnitureTypeToIndex('round-table'))
  })

  it('inserts into spatial hash', () => {
    bridge.addItem(makeChairItem('c1', 5, 5))
    const results = bridge.spatialHash.queryRadius(5, 5, 1)
    expect(results).toHaveLength(1)
  })

  // ─── removeItem ─────────────────────────────────────────────────────────

  it('removes an item by domain ID', () => {
    bridge.addItem(makeChairItem('c1', 5, 5))
    bridge.removeItem('c1')
    expect(bridge.size).toBe(0)
    expect(bridge.getEid('c1')).toBeUndefined()
  })

  it('removes from spatial hash on remove', () => {
    bridge.addItem(makeChairItem('c1', 5, 5))
    bridge.removeItem('c1')
    const results = bridge.spatialHash.queryRadius(5, 5, 1)
    expect(results).toHaveLength(0)
  })

  it('removing non-existent ID is a no-op', () => {
    bridge.removeItem('does-not-exist')
    expect(bridge.size).toBe(0)
  })

  // ─── moveItem ───────────────────────────────────────────────────────────

  it('moves an item to a new position', () => {
    const eid = bridge.addItem(makeChairItem('c1', 5, 5))
    bridge.moveItem('c1', [10, 0, 10])
    expect(Position.x[eid]).toBeCloseTo(10)
    expect(Position.z[eid]).toBeCloseTo(10)
  })

  it('updates spatial hash on move', () => {
    bridge.addItem(makeChairItem('c1', 5, 5))
    bridge.moveItem('c1', [50, 0, 50])

    // Old position: empty
    const oldResults = bridge.spatialHash.queryRadius(5, 5, 1)
    expect(oldResults).toHaveLength(0)

    // New position: has entity
    const newResults = bridge.spatialHash.queryRadius(50, 50, 1)
    expect(newResults).toHaveLength(1)
  })

  // ─── rotateItem ─────────────────────────────────────────────────────────

  it('rotates an item', () => {
    const eid = bridge.addItem(makeChairItem('c1', 0, 0))
    bridge.rotateItem('c1', [0, Math.PI, 0])
    expect(Rotation.y[eid]).toBeCloseTo(Math.PI)
  })

  // ─── moveItems (batch) ──────────────────────────────────────────────────

  it('batch moves multiple items', () => {
    bridge.addItem(makeChairItem('c1', 0, 0))
    bridge.addItem(makeChairItem('c2', 1, 1))

    bridge.moveItems([
      { id: 'c1', position: [10, 0, 10] },
      { id: 'c2', position: [20, 0, 20] },
    ])

    const eid1 = bridge.getEid('c1')!
    const eid2 = bridge.getEid('c2')!
    expect(Position.x[eid1]).toBeCloseTo(10)
    expect(Position.x[eid2]).toBeCloseTo(20)
  })

  // ─── syncAll ────────────────────────────────────────────────────────────

  it('syncAll replaces all entities', () => {
    bridge.addItem(makeChairItem('old-1', 0, 0))
    bridge.addItem(makeChairItem('old-2', 1, 1))
    expect(bridge.size).toBe(2)

    bridge.syncAll([
      makeChairItem('new-1', 5, 5),
      makeTableItem('new-2', 10, 10),
      makeChairItem('new-3', 15, 15),
    ])

    expect(bridge.size).toBe(3)
    expect(bridge.getEid('old-1')).toBeUndefined()
    expect(bridge.getEid('old-2')).toBeUndefined()
    expect(bridge.getEid('new-1')).toBeDefined()
    expect(bridge.getEid('new-2')).toBeDefined()
    expect(bridge.getEid('new-3')).toBeDefined()
  })

  it('syncAll updates spatial hash', () => {
    bridge.addItem(makeChairItem('c1', 0, 0))
    bridge.syncAll([makeChairItem('c2', 50, 50)])

    const oldResults = bridge.spatialHash.queryRadius(0, 0, 1)
    expect(oldResults).toHaveLength(0)

    const newResults = bridge.spatialHash.queryRadius(50, 50, 1)
    expect(newResults).toHaveLength(1)
  })

  // ─── readItem / readAll ─────────────────────────────────────────────────

  it('readItem returns domain-shaped data', () => {
    bridge.addItem(makeChairItem('c1', 7, 3))
    const item = bridge.readItem('c1')

    expect(item).toBeDefined()
    expect(item!.id).toBe('c1')
    expect(item!.type).toBe('chair')
    expect(item!.position[0]).toBeCloseTo(7)
    expect(item!.position[2]).toBeCloseTo(3)
  })

  it('readItem returns undefined for unknown ID', () => {
    expect(bridge.readItem('nope')).toBeUndefined()
  })

  it('readAll returns all items', () => {
    bridge.addItem(makeChairItem('c1', 0, 0))
    bridge.addItem(makeTableItem('t1', 5, 5))

    const all = bridge.readAll()
    expect(all).toHaveLength(2)

    const ids = all.map((i) => i.id).sort()
    expect(ids).toEqual(['c1', 't1'])
  })

  it('readItem reflects move changes', () => {
    bridge.addItem(makeChairItem('c1', 0, 0))
    bridge.moveItem('c1', [42, 1, 99])

    const item = bridge.readItem('c1')
    expect(item!.position[0]).toBeCloseTo(42)
    expect(item!.position[1]).toBeCloseTo(1)
    expect(item!.position[2]).toBeCloseTo(99)
  })

  // ─── Spatial query integration ──────────────────────────────────────────

  it('spatial queries work after addItem', () => {
    bridge.addItem(makeChairItem('c1', 5, 5))
    bridge.addItem(makeChairItem('c2', 6, 6))
    bridge.addItem(makeChairItem('c3', 50, 50))

    const selected = rectSelect(
      bridge.spatialHash,
      bridge.world,
      { minX: 4, minZ: 4, maxX: 7, maxZ: 7 },
    )
    expect(selected).toHaveLength(2)

    // Verify we can map back to domain IDs
    const domainIds = selected.map((eid) => bridge.getDomainId(eid)).sort()
    expect(domainIds).toEqual(['c1', 'c2'])
  })

  it('spatial queries work after syncAll', () => {
    const items: BridgeItem[] = []
    for (let i = 0; i < 50; i++) {
      items.push(makeChairItem(`c${i}`, i * 0.5, 0))
    }

    bridge.syncAll(items)

    const selected = rectSelect(
      bridge.spatialHash,
      bridge.world,
      { minX: 0, minZ: -1, maxX: 5, maxZ: 1 },
    )
    // Items at x=0, 0.5, 1, ... 5 → 11 items
    expect(selected).toHaveLength(11)
  })

  // ─── Groups ─────────────────────────────────────────────────────────────

  it('assigns consistent group eids for same groupId', () => {
    const item1: BridgeItem = { ...makeChairItem('c1', 0, 0), groupId: 'grp-a' }
    const item2: BridgeItem = { ...makeChairItem('c2', 1, 0), groupId: 'grp-a' }
    const item3: BridgeItem = { ...makeChairItem('c3', 2, 0), groupId: 'grp-b' }

    bridge.addItem(item1)
    bridge.addItem(item2)
    bridge.addItem(item3)

    // Items in same group should have same groupId eid
    const eid1 = bridge.getEid('c1')!
    const eid2 = bridge.getEid('c2')!
    const eid3 = bridge.getEid('c3')!

    expect(GroupMember.groupId[eid1]).toBe(GroupMember.groupId[eid2])
    expect(GroupMember.groupId[eid1]).not.toBe(GroupMember.groupId[eid3])
  })

  // ─── Performance ────────────────────────────────────────────────────────

  it('syncAll with 1000 items completes in < 100ms', () => {
    const items: BridgeItem[] = []
    for (let i = 0; i < 1000; i++) {
      items.push(makeChairItem(`c${i}`, (i % 100) * 0.5, Math.floor(i / 100) * 0.5))
    }

    const start = performance.now()
    bridge.syncAll(items)
    const elapsed = performance.now() - start

    expect(bridge.size).toBe(1000)
    expect(elapsed).toBeLessThan(100)
  })
})
