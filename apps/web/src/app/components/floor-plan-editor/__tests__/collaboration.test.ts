import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import * as Y from 'yjs'
import { useFloorPlanStore, type FloorPlanItem } from '../store'
import {
  syncItemsToDoc,
  readItemsFromDoc,
  floorPlanItemToYMap,
  yMapToFloorPlanItem,
  addItemToDoc,
  updateItemInDoc,
  removeItemsFromDoc,
  findItemIndex,
  getItemsArray,
  syncSettingsToDoc,
  readSettingsFromDoc,
} from '../collab/yjsModel'
import { createYjsBridge } from '../collab/yjsBridge'
import {
  setLocalPresence,
  getRemotePresences,
  getPresenceColor,
  type UserPresence,
} from '../collab/awareness'
import { Awareness } from 'y-protocols/awareness'

// ─── Helpers ─────────────────────────────────────────────────────────────────

function resetStore() {
  useFloorPlanStore.setState({
    items: [],
    selectedIds: [],
    past: [],
    future: [],
    batch: null,
    canUndo: false,
    canRedo: false,
    zoom: 1,
    panX: 0,
    panY: 0,
    gridSizeFt: 1,
    snapEnabled: false, // disable snap for predictable positions in tests
    tool: 'select',
    planWidthFt: 80,
    planHeightFt: 50,
  })
}

function makeItem(overrides: Partial<FloorPlanItem> = {}): FloorPlanItem {
  return {
    id: `item-${Math.random().toString(36).slice(2, 6)}`,
    name: 'Chair',
    category: 'chair',
    x: 10,
    y: 10,
    widthFt: 1.5,
    depthFt: 1.5,
    rotation: 0,
    locked: false,
    ...overrides,
  }
}

// ─── Y.Doc Data Model ────────────────────────────────────────────────────────

describe('Yjs data model', () => {
  let doc: Y.Doc

  beforeEach(() => {
    doc = new Y.Doc()
  })

  afterEach(() => {
    doc.destroy()
  })

  describe('floorPlanItemToYMap / yMapToFloorPlanItem', () => {
    it('round-trips a floor plan item', () => {
      const item = makeItem({ id: 'test-1', x: 15.5, y: 20.3, rotation: 45 })
      // Y.Map must be added to a doc before reading data
      const yItems = getItemsArray(doc)
      yItems.push([floorPlanItemToYMap(item)])
      const result = yMapToFloorPlanItem(yItems.get(0))
      expect(result).toEqual(item)
    })

    it('preserves all properties', () => {
      const item = makeItem({
        id: 'full',
        name: 'Round Table 6ft',
        category: 'table',
        x: 40,
        y: 25,
        widthFt: 6,
        depthFt: 6,
        rotation: 90,
        locked: true,
      })
      const yItems = getItemsArray(doc)
      yItems.push([floorPlanItemToYMap(item)])
      const result = yMapToFloorPlanItem(yItems.get(0))
      expect(result.id).toBe('full')
      expect(result.name).toBe('Round Table 6ft')
      expect(result.category).toBe('table')
      expect(result.locked).toBe(true)
    })
  })

  describe('syncItemsToDoc / readItemsFromDoc', () => {
    it('syncs items to doc and reads them back', () => {
      const items = [makeItem({ id: 'a' }), makeItem({ id: 'b' })]
      syncItemsToDoc(doc, items)
      const result = readItemsFromDoc(doc)
      expect(result).toHaveLength(2)
      expect(result[0]!.id).toBe('a')
      expect(result[1]!.id).toBe('b')
    })

    it('replaces existing items on re-sync', () => {
      syncItemsToDoc(doc, [makeItem({ id: 'old' })])
      syncItemsToDoc(doc, [makeItem({ id: 'new1' }), makeItem({ id: 'new2' })])
      const result = readItemsFromDoc(doc)
      expect(result).toHaveLength(2)
      expect(result[0]!.id).toBe('new1')
    })

    it('handles empty array', () => {
      syncItemsToDoc(doc, [])
      expect(readItemsFromDoc(doc)).toHaveLength(0)
    })
  })

  describe('addItemToDoc', () => {
    it('adds an item to the doc', () => {
      const item = makeItem({ id: 'added' })
      addItemToDoc(doc, item)
      const result = readItemsFromDoc(doc)
      expect(result).toHaveLength(1)
      expect(result[0]!.id).toBe('added')
    })

    it('appends to existing items', () => {
      syncItemsToDoc(doc, [makeItem({ id: 'first' })])
      addItemToDoc(doc, makeItem({ id: 'second' }))
      expect(readItemsFromDoc(doc)).toHaveLength(2)
    })
  })

  describe('updateItemInDoc', () => {
    it('updates item properties', () => {
      syncItemsToDoc(doc, [makeItem({ id: 'u1', x: 10, y: 10 })])
      updateItemInDoc(doc, 'u1', { x: 30, y: 40 })
      const result = readItemsFromDoc(doc)
      expect(result[0]!.x).toBe(30)
      expect(result[0]!.y).toBe(40)
    })

    it('does not change id', () => {
      syncItemsToDoc(doc, [makeItem({ id: 'keep-id' })])
      updateItemInDoc(doc, 'keep-id', { id: 'changed' } as Partial<FloorPlanItem>)
      expect(readItemsFromDoc(doc)[0]!.id).toBe('keep-id')
    })

    it('handles non-existent id gracefully', () => {
      syncItemsToDoc(doc, [makeItem({ id: 'exists' })])
      updateItemInDoc(doc, 'nope', { x: 99 })
      expect(readItemsFromDoc(doc)[0]!.x).toBe(10) // unchanged
    })
  })

  describe('removeItemsFromDoc', () => {
    it('removes items by id', () => {
      syncItemsToDoc(doc, [
        makeItem({ id: 'keep' }),
        makeItem({ id: 'remove1' }),
        makeItem({ id: 'remove2' }),
      ])
      removeItemsFromDoc(doc, ['remove1', 'remove2'])
      const result = readItemsFromDoc(doc)
      expect(result).toHaveLength(1)
      expect(result[0]!.id).toBe('keep')
    })

    it('handles removing non-existent ids', () => {
      syncItemsToDoc(doc, [makeItem({ id: 'a' })])
      removeItemsFromDoc(doc, ['nonexistent'])
      expect(readItemsFromDoc(doc)).toHaveLength(1)
    })
  })

  describe('findItemIndex', () => {
    it('finds item index by id', () => {
      syncItemsToDoc(doc, [
        makeItem({ id: 'first' }),
        makeItem({ id: 'second' }),
        makeItem({ id: 'third' }),
      ])
      expect(findItemIndex(doc, 'second')).toBe(1)
    })

    it('returns -1 for non-existent id', () => {
      syncItemsToDoc(doc, [makeItem({ id: 'only' })])
      expect(findItemIndex(doc, 'nope')).toBe(-1)
    })
  })

  describe('settings sync', () => {
    it('syncs and reads plan settings', () => {
      syncSettingsToDoc(doc, 100, 60)
      const settings = readSettingsFromDoc(doc)
      expect(settings).toEqual({ planWidthFt: 100, planHeightFt: 60 })
    })

    it('returns null when no settings set', () => {
      expect(readSettingsFromDoc(doc)).toBeNull()
    })
  })
})

// ─── Yjs ↔ Zustand Bridge ───────────────────────────────────────────────────

describe('Yjs bridge', () => {
  let doc: Y.Doc

  beforeEach(() => {
    doc = new Y.Doc()
    resetStore()
  })

  afterEach(() => {
    doc.destroy()
  })

  describe('pushToDoc', () => {
    it('pushes store items into Y.Doc', () => {
      useFloorPlanStore.getState().addItem({
        name: 'Chair', category: 'chair', x: 10, y: 10,
        widthFt: 1.5, depthFt: 1.5, rotation: 0,
      })

      const bridge = createYjsBridge(doc)
      bridge.pushToDoc()

      const items = readItemsFromDoc(doc)
      expect(items).toHaveLength(1)
      expect(items[0]!.name).toBe('Chair')
    })
  })

  describe('pullFromDoc', () => {
    it('pulls Y.Doc items into store', () => {
      syncItemsToDoc(doc, [
        makeItem({ id: 'remote-1', name: 'Remote Table', category: 'table' }),
      ])

      const bridge = createYjsBridge(doc)
      bridge.pullFromDoc()

      const items = useFloorPlanStore.getState().items
      expect(items).toHaveLength(1)
      expect(items[0]!.id).toBe('remote-1')
      expect(items[0]!.name).toBe('Remote Table')
    })
  })

  describe('bidirectional sync', () => {
    it('store changes propagate to Y.Doc', () => {
      const bridge = createYjsBridge(doc)
      bridge.connect()

      useFloorPlanStore.getState().addItem({
        name: 'Chair', category: 'chair', x: 20, y: 30,
        widthFt: 1.5, depthFt: 1.5, rotation: 0,
      })

      const items = readItemsFromDoc(doc)
      expect(items).toHaveLength(1)
      expect(items[0]!.x).toBe(20)

      bridge.disconnect()
    })

    it('Y.Doc changes propagate to store', () => {
      const bridge = createYjsBridge(doc)
      bridge.connect()

      addItemToDoc(doc, makeItem({ id: 'from-yjs', x: 50, y: 50 }))

      const items = useFloorPlanStore.getState().items
      expect(items).toHaveLength(1)
      expect(items[0]!.id).toBe('from-yjs')

      bridge.disconnect()
    })

    it('no infinite loop: store→doc→store stops', () => {
      const bridge = createYjsBridge(doc)
      bridge.connect()

      // This should not loop — echo suppression prevents it
      useFloorPlanStore.getState().addItem({
        name: 'Test', category: 'chair', x: 10, y: 10,
        widthFt: 1.5, depthFt: 1.5, rotation: 0,
      })

      expect(readItemsFromDoc(doc)).toHaveLength(1)
      expect(useFloorPlanStore.getState().items).toHaveLength(1)

      bridge.disconnect()
    })

    it('item removal syncs store to doc', () => {
      const bridge = createYjsBridge(doc)
      bridge.connect()

      const id = useFloorPlanStore.getState().addItem({
        name: 'Chair', category: 'chair', x: 10, y: 10,
        widthFt: 1.5, depthFt: 1.5, rotation: 0,
      })

      expect(readItemsFromDoc(doc)).toHaveLength(1)

      useFloorPlanStore.getState().removeItems([id])
      expect(readItemsFromDoc(doc)).toHaveLength(0)

      bridge.disconnect()
    })

    it('item update syncs store to doc', () => {
      const bridge = createYjsBridge(doc)
      bridge.connect()

      const id = useFloorPlanStore.getState().addItem({
        name: 'Chair', category: 'chair', x: 10, y: 10,
        widthFt: 1.5, depthFt: 1.5, rotation: 0,
      })

      useFloorPlanStore.getState().updateItems([{ id, changes: { x: 50, y: 60 } }])

      const docItems = readItemsFromDoc(doc)
      expect(docItems[0]!.x).toBe(50)
      expect(docItems[0]!.y).toBe(60)

      bridge.disconnect()
    })
  })

  describe('two-client sync', () => {
    it('two Y.Docs sync items via shared state', () => {
      // Simulate two clients with separate Y.Docs that share updates
      const doc1 = new Y.Doc()
      const doc2 = new Y.Doc()

      // Wire them together: doc1 updates → doc2, doc2 updates → doc1
      doc1.on('update', (update: Uint8Array) => {
        Y.applyUpdate(doc2, update)
      })
      doc2.on('update', (update: Uint8Array) => {
        Y.applyUpdate(doc1, update)
      })

      // Client 1 adds an item
      addItemToDoc(doc1, makeItem({ id: 'from-client-1', x: 10, y: 10 }))

      // Client 2 should see it
      const doc2Items = readItemsFromDoc(doc2)
      expect(doc2Items).toHaveLength(1)
      expect(doc2Items[0]!.id).toBe('from-client-1')

      // Client 2 adds another item
      addItemToDoc(doc2, makeItem({ id: 'from-client-2', x: 20, y: 20 }))

      // Both should see both items
      expect(readItemsFromDoc(doc1)).toHaveLength(2)
      expect(readItemsFromDoc(doc2)).toHaveLength(2)

      // Client 1 updates client 2's item
      updateItemInDoc(doc1, 'from-client-2', { x: 99 })

      // Client 2 sees the update
      const updated = readItemsFromDoc(doc2).find((i) => i.id === 'from-client-2')
      expect(updated!.x).toBe(99)

      doc1.destroy()
      doc2.destroy()
    })

    it('concurrent edits to different items merge correctly', () => {
      const doc1 = new Y.Doc()
      const doc2 = new Y.Doc()

      // Initial state
      const items = [
        makeItem({ id: 'item-a', x: 10 }),
        makeItem({ id: 'item-b', x: 20 }),
      ]
      syncItemsToDoc(doc1, items)
      Y.applyUpdate(doc2, Y.encodeStateAsUpdate(doc1))

      // Concurrent edits: client 1 moves item-a, client 2 moves item-b
      updateItemInDoc(doc1, 'item-a', { x: 50 })
      updateItemInDoc(doc2, 'item-b', { x: 60 })

      // Merge updates
      Y.applyUpdate(doc2, Y.encodeStateAsUpdate(doc1))
      Y.applyUpdate(doc1, Y.encodeStateAsUpdate(doc2))

      // Both docs should have both edits
      const result1 = readItemsFromDoc(doc1)
      const result2 = readItemsFromDoc(doc2)

      expect(result1.find((i) => i.id === 'item-a')!.x).toBe(50)
      expect(result1.find((i) => i.id === 'item-b')!.x).toBe(60)
      expect(result2.find((i) => i.id === 'item-a')!.x).toBe(50)
      expect(result2.find((i) => i.id === 'item-b')!.x).toBe(60)

      doc1.destroy()
      doc2.destroy()
    })

    it('concurrent edits to same item (last-writer-wins per property)', () => {
      const doc1 = new Y.Doc()
      const doc2 = new Y.Doc()

      syncItemsToDoc(doc1, [makeItem({ id: 'shared', x: 10, y: 10 })])
      Y.applyUpdate(doc2, Y.encodeStateAsUpdate(doc1))

      // Both clients edit x of same item
      updateItemInDoc(doc1, 'shared', { x: 50 })
      updateItemInDoc(doc2, 'shared', { x: 70 })

      // Merge (order matters for LWW, but both converge)
      Y.applyUpdate(doc2, Y.encodeStateAsUpdate(doc1))
      Y.applyUpdate(doc1, Y.encodeStateAsUpdate(doc2))

      // Both docs converge to same value
      const x1 = readItemsFromDoc(doc1).find((i) => i.id === 'shared')!.x
      const x2 = readItemsFromDoc(doc2).find((i) => i.id === 'shared')!.x
      expect(x1).toBe(x2) // converged (exact value depends on doc ordering)

      doc1.destroy()
      doc2.destroy()
    })
  })
})

// ─── Awareness / Presence ────────────────────────────────────────────────────

describe('awareness', () => {
  it('getPresenceColor returns consistent colors', () => {
    const color1 = getPresenceColor(0)
    const color2 = getPresenceColor(0)
    expect(color1).toBe(color2)
    expect(color1).toMatch(/^#[0-9a-fA-F]{6}$/)
  })

  it('different client IDs can get different colors', () => {
    const colors = new Set<string>()
    for (let i = 0; i < 12; i++) {
      colors.add(getPresenceColor(i))
    }
    expect(colors.size).toBe(12) // 12 unique colors in palette
  })

  it('setLocalPresence sets awareness state', () => {
    const doc = new Y.Doc()
    const awareness = new Awareness(doc)

    setLocalPresence(awareness, {
      name: 'Alice',
      color: '#ff0000',
      cursor: { x: 10, y: 20 },
      selectedIds: ['item-1'],
    })

    const state = awareness.getLocalState() as UserPresence
    expect(state.name).toBe('Alice')
    expect(state.cursor).toEqual({ x: 10, y: 20 })
    expect(state.selectedIds).toEqual(['item-1'])
    expect(state.lastUpdate).toBeGreaterThan(0)

    awareness.destroy()
    doc.destroy()
  })

  it('getRemotePresences excludes local user', () => {
    const doc = new Y.Doc()
    const awareness = new Awareness(doc)

    setLocalPresence(awareness, { name: 'Local', color: '#000', cursor: null, selectedIds: [] })

    const remote = getRemotePresences(awareness)
    expect(remote.size).toBe(0) // only local user, no remotes

    awareness.destroy()
    doc.destroy()
  })
})
