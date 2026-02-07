import { describe, it, expect, beforeEach } from 'vitest'
import { useVenueStore } from '../index'
import type { FurnitureItem } from '../types'

/**
 * Reset the store to a clean initial state before each test.
 * We use setState to wipe items, selection, and history so tests are isolated.
 */
function resetStore() {
    useVenueStore.getState().resetProject()
    // Also clear the history that resetProject itself records
    useVenueStore.setState({
        items: [],
        selectedIds: [],
        historyPast: [],
        historyFuture: [],
        historyBatch: null,
        canUndo: false,
        canRedo: false,
        inventoryWarning: null,
    })
}

describe('furniture slice', () => {
    beforeEach(() => {
        resetStore()
    })

    // ── addItem ────────────────────────────────────────────────────────

    describe('addItem', () => {
        it('creates an item with the correct type and default position/rotation', () => {
            useVenueStore.getState().addItem('round-table')

            const { items } = useVenueStore.getState()
            expect(items).toHaveLength(1)

            const item = items[0]!
            expect(item.type).toBe('round-table')
            expect(item.position).toEqual([0, 0, 0])
            expect(item.rotation).toEqual([0, 0, 0])
            expect(item.id).toBeTruthy()
        })

        it('creates an item with a specified position', () => {
            useVenueStore.getState().addItem('trestle-table', [5, 0, 10])

            const item = useVenueStore.getState().items[0]!
            expect(item.position).toEqual([5, 0, 10])
        })

        it('creates an item with a specified rotation', () => {
            useVenueStore.getState().addItem('round-table', [0, 0, 0], [0, Math.PI, 0])

            const item = useVenueStore.getState().items[0]!
            expect(item.rotation).toEqual([0, Math.PI, 0])
        })

        it('assigns a default chair rotation when type is chair and no rotation given', () => {
            useVenueStore.getState().addItem('chair')

            const item = useVenueStore.getState().items[0]!
            expect(item.type).toBe('chair')
            expect(item.rotation).toEqual([0, -Math.PI / 2, 0])
        })

        it('assigns groupId when provided', () => {
            const groupId = 'test-group-123'
            useVenueStore.getState().addItem('chair', [1, 0, 2], undefined, groupId)

            const item = useVenueStore.getState().items[0]!
            expect(item.groupId).toBe(groupId)
        })

        it('creates items with no groupId by default', () => {
            useVenueStore.getState().addItem('platform')

            const item = useVenueStore.getState().items[0]!
            expect(item.groupId).toBeUndefined()
        })

        it('clears selectedIds after adding an item', () => {
            // Seed a selection first
            useVenueStore.getState().addItem('round-table')
            const id = useVenueStore.getState().items[0]!.id
            useVenueStore.getState().setSelection([id])
            expect(useVenueStore.getState().selectedIds).toEqual([id])

            // Add another item — selection should be cleared
            useVenueStore.getState().addItem('chair')
            expect(useVenueStore.getState().selectedIds).toEqual([])
        })

        it('generates unique ids for each item', () => {
            useVenueStore.getState().addItem('round-table')
            useVenueStore.getState().addItem('round-table')
            useVenueStore.getState().addItem('chair')

            const ids = useVenueStore.getState().items.map((i) => i.id)
            const uniqueIds = new Set(ids)
            expect(uniqueIds.size).toBe(3)
        })
    })

    // ── removeItems ────────────────────────────────────────────────────

    describe('removeItems', () => {
        it('removes specified items by id', () => {
            useVenueStore.getState().addItem('round-table', [0, 0, 0])
            useVenueStore.getState().addItem('chair', [1, 0, 0])
            useVenueStore.getState().addItem('platform', [2, 0, 0])

            const items = useVenueStore.getState().items
            expect(items).toHaveLength(3)

            const idToRemove = items[1]!.id
            useVenueStore.getState().removeItems([idToRemove])

            const remaining = useVenueStore.getState().items
            expect(remaining).toHaveLength(2)
            expect(remaining.find((i) => i.id === idToRemove)).toBeUndefined()
        })

        it('removes multiple items at once', () => {
            useVenueStore.getState().addItem('round-table')
            useVenueStore.getState().addItem('chair')
            useVenueStore.getState().addItem('platform')

            const items = useVenueStore.getState().items
            const idsToRemove = [items[0]!.id, items[2]!.id]

            useVenueStore.getState().removeItems(idsToRemove)

            const remaining = useVenueStore.getState().items
            expect(remaining).toHaveLength(1)
            expect(remaining[0]!.type).toBe('chair')
        })

        it('does nothing when removing non-existent ids', () => {
            useVenueStore.getState().addItem('round-table')
            const before = useVenueStore.getState().items

            useVenueStore.getState().removeItems(['non-existent-id'])

            const after = useVenueStore.getState().items
            expect(after).toHaveLength(before.length)
        })

        it('does nothing when given an empty array', () => {
            useVenueStore.getState().addItem('round-table')
            const before = useVenueStore.getState().items

            useVenueStore.getState().removeItems([])

            const after = useVenueStore.getState().items
            expect(after).toHaveLength(before.length)
        })

        it('also removes deleted items from selectedIds', () => {
            useVenueStore.getState().addItem('round-table')
            useVenueStore.getState().addItem('chair')

            const items = useVenueStore.getState().items
            const idToRemove = items[0]!.id
            useVenueStore.getState().setSelection([idToRemove, items[1]!.id])

            useVenueStore.getState().removeItems([idToRemove])

            const { selectedIds } = useVenueStore.getState()
            expect(selectedIds).not.toContain(idToRemove)
            expect(selectedIds).toContain(items[1]!.id)
        })
    })

    // ── updateItems ────────────────────────────────────────────────────

    describe('updateItems', () => {
        it('updates positions of specified items', () => {
            useVenueStore.getState().addItem('round-table', [0, 0, 0])
            useVenueStore.getState().addItem('chair', [1, 0, 0])

            const items = useVenueStore.getState().items
            const id0 = items[0]!.id
            const id1 = items[1]!.id

            useVenueStore.getState().updateItems([
                { id: id0, changes: { position: [10, 0, 10] } },
                { id: id1, changes: { position: [20, 0, 20] } },
            ])

            const updated = useVenueStore.getState().items
            expect(updated.find((i) => i.id === id0)!.position).toEqual([10, 0, 10])
            expect(updated.find((i) => i.id === id1)!.position).toEqual([20, 0, 20])
        })

        it('updates rotation of a single item', () => {
            useVenueStore.getState().addItem('round-table', [0, 0, 0])
            const id = useVenueStore.getState().items[0]!.id

            useVenueStore.getState().updateItems([
                { id, changes: { rotation: [0, Math.PI / 4, 0] } },
            ])

            const updated = useVenueStore.getState().items.find((i) => i.id === id)!
            expect(updated.rotation).toEqual([0, Math.PI / 4, 0])
        })

        it('preserves other item properties when updating position', () => {
            useVenueStore.getState().addItem('trestle-table', [5, 0, 5], [0, 1, 0])
            const original = useVenueStore.getState().items[0]!

            useVenueStore.getState().updateItems([
                { id: original.id, changes: { position: [99, 0, 99] } },
            ])

            const updated = useVenueStore.getState().items[0]!
            expect(updated.type).toBe('trestle-table')
            expect(updated.rotation).toEqual([0, 1, 0])
            expect(updated.position).toEqual([99, 0, 99])
        })

        it('does nothing when given an empty updates array', () => {
            useVenueStore.getState().addItem('round-table')
            const before = useVenueStore.getState().items[0]!

            useVenueStore.getState().updateItems([])

            const after = useVenueStore.getState().items[0]!
            expect(after.position).toEqual(before.position)
        })

        it('does nothing when updating a non-existent id', () => {
            useVenueStore.getState().addItem('round-table', [1, 2, 3])
            const before = useVenueStore.getState().items[0]!

            useVenueStore.getState().updateItems([
                { id: 'non-existent', changes: { position: [99, 99, 99] } },
            ])

            const after = useVenueStore.getState().items[0]!
            expect(after.position).toEqual(before.position)
        })
    })

    // ── groupItems ─────────────────────────────────────────────────────

    describe('groupItems', () => {
        it('assigns a shared groupId to all specified items', () => {
            useVenueStore.getState().addItem('round-table')
            useVenueStore.getState().addItem('chair')
            useVenueStore.getState().addItem('chair')

            const items = useVenueStore.getState().items
            const ids = items.map((i) => i.id)

            useVenueStore.getState().groupItems([ids[0]!, ids[1]!])

            const updated = useVenueStore.getState().items
            const grouped0 = updated.find((i) => i.id === ids[0]!)!
            const grouped1 = updated.find((i) => i.id === ids[1]!)!
            const ungrouped = updated.find((i) => i.id === ids[2]!)!

            expect(grouped0.groupId).toBeTruthy()
            expect(grouped0.groupId).toBe(grouped1.groupId)
            expect(ungrouped.groupId).toBeUndefined()
        })

        it('does nothing when fewer than 2 ids are provided', () => {
            useVenueStore.getState().addItem('round-table')
            const id = useVenueStore.getState().items[0]!.id

            useVenueStore.getState().groupItems([id])

            const item = useVenueStore.getState().items[0]!
            expect(item.groupId).toBeUndefined()
        })

        it('does nothing with an empty array', () => {
            useVenueStore.getState().addItem('round-table')

            useVenueStore.getState().groupItems([])

            const item = useVenueStore.getState().items[0]!
            expect(item.groupId).toBeUndefined()
        })

        it('creates a unique groupId each time', () => {
            useVenueStore.getState().addItem('round-table')
            useVenueStore.getState().addItem('chair')
            useVenueStore.getState().addItem('platform')
            useVenueStore.getState().addItem('trestle-table')

            const items = useVenueStore.getState().items
            const ids = items.map((i) => i.id)

            useVenueStore.getState().groupItems([ids[0]!, ids[1]!])
            const groupId1 = useVenueStore.getState().items.find((i) => i.id === ids[0]!)!.groupId

            useVenueStore.getState().groupItems([ids[2]!, ids[3]!])
            const groupId2 = useVenueStore.getState().items.find((i) => i.id === ids[2]!)!.groupId

            expect(groupId1).toBeTruthy()
            expect(groupId2).toBeTruthy()
            expect(groupId1).not.toBe(groupId2)
        })
    })

    // ── ungroupItems ───────────────────────────────────────────────────

    describe('ungroupItems', () => {
        it('removes groupId from specified items', () => {
            useVenueStore.getState().addItem('round-table')
            useVenueStore.getState().addItem('chair')

            const items = useVenueStore.getState().items
            const ids = items.map((i) => i.id)

            // Group first
            useVenueStore.getState().groupItems([ids[0]!, ids[1]!])
            expect(useVenueStore.getState().items[0]!.groupId).toBeTruthy()

            // Ungroup
            useVenueStore.getState().ungroupItems([ids[0]!, ids[1]!])

            const updated = useVenueStore.getState().items
            expect(updated[0]!.groupId).toBeUndefined()
            expect(updated[1]!.groupId).toBeUndefined()
        })

        it('only removes groupId from specified items, leaving others grouped', () => {
            useVenueStore.getState().addItem('round-table')
            useVenueStore.getState().addItem('chair')
            useVenueStore.getState().addItem('platform')

            const items = useVenueStore.getState().items
            const ids = items.map((i) => i.id)

            useVenueStore.getState().groupItems([ids[0]!, ids[1]!, ids[2]!])
            const groupId = useVenueStore.getState().items[0]!.groupId

            // Ungroup only the first
            useVenueStore.getState().ungroupItems([ids[0]!])

            const updated = useVenueStore.getState().items
            expect(updated.find((i) => i.id === ids[0]!)!.groupId).toBeUndefined()
            expect(updated.find((i) => i.id === ids[1]!)!.groupId).toBe(groupId)
            expect(updated.find((i) => i.id === ids[2]!)!.groupId).toBe(groupId)
        })

        it('does nothing when items have no groupId', () => {
            useVenueStore.getState().addItem('round-table')
            const id = useVenueStore.getState().items[0]!.id

            useVenueStore.getState().ungroupItems([id])

            const item = useVenueStore.getState().items[0]!
            expect(item.groupId).toBeUndefined()
        })

        it('does nothing with an empty array', () => {
            useVenueStore.getState().addItem('round-table')
            useVenueStore.getState().addItem('chair')
            const ids = useVenueStore.getState().items.map((i) => i.id)

            useVenueStore.getState().groupItems(ids)
            const groupIdBefore = useVenueStore.getState().items[0]!.groupId

            useVenueStore.getState().ungroupItems([])

            const groupIdAfter = useVenueStore.getState().items[0]!.groupId
            expect(groupIdAfter).toBe(groupIdBefore)
        })
    })
})
