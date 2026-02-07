import { describe, it, expect, beforeEach } from 'vitest'
import { useVenueStore } from '../index'
import { HISTORY_LIMIT } from '../helpers'

/**
 * Reset the store to a clean initial state before each test.
 */
function resetStore() {
    useVenueStore.getState().resetProject()
    // Clear the history that resetProject itself records
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

describe('history slice', () => {
    beforeEach(() => {
        resetStore()
    })

    // ── History recording ──────────────────────────────────────────────

    describe('history recording', () => {
        it('records undo history when addItem is called', () => {
            expect(useVenueStore.getState().canUndo).toBe(false)

            useVenueStore.getState().addItem('round-table')

            expect(useVenueStore.getState().canUndo).toBe(true)
            expect(useVenueStore.getState().historyPast.length).toBe(1)
        })

        it('records undo history when removeItems is called', () => {
            useVenueStore.getState().addItem('round-table')
            const id = useVenueStore.getState().items[0]!.id
            const pastLengthBefore = useVenueStore.getState().historyPast.length

            useVenueStore.getState().removeItems([id])

            expect(useVenueStore.getState().historyPast.length).toBe(pastLengthBefore + 1)
        })

        it('records undo history when updateItems is called', () => {
            useVenueStore.getState().addItem('round-table')
            const id = useVenueStore.getState().items[0]!.id
            const pastLengthBefore = useVenueStore.getState().historyPast.length

            useVenueStore.getState().updateItems([
                { id, changes: { position: [5, 0, 5] } },
            ])

            expect(useVenueStore.getState().historyPast.length).toBe(pastLengthBefore + 1)
        })

        it('does not record history when recordHistory: false is specified', () => {
            useVenueStore.getState().addItem('round-table', undefined, undefined, undefined, { recordHistory: false })

            expect(useVenueStore.getState().canUndo).toBe(false)
            expect(useVenueStore.getState().historyPast.length).toBe(0)
        })

        it('clears redo history when a new mutation occurs', () => {
            // Create state, then undo, so we have redo history
            useVenueStore.getState().addItem('round-table')
            useVenueStore.getState().addItem('chair')
            useVenueStore.getState().undo()

            expect(useVenueStore.getState().canRedo).toBe(true)

            // New mutation should clear redo
            useVenueStore.getState().addItem('platform')

            expect(useVenueStore.getState().canRedo).toBe(false)
            expect(useVenueStore.getState().historyFuture.length).toBe(0)
        })
    })

    // ── Undo ───────────────────────────────────────────────────────────

    describe('undo', () => {
        it('restores the previous state', () => {
            useVenueStore.getState().addItem('round-table', [1, 0, 1])
            expect(useVenueStore.getState().items).toHaveLength(1)

            useVenueStore.getState().undo()

            expect(useVenueStore.getState().items).toHaveLength(0)
        })

        it('can undo multiple times in sequence', () => {
            useVenueStore.getState().addItem('round-table', [1, 0, 0])
            useVenueStore.getState().addItem('chair', [2, 0, 0])
            useVenueStore.getState().addItem('platform', [3, 0, 0])

            expect(useVenueStore.getState().items).toHaveLength(3)

            useVenueStore.getState().undo()
            expect(useVenueStore.getState().items).toHaveLength(2)

            useVenueStore.getState().undo()
            expect(useVenueStore.getState().items).toHaveLength(1)

            useVenueStore.getState().undo()
            expect(useVenueStore.getState().items).toHaveLength(0)
        })

        it('does nothing when there is nothing to undo', () => {
            expect(useVenueStore.getState().canUndo).toBe(false)

            useVenueStore.getState().undo()

            expect(useVenueStore.getState().items).toHaveLength(0)
            expect(useVenueStore.getState().canUndo).toBe(false)
        })

        it('updates canUndo and canRedo flags correctly', () => {
            useVenueStore.getState().addItem('round-table')

            expect(useVenueStore.getState().canUndo).toBe(true)
            expect(useVenueStore.getState().canRedo).toBe(false)

            useVenueStore.getState().undo()

            expect(useVenueStore.getState().canUndo).toBe(false)
            expect(useVenueStore.getState().canRedo).toBe(true)
        })

        it('restores item positions after updateItems + undo', () => {
            useVenueStore.getState().addItem('round-table', [0, 0, 0])
            const id = useVenueStore.getState().items[0]!.id

            useVenueStore.getState().updateItems([
                { id, changes: { position: [50, 0, 50] } },
            ])
            expect(useVenueStore.getState().items[0]!.position).toEqual([50, 0, 50])

            useVenueStore.getState().undo()
            expect(useVenueStore.getState().items[0]!.position).toEqual([0, 0, 0])
        })
    })

    // ── Redo ───────────────────────────────────────────────────────────

    describe('redo', () => {
        it('restores the undone state', () => {
            useVenueStore.getState().addItem('round-table', [1, 0, 1])
            expect(useVenueStore.getState().items).toHaveLength(1)

            useVenueStore.getState().undo()
            expect(useVenueStore.getState().items).toHaveLength(0)

            useVenueStore.getState().redo()
            expect(useVenueStore.getState().items).toHaveLength(1)
            expect(useVenueStore.getState().items[0]!.position).toEqual([1, 0, 1])
        })

        it('can redo multiple times in sequence', () => {
            useVenueStore.getState().addItem('round-table')
            useVenueStore.getState().addItem('chair')

            useVenueStore.getState().undo()
            useVenueStore.getState().undo()

            expect(useVenueStore.getState().items).toHaveLength(0)

            useVenueStore.getState().redo()
            expect(useVenueStore.getState().items).toHaveLength(1)

            useVenueStore.getState().redo()
            expect(useVenueStore.getState().items).toHaveLength(2)
        })

        it('does nothing when there is nothing to redo', () => {
            useVenueStore.getState().addItem('round-table')

            expect(useVenueStore.getState().canRedo).toBe(false)

            useVenueStore.getState().redo()

            expect(useVenueStore.getState().items).toHaveLength(1)
            expect(useVenueStore.getState().canRedo).toBe(false)
        })

        it('updates canUndo and canRedo flags correctly after redo', () => {
            useVenueStore.getState().addItem('round-table')
            useVenueStore.getState().undo()

            expect(useVenueStore.getState().canUndo).toBe(false)
            expect(useVenueStore.getState().canRedo).toBe(true)

            useVenueStore.getState().redo()

            expect(useVenueStore.getState().canUndo).toBe(true)
            expect(useVenueStore.getState().canRedo).toBe(false)
        })
    })

    // ── Undo / Redo round-trip ─────────────────────────────────────────

    describe('undo/redo round-trip', () => {
        it('preserves item data through a full undo + redo cycle', () => {
            useVenueStore.getState().addItem('trestle-table', [7, 0, 3], [0, 1.5, 0])
            const originalItem = { ...useVenueStore.getState().items[0]! }

            useVenueStore.getState().undo()
            expect(useVenueStore.getState().items).toHaveLength(0)

            useVenueStore.getState().redo()
            const restoredItem = useVenueStore.getState().items[0]!

            expect(restoredItem.id).toBe(originalItem.id)
            expect(restoredItem.type).toBe(originalItem.type)
            expect(restoredItem.position).toEqual(originalItem.position)
            expect(restoredItem.rotation).toEqual(originalItem.rotation)
        })
    })

    // ── Batch history ──────────────────────────────────────────────────

    describe('batch history', () => {
        it('beginHistoryBatch + multiple addItems + endHistoryBatch creates a single undo entry', () => {
            useVenueStore.getState().beginHistoryBatch()

            useVenueStore.getState().addItem('round-table', [0, 0, 0])
            useVenueStore.getState().addItem('chair', [1, 0, 0])
            useVenueStore.getState().addItem('chair', [2, 0, 0])

            useVenueStore.getState().endHistoryBatch()

            expect(useVenueStore.getState().items).toHaveLength(3)
            // Only one history entry from the batch
            expect(useVenueStore.getState().historyPast.length).toBe(1)

            // A single undo should revert all three adds
            useVenueStore.getState().undo()
            expect(useVenueStore.getState().items).toHaveLength(0)
        })

        it('does not record history for individual mutations inside a batch', () => {
            useVenueStore.getState().beginHistoryBatch()

            useVenueStore.getState().addItem('round-table')
            // During the batch, addItem should not push to historyPast because historyBatch is set
            expect(useVenueStore.getState().historyPast.length).toBe(0)

            useVenueStore.getState().addItem('chair')
            expect(useVenueStore.getState().historyPast.length).toBe(0)

            useVenueStore.getState().endHistoryBatch()
            // After the batch ends, exactly one snapshot is pushed
            expect(useVenueStore.getState().historyPast.length).toBe(1)
        })

        it('endHistoryBatch with no changes creates no history entry', () => {
            useVenueStore.getState().beginHistoryBatch()
            // No mutations inside the batch
            useVenueStore.getState().endHistoryBatch()

            expect(useVenueStore.getState().historyPast.length).toBe(0)
            expect(useVenueStore.getState().canUndo).toBe(false)
        })

        it('beginHistoryBatch is idempotent when already in a batch', () => {
            useVenueStore.getState().beginHistoryBatch()
            const batchSnapshot1 = useVenueStore.getState().historyBatch

            useVenueStore.getState().addItem('round-table')

            // Calling begin again should be a no-op
            useVenueStore.getState().beginHistoryBatch()
            const batchSnapshot2 = useVenueStore.getState().historyBatch

            // The batch snapshot should still be the original one (before addItem)
            expect(batchSnapshot2).toBe(batchSnapshot1)

            useVenueStore.getState().endHistoryBatch()
            expect(useVenueStore.getState().historyPast.length).toBe(1)
        })

        it('batch works with updateItems', () => {
            useVenueStore.getState().addItem('round-table', [0, 0, 0])
            const id = useVenueStore.getState().items[0]!.id
            const pastLengthBefore = useVenueStore.getState().historyPast.length

            useVenueStore.getState().beginHistoryBatch()

            useVenueStore.getState().updateItems([
                { id, changes: { position: [1, 0, 0] } },
            ])
            useVenueStore.getState().updateItems([
                { id, changes: { position: [2, 0, 0] } },
            ])
            useVenueStore.getState().updateItems([
                { id, changes: { position: [3, 0, 0] } },
            ])

            useVenueStore.getState().endHistoryBatch()

            // Only one history entry for the batch
            expect(useVenueStore.getState().historyPast.length).toBe(pastLengthBefore + 1)
            expect(useVenueStore.getState().items[0]!.position).toEqual([3, 0, 0])

            // Undo should go back to original position
            useVenueStore.getState().undo()
            expect(useVenueStore.getState().items[0]!.position).toEqual([0, 0, 0])
        })
    })

    // ── History limit ──────────────────────────────────────────────────

    describe('history limit', () => {
        it('HISTORY_LIMIT is a positive number', () => {
            expect(HISTORY_LIMIT).toBeGreaterThan(0)
            expect(HISTORY_LIMIT).toBe(200)
        })

        it('historyPast does not exceed HISTORY_LIMIT', () => {
            // Perform more mutations than the limit
            const count = HISTORY_LIMIT + 20
            for (let i = 0; i < count; i++) {
                useVenueStore.getState().addItem('chair', [i, 0, 0])
            }

            const { historyPast } = useVenueStore.getState()
            expect(historyPast.length).toBeLessThanOrEqual(HISTORY_LIMIT)
        })

        it('old history entries are dropped when the limit is reached', () => {
            // Fill up history well beyond the limit
            const count = HISTORY_LIMIT + 10
            for (let i = 0; i < count; i++) {
                useVenueStore.getState().addItem('chair', [i, 0, 0])
            }

            const { historyPast } = useVenueStore.getState()
            expect(historyPast.length).toBe(HISTORY_LIMIT)

            // The earliest entries should have been dropped; the first history
            // snapshot should have items (not be the empty initial state)
            const earliest = historyPast[0]!
            expect(earliest.items.length).toBeGreaterThan(0)
        })

        it('historyFuture does not exceed HISTORY_LIMIT after many undos', () => {
            // Add items to build history
            const count = HISTORY_LIMIT + 20
            for (let i = 0; i < count; i++) {
                useVenueStore.getState().addItem('chair', [i, 0, 0])
            }

            // Undo as many times as we can
            let undoCount = 0
            while (useVenueStore.getState().canUndo && undoCount < count + 10) {
                useVenueStore.getState().undo()
                undoCount++
            }

            const { historyFuture } = useVenueStore.getState()
            expect(historyFuture.length).toBeLessThanOrEqual(HISTORY_LIMIT)
        })
    })
})
