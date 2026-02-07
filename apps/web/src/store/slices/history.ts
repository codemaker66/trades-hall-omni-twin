import type { VenueSnapshot, StoreSet, StoreGet } from '../types'
import {
    createVenueSnapshot, snapshotEquals, cloneItems, historyPatch,
    HISTORY_LIMIT, computeInventoryUsage, getInventoryWarning, syncActiveScenario
} from '../helpers'

export const createHistorySlice = (set: StoreSet, _get: StoreGet) => ({
    canUndo: false,
    canRedo: false,
    historyPast: [] as VenueSnapshot[],
    historyFuture: [] as VenueSnapshot[],
    historyBatch: null as VenueSnapshot | null,

    beginHistoryBatch: () => set((state) => {
        if (state.historyBatch) return {}
        return { historyBatch: createVenueSnapshot(state) }
    }),

    endHistoryBatch: () => set((state) => {
        if (!state.historyBatch) return {}

        const baseline = state.historyBatch
        const current = createVenueSnapshot(state)
        if (snapshotEquals(baseline, current)) {
            return { historyBatch: null }
        }

        const historyPast = [...state.historyPast, baseline].slice(-HISTORY_LIMIT)
        return {
            ...historyPatch(historyPast, []),
            historyBatch: null
        }
    }),

    undo: () => set((state) => {
        if (state.historyPast.length === 0) return {}

        const current = createVenueSnapshot(state)
        const previous = state.historyPast[state.historyPast.length - 1]
        if (!previous) return {}
        const past = state.historyPast.slice(0, -1)
        const future = [current, ...state.historyFuture].slice(0, HISTORY_LIMIT)
        const items = cloneItems(previous.items)

        return {
            items,
            selectedIds: [...previous.selectedIds],
            transformMode: previous.transformMode,
            draggedItemType: null,
            chairPrompt: null,
            isDragging: false,
            historyBatch: null,
            inventoryWarning: getInventoryWarning(state.inventoryCatalog, computeInventoryUsage(items)),
            ...syncActiveScenario(state, items, previous.transformMode),
            ...historyPatch(past, future)
        }
    }),

    redo: () => set((state) => {
        if (state.historyFuture.length === 0) return {}

        const current = createVenueSnapshot(state)
        const next = state.historyFuture[0]
        if (!next) return {}
        const past = [...state.historyPast, current].slice(-HISTORY_LIMIT)
        const future = state.historyFuture.slice(1)
        const items = cloneItems(next.items)

        return {
            items,
            selectedIds: [...next.selectedIds],
            transformMode: next.transformMode,
            draggedItemType: null,
            chairPrompt: null,
            isDragging: false,
            historyBatch: null,
            inventoryWarning: getInventoryWarning(state.inventoryCatalog, computeInventoryUsage(items)),
            ...syncActiveScenario(state, items, next.transformMode),
            ...historyPatch(past, future)
        }
    })
})
