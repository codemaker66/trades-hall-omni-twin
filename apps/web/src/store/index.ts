import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'
import type { VenueState } from './types'
import {
    VENUE_STORE_STORAGE_KEY, VENUE_STORE_VERSION,
    buildInitialState, createPersistedStateFromVenueState,
    mergeRehydratedState, sanitizePersistedState
} from './helpers'
import { createFurnitureSlice } from './slices/furniture'
import { createSelectionSlice } from './slices/selection'
import { createInventorySlice } from './slices/inventory'
import { createScenariosSlice } from './slices/scenarios'
import { createHistorySlice } from './slices/history'

export const useVenueStore = create<VenueState>()(
    persist((set, get) => ({
        ...buildInitialState(),
        ...createFurnitureSlice(set, get),
        ...createSelectionSlice(set, get),
        ...createInventorySlice(set, get),
        ...createScenariosSlice(set, get),
        ...createHistorySlice(set, get),
    }),
    {
        name: VENUE_STORE_STORAGE_KEY,
        version: VENUE_STORE_VERSION,
        storage: createJSONStorage(() => localStorage),
        partialize: (state) => createPersistedStateFromVenueState(state),
        merge: (persistedState, currentState) => mergeRehydratedState(persistedState, currentState),
        migrate: (persistedState) => sanitizePersistedState(persistedState)
    })
)

// Re-export types for consumer convenience
export type {
    FurnitureType, TransformMode, ScenarioStatus, ProjectImportMode, ProjectImportResult,
    FurnitureItem, InventoryItem, InventoryUsage, LayoutMetrics, MutationOptions,
    InventoryUpdate, VenueSnapshot, ScenarioSnapshot, Scenario, VenueState,
    PersistedVenueState
} from './types'
