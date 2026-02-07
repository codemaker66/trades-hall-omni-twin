import type { StoreSet, StoreGet, TransformMode } from '../types'
import { syncActiveScenario } from '../helpers'

export const createSelectionSlice = (set: StoreSet, _get: StoreGet) => ({
    selectedIds: [] as string[],
    snappingEnabled: true,
    snapGrid: 0.5,
    transformMode: 'translate' as TransformMode,
    isDragging: false,
    shortcutsHelpOpen: false,

    setIsDragging: (isDragging: boolean) => set({ isDragging }),
    setSelection: (ids: string[]) => set({ selectedIds: ids }),
    setShortcutsHelpOpen: (open: boolean) => set({ shortcutsHelpOpen: open }),
    toggleSnapping: () => set((state) => ({ snappingEnabled: !state.snappingEnabled })),

    toggleTransformMode: () => set((state) => {
        const transformMode: TransformMode = state.transformMode === 'translate' ? 'rotate' : 'translate'
        return {
            transformMode,
            ...syncActiveScenario(state, state.items, transformMode)
        }
    })
})
