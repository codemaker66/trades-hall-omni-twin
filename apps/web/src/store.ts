import { create } from 'zustand'

export type FurnitureType = 'round-table' | 'trestle-table' | 'chair' | 'platform'

export interface FurnitureItem {
    id: string
    groupId?: string // New: For grouping items
    type: FurnitureType
    position: [number, number, number]
    rotation: [number, number, number]
}

interface VenueState {
    items: FurnitureItem[]
    selectedIds: string[]
    snappingEnabled: boolean
    snapGrid: number
    transformMode: 'translate' | 'rotate' // New: Toggle between translation and rotation
    toggleTransformMode: () => void
    isDragging: boolean // New: Track drag state to disable camera
    setIsDragging: (isDragging: boolean) => void

    draggedItemType: FurnitureType | null

    chairPrompt: {
        visible: boolean
        type: FurnitureType
        tableId?: string // If present, we are reconfiguring existing table
    } | null
    openChairPrompt: (type: FurnitureType, tableId?: string) => void
    closeChairPrompt: () => void

    setDraggedItem: (type: FurnitureType | null) => void

    addItem: (type: FurnitureType, position?: [number, number, number], rotation?: [number, number, number], groupId?: string) => void
    updateItem: (id: string, updates: Partial<FurnitureItem>) => void
    updateItems: (updates: { id: string, changes: Partial<FurnitureItem> }[]) => void // Batch update
    removeItems: (ids: string[]) => void
    ungroupItems: (ids: string[]) => void // Remove groupId
    setSelection: (ids: string[]) => void
    toggleSnapping: () => void
    rotateSelection: (amountDegrees: number) => void // New: 90 degree rotation
    groupItems: (ids: string[]) => void // New: Combine items into a group
}

export const useVenueStore = create<VenueState>((set) => ({
    items: [],
    selectedIds: [],
    snappingEnabled: true,
    snapGrid: 0.5,
    transformMode: 'translate',
    draggedItemType: null,
    isDragging: false,
    chairPrompt: null,

    setIsDragging: (isDragging) => set({ isDragging }),
    openChairPrompt: (type, tableId) => set({ chairPrompt: { visible: true, type, tableId } }),
    closeChairPrompt: () => set({ chairPrompt: null }),
    setDraggedItem: (type: FurnitureType | null) => set({ draggedItemType: type }),

    addItem: (type, position = [0, 0, 0], rotation = [0, 0, 0], groupId) => set((state) => ({
        items: [...state.items, {
            id: crypto.randomUUID(),
            groupId, // Optional group ID
            type,
            position,
            rotation
        }],
        selectedIds: []
    })),

    updateItem: (id, updates) => set((state) => ({
        items: state.items.map((item) =>
            item.id === id ? { ...item, ...updates } : item
        )
    })),

    updateItems: (updates) => set((state) => ({
        items: state.items.map((item) => {
            const update = updates.find(u => u.id === item.id)
            return update ? { ...item, ...update.changes } : item
        })
    })),

    removeItems: (ids) => set((state) => ({
        items: state.items.filter((item) => !ids.includes(item.id)),
        selectedIds: state.selectedIds.filter(id => !ids.includes(id))
    })),

    ungroupItems: (ids) => set((state) => ({
        items: state.items.map(item =>
            ids.includes(item.id) ? { ...item, groupId: undefined } : item
        )
    })),

    groupItems: (ids) => set((state) => {
        const groupId = crypto.randomUUID()
        return {
            items: state.items.map(item =>
                ids.includes(item.id) ? { ...item, groupId } : item
            )
        }
    }),

    setSelection: (ids) => set({ selectedIds: ids }),

    toggleSnapping: () => set((state) => ({ snappingEnabled: !state.snappingEnabled })),

    toggleTransformMode: () => set((state) => ({
        transformMode: state.transformMode === 'translate' ? 'rotate' : 'translate'
    })),

    rotateSelection: (amountDegrees) => set((state) => {
        if (state.selectedIds.length === 0) return {}

        // 1. Calculate Center of Selection
        const selectedItems = state.items.filter(i => state.selectedIds.includes(i.id))
        if (selectedItems.length === 0) return {}

        let centerX = 0
        let centerZ = 0

        selectedItems.forEach(item => {
            centerX += item.position[0]
            centerZ += item.position[2]
        })

        centerX /= selectedItems.length
        centerZ /= selectedItems.length

        // 2. Rotate each item around center
        const rad = (amountDegrees * Math.PI) / 180
        const cos = Math.cos(rad)
        const sin = Math.sin(rad)

        const newItems = state.items.map(item => {
            if (!state.selectedIds.includes(item.id)) return item

            // Relative Pos
            const dx = item.position[0] - centerX
            const dz = item.position[2] - centerZ

            // Rotate Point
            const rotX = dx * cos - dz * sin
            const rotZ = dx * sin + dz * cos

            // New Pos
            const finalX = centerX + rotX
            const finalZ = centerZ + rotZ

            // New Rotation (Add to existing Y rotation)
            const currentRotY = item.rotation[1]
            const newRotY = currentRotY + rad

            return {
                ...item,
                position: [finalX, item.position[1], finalZ] as [number, number, number],
                rotation: [item.rotation[0], newRotY, item.rotation[2]] as [number, number, number]
            }
        })

        return { items: newItems }
    })
}))
