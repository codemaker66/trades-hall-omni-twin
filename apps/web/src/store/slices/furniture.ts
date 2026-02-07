import type { FurnitureItem, FurnitureType, StoreSet, StoreGet, MutationOptions } from '../types'
import {
    historyBeforeMutation, recordHistory, syncActiveScenario,
    computeInventoryUsage, getInventoryWarning
} from '../helpers'

export const createFurnitureSlice = (set: StoreSet, _get: StoreGet) => ({
    items: [] as FurnitureItem[],
    draggedItemType: null as FurnitureType | null,
    chairPrompt: null as { visible: boolean, type: FurnitureType, tableId?: string } | null,

    setDraggedItem: (type: FurnitureType | null) => set({ draggedItemType: type }),
    openChairPrompt: (type: FurnitureType, tableId?: string) => set({ chairPrompt: { visible: true, type, tableId } }),
    closeChairPrompt: () => set({ chairPrompt: null }),

    addItem: (
        type: FurnitureType,
        position: [number, number, number] = [0, 0, 0],
        rotation?: [number, number, number],
        groupId?: string,
        options?: MutationOptions
    ) => set((state) => {
        const usage = computeInventoryUsage(state.items)
        const inventoryItem = state.inventoryCatalog.find((item) => item.furnitureType === type)
        if (inventoryItem) {
            const available = Math.max(0, inventoryItem.quantityTotal - inventoryItem.quantityReserved)
            if (usage[type] >= available) {
                return {
                    inventoryWarning: `${inventoryItem.name} inventory limit reached (${available} available).`
                }
            }
        }

        const items = [...state.items, {
            id: crypto.randomUUID(),
            groupId,
            type,
            position,
            rotation: rotation ?? (type === 'chair' ? [0, -Math.PI / 2, 0] as [number, number, number] : [0, 0, 0] as [number, number, number])
        }]

        return {
            ...historyBeforeMutation(state, recordHistory(options)),
            items,
            selectedIds: [],
            inventoryWarning: getInventoryWarning(state.inventoryCatalog, computeInventoryUsage(items)),
            ...syncActiveScenario(state, items)
        }
    }),

    updateItem: (id: string, updates: Partial<FurnitureItem>, options?: MutationOptions) => set((state) => {
        if (Object.keys(updates).length === 0) return {}

        let changed = false
        const items = state.items.map((item) => {
            if (item.id !== id) return item
            changed = true
            return { ...item, ...updates }
        })

        if (!changed) return {}

        return {
            ...historyBeforeMutation(state, recordHistory(options)),
            items,
            inventoryWarning: getInventoryWarning(state.inventoryCatalog, computeInventoryUsage(items)),
            ...syncActiveScenario(state, items)
        }
    }),

    updateItems: (updates: { id: string, changes: Partial<FurnitureItem> }[], options?: MutationOptions) => set((state) => {
        if (updates.length === 0) return {}

        const updatesById = new Map<string, Partial<FurnitureItem>>()
        for (const update of updates) {
            updatesById.set(update.id, update.changes)
        }

        let changed = false
        const items = state.items.map((item) => {
            const itemChanges = updatesById.get(item.id)
            if (!itemChanges) return item
            changed = true
            return { ...item, ...itemChanges }
        })

        if (!changed) return {}

        return {
            ...historyBeforeMutation(state, recordHistory(options)),
            items,
            inventoryWarning: getInventoryWarning(state.inventoryCatalog, computeInventoryUsage(items)),
            ...syncActiveScenario(state, items)
        }
    }),

    removeItems: (ids: string[], options?: MutationOptions) => set((state) => {
        if (ids.length === 0) return {}

        const idSet = new Set(ids)
        const items = state.items.filter((item) => !idSet.has(item.id))
        const selectedIds = state.selectedIds.filter((id) => !idSet.has(id))

        if (items.length === state.items.length && selectedIds.length === state.selectedIds.length) {
            return {}
        }

        return {
            ...historyBeforeMutation(state, recordHistory(options)),
            items,
            selectedIds,
            inventoryWarning: getInventoryWarning(state.inventoryCatalog, computeInventoryUsage(items)),
            ...syncActiveScenario(state, items)
        }
    }),

    ungroupItems: (ids: string[], options?: MutationOptions) => set((state) => {
        if (ids.length === 0) return {}
        const idSet = new Set(ids)

        let changed = false
        const items = state.items.map((item) => {
            if (!idSet.has(item.id) || item.groupId === undefined) return item
            changed = true
            return { ...item, groupId: undefined }
        })

        if (!changed) return {}

        return {
            ...historyBeforeMutation(state, recordHistory(options)),
            items,
            ...syncActiveScenario(state, items)
        }
    }),

    groupItems: (ids: string[], options?: MutationOptions) => set((state) => {
        if (ids.length < 2) return {}
        const idSet = new Set(ids)
        const groupId = crypto.randomUUID()

        let changed = false
        const items = state.items.map((item) => {
            if (!idSet.has(item.id)) return item
            changed = true
            return { ...item, groupId }
        })

        if (!changed) return {}

        return {
            ...historyBeforeMutation(state, recordHistory(options)),
            items,
            ...syncActiveScenario(state, items)
        }
    }),

    rotateSelection: (amountDegrees: number, options?: MutationOptions) => set((state) => {
        if (state.selectedIds.length === 0) return {}

        const selectedIdSet = new Set(state.selectedIds)
        const selectedItems = state.items.filter((item) => selectedIdSet.has(item.id))
        if (selectedItems.length === 0) return {}

        const rad = (amountDegrees * Math.PI) / 180
        const chairsOnly = selectedItems.every((item) => item.type === 'chair')

        if (chairsOnly) {
            const items = state.items.map((item) => {
                if (!selectedIdSet.has(item.id)) return item

                const newRotY = item.rotation[1] + rad
                return {
                    ...item,
                    rotation: [item.rotation[0], newRotY, item.rotation[2]] as [number, number, number]
                }
            })

            return {
                ...historyBeforeMutation(state, recordHistory(options)),
                items,
                ...syncActiveScenario(state, items)
            }
        }

        let centerX = 0
        let centerZ = 0
        for (const item of selectedItems) {
            centerX += item.position[0]
            centerZ += item.position[2]
        }

        centerX /= selectedItems.length
        centerZ /= selectedItems.length

        const cos = Math.cos(rad)
        const sin = Math.sin(rad)

        const items = state.items.map((item) => {
            if (!selectedIdSet.has(item.id)) return item

            const dx = item.position[0] - centerX
            const dz = item.position[2] - centerZ
            const rotX = dx * cos - dz * sin
            const rotZ = dx * sin + dz * cos
            const finalX = centerX + rotX
            const finalZ = centerZ + rotZ
            const newRotY = item.rotation[1] + rad

            return {
                ...item,
                position: [finalX, item.position[1], finalZ] as [number, number, number],
                rotation: [item.rotation[0], newRotY, item.rotation[2]] as [number, number, number]
            }
        })

        return {
            ...historyBeforeMutation(state, recordHistory(options)),
            items,
            ...syncActiveScenario(state, items)
        }
    })
})
