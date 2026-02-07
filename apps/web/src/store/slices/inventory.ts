import type { InventoryItem, InventoryUpdate, InventoryUsage, LayoutMetrics, FurnitureType, StoreSet, StoreGet } from '../types'
import { computeInventoryUsage, computeLayoutMetrics, getInventoryWarning, cloneInventory, DEFAULT_INVENTORY } from '../helpers'

export const createInventorySlice = (set: StoreSet, get: StoreGet) => ({
    inventoryCatalog: cloneInventory(DEFAULT_INVENTORY),
    inventoryWarning: null as string | null,

    updateInventoryItem: (id: string, updates: InventoryUpdate) => set((state) => {
        const index = state.inventoryCatalog.findIndex((item) => item.id === id)
        if (index === -1) return {}

        const current = state.inventoryCatalog[index]!
        const quantityTotalRaw = updates.quantityTotal ?? current.quantityTotal
        const quantityTotal = Math.max(0, Math.floor(quantityTotalRaw))
        const quantityReservedRaw = updates.quantityReserved ?? current.quantityReserved
        const quantityReserved = Math.max(0, Math.min(quantityTotal, Math.floor(quantityReservedRaw)))

        const nextItem: InventoryItem = {
            id: current.id,
            furnitureType: current.furnitureType,
            category: current.category,
            name: updates.name ?? current.name,
            quantityTotal,
            quantityReserved,
            seatsPerItem: updates.seatsPerItem ?? current.seatsPerItem
        }

        const inventoryCatalog = [...state.inventoryCatalog]
        inventoryCatalog[index] = nextItem

        return {
            inventoryCatalog,
            inventoryWarning: getInventoryWarning(inventoryCatalog, computeInventoryUsage(state.items))
        }
    }),

    clearInventoryWarning: () => set({ inventoryWarning: null }),
    getInventoryUsage: (): InventoryUsage => computeInventoryUsage(get().items),
    getLayoutMetrics: (): LayoutMetrics => computeLayoutMetrics(get().items),
    hasInventoryForType: (type: FurnitureType): boolean => {
        const state = get()
        const usage = computeInventoryUsage(state.items)
        const inventoryItem = state.inventoryCatalog.find((item) => item.furnitureType === type)
        if (!inventoryItem) return true

        const available = Math.max(0, inventoryItem.quantityTotal - inventoryItem.quantityReserved)
        return usage[type] < available
    }
})
