export type FurnitureType = 'round-table' | 'trestle-table' | 'chair' | 'platform'
export type TransformMode = 'translate' | 'rotate'
export type ScenarioStatus = 'draft' | 'review' | 'approved'
export type ProjectImportMode = 'replace' | 'merge'
export type ProjectImportResult = {
    ok: boolean
    message: string
}

export interface FurnitureItem {
    id: string
    groupId?: string
    type: FurnitureType
    position: [number, number, number]
    rotation: [number, number, number]
}

export interface InventoryItem {
    id: string
    name: string
    furnitureType: FurnitureType
    category: 'table' | 'chair' | 'other'
    quantityTotal: number
    quantityReserved: number
    seatsPerItem?: number
}

export type InventoryUsage = Record<FurnitureType, number>

export interface LayoutMetrics {
    seatCount: number
    tableCount: number
    chairCount: number
    platformCount: number
}

export type MutationOptions = {
    recordHistory?: boolean
}

export type InventoryUpdate = Partial<Pick<InventoryItem, 'name' | 'quantityTotal' | 'quantityReserved' | 'seatsPerItem'>>

export type VenueSnapshot = {
    items: FurnitureItem[]
    selectedIds: string[]
    transformMode: TransformMode
}

export type ScenarioSnapshot = {
    items: FurnitureItem[]
    transformMode: TransformMode
}

export interface Scenario {
    id: string
    name: string
    status: ScenarioStatus
    createdAt: string
    updatedAt: string
    snapshot: ScenarioSnapshot
}

export interface VenueState {
    items: FurnitureItem[]
    selectedIds: string[]
    snappingEnabled: boolean
    snapGrid: number
    transformMode: TransformMode
    toggleTransformMode: () => void
    isDragging: boolean
    setIsDragging: (isDragging: boolean) => void

    draggedItemType: FurnitureType | null

    chairPrompt: {
        visible: boolean
        type: FurnitureType
        tableId?: string
    } | null
    openChairPrompt: (type: FurnitureType, tableId?: string) => void
    closeChairPrompt: () => void

    inventoryCatalog: InventoryItem[]
    inventoryWarning: string | null
    updateInventoryItem: (id: string, updates: InventoryUpdate) => void
    clearInventoryWarning: () => void
    getInventoryUsage: () => InventoryUsage
    getLayoutMetrics: () => LayoutMetrics
    hasInventoryForType: (type: FurnitureType) => boolean

    scenarios: Scenario[]
    activeScenarioId: string | null
    saveScenario: (name?: string) => string
    loadScenario: (id: string, options?: MutationOptions) => void
    deleteScenario: (id: string) => void
    renameScenario: (id: string, name: string) => void
    setScenarioStatus: (id: string, status: ScenarioStatus) => void
    resetProject: () => void
    exportProject: () => string
    importProject: (payload: string, options?: { mode?: ProjectImportMode }) => ProjectImportResult

    canUndo: boolean
    canRedo: boolean
    historyPast: VenueSnapshot[]
    historyFuture: VenueSnapshot[]
    historyBatch: VenueSnapshot | null
    beginHistoryBatch: () => void
    endHistoryBatch: () => void
    undo: () => void
    redo: () => void

    shortcutsHelpOpen: boolean
    setShortcutsHelpOpen: (open: boolean) => void

    setDraggedItem: (type: FurnitureType | null) => void

    addItem: (
        type: FurnitureType,
        position?: [number, number, number],
        rotation?: [number, number, number],
        groupId?: string,
        options?: MutationOptions
    ) => void
    updateItem: (id: string, updates: Partial<FurnitureItem>, options?: MutationOptions) => void
    updateItems: (updates: { id: string, changes: Partial<FurnitureItem> }[], options?: MutationOptions) => void
    removeItems: (ids: string[], options?: MutationOptions) => void
    ungroupItems: (ids: string[], options?: MutationOptions) => void
    setSelection: (ids: string[]) => void
    toggleSnapping: () => void
    rotateSelection: (amountDegrees: number, options?: MutationOptions) => void
    groupItems: (ids: string[], options?: MutationOptions) => void
}

export type PersistedVenueState = Pick<
    VenueState,
    'items' | 'snappingEnabled' | 'snapGrid' | 'transformMode' | 'inventoryCatalog' | 'scenarios' | 'activeScenarioId'
>

export type ProjectExportDocument = {
    format: typeof PROJECT_EXPORT_FORMAT
    version: number
    exportedAt: string
    data: PersistedVenueState
}

export const PROJECT_EXPORT_FORMAT = 'omnitwin-venue-project'
export const PROJECT_EXPORT_VERSION = 1

export type StoreSet = (partial: Partial<VenueState> | ((state: VenueState) => Partial<VenueState>)) => void
export type StoreGet = () => VenueState
