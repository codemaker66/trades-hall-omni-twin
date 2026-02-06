import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'

export type FurnitureType = 'round-table' | 'trestle-table' | 'chair' | 'platform'
export type TransformMode = 'translate' | 'rotate'
export type ScenarioStatus = 'draft' | 'review' | 'approved'

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

type MutationOptions = {
    recordHistory?: boolean
}

type InventoryUpdate = Partial<Pick<InventoryItem, 'name' | 'quantityTotal' | 'quantityReserved' | 'seatsPerItem'>>

type VenueSnapshot = {
    items: FurnitureItem[]
    selectedIds: string[]
    transformMode: TransformMode
}

type ScenarioSnapshot = {
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

const HISTORY_LIMIT = 200
const VENUE_STORE_STORAGE_KEY = 'omnitwin_venue_store_v1'
const VENUE_STORE_VERSION = 1

const DEFAULT_INVENTORY: InventoryItem[] = [
    {
        id: 'inv_round_table',
        name: '6ft Round Table',
        furnitureType: 'round-table',
        category: 'table',
        quantityTotal: 40,
        quantityReserved: 2,
        seatsPerItem: 8
    },
    {
        id: 'inv_trestle_table',
        name: '6ft Trestle Table',
        furnitureType: 'trestle-table',
        category: 'table',
        quantityTotal: 60,
        quantityReserved: 4,
        seatsPerItem: 6
    },
    {
        id: 'inv_chair',
        name: 'Banquet Chair',
        furnitureType: 'chair',
        category: 'chair',
        quantityTotal: 400,
        quantityReserved: 30,
        seatsPerItem: 1
    },
    {
        id: 'inv_platform',
        name: 'Platform Segment',
        furnitureType: 'platform',
        category: 'other',
        quantityTotal: 12,
        quantityReserved: 1
    }
]

const nowIso = () => new Date().toISOString()

const cloneTuple3 = ([x, y, z]: [number, number, number]): [number, number, number] => [x, y, z]

const cloneItem = (item: FurnitureItem): FurnitureItem => ({
    ...item,
    position: cloneTuple3(item.position),
    rotation: cloneTuple3(item.rotation)
})

const cloneItems = (items: FurnitureItem[]) => items.map(cloneItem)

const cloneInventory = (inventory: InventoryItem[]) => inventory.map((item) => ({ ...item }))

const emptyInventoryUsage = (): InventoryUsage => ({
    'round-table': 0,
    'trestle-table': 0,
    'chair': 0,
    'platform': 0
})

const computeInventoryUsage = (items: FurnitureItem[]): InventoryUsage => {
    const usage = emptyInventoryUsage()
    for (const item of items) {
        usage[item.type] += 1
    }
    return usage
}

const computeLayoutMetrics = (items: FurnitureItem[]): LayoutMetrics => {
    let tableCount = 0
    let chairCount = 0
    let platformCount = 0

    for (const item of items) {
        if (item.type === 'round-table' || item.type === 'trestle-table') {
            tableCount += 1
        } else if (item.type === 'chair') {
            chairCount += 1
        } else if (item.type === 'platform') {
            platformCount += 1
        }
    }

    return {
        seatCount: chairCount,
        tableCount,
        chairCount,
        platformCount
    }
}

const createVenueSnapshot = (state: Pick<VenueState, 'items' | 'selectedIds' | 'transformMode'>): VenueSnapshot => ({
    items: cloneItems(state.items),
    selectedIds: [...state.selectedIds],
    transformMode: state.transformMode
})

const createScenarioSnapshot = (state: Pick<VenueState, 'items' | 'transformMode'>): ScenarioSnapshot => ({
    items: cloneItems(state.items),
    transformMode: state.transformMode
})

const snapshotEquals = (a: VenueSnapshot, b: VenueSnapshot): boolean => {
    if (a.transformMode !== b.transformMode) return false
    if (a.selectedIds.length !== b.selectedIds.length) return false
    if (a.items.length !== b.items.length) return false

    for (let i = 0; i < a.selectedIds.length; i++) {
        if (a.selectedIds[i] !== b.selectedIds[i]) return false
    }

    for (let i = 0; i < a.items.length; i++) {
        const left = a.items[i]
        const right = b.items[i]

        if (left.id !== right.id) return false
        if (left.type !== right.type) return false
        if (left.groupId !== right.groupId) return false
        if (left.position[0] !== right.position[0] || left.position[1] !== right.position[1] || left.position[2] !== right.position[2]) return false
        if (left.rotation[0] !== right.rotation[0] || left.rotation[1] !== right.rotation[1] || left.rotation[2] !== right.rotation[2]) return false
    }

    return true
}

const getInventoryWarning = (inventoryCatalog: InventoryItem[], usage: InventoryUsage): string | null => {
    for (const catalogItem of inventoryCatalog) {
        const available = Math.max(0, catalogItem.quantityTotal - catalogItem.quantityReserved)
        const used = usage[catalogItem.furnitureType]

        if (used > available) {
            return `${catalogItem.name} is over-allocated (${used}/${available} available).`
        }
    }

    return null
}

const historyPatch = (historyPast: VenueSnapshot[], historyFuture: VenueSnapshot[]) => ({
    historyPast,
    historyFuture,
    canUndo: historyPast.length > 0,
    canRedo: historyFuture.length > 0
})

const recordHistory = (options?: MutationOptions) => options?.recordHistory !== false

const isFurnitureType = (value: unknown): value is FurnitureType =>
    value === 'round-table' || value === 'trestle-table' || value === 'chair' || value === 'platform'

const isTransformMode = (value: unknown): value is TransformMode =>
    value === 'translate' || value === 'rotate'

const isScenarioStatus = (value: unknown): value is ScenarioStatus =>
    value === 'draft' || value === 'review' || value === 'approved'

const toTuple3 = (
    value: unknown,
    fallback: [number, number, number]
): [number, number, number] => {
    if (!Array.isArray(value) || value.length !== 3) return fallback

    const [x, y, z] = value
    if (typeof x !== 'number' || typeof y !== 'number' || typeof z !== 'number') return fallback
    return [x, y, z]
}

const coercePositiveInt = (value: unknown, fallback: number): number => {
    if (typeof value !== 'number' || !Number.isFinite(value)) return fallback
    return Math.max(0, Math.floor(value))
}

const sanitizeFurnitureItem = (raw: unknown): FurnitureItem | null => {
    if (!raw || typeof raw !== 'object') return null
    const candidate = raw as Partial<FurnitureItem>
    if (typeof candidate.id !== 'string' || candidate.id.length === 0) return null
    if (!isFurnitureType(candidate.type)) return null

    const groupId = typeof candidate.groupId === 'string' && candidate.groupId.length > 0
        ? candidate.groupId
        : undefined

    return {
        id: candidate.id,
        groupId,
        type: candidate.type,
        position: toTuple3(candidate.position, [0, 0, 0]),
        rotation: toTuple3(candidate.rotation, [0, 0, 0])
    }
}

const sanitizeInventoryItem = (raw: unknown, fallback: InventoryItem): InventoryItem => {
    if (!raw || typeof raw !== 'object') return { ...fallback }

    const candidate = raw as Partial<InventoryItem>
    const quantityTotal = coercePositiveInt(candidate.quantityTotal, fallback.quantityTotal)
    const quantityReserved = Math.min(quantityTotal, coercePositiveInt(candidate.quantityReserved, fallback.quantityReserved))
    const seatsPerItem = candidate.seatsPerItem === undefined
        ? fallback.seatsPerItem
        : coercePositiveInt(candidate.seatsPerItem, fallback.seatsPerItem ?? 0)
    const normalizedSeatsPerItem = typeof seatsPerItem === 'number' && seatsPerItem > 0
        ? seatsPerItem
        : undefined

    return {
        ...fallback,
        id: fallback.id,
        furnitureType: fallback.furnitureType,
        category: fallback.category,
        name: typeof candidate.name === 'string' && candidate.name.trim().length > 0
            ? candidate.name.trim()
            : fallback.name,
        quantityTotal,
        quantityReserved,
        seatsPerItem: normalizedSeatsPerItem
    }
}

const sanitizeScenario = (raw: unknown): Scenario | null => {
    if (!raw || typeof raw !== 'object') return null
    const candidate = raw as Partial<Scenario>
    if (typeof candidate.id !== 'string' || candidate.id.length === 0) return null
    if (typeof candidate.name !== 'string' || candidate.name.trim().length === 0) return null
    if (!isScenarioStatus(candidate.status)) return null

    const snapshot = candidate.snapshot
    if (!snapshot || typeof snapshot !== 'object') return null
    const snapshotCandidate = snapshot as Partial<ScenarioSnapshot>
    if (!Array.isArray(snapshotCandidate.items)) return null
    if (!isTransformMode(snapshotCandidate.transformMode)) return null

    const items = snapshotCandidate.items
        .map(sanitizeFurnitureItem)
        .filter((item): item is FurnitureItem => item !== null)

    const createdAt = typeof candidate.createdAt === 'string' ? candidate.createdAt : nowIso()
    const updatedAt = typeof candidate.updatedAt === 'string' ? candidate.updatedAt : createdAt

    return {
        id: candidate.id,
        name: candidate.name.trim(),
        status: candidate.status,
        createdAt,
        updatedAt,
        snapshot: {
            items,
            transformMode: snapshotCandidate.transformMode
        }
    }
}

interface VenueState {
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

    canUndo: boolean
    canRedo: boolean
    historyPast: VenueSnapshot[]
    historyFuture: VenueSnapshot[]
    historyBatch: VenueSnapshot | null
    beginHistoryBatch: () => void
    endHistoryBatch: () => void
    undo: () => void
    redo: () => void

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

type PersistedVenueState = Pick<
    VenueState,
    'items' | 'snappingEnabled' | 'snapGrid' | 'transformMode' | 'inventoryCatalog' | 'scenarios' | 'activeScenarioId'
>

const buildDefaultPersistedState = (): PersistedVenueState => ({
    items: [],
    snappingEnabled: true,
    snapGrid: 0.5,
    transformMode: 'translate',
    inventoryCatalog: cloneInventory(DEFAULT_INVENTORY),
    scenarios: [],
    activeScenarioId: null
})

const sanitizePersistedState = (raw: unknown): PersistedVenueState => {
    const defaults = buildDefaultPersistedState()
    if (!raw || typeof raw !== 'object') return defaults

    const candidate = raw as Partial<PersistedVenueState>

    const items = Array.isArray(candidate.items)
        ? candidate.items.map(sanitizeFurnitureItem).filter((item): item is FurnitureItem => item !== null)
        : defaults.items

    const snapGridRaw = typeof candidate.snapGrid === 'number' && Number.isFinite(candidate.snapGrid)
        ? candidate.snapGrid
        : defaults.snapGrid
    const snapGrid = Math.max(0.1, snapGridRaw)

    const inventoryByType = new Map<FurnitureType, InventoryItem>()
    if (Array.isArray(candidate.inventoryCatalog)) {
        for (const fallback of DEFAULT_INVENTORY) {
            const rawItem = candidate.inventoryCatalog.find((entry) => {
                if (!entry || typeof entry !== 'object') return false
                return (entry as Partial<InventoryItem>).furnitureType === fallback.furnitureType
            })
            inventoryByType.set(fallback.furnitureType, sanitizeInventoryItem(rawItem, fallback))
        }
    } else {
        for (const fallback of DEFAULT_INVENTORY) {
            inventoryByType.set(fallback.furnitureType, { ...fallback })
        }
    }

    const scenarios = Array.isArray(candidate.scenarios)
        ? candidate.scenarios
            .map(sanitizeScenario)
            .filter((scenario): scenario is Scenario => scenario !== null)
        : defaults.scenarios

    const scenarioIdSet = new Set(scenarios.map((scenario) => scenario.id))
    const activeScenarioId = typeof candidate.activeScenarioId === 'string' && scenarioIdSet.has(candidate.activeScenarioId)
        ? candidate.activeScenarioId
        : null

    return {
        items,
        snappingEnabled: typeof candidate.snappingEnabled === 'boolean'
            ? candidate.snappingEnabled
            : defaults.snappingEnabled,
        snapGrid,
        transformMode: isTransformMode(candidate.transformMode)
            ? candidate.transformMode
            : defaults.transformMode,
        inventoryCatalog: DEFAULT_INVENTORY.map((fallback) => {
            const match = inventoryByType.get(fallback.furnitureType)
            return match ? { ...match } : { ...fallback }
        }),
        scenarios,
        activeScenarioId
    }
}

const historyBeforeMutation = (state: VenueState, shouldRecordHistory: boolean) => {
    if (!shouldRecordHistory || state.historyBatch) {
        return {}
    }

    const historyPast = [...state.historyPast, createVenueSnapshot(state)].slice(-HISTORY_LIMIT)
    return historyPatch(historyPast, [])
}

const syncActiveScenario = (state: VenueState, items: FurnitureItem[], transformMode: TransformMode = state.transformMode) => {
    if (!state.activeScenarioId) return {}

    const scenarioIndex = state.scenarios.findIndex((scenario) => scenario.id === state.activeScenarioId)
    if (scenarioIndex === -1) return {}

    const scenarios = [...state.scenarios]
    scenarios[scenarioIndex] = {
        ...scenarios[scenarioIndex],
        updatedAt: nowIso(),
        snapshot: {
            items: cloneItems(items),
            transformMode
        }
    }

    return { scenarios }
}

const INITIAL_TRANSIENT_STATE = {
    selectedIds: [] as string[],
    draggedItemType: null as FurnitureType | null,
    isDragging: false,
    chairPrompt: null as VenueState['chairPrompt'],
    inventoryWarning: null as string | null,
    canUndo: false,
    canRedo: false,
    historyPast: [] as VenueSnapshot[],
    historyFuture: [] as VenueSnapshot[],
    historyBatch: null as VenueSnapshot | null
}

const buildInitialState = (): Pick<
    VenueState,
    keyof PersistedVenueState |
    'selectedIds' |
    'draggedItemType' |
    'isDragging' |
    'chairPrompt' |
    'inventoryWarning' |
    'canUndo' |
    'canRedo' |
    'historyPast' |
    'historyFuture' |
    'historyBatch'
> => ({
    ...buildDefaultPersistedState(),
    ...INITIAL_TRANSIENT_STATE
})

const mergeRehydratedState = (persistedRaw: unknown, currentState: VenueState): VenueState => {
    const persisted = sanitizePersistedState(persistedRaw)
    const inventoryWarning = getInventoryWarning(persisted.inventoryCatalog, computeInventoryUsage(persisted.items))

    return {
        ...currentState,
        ...persisted,
        ...INITIAL_TRANSIENT_STATE,
        inventoryWarning
    }
}

export const useVenueStore = create<VenueState>()(
    persist((set, get) => ({
    ...buildInitialState(),

    setIsDragging: (isDragging) => set({ isDragging }),
    openChairPrompt: (type, tableId) => set({ chairPrompt: { visible: true, type, tableId } }),
    closeChairPrompt: () => set({ chairPrompt: null }),
    setDraggedItem: (type: FurnitureType | null) => set({ draggedItemType: type }),

    updateInventoryItem: (id, updates) => set((state) => {
        const index = state.inventoryCatalog.findIndex((item) => item.id === id)
        if (index === -1) return {}

        const current = state.inventoryCatalog[index]
        const quantityTotalRaw = updates.quantityTotal ?? current.quantityTotal
        const quantityTotal = Math.max(0, Math.floor(quantityTotalRaw))
        const quantityReservedRaw = updates.quantityReserved ?? current.quantityReserved
        const quantityReserved = Math.max(0, Math.min(quantityTotal, Math.floor(quantityReservedRaw)))

        const nextItem: InventoryItem = {
            ...current,
            ...updates,
            quantityTotal,
            quantityReserved,
            name: updates.name ?? current.name,
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
    getInventoryUsage: () => computeInventoryUsage(get().items),
    getLayoutMetrics: () => computeLayoutMetrics(get().items),
    hasInventoryForType: (type) => {
        const state = get()
        const usage = computeInventoryUsage(state.items)
        const inventoryItem = state.inventoryCatalog.find((item) => item.furnitureType === type)
        if (!inventoryItem) return true

        const available = Math.max(0, inventoryItem.quantityTotal - inventoryItem.quantityReserved)
        return usage[type] < available
    },

    saveScenario: (name) => {
        let createdScenarioId = ''
        set((state) => {
            const now = nowIso()
            createdScenarioId = crypto.randomUUID()
            const trimmedName = name?.trim()
            const fallbackName = `Scenario ${state.scenarios.length + 1}`
            const scenarioName = trimmedName && trimmedName.length > 0 ? trimmedName : fallbackName

            const scenario: Scenario = {
                id: createdScenarioId,
                name: scenarioName,
                status: 'draft',
                createdAt: now,
                updatedAt: now,
                snapshot: createScenarioSnapshot(state)
            }

            return {
                scenarios: [...state.scenarios, scenario],
                activeScenarioId: createdScenarioId
            }
        })

        return createdScenarioId
    },

    loadScenario: (id, options) => set((state) => {
        const scenario = state.scenarios.find((candidate) => candidate.id === id)
        if (!scenario) return {}

        const items = cloneItems(scenario.snapshot.items)
        const transformMode = scenario.snapshot.transformMode
        const scenarios = state.scenarios.map((entry) =>
            entry.id === id ? { ...entry, updatedAt: nowIso() } : entry
        )

        return {
            ...historyBeforeMutation(state, recordHistory(options)),
            items,
            selectedIds: [],
            transformMode,
            draggedItemType: null,
            chairPrompt: null,
            isDragging: false,
            scenarios,
            activeScenarioId: id,
            inventoryWarning: getInventoryWarning(state.inventoryCatalog, computeInventoryUsage(items))
        }
    }),

    deleteScenario: (id) => set((state) => {
        const scenarios = state.scenarios.filter((scenario) => scenario.id !== id)
        if (scenarios.length === state.scenarios.length) return {}

        const activeScenarioId = state.activeScenarioId === id
            ? (scenarios[0]?.id ?? null)
            : state.activeScenarioId

        return {
            scenarios,
            activeScenarioId
        }
    }),

    renameScenario: (id, name) => set((state) => {
        const trimmed = name.trim()
        if (!trimmed) return {}

        const index = state.scenarios.findIndex((scenario) => scenario.id === id)
        if (index === -1) return {}

        const scenarios = [...state.scenarios]
        scenarios[index] = {
            ...scenarios[index],
            name: trimmed,
            updatedAt: nowIso()
        }

        return { scenarios }
    }),

    setScenarioStatus: (id, status) => set((state) => {
        const index = state.scenarios.findIndex((scenario) => scenario.id === id)
        if (index === -1) return {}

        const scenarios = [...state.scenarios]
        scenarios[index] = {
            ...scenarios[index],
            status,
            updatedAt: nowIso()
        }

        return { scenarios }
    }),

    resetProject: () => set((state) => ({
        ...historyBeforeMutation(state, true),
        ...buildDefaultPersistedState(),
        ...INITIAL_TRANSIENT_STATE
    })),

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
        const historyPast = state.historyPast.slice(0, -1)
        const historyFuture = [current, ...state.historyFuture].slice(0, HISTORY_LIMIT)
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
            ...historyPatch(historyPast, historyFuture)
        }
    }),

    redo: () => set((state) => {
        if (state.historyFuture.length === 0) return {}

        const current = createVenueSnapshot(state)
        const next = state.historyFuture[0]
        const historyPast = [...state.historyPast, current].slice(-HISTORY_LIMIT)
        const historyFuture = state.historyFuture.slice(1)
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
            ...historyPatch(historyPast, historyFuture)
        }
    }),

    addItem: (type, position = [0, 0, 0], rotation, groupId, options) => set((state) => {
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
            rotation: rotation ?? (type === 'chair' ? [0, -Math.PI / 2, 0] : [0, 0, 0])
        }]

        return {
            ...historyBeforeMutation(state, recordHistory(options)),
            items,
            selectedIds: [],
            inventoryWarning: getInventoryWarning(state.inventoryCatalog, computeInventoryUsage(items)),
            ...syncActiveScenario(state, items)
        }
    }),

    updateItem: (id, updates, options) => set((state) => {
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

    updateItems: (updates, options) => set((state) => {
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

    removeItems: (ids, options) => set((state) => {
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

    ungroupItems: (ids, options) => set((state) => {
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

    groupItems: (ids, options) => set((state) => {
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

    setSelection: (ids) => set({ selectedIds: ids }),

    toggleSnapping: () => set((state) => ({ snappingEnabled: !state.snappingEnabled })),

    toggleTransformMode: () => set((state) => {
        const transformMode: TransformMode = state.transformMode === 'translate' ? 'rotate' : 'translate'
        return {
            transformMode,
            ...syncActiveScenario(state, state.items, transformMode)
        }
    }),

    rotateSelection: (amountDegrees, options) => set((state) => {
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
}),
    {
        name: VENUE_STORE_STORAGE_KEY,
        version: VENUE_STORE_VERSION,
        storage: createJSONStorage(() => localStorage),
        partialize: (state) => ({
            items: cloneItems(state.items),
            snappingEnabled: state.snappingEnabled,
            snapGrid: state.snapGrid,
            transformMode: state.transformMode,
            inventoryCatalog: cloneInventory(state.inventoryCatalog),
            scenarios: state.scenarios.map((scenario) => ({
                ...scenario,
                snapshot: {
                    items: cloneItems(scenario.snapshot.items),
                    transformMode: scenario.snapshot.transformMode
                }
            })),
            activeScenarioId: state.activeScenarioId
        }),
        merge: (persistedState, currentState) => mergeRehydratedState(persistedState, currentState),
        migrate: (persistedState) => sanitizePersistedState(persistedState)
    })
)
