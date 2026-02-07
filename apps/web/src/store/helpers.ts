import type {
    FurnitureItem, FurnitureType, InventoryItem, InventoryUsage,
    LayoutMetrics, MutationOptions, PersistedVenueState, Scenario,
    ScenarioSnapshot, ScenarioStatus, TransformMode, VenueSnapshot, VenueState
} from './types'

// ── Constants ────────────────────────────────────────────────────────────────

export const HISTORY_LIMIT = 200
export const VENUE_STORE_STORAGE_KEY = 'omnitwin_venue_store_v1'
export const VENUE_STORE_VERSION = 1

export const DEFAULT_INVENTORY: InventoryItem[] = [
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

// ── Pure Utilities ───────────────────────────────────────────────────────────

export const nowIso = () => new Date().toISOString()

export const cloneTuple3 = ([x, y, z]: [number, number, number]): [number, number, number] => [x, y, z]

export const cloneItem = (item: FurnitureItem): FurnitureItem => ({
    ...item,
    position: cloneTuple3(item.position),
    rotation: cloneTuple3(item.rotation)
})

export const cloneItems = (items: FurnitureItem[]) => items.map(cloneItem)

export const cloneInventory = (inventory: InventoryItem[]) => inventory.map((item) => ({ ...item }))

export const cloneScenario = (scenario: Scenario): Scenario => ({
    ...scenario,
    snapshot: {
        items: cloneItems(scenario.snapshot.items),
        transformMode: scenario.snapshot.transformMode
    }
})

export const cloneScenarios = (scenarios: Scenario[]) => scenarios.map(cloneScenario)

export const emptyInventoryUsage = (): InventoryUsage => ({
    'round-table': 0,
    'trestle-table': 0,
    'chair': 0,
    'platform': 0
})

export const computeInventoryUsage = (items: FurnitureItem[]): InventoryUsage => {
    const usage = emptyInventoryUsage()
    for (const item of items) {
        usage[item.type] += 1
    }
    return usage
}

export const computeLayoutMetrics = (items: FurnitureItem[]): LayoutMetrics => {
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

    return { seatCount: chairCount, tableCount, chairCount, platformCount }
}

export const createVenueSnapshot = (state: Pick<VenueState, 'items' | 'selectedIds' | 'transformMode'>): VenueSnapshot => ({
    items: cloneItems(state.items),
    selectedIds: [...state.selectedIds],
    transformMode: state.transformMode
})

export const createScenarioSnapshot = (state: Pick<VenueState, 'items' | 'transformMode'>): ScenarioSnapshot => ({
    items: cloneItems(state.items),
    transformMode: state.transformMode
})

export const snapshotEquals = (a: VenueSnapshot, b: VenueSnapshot): boolean => {
    if (a.transformMode !== b.transformMode) return false
    if (a.selectedIds.length !== b.selectedIds.length) return false
    if (a.items.length !== b.items.length) return false

    for (let i = 0; i < a.selectedIds.length; i++) {
        if (a.selectedIds[i] !== b.selectedIds[i]) return false
    }

    for (let i = 0; i < a.items.length; i++) {
        const left = a.items[i]!
        const right = b.items[i]!

        if (left.id !== right.id) return false
        if (left.type !== right.type) return false
        if (left.groupId !== right.groupId) return false
        if (left.position[0] !== right.position[0] || left.position[1] !== right.position[1] || left.position[2] !== right.position[2]) return false
        if (left.rotation[0] !== right.rotation[0] || left.rotation[1] !== right.rotation[1] || left.rotation[2] !== right.rotation[2]) return false
    }

    return true
}

export const getInventoryWarning = (inventoryCatalog: InventoryItem[], usage: InventoryUsage): string | null => {
    for (const catalogItem of inventoryCatalog) {
        const available = Math.max(0, catalogItem.quantityTotal - catalogItem.quantityReserved)
        const used = usage[catalogItem.furnitureType]

        if (used > available) {
            return `${catalogItem.name} is over-allocated (${used}/${available} available).`
        }
    }

    return null
}

export const historyPatch = (historyPast: VenueSnapshot[], historyFuture: VenueSnapshot[]) => ({
    historyPast,
    historyFuture,
    canUndo: historyPast.length > 0,
    canRedo: historyFuture.length > 0
})

export const recordHistory = (options?: MutationOptions) => options?.recordHistory !== false

export const historyBeforeMutation = (state: VenueState, shouldRecordHistory: boolean) => {
    if (!shouldRecordHistory || state.historyBatch) {
        return {}
    }

    const historyPast = [...state.historyPast, createVenueSnapshot(state)].slice(-HISTORY_LIMIT)
    return historyPatch(historyPast, [])
}

export const syncActiveScenario = (state: VenueState, items: FurnitureItem[], transformMode: TransformMode = state.transformMode) => {
    if (!state.activeScenarioId) return {}

    const scenarioIndex = state.scenarios.findIndex((scenario) => scenario.id === state.activeScenarioId)
    if (scenarioIndex === -1) return {}

    const scenarios = [...state.scenarios]
    const existing = scenarios[scenarioIndex]!
    scenarios[scenarioIndex] = {
        ...existing,
        updatedAt: nowIso(),
        snapshot: {
            items: cloneItems(items),
            transformMode
        }
    }

    return { scenarios }
}

export const createStoreId = (): string => {
    if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
        return crypto.randomUUID()
    }

    return `id_${Date.now()}_${Math.random().toString(16).slice(2)}`
}

// ── Sanitization ─────────────────────────────────────────────────────────────

export const isFurnitureType = (value: unknown): value is FurnitureType =>
    value === 'round-table' || value === 'trestle-table' || value === 'chair' || value === 'platform'

export const isTransformMode = (value: unknown): value is TransformMode =>
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

export const sanitizeFurnitureItem = (raw: unknown): FurnitureItem | null => {
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

// ── Persistence ──────────────────────────────────────────────────────────────

export const buildDefaultPersistedState = (): PersistedVenueState => ({
    items: [],
    snappingEnabled: true,
    snapGrid: 0.5,
    transformMode: 'translate',
    inventoryCatalog: cloneInventory(DEFAULT_INVENTORY),
    scenarios: [],
    activeScenarioId: null
})

export const clonePersistedVenueState = (state: PersistedVenueState): PersistedVenueState => ({
    items: cloneItems(state.items),
    snappingEnabled: state.snappingEnabled,
    snapGrid: state.snapGrid,
    transformMode: state.transformMode,
    inventoryCatalog: cloneInventory(state.inventoryCatalog),
    scenarios: cloneScenarios(state.scenarios),
    activeScenarioId: state.activeScenarioId
})

const isObjectRecord = (value: unknown): value is Record<string, unknown> =>
    typeof value === 'object' && value !== null

const looksLikePersistedVenueState = (value: unknown): value is Partial<PersistedVenueState> => {
    if (!isObjectRecord(value)) return false
    return (
        'items' in value ||
        'inventoryCatalog' in value ||
        'scenarios' in value ||
        'transformMode' in value ||
        'snappingEnabled' in value ||
        'snapGrid' in value ||
        'activeScenarioId' in value
    )
}

export const sanitizePersistedState = (raw: unknown): PersistedVenueState => {
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

export const createPersistedStateFromVenueState = (state: Pick<
    VenueState,
    'items' | 'snappingEnabled' | 'snapGrid' | 'transformMode' | 'inventoryCatalog' | 'scenarios' | 'activeScenarioId'
>): PersistedVenueState => ({
    items: cloneItems(state.items),
    snappingEnabled: state.snappingEnabled,
    snapGrid: state.snapGrid,
    transformMode: state.transformMode,
    inventoryCatalog: cloneInventory(state.inventoryCatalog),
    scenarios: cloneScenarios(state.scenarios),
    activeScenarioId: state.activeScenarioId
})

// ── Project Import/Export ────────────────────────────────────────────────────

export const createProjectExportDocument = (state: PersistedVenueState) => ({
    format: 'omnitwin-venue-project' as const,
    version: 1,
    exportedAt: nowIso(),
    data: clonePersistedVenueState(state)
})

export const parseImportedPersistedState = (
    payload: string
): { ok: true, state: PersistedVenueState } | { ok: false, message: string } => {
    const trimmed = payload.trim()
    if (!trimmed) {
        return { ok: false, message: 'Import failed: the file is empty.' }
    }

    let parsed: unknown
    try {
        parsed = JSON.parse(trimmed)
    } catch {
        return { ok: false, message: 'Import failed: invalid JSON file.' }
    }

    if (!isObjectRecord(parsed)) {
        return { ok: false, message: 'Import failed: unsupported project payload.' }
    }

    let rawState: unknown = parsed
    if ('data' in parsed || 'format' in parsed || 'version' in parsed) {
        const format = parsed.format
        if (typeof format === 'string' && format !== 'omnitwin-venue-project') {
            return { ok: false, message: `Import failed: unsupported project format "${format}".` }
        }

        const version = parsed.version
        if (typeof version === 'number' && version > 1) {
            return {
                ok: false,
                message: `Import failed: project version ${version} is newer than supported version 1.`
            }
        }

        if (!('data' in parsed)) {
            return { ok: false, message: 'Import failed: missing project data payload.' }
        }

        rawState = parsed.data
    }

    if (!looksLikePersistedVenueState(rawState)) {
        return { ok: false, message: 'Import failed: file does not look like an OmniTwin project export.' }
    }

    return { ok: true, state: sanitizePersistedState(rawState) }
}

const dedupeScenarioName = (name: string, usedNames: Set<string>): string => {
    const baseName = name.trim().length > 0 ? name.trim() : 'Scenario'
    const lowerBase = baseName.toLowerCase()
    if (!usedNames.has(lowerBase)) {
        usedNames.add(lowerBase)
        return baseName
    }

    let suffix = 2
    while (true) {
        const candidate = `${baseName} (${suffix})`
        const lowerCandidate = candidate.toLowerCase()
        if (!usedNames.has(lowerCandidate)) {
            usedNames.add(lowerCandidate)
            return candidate
        }
        suffix += 1
    }
}

const mergeInventoryCatalogForImport = (
    currentInventory: InventoryItem[],
    importedInventory: InventoryItem[]
): InventoryItem[] => {
    const currentByType = new Map(currentInventory.map((item) => [item.furnitureType, item]))
    const importedByType = new Map(importedInventory.map((item) => [item.furnitureType, item]))

    return DEFAULT_INVENTORY.map((fallback) => {
        const currentItem = currentByType.get(fallback.furnitureType) ?? fallback
        const importedItem = importedByType.get(fallback.furnitureType)
        if (!importedItem) {
            return { ...currentItem }
        }

        const quantityTotal = Math.max(currentItem.quantityTotal, importedItem.quantityTotal)
        const quantityReserved = Math.min(
            quantityTotal,
            Math.max(currentItem.quantityReserved, importedItem.quantityReserved)
        )
        const seatsPerItem = currentItem.seatsPerItem ?? importedItem.seatsPerItem

        return {
            ...currentItem,
            quantityTotal,
            quantityReserved,
            seatsPerItem
        }
    })
}

export const mergePersistedVenueStates = (
    current: PersistedVenueState,
    imported: PersistedVenueState
): PersistedVenueState => {
    const mergedItems = cloneItems(current.items)
    const usedItemIds = new Set(mergedItems.map((item) => item.id))
    for (const importedItem of imported.items) {
        let nextId = importedItem.id
        while (!nextId || usedItemIds.has(nextId)) {
            nextId = createStoreId()
        }
        usedItemIds.add(nextId)
        mergedItems.push({
            ...importedItem,
            id: nextId,
            position: cloneTuple3(importedItem.position),
            rotation: cloneTuple3(importedItem.rotation)
        })
    }

    const mergedScenarios = cloneScenarios(current.scenarios)
    const usedScenarioIds = new Set(mergedScenarios.map((scenario) => scenario.id))
    const usedScenarioNames = new Set(mergedScenarios.map((scenario) => scenario.name.trim().toLowerCase()))
    let importedActiveScenarioId: string | null = null

    for (const importedScenario of imported.scenarios) {
        let nextScenarioId = importedScenario.id
        while (!nextScenarioId || usedScenarioIds.has(nextScenarioId)) {
            nextScenarioId = createStoreId()
        }
        usedScenarioIds.add(nextScenarioId)

        const nextScenarioName = dedupeScenarioName(importedScenario.name, usedScenarioNames)
        const nextScenario = cloneScenario(importedScenario)
        nextScenario.id = nextScenarioId
        nextScenario.name = nextScenarioName

        if (imported.activeScenarioId === importedScenario.id) {
            importedActiveScenarioId = nextScenarioId
        }

        mergedScenarios.push(nextScenario)
    }

    return {
        items: mergedItems,
        snappingEnabled: current.snappingEnabled,
        snapGrid: current.snapGrid,
        transformMode: current.transformMode,
        inventoryCatalog: mergeInventoryCatalogForImport(current.inventoryCatalog, imported.inventoryCatalog),
        scenarios: mergedScenarios,
        activeScenarioId: current.activeScenarioId ?? importedActiveScenarioId
    }
}

// ── Initial State ────────────────────────────────────────────────────────────

export const INITIAL_TRANSIENT_STATE = {
    selectedIds: [] as string[],
    draggedItemType: null as FurnitureType | null,
    isDragging: false,
    shortcutsHelpOpen: false,
    chairPrompt: null as VenueState['chairPrompt'],
    inventoryWarning: null as string | null,
    canUndo: false,
    canRedo: false,
    historyPast: [] as VenueSnapshot[],
    historyFuture: [] as VenueSnapshot[],
    historyBatch: null as VenueSnapshot | null
}

export const buildInitialState = () => ({
    ...buildDefaultPersistedState(),
    ...INITIAL_TRANSIENT_STATE
})

export const mergeRehydratedState = (persistedRaw: unknown, currentState: VenueState): VenueState => {
    const persisted = sanitizePersistedState(persistedRaw)
    const inventoryWarning = getInventoryWarning(persisted.inventoryCatalog, computeInventoryUsage(persisted.items))

    return {
        ...currentState,
        ...persisted,
        ...INITIAL_TRANSIENT_STATE,
        inventoryWarning
    }
}
