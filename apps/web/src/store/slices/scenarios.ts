import type { Scenario, ScenarioStatus, MutationOptions, ProjectImportMode, ProjectImportResult, StoreSet, StoreGet } from '../types'
import {
    nowIso, cloneItems, cloneScenarios, cloneInventory,
    createScenarioSnapshot, createPersistedStateFromVenueState, createProjectExportDocument,
    parseImportedPersistedState, mergePersistedVenueStates, clonePersistedVenueState,
    historyBeforeMutation, recordHistory, computeInventoryUsage, getInventoryWarning,
    buildDefaultPersistedState, INITIAL_TRANSIENT_STATE
} from '../helpers'

export const createScenariosSlice = (set: StoreSet, get: StoreGet) => ({
    scenarios: [] as Scenario[],
    activeScenarioId: null as string | null,

    saveScenario: (name?: string): string => {
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

    loadScenario: (id: string, options?: MutationOptions) => set((state) => {
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

    deleteScenario: (id: string) => set((state) => {
        const scenarios = state.scenarios.filter((scenario) => scenario.id !== id)
        if (scenarios.length === state.scenarios.length) return {}

        const activeScenarioId = state.activeScenarioId === id
            ? (scenarios[0]?.id ?? null)
            : state.activeScenarioId

        return { scenarios, activeScenarioId }
    }),

    renameScenario: (id: string, name: string) => set((state) => {
        const trimmed = name.trim()
        if (!trimmed) return {}

        const index = state.scenarios.findIndex((scenario) => scenario.id === id)
        if (index === -1) return {}

        const scenarios = [...state.scenarios]
        const existing = scenarios[index]!
        scenarios[index] = {
            ...existing,
            name: trimmed,
            updatedAt: nowIso()
        }

        return { scenarios }
    }),

    setScenarioStatus: (id: string, status: ScenarioStatus) => set((state) => {
        const index = state.scenarios.findIndex((scenario) => scenario.id === id)
        if (index === -1) return {}

        const scenarios = [...state.scenarios]
        const existing = scenarios[index]!
        scenarios[index] = {
            ...existing,
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

    exportProject: (): string => {
        const persistedState = createPersistedStateFromVenueState(get())
        const projectDocument = createProjectExportDocument(persistedState)
        return JSON.stringify(projectDocument, null, 2)
    },

    importProject: (payload: string, options?: { mode?: ProjectImportMode }): ProjectImportResult => {
        const mode: ProjectImportMode = options?.mode === 'merge' ? 'merge' : 'replace'
        const parsedImport = parseImportedPersistedState(payload)
        if (!parsedImport.ok) {
            return parsedImport
        }

        set((state) => {
            const currentPersisted = createPersistedStateFromVenueState(state)
            const nextPersisted = mode === 'merge'
                ? mergePersistedVenueStates(currentPersisted, parsedImport.state)
                : clonePersistedVenueState(parsedImport.state)

            return {
                ...historyBeforeMutation(state, true),
                ...nextPersisted,
                ...INITIAL_TRANSIENT_STATE,
                inventoryWarning: getInventoryWarning(nextPersisted.inventoryCatalog, computeInventoryUsage(nextPersisted.items))
            }
        })

        return {
            ok: true,
            message: mode === 'merge'
                ? 'Project imported in merge mode.'
                : 'Project imported and replaced current project.'
        }
    }
})
