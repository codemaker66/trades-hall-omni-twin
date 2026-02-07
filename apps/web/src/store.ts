// Barrel re-export — all consumers continue to import from '../../store'
export { useVenueStore } from './store/index'
export type {
    FurnitureType, TransformMode, ScenarioStatus, ProjectImportMode, ProjectImportResult,
    FurnitureItem, InventoryItem, InventoryUsage, LayoutMetrics, MutationOptions,
    InventoryUpdate, VenueSnapshot, ScenarioSnapshot, Scenario, VenueState,
    PersistedVenueState
} from './store/types'
