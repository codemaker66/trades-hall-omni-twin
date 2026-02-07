/**
 * Schema versioning for domain events.
 * Each event type has a schema version that may evolve over time.
 * When reading old events, migrations transform them to the current schema.
 */

import type { DomainEventType } from './events'

/** Current schema version per event type. */
export const CURRENT_SCHEMA_VERSIONS: Record<DomainEventType, number> = {
  VenueCreated: 1,
  VenueRenamed: 1,
  VenueArchived: 1,
  ItemPlaced: 1,
  ItemMoved: 1,
  ItemRotated: 1,
  ItemScaled: 1,
  ItemRemoved: 1,
  ItemsBatchMoved: 1,
  ItemsBatchRotated: 1,
  GroupCreated: 1,
  GroupDissolved: 1,
  ItemsGrouped: 1,
  ItemsUngrouped: 1,
  ScenarioCreated: 1,
  ScenarioActivated: 1,
  ScenarioDeleted: 1,
  LayoutSnapshotCreated: 1,
  LayoutImported: 1,
}

/** Schema version metadata attached to stored events. */
export interface SchemaVersionMeta {
  schemaVersion: number
  eventType: DomainEventType
}
