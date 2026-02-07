/**
 * Event migrator: transforms events from old schema versions to current.
 * Migrations are registered per event type and run in sequence.
 */

import type { DomainEvent, DomainEventType } from '@omni-twin/types'
import { CURRENT_SCHEMA_VERSIONS } from '@omni-twin/types/schema-version'

// ─── Types ───────────────────────────────────────────────────────────────────

/** A single migration step for one event type. */
export interface EventMigration {
  eventType: DomainEventType
  fromVersion: number
  toVersion: number
  migrate: (event: VersionedEvent) => VersionedEvent
}

/** An event with a schema version attached. */
export interface VersionedEvent {
  type: string
  schemaVersion: number
  payload: unknown
  [key: string]: unknown
}

// ─── Migration Registry ──────────────────────────────────────────────────────

const migrations: EventMigration[] = []

/**
 * Register a migration for a specific event type.
 * Migrations must be registered in order (fromVersion → toVersion).
 */
export function registerMigration(migration: EventMigration): void {
  migrations.push(migration)
  // Keep sorted by eventType + fromVersion for efficient lookup
  migrations.sort((a, b) => {
    if (a.eventType < b.eventType) return -1
    if (a.eventType > b.eventType) return 1
    return a.fromVersion - b.fromVersion
  })
}

/**
 * Clear all registered migrations. Useful for testing.
 */
export function clearMigrations(): void {
  migrations.length = 0
}

// ─── Migration Execution ─────────────────────────────────────────────────────

/**
 * Migrate a versioned event to the current schema version.
 * Applies all necessary migrations in sequence.
 *
 * @returns The migrated event, or the original if already current.
 */
export function migrateEvent(event: VersionedEvent): VersionedEvent {
  const eventType = event.type as DomainEventType
  const currentVersion = CURRENT_SCHEMA_VERSIONS[eventType]

  if (currentVersion === undefined) {
    // Unknown event type — return as-is
    return event
  }

  let migrated = { ...event }

  while (migrated.schemaVersion < currentVersion) {
    const migration = migrations.find(
      (m) => m.eventType === eventType && m.fromVersion === migrated.schemaVersion
    )

    if (!migration) {
      throw new Error(
        `No migration found for ${eventType} from version ${migrated.schemaVersion} to ${migrated.schemaVersion + 1}`
      )
    }

    migrated = migration.migrate(migrated)
    if (migrated.schemaVersion !== migration.toVersion) {
      throw new Error(
        `Migration for ${eventType} v${migration.fromVersion}→v${migration.toVersion} did not update schemaVersion`
      )
    }
  }

  return migrated
}

/**
 * Migrate a batch of versioned events.
 */
export function migrateEvents(events: VersionedEvent[]): VersionedEvent[] {
  return events.map(migrateEvent)
}

/**
 * Check if an event needs migration.
 */
export function needsMigration(event: VersionedEvent): boolean {
  const eventType = event.type as DomainEventType
  const currentVersion = CURRENT_SCHEMA_VERSIONS[eventType]
  if (currentVersion === undefined) return false
  return event.schemaVersion < currentVersion
}

/**
 * Convert a DomainEvent to a VersionedEvent (at current schema version).
 */
export function toVersionedEvent(event: DomainEvent): VersionedEvent {
  return {
    ...event,
    schemaVersion: CURRENT_SCHEMA_VERSIONS[event.type] ?? 1,
  }
}
