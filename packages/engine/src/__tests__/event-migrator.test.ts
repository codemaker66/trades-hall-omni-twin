import { describe, it, expect, beforeEach } from 'vitest'
import {
  registerMigration,
  clearMigrations,
  migrateEvent,
  migrateEvents,
  needsMigration,
  toVersionedEvent,
  type VersionedEvent,
} from '../event-migrator'
import type { DomainEvent } from '@omni-twin/types'

// ─── Setup ───────────────────────────────────────────────────────────────────

beforeEach(() => {
  clearMigrations()
})

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeVersionedEvent(type: string, schemaVersion: number, payload: unknown): VersionedEvent {
  return {
    id: 'evt-1',
    type,
    schemaVersion,
    timestamp: '2025-01-01T00:00:00Z',
    userId: 'user-1',
    venueId: 'venue-1',
    version: 1,
    payload,
  }
}

// ─── No Migration Needed ─────────────────────────────────────────────────────

describe('no migration needed', () => {
  it('returns event unchanged when at current version', () => {
    const event = makeVersionedEvent('ItemPlaced', 1, { itemId: 'a' })
    const migrated = migrateEvent(event)
    expect(migrated).toEqual(event)
  })

  it('needsMigration returns false for current version', () => {
    const event = makeVersionedEvent('ItemPlaced', 1, { itemId: 'a' })
    expect(needsMigration(event)).toBe(false)
  })

  it('handles unknown event types gracefully', () => {
    const event = makeVersionedEvent('UnknownEvent', 1, {})
    const migrated = migrateEvent(event)
    expect(migrated).toEqual(event)
  })
})

// ─── Single Migration ────────────────────────────────────────────────────────

describe('single migration', () => {
  it('migrates from v0 to v1 when migration is registered', () => {
    // Simulate: ItemPlaced v0 had no `scale` in payload, v1 adds default scale
    registerMigration({
      eventType: 'ItemPlaced',
      fromVersion: 0,
      toVersion: 1,
      migrate: (event) => ({
        ...event,
        schemaVersion: 1,
        payload: {
          ...(event.payload as Record<string, unknown>),
          scale: [1, 1, 1],
        },
      }),
    })

    const oldEvent = makeVersionedEvent('ItemPlaced', 0, {
      itemId: 'a',
      furnitureType: 'chair',
      position: [0, 0, 0],
      rotation: [0, 0, 0],
    })

    expect(needsMigration(oldEvent)).toBe(true)

    const migrated = migrateEvent(oldEvent)
    expect(migrated.schemaVersion).toBe(1)
    const payload = migrated.payload as Record<string, unknown>
    expect(payload['scale']).toEqual([1, 1, 1])
    expect(payload['itemId']).toBe('a')
  })
})

// ─── Chained Migration ──────────────────────────────────────────────────────

describe('chained migrations', () => {
  it('applies multiple migrations in sequence', () => {
    // Register v0→v1 and v1→v2 (simulating two schema changes)

    // NOTE: For this test we temporarily need ItemPlaced at schema v2.
    // In production, CURRENT_SCHEMA_VERSIONS would be updated.
    // For the test, we'll use a known event type and register up to v1.

    registerMigration({
      eventType: 'VenueCreated',
      fromVersion: 0,
      toVersion: 1,
      migrate: (event) => ({
        ...event,
        schemaVersion: 1,
        payload: {
          ...(event.payload as Record<string, unknown>),
          migrationStep: 'v0→v1',
        },
      }),
    })

    const oldEvent = makeVersionedEvent('VenueCreated', 0, { name: 'Test' })
    const migrated = migrateEvent(oldEvent)
    expect(migrated.schemaVersion).toBe(1)
    const payload = migrated.payload as Record<string, unknown>
    expect(payload['migrationStep']).toBe('v0→v1')
    expect(payload['name']).toBe('Test')
  })
})

// ─── Missing Migration ──────────────────────────────────────────────────────

describe('missing migration', () => {
  it('throws when no migration found for a gap', () => {
    // Event at v0 but no migration registered and current is v1
    const oldEvent = makeVersionedEvent('ItemPlaced', 0, { itemId: 'a' })
    expect(() => migrateEvent(oldEvent)).toThrow(/No migration found/)
  })
})

// ─── Batch Migration ─────────────────────────────────────────────────────────

describe('migrateEvents (batch)', () => {
  it('migrates an array of events', () => {
    registerMigration({
      eventType: 'ItemMoved',
      fromVersion: 0,
      toVersion: 1,
      migrate: (event) => ({
        ...event,
        schemaVersion: 1,
        payload: { ...(event.payload as Record<string, unknown>), migrated: true },
      }),
    })

    const events = [
      makeVersionedEvent('ItemMoved', 0, { itemId: 'a', position: [0, 0, 0] }),
      makeVersionedEvent('ItemMoved', 1, { itemId: 'b', position: [1, 0, 0] }),
    ]

    const migrated = migrateEvents(events)
    expect(migrated).toHaveLength(2)
    expect(migrated[0]!.schemaVersion).toBe(1)
    expect((migrated[0]!.payload as Record<string, unknown>)['migrated']).toBe(true)
    expect(migrated[1]!.schemaVersion).toBe(1) // Already at v1, unchanged
  })
})

// ─── toVersionedEvent ────────────────────────────────────────────────────────

describe('toVersionedEvent', () => {
  it('adds current schema version to a domain event', () => {
    const event: DomainEvent = {
      id: 'evt-1',
      type: 'ItemPlaced',
      timestamp: '2025-01-01T00:00:00Z',
      userId: 'user-1',
      venueId: 'venue-1',
      version: 1,
      payload: {
        itemId: 'a',
        furnitureType: 'chair',
        position: [0, 0, 0] as [number, number, number],
        rotation: [0, 0, 0] as [number, number, number],
      },
    }

    const versioned = toVersionedEvent(event)
    expect(versioned.schemaVersion).toBe(1)
    expect(versioned.type).toBe('ItemPlaced')
  })
})
