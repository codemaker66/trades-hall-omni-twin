/**
 * Event store for venue domain events.
 * Provides append (with optimistic concurrency), replay, and snapshot operations.
 */

import { eq, and, gt, asc, desc } from 'drizzle-orm'
import { db } from './client'
import { venueEvents, venueSnapshots } from './schema/index'
import type { DomainEvent, DomainEventBase } from '@omni-twin/types'

// ─── Types ───────────────────────────────────────────────────────────────────

/** A stored event row from the database. */
export interface StoredEvent {
  id: number
  venueId: string
  version: number
  type: string
  payload: unknown
  userId: string | null
  timestamp: Date
}

/** A stored snapshot row from the database. */
export interface StoredSnapshot {
  id: string
  venueId: string
  version: number
  state: unknown
  createdAt: Date
}

/** Error when optimistic concurrency check fails. */
export class ConcurrencyError extends Error {
  constructor(
    public readonly venueId: string,
    public readonly expectedVersion: number,
    public readonly actualVersion: number,
  ) {
    super(
      `Concurrency conflict for venue ${venueId}: expected version ${expectedVersion}, actual ${actualVersion}`
    )
    this.name = 'ConcurrencyError'
  }
}

// ─── Append ──────────────────────────────────────────────────────────────────

/**
 * Append events to a venue's event stream with optimistic concurrency.
 * Events are assigned sequential versions starting from expectedVersion + 1.
 *
 * @param venueId - The venue's unique ID
 * @param expectedVersion - The last known version (0 for new streams)
 * @param events - Events to append (version + id will be assigned)
 * @throws ConcurrencyError if the expected version doesn't match
 */
export async function appendEvents(
  venueId: string,
  expectedVersion: number,
  events: Array<Omit<DomainEventBase, 'id' | 'version'> & { type: string; payload: unknown }>,
): Promise<StoredEvent[]> {
  if (events.length === 0) return []

  // Use a transaction to ensure atomicity
  return await db.transaction(async (tx) => {
    // Check current max version
    const latest = await tx
      .select({ version: venueEvents.version })
      .from(venueEvents)
      .where(eq(venueEvents.venueId, venueId))
      .orderBy(desc(venueEvents.version))
      .limit(1)

    const currentVersion = latest[0]?.version ?? 0

    if (currentVersion !== expectedVersion) {
      throw new ConcurrencyError(venueId, expectedVersion, currentVersion)
    }

    // Insert events with sequential versions
    const rows = events.map((event, i) => ({
      venueId,
      version: expectedVersion + i + 1,
      type: event.type,
      payload: event.payload,
      userId: event.userId,
      timestamp: new Date(event.timestamp),
    }))

    const inserted = await tx.insert(venueEvents).values(rows).returning()

    return inserted.map((row) => ({
      id: row.id,
      venueId: row.venueId,
      version: row.version,
      type: row.type,
      payload: row.payload,
      userId: row.userId,
      timestamp: row.timestamp,
    }))
  })
}

// ─── Replay ──────────────────────────────────────────────────────────────────

/**
 * Get all events for a venue, optionally from a specific version.
 * Events are returned in version order (ascending).
 */
export async function getEvents(
  venueId: string,
  fromVersion?: number,
): Promise<StoredEvent[]> {
  const conditions = [eq(venueEvents.venueId, venueId)]
  if (fromVersion !== undefined && fromVersion > 0) {
    conditions.push(gt(venueEvents.version, fromVersion))
  }

  const rows = await db
    .select()
    .from(venueEvents)
    .where(and(...conditions))
    .orderBy(asc(venueEvents.version))

  return rows.map((row) => ({
    id: row.id,
    venueId: row.venueId,
    version: row.version,
    type: row.type,
    payload: row.payload,
    userId: row.userId,
    timestamp: row.timestamp,
  }))
}

/**
 * Get the current (latest) version number for a venue's event stream.
 * Returns 0 if no events exist.
 */
export async function getCurrentVersion(venueId: string): Promise<number> {
  const latest = await db
    .select({ version: venueEvents.version })
    .from(venueEvents)
    .where(eq(venueEvents.venueId, venueId))
    .orderBy(desc(venueEvents.version))
    .limit(1)

  return latest[0]?.version ?? 0
}

// ─── Snapshots ───────────────────────────────────────────────────────────────

/**
 * Get the latest snapshot for a venue.
 * Returns null if no snapshot exists.
 */
export async function getSnapshot(venueId: string): Promise<StoredSnapshot | null> {
  const rows = await db
    .select()
    .from(venueSnapshots)
    .where(eq(venueSnapshots.venueId, venueId))
    .orderBy(desc(venueSnapshots.version))
    .limit(1)

  const row = rows[0]
  if (!row) return null

  return {
    id: row.id,
    venueId: row.venueId,
    version: row.version,
    state: row.state,
    createdAt: row.createdAt,
  }
}

/**
 * Save a snapshot of the venue's state at a specific version.
 * Used to speed up state reconstruction by avoiding full replay.
 */
export async function saveSnapshot(
  venueId: string,
  version: number,
  state: unknown,
): Promise<StoredSnapshot> {
  const [row] = await db
    .insert(venueSnapshots)
    .values({ venueId, version, state })
    .returning()

  if (!row) throw new Error('Failed to insert snapshot')

  return {
    id: row.id,
    venueId: row.venueId,
    version: row.version,
    state: row.state,
    createdAt: row.createdAt,
  }
}

// ─── Reconstruct Helper ─────────────────────────────────────────────────────

/**
 * Convert a stored event row back to a typed DomainEvent.
 * The caller is responsible for validating the event shape.
 */
export function toDomainEvent(stored: StoredEvent): DomainEvent {
  return {
    id: String(stored.id),
    type: stored.type,
    timestamp: stored.timestamp.toISOString(),
    userId: stored.userId ?? '',
    venueId: stored.venueId,
    version: stored.version,
    payload: stored.payload,
  } as DomainEvent
}
