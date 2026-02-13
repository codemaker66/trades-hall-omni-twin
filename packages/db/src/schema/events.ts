import { pgTable, uuid, text, integer, jsonb, timestamp, bigserial, uniqueIndex, index } from 'drizzle-orm/pg-core'
import { users } from './users'

/** Append-only event store for venue domain events. */
export const venueEvents = pgTable(
  'venue_events',
  {
    id: bigserial('id', { mode: 'number' }).primaryKey(),
    venueId: uuid('venue_id').notNull(),
    version: integer('version').notNull(),
    type: text('type').notNull(),
    payload: jsonb('payload').notNull(),
    userId: uuid('user_id').references(() => users.id),
    timestamp: timestamp('timestamp', { withTimezone: true }).notNull().defaultNow(),
  },
  (table) => [
    uniqueIndex('venue_events_venue_version_idx').on(table.venueId, table.version),
  ]
)

/** Periodic materialized snapshots for fast state reconstruction. */
export const venueSnapshots = pgTable('venue_snapshots', {
  id: uuid('id').primaryKey().defaultRandom(),
  venueId: uuid('venue_id').notNull(),
  version: integer('version').notNull(),
  state: jsonb('state').notNull(),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  index('venue_snapshots_venue_version_idx').on(table.venueId, table.version),
])
