import { pgTable, uuid, text, integer, real, jsonb, timestamp } from 'drizzle-orm/pg-core'
import { users } from './users'

export const venues = pgTable('venues', {
  id: uuid('id').primaryKey().defaultRandom(),
  name: text('name').notNull(),
  slug: text('slug').notNull().unique(),
  description: text('description'),
  address: text('address'),
  latitude: real('latitude'),
  longitude: real('longitude'),
  capacity: integer('capacity'),
  squareFootage: real('square_footage'),
  venueType: text('venue_type', {
    enum: ['ballroom', 'conference', 'outdoor', 'theater', 'restaurant', 'warehouse', 'other'],
  }),
  ownerId: uuid('owner_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  pricingModel: text('pricing_model', { enum: ['hourly', 'daily', 'flat', 'custom'] }),
  basePrice: real('base_price'),
  amenities: jsonb('amenities').$type<string[]>().default([]),
  images: jsonb('images').$type<string[]>().default([]),
  status: text('status', { enum: ['draft', 'published', 'archived'] }).notNull().default('draft'),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
  archivedAt: timestamp('archived_at', { withTimezone: true }),
})

export const venuePermissions = pgTable('venue_permissions', {
  id: uuid('id').primaryKey().defaultRandom(),
  venueId: uuid('venue_id')
    .notNull()
    .references(() => venues.id, { onDelete: 'cascade' }),
  userId: uuid('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  role: text('role', { enum: ['owner', 'editor', 'viewer', 'commenter'] }).notNull(),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
})
