import { pgTable, uuid, text, integer, real, boolean, jsonb, timestamp, index } from 'drizzle-orm/pg-core'
import { users } from './users'
import { venues } from './venues'

export const floorPlans = pgTable('floor_plans', {
  id: uuid('id').primaryKey().defaultRandom(),
  venueId: uuid('venue_id')
    .notNull()
    .references(() => venues.id, { onDelete: 'cascade' }),
  name: text('name').notNull(),
  version: integer('version').notNull().default(1),
  widthFt: real('width_ft').notNull(),
  heightFt: real('height_ft').notNull(),
  backgroundImageUrl: text('background_image_url'),
  objects: jsonb('objects').notNull().$type<unknown[]>().default([]),
  isTemplate: boolean('is_template').notNull().default(false),
  createdBy: uuid('created_by').references(() => users.id),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  index('floor_plans_venue_id_idx').on(table.venueId),
])

export const furnitureCatalog = pgTable('furniture_catalog', {
  id: uuid('id').primaryKey().defaultRandom(),
  name: text('name').notNull(),
  category: text('category', {
    enum: ['table', 'chair', 'stage', 'decor', 'equipment'],
  }).notNull(),
  modelUrl: text('model_url'),
  thumbnailUrl: text('thumbnail_url'),
  widthFt: real('width_ft').notNull(),
  depthFt: real('depth_ft').notNull(),
  heightFt: real('height_ft').notNull(),
  capacity: integer('capacity'),
  stackable: boolean('stackable').notNull().default(false),
  isCustom: boolean('is_custom').notNull().default(false),
  createdBy: uuid('created_by').references(() => users.id),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
})
