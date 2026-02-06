import { pgTable, uuid, text, integer, real, jsonb, timestamp } from 'drizzle-orm/pg-core'
import { users } from './users'
import { venues } from './venues'
import { floorPlans } from './floor-plans'

/**
 * An "occasion" is a booking event at a venue (wedding, conference, gala, etc.).
 * Named `occasions` to avoid confusion with `venueEvents` (event sourcing table).
 */
export const occasions = pgTable('occasions', {
  id: uuid('id').primaryKey().defaultRandom(),
  venueId: uuid('venue_id')
    .notNull()
    .references(() => venues.id, { onDelete: 'cascade' }),
  organizerId: uuid('organizer_id')
    .notNull()
    .references(() => users.id),
  name: text('name').notNull(),
  type: text('type', {
    enum: ['wedding', 'corporate', 'social', 'conference', 'gala', 'other'],
  }).notNull(),
  dateStart: timestamp('date_start', { withTimezone: true }).notNull(),
  dateEnd: timestamp('date_end', { withTimezone: true }).notNull(),
  setupTime: timestamp('setup_time', { withTimezone: true }),
  teardownTime: timestamp('teardown_time', { withTimezone: true }),
  guestCount: integer('guest_count').notNull(),
  status: text('status', {
    enum: ['inquiry', 'confirmed', 'completed', 'cancelled'],
  }).notNull().default('inquiry'),
  floorPlanId: uuid('floor_plan_id').references(() => floorPlans.id),
  budget: real('budget'),
  notes: text('notes'),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
})

export const proposals = pgTable('proposals', {
  id: uuid('id').primaryKey().defaultRandom(),
  occasionId: uuid('occasion_id')
    .notNull()
    .references(() => occasions.id, { onDelete: 'cascade' }),
  venueId: uuid('venue_id')
    .notNull()
    .references(() => venues.id, { onDelete: 'cascade' }),
  status: text('status', {
    enum: ['draft', 'sent', 'viewed', 'accepted', 'declined'],
  }).notNull().default('draft'),
  pricingBreakdown: jsonb('pricing_breakdown'),
  customMessage: text('custom_message'),
  validUntil: timestamp('valid_until', { withTimezone: true }),
  sentAt: timestamp('sent_at', { withTimezone: true }),
  viewedAt: timestamp('viewed_at', { withTimezone: true }),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
})

export const bookings = pgTable('bookings', {
  id: uuid('id').primaryKey().defaultRandom(),
  occasionId: uuid('occasion_id')
    .notNull()
    .references(() => occasions.id, { onDelete: 'cascade' }),
  venueId: uuid('venue_id')
    .notNull()
    .references(() => venues.id, { onDelete: 'cascade' }),
  proposalId: uuid('proposal_id').references(() => proposals.id),
  status: text('status', {
    enum: ['pending', 'confirmed', 'deposit_paid', 'completed', 'cancelled'],
  }).notNull().default('pending'),
  totalAmount: real('total_amount'),
  depositAmount: real('deposit_amount'),
  paymentStatus: text('payment_status', {
    enum: ['unpaid', 'deposit', 'partial', 'paid', 'refunded'],
  }).notNull().default('unpaid'),
  contractUrl: text('contract_url'),
  signedAt: timestamp('signed_at', { withTimezone: true }),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
})
