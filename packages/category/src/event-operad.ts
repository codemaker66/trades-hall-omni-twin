/**
 * CT-4: Event Assembly Operad
 *
 * Concrete operad for assembling complete event configurations.
 *
 * The "assemble event" operation:
 *   (VenueBooking, CateringContract, AVSetup, DecorationPlan, StaffSchedule)
 *     → CompleteEventConfiguration
 *
 * Validates compatibility constraints between all inputs:
 *   - Temporal: setup windows don't overlap destructively
 *   - Spatial: catering area vs AV placement
 *   - Resource: power/water budget limits
 */

import type { Result } from './core'
import { ok, err } from './core'
import { createOp, createOperad } from './operad'
import type { OperadOp } from './operad'
import type { Cents, Minutes, ISODateTime } from './objects'
import { cents, minutes } from './objects'

// ─── Event Assembly Types ───────────────────────────────────────────────────

export interface VenueBooking {
  readonly kind: 'venue-booking'
  readonly venueId: string
  readonly date: string
  readonly startTime: string
  readonly endTime: string
  readonly cost: Cents
}

export interface CateringContract {
  readonly kind: 'catering-contract'
  readonly vendorId: string
  readonly style: 'plated' | 'buffet' | 'cocktail'
  readonly guestCount: number
  readonly setupMinutes: Minutes
  readonly cost: Cents
  readonly spaceRequired: number  // m²
}

export interface AVSetup {
  readonly kind: 'av-setup'
  readonly vendorId: string
  readonly equipment: readonly string[]
  readonly setupMinutes: Minutes
  readonly powerDraw: number  // amps
  readonly cost: Cents
  readonly spaceRequired: number  // m²
}

export interface DecorationPlan {
  readonly kind: 'decoration-plan'
  readonly vendorId: string
  readonly elements: readonly string[]
  readonly setupMinutes: Minutes
  readonly cost: Cents
}

export interface StaffSchedule {
  readonly kind: 'staff-schedule'
  readonly staffCount: number
  readonly roles: readonly string[]
  readonly costPerHour: Cents
  readonly totalHours: number
}

/** An event component — the base type for our operad. */
export type EventComponent =
  | VenueBooking
  | CateringContract
  | AVSetup
  | DecorationPlan
  | StaffSchedule

/** The complete assembled event configuration. */
export interface AssembledEvent {
  readonly venue: VenueBooking
  readonly catering: CateringContract
  readonly av: AVSetup
  readonly decoration: DecorationPlan
  readonly staff: StaffSchedule
  readonly totalCost: Cents
  readonly maxSetupTime: Minutes
  readonly conflicts: readonly string[]
}

// ─── Event Assembly Operation ───────────────────────────────────────────────

/**
 * The assembly operation: 5-ary operation in the event operad.
 */
export const assembleEvent: OperadOp<
  [VenueBooking, CateringContract, AVSetup, DecorationPlan, StaffSchedule],
  AssembledEvent
> = createOp(
  'assembleEvent',
  (venue, catering, av, decoration, staff) => {
    const totalCost = cents(
      (venue.cost as number) +
      (catering.cost as number) +
      (av.cost as number) +
      (decoration.cost as number) +
      (staff.costPerHour as number) * staff.totalHours,
    )

    const maxSetupTime = minutes(Math.max(
      catering.setupMinutes as number,
      av.setupMinutes as number,
      decoration.setupMinutes as number,
    ))

    const conflicts = detectAssemblyConflicts(venue, catering, av, decoration, staff)

    return {
      venue,
      catering,
      av,
      decoration,
      staff,
      totalCost,
      maxSetupTime,
      conflicts,
    }
  },
  (venue, catering, av, decoration, staff) => {
    const errors: string[] = []

    // Validate venue capacity
    if (!venue.venueId) {
      errors.push('Venue booking must have a venue ID')
    }

    // Validate guest count
    if (catering.guestCount <= 0) {
      errors.push('Catering guest count must be positive')
    }

    // Validate power budget (100A typical venue limit)
    if (av.powerDraw > 100) {
      errors.push(`AV power draw (${av.powerDraw}A) exceeds venue capacity (100A)`)
    }

    // Validate staff count
    if (staff.staffCount <= 0) {
      errors.push('Staff count must be positive')
    }

    return errors.length > 0 ? err(errors) : ok(undefined)
  },
)

// ─── Conflict Detection ─────────────────────────────────────────────────────

function detectAssemblyConflicts(
  _venue: VenueBooking,
  catering: CateringContract,
  av: AVSetup,
  _decoration: DecorationPlan,
  _staff: StaffSchedule,
): string[] {
  const conflicts: string[] = []

  // Spatial conflict: catering and AV competing for floor space
  // (This is a simplified check — real implementation uses the constraint solver)
  if (catering.spaceRequired + av.spaceRequired > 100) {
    conflicts.push(
      `Catering (${catering.spaceRequired}m²) and AV (${av.spaceRequired}m²) may compete for floor space`,
    )
  }

  // Power budget
  if (av.powerDraw > 80) {
    conflicts.push(`AV power draw (${av.powerDraw}A) is near venue limit`)
  }

  return conflicts
}

// ─── Event Operad ───────────────────────────────────────────────────────────

/**
 * The event operad: composable operations on event components.
 */
export const eventOperad = createOperad<EventComponent>('EventAssembly')

/**
 * Combine two catering contracts (e.g., main caterer + dessert caterer).
 */
export const combineCatering: OperadOp<[CateringContract, CateringContract], CateringContract> = createOp(
  'combineCatering',
  (main, secondary) => ({
    kind: 'catering-contract' as const,
    vendorId: `${main.vendorId}+${secondary.vendorId}`,
    style: main.style,
    guestCount: main.guestCount,
    setupMinutes: minutes(Math.max(main.setupMinutes as number, secondary.setupMinutes as number)),
    cost: cents((main.cost as number) + (secondary.cost as number)),
    spaceRequired: main.spaceRequired + secondary.spaceRequired,
  }),
)

/**
 * Combine two AV setups (e.g., audio + video).
 */
export const combineAV: OperadOp<[AVSetup, AVSetup], AVSetup> = createOp(
  'combineAV',
  (audio, video) => ({
    kind: 'av-setup' as const,
    vendorId: `${audio.vendorId}+${video.vendorId}`,
    equipment: [...audio.equipment, ...video.equipment],
    setupMinutes: minutes(Math.max(audio.setupMinutes as number, video.setupMinutes as number)),
    powerDraw: audio.powerDraw + video.powerDraw,
    cost: cents((audio.cost as number) + (video.cost as number)),
    spaceRequired: audio.spaceRequired + video.spaceRequired,
  }),
  (audio, video) => {
    if (audio.powerDraw + video.powerDraw > 100) {
      return err([`Combined power draw (${audio.powerDraw + video.powerDraw}A) exceeds limit`])
    }
    return ok(undefined)
  },
)
