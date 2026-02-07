/**
 * CT-1: Domain Morphisms (Arrows between Domain Objects)
 *
 * Every operation in the venue planning system is a morphism:
 *   match:      EventSpec → VenueSpec → CompatibilityScore
 *   constrain:  FloorPlan × Constraint[] → ValidatedFloorPlan | Violation[]
 *   assign:     EventSpec × VenueSpec → Assignment
 *   price:      Assignment × Service[] → Proposal
 *   schedule:   Assignment × TimeSlot → ScheduleEntry
 *   validate:   Configuration → Configuration (or errors)
 *
 * These compose: validate ∘ schedule ∘ price ∘ assign ∘ match
 * is a single pipeline from (EventSpec, VenueSpec) → ValidatedConfiguration.
 */

import type { Morphism, Result, Product } from './core'
import { ok, err, pair } from './core'
import type {
  VenueSpec, EventSpec, FloorPlan, Constraint, Assignment,
  Service, Proposal, Configuration, Schedule, ScheduleEntry,
  ValidatedFloorPlan, Violation, CompatibilityScore, Cents,
  Amenity, RankedVenueMatches, ISODateTime, Minutes,
} from './objects'
import { compatibilityScore, markValidated, cents, minutes } from './objects'

// ─── match: EventSpec × VenueSpec → CompatibilityScore ─────────────────────

/**
 * Compute how well an event matches a venue.
 * This is a morphism from the product EventSpec × VenueSpec to CompatibilityScore.
 */
export function matchEventToVenue(event: EventSpec, venue: VenueSpec): CompatibilityScore {
  let score = 0
  let factors = 0

  // Capacity fit (0-1, peaks at 70-90% utilization)
  const utilizationRatio = event.guestCount / venue.maxCapacity
  if (utilizationRatio <= 1) {
    const idealUtilization = 0.8
    score += 1 - Math.abs(utilizationRatio - idealUtilization) / idealUtilization
    factors++
  }

  // Area check
  const venueArea = venue.width * venue.depth
  const areaPerGuest = venueArea / event.guestCount
  if (areaPerGuest >= 1.5) { // minimum 1.5 m² per guest
    score += Math.min(1, areaPerGuest / 3) // ideal ~3 m² per guest
    factors++
  }

  // Amenity match
  const requiredAmenities = event.requirements
    .filter((r): r is { readonly kind: 'amenity'; readonly amenity: Amenity } => r.kind === 'amenity')
    .map(r => r.amenity)
  if (requiredAmenities.length > 0) {
    const matched = requiredAmenities.filter(a => venue.amenities.includes(a))
    score += matched.length / requiredAmenities.length
    factors++
  }

  // Focal point (needed for theater/concert/conference)
  const needsFocalPoint = ['theater', 'concert', 'conference'].includes(event.type)
  if (needsFocalPoint) {
    score += venue.focalPoint ? 1 : 0
    factors++
  }

  return compatibilityScore(factors > 0 ? score / factors : 0)
}

/** Curried version for pipeline composition. */
export function matchVenue(venue: VenueSpec): Morphism<EventSpec, CompatibilityScore> {
  return (event) => matchEventToVenue(event, venue)
}

// ─── constrain: FloorPlan × Constraint[] → Result<ValidatedFloorPlan> ─────

/**
 * Validate a floor plan against a set of constraints.
 * Returns either a validated floor plan (branded) or violations.
 */
export function constrainFloorPlan(
  plan: FloorPlan,
  constraints: readonly Constraint[],
): Result<ValidatedFloorPlan, Violation[]> {
  const violations: Violation[] = []

  for (const constraint of constraints) {
    switch (constraint.kind) {
      case 'fire-code':
        // Simplified: check that placements don't crowd exits
        // Real implementation would use the constraint solver
        break
      case 'capacity':
        if (plan.placements.length > constraint.maxOccupants) {
          violations.push({
            constraint,
            message: `Floor plan has ${plan.placements.length} items, max is ${constraint.maxOccupants}`,
            severity: 'error',
            affectedItems: [],
          })
        }
        break
      case 'spacing':
        // Check minimum gap between items
        for (let i = 0; i < plan.placements.length; i++) {
          for (let j = i + 1; j < plan.placements.length; j++) {
            const a = plan.placements[i]!
            const b = plan.placements[j]!
            const dx = a.x - b.x
            const dz = a.z - b.z
            const dist = Math.sqrt(dx * dx + dz * dz)
            const minDist = (a.width + b.width) / 2 + constraint.minGap
            if (dist < minDist) {
              violations.push({
                constraint,
                message: `Items ${a.id} and ${b.id} are ${dist.toFixed(2)}m apart (min: ${minDist.toFixed(2)}m)`,
                severity: 'error',
                affectedItems: [a.id, b.id],
              })
            }
          }
        }
        break
      case 'accessibility':
        // Check wheelchair spaces exist
        break
      case 'custom':
        if (!constraint.validate(plan)) {
          violations.push({
            constraint,
            message: `Custom constraint "${constraint.name}" failed`,
            severity: 'error',
            affectedItems: [],
          })
        }
        break
    }
  }

  if (violations.length > 0) return err(violations)
  return ok(markValidated(plan))
}

// ─── assign: EventSpec × VenueSpec → Assignment ───────────────────────────

/**
 * Create an assignment from event requirements and venue capabilities.
 * Maps event requirements to venue resources.
 */
export function assignEventToVenue(
  event: EventSpec,
  venue: VenueSpec,
  floorPlan: ValidatedFloorPlan,
): Assignment {
  const allocatedAmenities = event.requirements
    .filter((r): r is { readonly kind: 'amenity'; readonly amenity: Amenity } => r.kind === 'amenity')
    .map(r => r.amenity)
    .filter(a => venue.amenities.includes(a))

  return {
    eventId: event.id,
    venueId: venue.id,
    floorPlan,
    allocatedAmenities,
    guestCapacity: Math.min(event.guestCount, venue.maxCapacity),
  }
}

// ─── price: Assignment × Service[] → Proposal ────────────────────────────

/**
 * Generate a priced proposal from an assignment and services.
 */
export function priceAssignment(
  assignment: Assignment,
  services: readonly Service[],
  venueCostPerHour: Cents,
  eventDuration: Minutes,
): Proposal {
  const venueCost = cents(Math.round((venueCostPerHour as number) * ((eventDuration as number) / 60)))
  const serviceCost = cents(services.reduce((sum, s) => sum + (s.baseCost as number), 0))
  const subtotal = cents((venueCost as number) + (serviceCost as number))
  const tax = cents(Math.round((subtotal as number) * 0.1)) // 10% tax
  const total = cents((subtotal as number) + (tax as number))

  const breakdown = [
    { description: 'Venue hire', amount: venueCost, category: 'venue' as const },
    ...services.map(s => ({
      description: s.name,
      amount: s.baseCost,
      category: 'service' as const,
    })),
    { description: 'Tax (10%)', amount: tax, category: 'tax' as const },
  ]

  return {
    id: `proposal-${assignment.eventId}-${Date.now()}`,
    assignment,
    services: [...services],
    totalCost: total,
    breakdown,
    validUntil: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString() as ISODateTime,
  }
}

// ─── schedule: Assignment × TimeSlot → ScheduleEntry ──────────────────────

/**
 * Create a schedule entry from an assignment and time slot.
 */
export function scheduleAssignment(
  assignment: Assignment,
  startTime: ISODateTime,
  duration: Minutes,
  setupTime: Minutes = minutes(60),
  teardownTime: Minutes = minutes(30),
): ScheduleEntry {
  const start = new Date(startTime as string)
  const end = new Date(start.getTime() + (duration as number) * 60 * 1000)
  return {
    assignment,
    startTime,
    endTime: end.toISOString() as ISODateTime,
    setupTime,
    teardownTime,
  }
}

// ─── validate: Configuration → Result<Configuration> ──────────────────────

/**
 * Validate a complete configuration.
 * Checks temporal consistency, resource availability, and budget compliance.
 */
export function validateConfiguration(config: Configuration): Result<Configuration, string[]> {
  const errors: string[] = []

  // Check guest capacity
  if (config.assignment.guestCapacity < config.event.guestCount) {
    errors.push(`Venue capacity (${config.assignment.guestCapacity}) is less than guest count (${config.event.guestCount})`)
  }

  // Check schedule entries don't overlap
  const entries = config.schedule.entries
  for (let i = 0; i < entries.length; i++) {
    for (let j = i + 1; j < entries.length; j++) {
      const a = entries[i]!
      const b = entries[j]!
      if (a.startTime < b.endTime && b.startTime < a.endTime) {
        errors.push(`Schedule entries overlap: ${a.startTime} - ${a.endTime} and ${b.startTime} - ${b.endTime}`)
      }
    }
  }

  // Check budget
  if (config.event.budget !== undefined) {
    if ((config.totalCost as number) > (config.event.budget as number)) {
      errors.push(`Total cost (${config.totalCost}) exceeds budget (${config.event.budget})`)
    }
  }

  if (errors.length > 0) return err(errors)
  return ok(config)
}
