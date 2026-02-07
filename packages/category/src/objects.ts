/**
 * CT-1: Domain Objects as Categorical Objects
 *
 * Every type in the venue planning domain is an object in our category.
 * We use branded types (phantom types) to make illegal states unrepresentable.
 *
 * Objects in our category:
 *   VenueSpec, EventSpec, FloorPlan, Constraint, Assignment,
 *   Schedule, Proposal, Configuration, Service
 */

// ─── Branded Type Machinery ────────────────────────────────────────────────

declare const __brand: unique symbol
type Brand<T, B extends string> = T & { readonly [__brand]: B }

// ─── Time Types ────────────────────────────────────────────────────────────

/** ISO-8601 datetime string. */
export type ISODateTime = Brand<string, 'ISODateTime'>

/** Duration in minutes. */
export type Minutes = Brand<number, 'Minutes'>

/** Price in cents (avoid floating-point). */
export type Cents = Brand<number, 'Cents'>

export function isoDateTime(s: string): ISODateTime { return s as ISODateTime }
export function minutes(n: number): Minutes { return n as Minutes }
export function cents(n: number): Cents { return Math.round(n) as Cents }

// ─── VenueSpec ─────────────────────────────────────────────────────────────

/** Specification of a venue (the space itself). */
export interface VenueSpec {
  readonly id: string
  readonly name: string
  readonly width: number   // meters
  readonly depth: number   // meters
  readonly height: number  // meters
  readonly maxCapacity: number
  readonly exits: readonly ExitSpec[]
  readonly obstacles: readonly ObstacleSpec[]
  readonly amenities: readonly Amenity[]
  readonly focalPoint?: { readonly x: number; readonly z: number }
}

export interface ExitSpec {
  readonly x: number
  readonly z: number
  readonly width: number
  readonly facing: number  // radians
}

export interface ObstacleSpec {
  readonly x: number
  readonly z: number
  readonly width: number
  readonly depth: number
}

export type Amenity =
  | 'stage' | 'bar' | 'kitchen' | 'restrooms'
  | 'av-system' | 'lighting' | 'parking' | 'wifi'
  | 'outdoor-space' | 'accessible'

// ─── EventSpec ─────────────────────────────────────────────────────────────

/** Specification of an event (what the client wants). */
export interface EventSpec {
  readonly id: string
  readonly name: string
  readonly type: EventType
  readonly guestCount: number
  readonly startTime: ISODateTime
  readonly duration: Minutes
  readonly requirements: readonly Requirement[]
  readonly budget?: Cents
  readonly style?: EventStyle
}

export type EventType =
  | 'wedding' | 'conference' | 'banquet' | 'cocktail'
  | 'theater' | 'workshop' | 'exhibition' | 'concert'

export type EventStyle = 'formal' | 'casual' | 'corporate' | 'festive'

export type Requirement =
  | { readonly kind: 'amenity'; readonly amenity: Amenity }
  | { readonly kind: 'furniture'; readonly furnitureType: string; readonly count: number }
  | { readonly kind: 'space'; readonly minArea: number }
  | { readonly kind: 'catering'; readonly style: CateringStyle }
  | { readonly kind: 'av'; readonly needs: readonly AVNeed[] }

export type CateringStyle = 'plated' | 'buffet' | 'cocktail' | 'family-style'
export type AVNeed = 'microphone' | 'projector' | 'speakers' | 'lighting-rig' | 'screen'

// ─── FloorPlan ─────────────────────────────────────────────────────────────

/** A spatial arrangement of furniture — the output of layout solving. */
export interface FloorPlan {
  readonly venueId: string
  readonly placements: readonly FloorPlanItem[]
  readonly groupings: readonly TableGrouping[]
}

export interface FloorPlanItem {
  readonly id: string
  readonly type: string
  readonly x: number
  readonly z: number
  readonly rotation: number
  readonly width: number
  readonly depth: number
}

export interface TableGrouping {
  readonly tableId: string
  readonly chairIds: readonly string[]
}

// ─── Constraint ────────────────────────────────────────────────────────────

/** A rule that a floor plan must satisfy. */
export type Constraint =
  | { readonly kind: 'fire-code'; readonly minAisle: number; readonly exitClearance: number }
  | { readonly kind: 'capacity'; readonly maxOccupants: number }
  | { readonly kind: 'spacing'; readonly minGap: number }
  | { readonly kind: 'accessibility'; readonly wheelchairSpaces: number; readonly accessibleRouteWidth: number }
  | { readonly kind: 'custom'; readonly name: string; readonly validate: (plan: FloorPlan) => boolean }

// ─── Violation ─────────────────────────────────────────────────────────────

/** A constraint violation. */
export interface Violation {
  readonly constraint: Constraint
  readonly message: string
  readonly severity: 'error' | 'warning'
  readonly affectedItems: readonly string[]
}

// ─── ValidatedFloorPlan ────────────────────────────────────────────────────

declare const __validated: unique symbol

/** A floor plan that has passed all constraints — branded type. */
export type ValidatedFloorPlan = FloorPlan & { readonly [__validated]: true }

export function markValidated(plan: FloorPlan): ValidatedFloorPlan {
  return plan as ValidatedFloorPlan
}

// ─── Assignment ────────────────────────────────────────────────────────────

/** A mapping from event requirements to venue resources. */
export interface Assignment {
  readonly eventId: string
  readonly venueId: string
  readonly floorPlan: ValidatedFloorPlan
  readonly allocatedAmenities: readonly Amenity[]
  readonly guestCapacity: number
}

// ─── Schedule ──────────────────────────────────────────────────────────────

/** A time-indexed collection of assignments. */
export interface Schedule {
  readonly entries: readonly ScheduleEntry[]
}

export interface ScheduleEntry {
  readonly assignment: Assignment
  readonly startTime: ISODateTime
  readonly endTime: ISODateTime
  readonly setupTime: Minutes
  readonly teardownTime: Minutes
}

// ─── Service ───────────────────────────────────────────────────────────────

/** A vendored service for an event. */
export interface Service {
  readonly id: string
  readonly name: string
  readonly type: ServiceType
  readonly baseCost: Cents
  readonly setupTime: Minutes
  readonly teardownTime: Minutes
  readonly requirements: readonly ServiceRequirement[]
}

export type ServiceType =
  | 'catering' | 'av' | 'decoration' | 'photography'
  | 'music' | 'lighting' | 'staffing' | 'security'

export interface ServiceRequirement {
  readonly kind: 'space' | 'power' | 'water' | 'internet' | 'time'
  readonly description: string
  readonly quantity: number
}

// ─── Proposal ──────────────────────────────────────────────────────────────

/** A priced assignment sent to a client. */
export interface Proposal {
  readonly id: string
  readonly assignment: Assignment
  readonly services: readonly Service[]
  readonly totalCost: Cents
  readonly breakdown: readonly CostLineItem[]
  readonly validUntil: ISODateTime
}

export interface CostLineItem {
  readonly description: string
  readonly amount: Cents
  readonly category: 'venue' | 'service' | 'equipment' | 'staff' | 'tax'
}

// ─── Configuration ─────────────────────────────────────────────────────────

/** A complete event setup — the terminal object of the planning pipeline. */
export interface Configuration {
  readonly event: EventSpec
  readonly venue: VenueSpec
  readonly assignment: Assignment
  readonly services: readonly Service[]
  readonly schedule: Schedule
  readonly totalCost: Cents
  readonly status: ConfigurationStatus
}

export type ConfigurationStatus = 'draft' | 'proposed' | 'confirmed' | 'completed' | 'cancelled'

// ─── Compatibility ─────────────────────────────────────────────────────────

/** Compatibility score: how well an event matches a venue. 0-1. */
export type CompatibilityScore = Brand<number, 'CompatibilityScore'>

export function compatibilityScore(n: number): CompatibilityScore {
  return Math.max(0, Math.min(1, n)) as CompatibilityScore
}

// ─── Ranked Matches ────────────────────────────────────────────────────────

export interface RankedVenueMatch {
  readonly venue: VenueSpec
  readonly score: CompatibilityScore
  readonly reasons: readonly string[]
}

export type RankedVenueMatches = readonly RankedVenueMatch[]
