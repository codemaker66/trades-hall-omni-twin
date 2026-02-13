/**
 * PS-11 (partial): Schedule Energy Function for SA/PT Integration
 *
 * Computes a penalty-based energy for event->(room,timeslot) assignments,
 * suitable for continuous-relaxation solvers like Simulated Annealing or
 * Parallel Tempering.
 *
 * The assignment vector is a flat Float64Array where every pair of values
 * encodes one event's room index and timeslot index (continuous, rounded
 * to nearest integer for evaluation):
 *
 *   [event0_room, event0_timeslot, event1_room, event1_timeslot, ...]
 *
 * Energy terms:
 *   H_obj          - negative preference score (lower is better)
 *   H_one_room     - penalty for out-of-range room indices
 *   H_no_conflict  - penalty for double-booking (same room + timeslot)
 *   H_capacity     - penalty for room capacity violations
 *
 *   Total = H_obj + penaltyWeight * (H_one_room + H_no_conflict + H_capacity)
 */

import type { EventSpec, RoomSpec, TimeslotSpec, PRNG } from '../types.js'

// ---------------------------------------------------------------------------
// Energy computation
// ---------------------------------------------------------------------------

/**
 * Compute scheduling energy for a continuous assignment vector.
 *
 * @param assignment     Flat array: [room0, timeslot0, room1, timeslot1, ...]
 * @param events         Event specifications with guest counts and preferences
 * @param rooms          Room specifications with capacities
 * @param timeslots      Available timeslot specifications
 * @param penaltyWeight  Multiplier for constraint violation penalties
 * @returns Total energy (lower is better)
 */
export function computeScheduleEnergy(
  /** Flat assignment: [event0_room, event0_timeslot, event1_room, event1_timeslot, ...] */
  assignment: Float64Array,
  events: EventSpec[],
  rooms: RoomSpec[],
  timeslots: TimeslotSpec[],
  penaltyWeight: number,
): number {
  const nEvents = events.length
  const nRooms = rooms.length
  const nTimeslots = timeslots.length

  let hObj = 0
  let hOneRoom = 0
  let hCapacity = 0

  // Track room-timeslot bookings for conflict detection.
  // Key: "roomIdx:timeslotIdx", value: count of events assigned there.
  const bookingCounts = new Map<string, number>()

  for (let e = 0; e < nEvents; e++) {
    const rawRoom = assignment[e * 2]!
    const rawTimeslot = assignment[e * 2 + 1]!

    // Round to nearest integer index
    const roomIdx = Math.round(rawRoom)
    const timeslotIdx = Math.round(rawTimeslot)

    // --- H_one_room: penalize out-of-range indices ---
    if (roomIdx < 0 || roomIdx >= nRooms) {
      hOneRoom += 1
      // Cannot evaluate preference or capacity for invalid room
      continue
    }
    if (timeslotIdx < 0 || timeslotIdx >= nTimeslots) {
      hOneRoom += 1
      continue
    }

    const event = events[e]!
    const room = rooms[roomIdx]!
    const timeslot = timeslots[timeslotIdx]!

    // --- H_obj: negative preference score ---
    // Look up preference for this (room, timeslot) pair.
    // Preferences are keyed by "roomId:timeslotId" in the event spec.
    const prefKey = `${room.id}:${timeslot.id}`
    const prefScore = event.preferences[prefKey] ?? 0
    hObj -= prefScore

    // --- H_capacity: penalize when guests exceed room capacity ---
    if (event.guests > room.capacity) {
      hCapacity += (event.guests - room.capacity)
    }

    // --- Track bookings for conflict detection ---
    const bookingKey = `${roomIdx}:${timeslotIdx}`
    const currentCount = bookingCounts.get(bookingKey) ?? 0
    bookingCounts.set(bookingKey, currentCount + 1)
  }

  // --- H_no_conflict: penalize double-bookings ---
  let hNoConflict = 0
  for (const count of bookingCounts.values()) {
    if (count > 1) {
      // Each extra event beyond the first is a conflict
      hNoConflict += (count - 1)
    }
  }

  return hObj + penaltyWeight * (hOneRoom + hNoConflict + hCapacity)
}

// ---------------------------------------------------------------------------
// Neighbor generation
// ---------------------------------------------------------------------------

/**
 * Generate a neighbor state by randomly reassigning one event's room or timeslot.
 *
 * Picks a random event index, then with 50/50 probability either:
 *   - Reassigns its room index to a uniform random room
 *   - Reassigns its timeslot index to a uniform random timeslot
 *
 * @param state      Current assignment vector (not mutated)
 * @param nEvents    Number of events
 * @param nRooms     Number of available rooms
 * @param nTimeslots Number of available timeslots
 * @param rng        Seeded PRNG
 * @returns New assignment vector with one perturbation
 */
export function scheduleNeighbor(
  state: Float64Array,
  nEvents: number,
  nRooms: number,
  nTimeslots: number,
  rng: PRNG,
): Float64Array {
  const next = new Float64Array(state)

  // Pick a random event
  const eventIdx = Math.floor(rng.random() * nEvents)
  const baseOffset = eventIdx * 2

  if (rng.random() < 0.5) {
    // Reassign room: pick a uniform random room index
    next[baseOffset] = Math.floor(rng.random() * nRooms)
  } else {
    // Reassign timeslot: pick a uniform random timeslot index
    next[baseOffset + 1] = Math.floor(rng.random() * nTimeslots)
  }

  return next
}
