/**
 * PS-10: MIP Scheduling via HiGHS
 *
 * Exact optimal scheduling for hard constraints using the HiGHS MIP solver
 * (WASM-compiled, MIT license). Runs in the browser.
 *
 * Formulation:
 *   min  -sum w_{e,r,t} * x_{e,r,t}       (maximize preference scores)
 *   s.t. sum_{r,t} x_{e,r,t} = 1           for all events (each assigned once)
 *        sum_e x_{e,r,t} <= 1               for all (r,t) (no conflicts)
 *        guests_e * x_{e,r,t} <= capacity_r  for all (e,r,t) (capacity)
 *        x_{e,r,t} in {0,1}
 *
 * Falls back to greedy solver when HiGHS is not available.
 */

import type {
  EventSpec, RoomSpec, TimeslotSpec,
  ScheduleResult, ScheduleAssignment,
} from './types.js'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Variable {
  eventIdx: number
  roomIdx: number
  timeslotIdx: number
  varName: string
  preference: number
}

// ---------------------------------------------------------------------------
// LP Format Builder
// ---------------------------------------------------------------------------

/**
 * Build a complete LP-format string for the scheduling problem.
 * Compatible with HiGHS-js solver.
 */
export function buildScheduleLP(
  events: EventSpec[],
  rooms: RoomSpec[],
  timeslots: TimeslotSpec[],
): { lp: string; variables: Variable[] } {
  const variables: Variable[] = []

  // Enumerate valid (event, room, timeslot) triples
  for (let e = 0; e < events.length; e++) {
    const event = events[e]!
    for (let r = 0; r < rooms.length; r++) {
      const room = rooms[r]!
      // Skip if room can't hold the event
      if (event.guests > room.capacity) continue

      for (let t = 0; t < timeslots.length; t++) {
        const varName = `x_${e}_${r}_${t}`
        const preference = event.preferences[room.id] ?? 1.0
        variables.push({ eventIdx: e, roomIdx: r, timeslotIdx: t, varName, preference })
      }
    }
  }

  // Build LP format
  const lines: string[] = []

  // Objective: minimize negative preferences (= maximize preferences)
  lines.push('Minimize')
  const objTerms = variables.map((v) => `${(-v.preference).toFixed(4)} ${v.varName}`)
  lines.push(`  obj: ${objTerms.join(' + ').replace(/\+ -/g, '- ')}`)

  lines.push('Subject To')

  // Constraint 1: Each event assigned exactly once
  for (let e = 0; e < events.length; e++) {
    const evVars = variables.filter((v) => v.eventIdx === e)
    if (evVars.length === 0) continue
    const terms = evVars.map((v) => v.varName).join(' + ')
    lines.push(`  assign_e${e}: ${terms} = 1`)
  }

  // Constraint 2: No room conflicts (at most one event per room-timeslot)
  for (let r = 0; r < rooms.length; r++) {
    for (let t = 0; t < timeslots.length; t++) {
      const rtVars = variables.filter((v) => v.roomIdx === r && v.timeslotIdx === t)
      if (rtVars.length <= 1) continue
      const terms = rtVars.map((v) => v.varName).join(' + ')
      lines.push(`  conflict_r${r}_t${t}: ${terms} <= 1`)
    }
  }

  // Binary declarations
  lines.push('Binary')
  lines.push(`  ${variables.map((v) => v.varName).join(' ')}`)
  lines.push('End')

  return { lp: lines.join('\n'), variables }
}

// ---------------------------------------------------------------------------
// Greedy Solver (Fallback)
// ---------------------------------------------------------------------------

/**
 * Greedy scheduler: assign events largest-first to best available room-timeslot.
 * Used when HiGHS is not available or as a warm start for MIP.
 */
export function solveScheduleGreedy(
  events: EventSpec[],
  rooms: RoomSpec[],
  timeslots: TimeslotSpec[],
): ScheduleResult {
  const start = performance.now()

  // Sort events by guests descending (hardest to place first)
  const sortedEventIndices = events
    .map((_, i) => i)
    .sort((a, b) => events[b]!.guests - events[a]!.guests)

  // Track occupied (room, timeslot) pairs
  const occupied = new Set<string>()
  const assignments: ScheduleAssignment[] = []
  let totalPreference = 0

  for (const eIdx of sortedEventIndices) {
    const event = events[eIdx]!
    let bestScore = -Infinity
    let bestRoom = -1
    let bestTimeslot = -1

    for (let r = 0; r < rooms.length; r++) {
      const room = rooms[r]!
      if (event.guests > room.capacity) continue

      for (let t = 0; t < timeslots.length; t++) {
        const key = `${r}_${t}`
        if (occupied.has(key)) continue

        const preference = event.preferences[room.id] ?? 1.0
        if (preference > bestScore) {
          bestScore = preference
          bestRoom = r
          bestTimeslot = t
        }
      }
    }

    if (bestRoom >= 0 && bestTimeslot >= 0) {
      occupied.add(`${bestRoom}_${bestTimeslot}`)
      assignments.push({
        eventId: event.id,
        roomId: rooms[bestRoom]!.id,
        timeslotId: timeslots[bestTimeslot]!.id,
      })
      totalPreference += bestScore
    }
  }

  return {
    assignments,
    objectiveValue: totalPreference,
    feasible: assignments.length === events.length,
    solveDurationMs: performance.now() - start,
  }
}

// ---------------------------------------------------------------------------
// Main MIP Solver
// ---------------------------------------------------------------------------

/**
 * Solve scheduling MIP via HiGHS if available, else fall back to greedy.
 */
export async function solveScheduleMIP(
  events: EventSpec[],
  rooms: RoomSpec[],
  timeslots: TimeslotSpec[],
): Promise<ScheduleResult> {
  if (events.length === 0) {
    return { assignments: [], objectiveValue: 0, feasible: true, solveDurationMs: 0 }
  }

  const start = performance.now()

  // Try HiGHS
  try {
    const highs = await import('highs' as string).then((m) => m.default ?? m)
    const solver = await highs()

    const { lp, variables } = buildScheduleLP(events, rooms, timeslots)
    const result = solver.solve(lp)

    if (result.Status === 'Optimal') {
      return parseHiGHSResult(result, variables, events, rooms, timeslots, start)
    }
  } catch {
    // HiGHS not available — fall back to greedy
  }

  return solveScheduleGreedy(events, rooms, timeslots)
}

function parseHiGHSResult(
  result: Record<string, unknown>,
  variables: Variable[],
  events: EventSpec[],
  rooms: RoomSpec[],
  timeslots: TimeslotSpec[],
  start: number,
): ScheduleResult {
  const columns = result['Columns'] as Record<string, { Primal: number }> | undefined
  const assignments: ScheduleAssignment[] = []
  let totalPreference = 0

  if (columns) {
    for (const v of variables) {
      const col = columns[v.varName]
      if (col && col.Primal > 0.5) {
        assignments.push({
          eventId: events[v.eventIdx]!.id,
          roomId: rooms[v.roomIdx]!.id,
          timeslotId: timeslots[v.timeslotIdx]!.id,
        })
        totalPreference += v.preference
      }
    }
  }

  return {
    assignments,
    objectiveValue: totalPreference,
    feasible: assignments.length === events.length,
    solveDurationMs: performance.now() - start,
  }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/**
 * Validate a schedule: check each event assigned once, no conflicts, capacity met.
 */
export function validateSchedule(
  assignments: ScheduleAssignment[],
  events: EventSpec[],
  rooms: RoomSpec[],
  _timeslots: TimeslotSpec[],
): { valid: boolean; violations: string[] } {
  const violations: string[] = []

  // Check each event assigned exactly once
  const eventAssignments = new Map<string, number>()
  for (const a of assignments) {
    eventAssignments.set(a.eventId, (eventAssignments.get(a.eventId) ?? 0) + 1)
  }
  for (const event of events) {
    const count = eventAssignments.get(event.id) ?? 0
    if (count === 0) violations.push(`Event ${event.id} not assigned`)
    if (count > 1) violations.push(`Event ${event.id} assigned ${count} times`)
  }

  // Check no room-timeslot conflicts
  const rtMap = new Map<string, string[]>()
  for (const a of assignments) {
    const key = `${a.roomId}_${a.timeslotId}`
    const existing = rtMap.get(key) ?? []
    existing.push(a.eventId)
    rtMap.set(key, existing)
  }
  for (const [key, eventIds] of rtMap) {
    if (eventIds.length > 1) {
      violations.push(`Room-timeslot ${key} has ${eventIds.length} events: ${eventIds.join(', ')}`)
    }
  }

  // Check capacity constraints
  const roomMap = new Map(rooms.map((r) => [r.id, r]))
  const eventMap = new Map(events.map((e) => [e.id, e]))
  for (const a of assignments) {
    const room = roomMap.get(a.roomId)
    const event = eventMap.get(a.eventId)
    if (room && event && event.guests > room.capacity) {
      violations.push(`Event ${a.eventId} (${event.guests} guests) exceeds room ${a.roomId} capacity (${room.capacity})`)
    }
  }

  return { valid: violations.length === 0, violations }
}

// ---------------------------------------------------------------------------
// Schedule Energy (for SA/PT integration)
// ---------------------------------------------------------------------------

/**
 * Compute schedule energy for use with SA/PT solvers.
 * State is a flat array: [event0_room, event0_timeslot, event1_room, event1_timeslot, ...]
 */
export function computeScheduleEnergy(
  assignment: Float64Array,
  events: EventSpec[],
  rooms: RoomSpec[],
  timeslots: TimeslotSpec[],
  penaltyWeight: number,
): number {
  const nEvents = events.length
  let objective = 0
  let penalty = 0

  // Track room-timeslot occupancy
  const occupancy = new Map<string, number>()

  for (let e = 0; e < nEvents; e++) {
    const rIdx = Math.round(assignment[e * 2]!)
    const tIdx = Math.round(assignment[e * 2 + 1]!)
    const event = events[e]!

    // Bounds check
    if (rIdx < 0 || rIdx >= rooms.length || tIdx < 0 || tIdx >= timeslots.length) {
      penalty += 1e6
      continue
    }

    const room = rooms[rIdx]!

    // Preference score (negate for minimization)
    const pref = event.preferences[room.id] ?? 1.0
    objective -= pref

    // Capacity violation
    if (event.guests > room.capacity) {
      const excess = event.guests - room.capacity
      penalty += excess * excess
    }

    // Conflict detection
    const key = `${rIdx}_${tIdx}`
    occupancy.set(key, (occupancy.get(key) ?? 0) + 1)
  }

  // Conflict penalties
  for (const count of occupancy.values()) {
    if (count > 1) {
      penalty += (count - 1) * (count - 1)
    }
  }

  return objective + penaltyWeight * penalty
}
