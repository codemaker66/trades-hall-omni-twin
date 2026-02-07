/**
 * Timeline data structure: branch tree with snapshots for fast reconstruction.
 *
 * The timeline is a persistent (immutable) tree where branches share common
 * ancestry. Periodic snapshots (every SNAPSHOT_INTERVAL events) enable O(1)
 * state lookup via binary search + replay from nearest snapshot.
 */

import type { DomainEvent } from '@omni-twin/types'
import type { ProjectedVenueState } from '../projector'
import { applyEvent, emptyVenueState } from '../projector'
import type {
  Timeline,
  Branch,
  TimelineSnapshot,
  EventMarker,
  EventCategory,
} from './types'

// ─── Constants ───────────────────────────────────────────────────────────────

/** Create a snapshot every N events for fast reconstruction */
export const SNAPSHOT_INTERVAL = 50

// ─── Event Classification ────────────────────────────────────────────────────

export function classifyEvent(event: DomainEvent): EventCategory {
  switch (event.type) {
    case 'ItemPlaced':
      return 'add'
    case 'ItemRemoved':
      return 'remove'
    case 'ItemMoved':
    case 'ItemsBatchMoved':
      return 'move'
    case 'ItemRotated':
    case 'ItemsBatchRotated':
      return 'rotate'
    case 'ItemScaled':
      return 'scale'
    case 'GroupCreated':
    case 'GroupDissolved':
    case 'ItemsGrouped':
    case 'ItemsUngrouped':
      return 'group'
    default:
      return 'other'
  }
}

// ─── Timeline Creation ───────────────────────────────────────────────────────

let _branchCounter = 0

function nextBranchId(): string {
  return `branch_${++_branchCounter}`
}

/** Reset branch counter (for tests) */
export function resetBranchCounter(): void {
  _branchCounter = 0
}

/** Create a new timeline with a root branch from a venue ID. */
export function createTimeline(venueId: string): Timeline {
  const rootId = nextBranchId()
  const root: Branch = {
    id: rootId,
    name: 'main',
    parentId: null,
    forkIndex: 0,
    events: [],
    snapshots: [{
      atIndex: -1,
      version: 0,
      state: emptyVenueState(venueId),
    }],
  }

  return {
    rootBranchId: rootId,
    branches: new Map([[rootId, root]]),
    activeBranchId: rootId,
    cursor: -1,
  }
}

// ─── Snapshot Management ─────────────────────────────────────────────────────

/**
 * Binary search for the nearest snapshot at or before the given event index.
 * Returns the snapshot and its index within the snapshots array.
 */
export function findNearestSnapshot(
  snapshots: readonly TimelineSnapshot[],
  targetIndex: number,
): TimelineSnapshot {
  let lo = 0
  let hi = snapshots.length - 1
  let best = snapshots[0]!

  while (lo <= hi) {
    const mid = (lo + hi) >>> 1
    const snap = snapshots[mid]!
    if (snap.atIndex <= targetIndex) {
      best = snap
      lo = mid + 1
    } else {
      hi = mid - 1
    }
  }

  return best
}

/**
 * Build snapshots for a branch up to the given state.
 * Called after appending events to keep snapshot coverage current.
 */
function buildSnapshotsIfNeeded(
  branch: Branch,
  ancestorState: ProjectedVenueState,
): TimelineSnapshot[] {
  const snapshots = [...branch.snapshots]
  const lastSnap = snapshots[snapshots.length - 1]!
  const lastSnapIndex = lastSnap.atIndex

  // Determine how many new snapshots we need
  const eventCount = branch.events.length
  const nextSnapAt = lastSnapIndex + SNAPSHOT_INTERVAL - (lastSnapIndex % SNAPSHOT_INTERVAL || SNAPSHOT_INTERVAL)

  if (eventCount - 1 < nextSnapAt) return snapshots

  // Reconstruct state from last snapshot to build new ones
  let state = lastSnap.atIndex === -1
    ? ancestorState
    : lastSnap.state
  const startReplay = lastSnap.atIndex + 1

  for (let i = startReplay; i < eventCount; i++) {
    state = applyEvent(state, branch.events[i]!)
    if ((i + 1) % SNAPSHOT_INTERVAL === 0) {
      snapshots.push({
        atIndex: i,
        version: state.version,
        state,
      })
    }
  }

  return snapshots
}

// ─── State Reconstruction ────────────────────────────────────────────────────

/**
 * Collect the full event chain for a branch (ancestor events + own events).
 * Returns the events array and the base state at the fork point.
 */
function getAncestorChain(
  timeline: Timeline,
  branchId: string,
): { events: DomainEvent[]; baseState: ProjectedVenueState } {
  const branch = timeline.branches.get(branchId)
  if (!branch) throw new Error(`Branch not found: ${branchId}`)

  if (branch.parentId === null) {
    // Root branch — base state is the initial snapshot
    return {
      events: branch.events,
      baseState: branch.snapshots[0]!.state,
    }
  }

  // Recursively get parent chain
  const parent = getAncestorChain(timeline, branch.parentId)
  // Only include parent events up to fork point
  const parentEvents = parent.events.slice(0, branch.forkIndex)

  return {
    events: [...parentEvents, ...branch.events],
    baseState: parent.baseState,
  }
}

/**
 * Reconstruct the state at a specific position within a branch.
 * Uses snapshots for O(SNAPSHOT_INTERVAL) replay rather than full O(N).
 */
export function reconstructAt(
  timeline: Timeline,
  branchId: string,
  eventIndex: number,
): ProjectedVenueState {
  const branch = timeline.branches.get(branchId)
  if (!branch) throw new Error(`Branch not found: ${branchId}`)

  // For the root branch, we can use local snapshots directly
  if (branch.parentId === null) {
    const snapshot = findNearestSnapshot(branch.snapshots, eventIndex)
    let state = snapshot.state
    const startReplay = snapshot.atIndex + 1
    for (let i = startReplay; i <= eventIndex && i < branch.events.length; i++) {
      state = applyEvent(state, branch.events[i]!)
    }
    return state
  }

  // For child branches, we need the parent state at fork point + branch events
  // First, reconstruct the parent state at the fork point
  const parentState = reconstructAt(timeline, branch.parentId, branch.forkIndex - 1)

  // Then use local snapshots to find nearest and replay
  const snapshot = findNearestSnapshot(branch.snapshots, eventIndex)
  let state: ProjectedVenueState
  let startReplay: number

  if (snapshot.atIndex >= 0) {
    state = snapshot.state
    startReplay = snapshot.atIndex + 1
  } else {
    state = parentState
    startReplay = 0
  }

  for (let i = startReplay; i <= eventIndex && i < branch.events.length; i++) {
    state = applyEvent(state, branch.events[i]!)
  }

  return state
}

/** Reconstruct the state at the timeline cursor. */
export function reconstructCurrent(timeline: Timeline): ProjectedVenueState {
  return reconstructAt(timeline, timeline.activeBranchId, timeline.cursor)
}

// ─── Timeline Mutations ──────────────────────────────────────────────────────

/** Append an event to the active branch. Returns new timeline. */
export function appendEvent(timeline: Timeline, event: DomainEvent): Timeline {
  const branch = timeline.branches.get(timeline.activeBranchId)
  if (!branch) throw new Error(`Active branch not found: ${timeline.activeBranchId}`)

  // If cursor is not at the end, truncate future events
  const events = timeline.cursor < branch.events.length - 1
    ? [...branch.events.slice(0, timeline.cursor + 1), event]
    : [...branch.events, event]

  // Truncate snapshots that are beyond the new event range
  const snapshots = branch.snapshots.filter(s => s.atIndex < events.length - 1 || s.atIndex === -1)

  // Build ancestor state for snapshot building
  let ancestorState: ProjectedVenueState
  if (branch.parentId === null) {
    ancestorState = branch.snapshots[0]!.state
  } else {
    ancestorState = reconstructAt(timeline, branch.parentId, branch.forkIndex - 1)
  }

  const updatedBranch: Branch = {
    ...branch,
    events,
    snapshots: buildSnapshotsIfNeeded({ ...branch, events, snapshots }, ancestorState),
  }

  const branches = new Map(timeline.branches)
  branches.set(updatedBranch.id, updatedBranch)

  return {
    ...timeline,
    branches,
    cursor: events.length - 1,
  }
}

/** Append multiple events at once. */
export function appendEvents(timeline: Timeline, events: DomainEvent[]): Timeline {
  let t = timeline
  for (const event of events) {
    t = appendEvent(t, event)
  }
  return t
}

/** Move the cursor to a specific event index. */
export function seekTo(timeline: Timeline, eventIndex: number): Timeline {
  const branch = timeline.branches.get(timeline.activeBranchId)
  if (!branch) throw new Error(`Active branch not found: ${timeline.activeBranchId}`)

  const clamped = Math.max(-1, Math.min(eventIndex, branch.events.length - 1))
  return { ...timeline, cursor: clamped }
}

/** Create a new branch at the current cursor position. */
export function createBranch(
  timeline: Timeline,
  name: string,
): Timeline {
  const currentBranch = timeline.branches.get(timeline.activeBranchId)
  if (!currentBranch) throw new Error(`Active branch not found: ${timeline.activeBranchId}`)

  const branchId = nextBranchId()

  // The fork point is the current cursor position + 1 (events up to cursor are shared)
  const forkIndex = timeline.cursor + 1

  // Get the state at the fork point for the initial snapshot
  const forkState = reconstructCurrent(timeline)

  const newBranch: Branch = {
    id: branchId,
    name,
    parentId: timeline.activeBranchId,
    forkIndex,
    events: [],
    snapshots: [{
      atIndex: -1,
      version: forkState.version,
      state: forkState,
    }],
  }

  const branches = new Map(timeline.branches)
  branches.set(branchId, newBranch)

  return {
    ...timeline,
    branches,
    activeBranchId: branchId,
    cursor: -1,
  }
}

/** Switch to a different branch. */
export function switchBranch(timeline: Timeline, branchId: string): Timeline {
  const branch = timeline.branches.get(branchId)
  if (!branch) throw new Error(`Branch not found: ${branchId}`)

  return {
    ...timeline,
    activeBranchId: branchId,
    cursor: branch.events.length - 1,
  }
}

// ─── Event Markers ───────────────────────────────────────────────────────────

/** Get event markers for the active branch. */
export function getEventMarkers(timeline: Timeline): EventMarker[] {
  const branch = timeline.branches.get(timeline.activeBranchId)
  if (!branch) return []

  return branch.events.map((event, index) => ({
    index,
    category: classifyEvent(event),
    event,
  }))
}

/** Get the total event count for the active branch (including ancestors). */
export function getActiveBranchLength(timeline: Timeline): number {
  const branch = timeline.branches.get(timeline.activeBranchId)
  if (!branch) return 0
  return branch.events.length
}

/** List all branches. */
export function listBranches(timeline: Timeline): Branch[] {
  return Array.from(timeline.branches.values())
}
