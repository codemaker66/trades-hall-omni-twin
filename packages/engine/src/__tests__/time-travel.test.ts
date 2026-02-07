import { describe, it, expect, beforeEach } from 'vitest'
import fc from 'fast-check'

import type { DomainEvent } from '@omni-twin/types'
import type { ProjectedVenueState } from '../projector'
import { emptyVenueState, applyEvent } from '../projector'
import {
  SNAPSHOT_INTERVAL,
  classifyEvent,
  resetBranchCounter,
  createTimeline,
  findNearestSnapshot,
  reconstructAt,
  reconstructCurrent,
  appendEvent,
  appendEvents,
  seekTo,
  createBranch,
  switchBranch,
  getEventMarkers,
  getActiveBranchLength,
  listBranches,
} from '../time-travel/timeline'
import { computeDiff, changedOnly, filterByStatus } from '../time-travel/diff'
import { threeWayMerge, resolveConflict } from '../time-travel/merge'
import type { TimelineSnapshot } from '../time-travel/types'

// ─── Test Helpers ────────────────────────────────────────────────────────────

const VENUE_ID = 'venue-1'
const USER_ID = 'user-1'
let eventVersion = 0

beforeEach(() => {
  resetBranchCounter()
  eventVersion = 0
})

function makeEvent(
  type: DomainEvent['type'],
  payload: Record<string, unknown>,
): DomainEvent {
  eventVersion++
  return {
    id: `evt-${eventVersion}`,
    type,
    timestamp: new Date(1000 + eventVersion * 100).toISOString(),
    userId: USER_ID,
    venueId: VENUE_ID,
    version: eventVersion,
    payload,
  } as DomainEvent
}

function placeEvent(itemId: string, x = 0, y = 0, z = 0): DomainEvent {
  return makeEvent('ItemPlaced', {
    itemId,
    furnitureType: 'chair',
    position: [x, y, z],
    rotation: [0, 0, 0],
  })
}

function moveEvent(itemId: string, x: number, y: number, z: number): DomainEvent {
  return makeEvent('ItemMoved', { itemId, position: [x, y, z] })
}

function removeEvent(itemId: string): DomainEvent {
  return makeEvent('ItemRemoved', { itemId })
}

function rotateEvent(itemId: string, rx: number, ry: number, rz: number): DomainEvent {
  return makeEvent('ItemRotated', { itemId, rotation: [rx, ry, rz] })
}

function scaleEvent(itemId: string, sx: number, sy: number, sz: number): DomainEvent {
  return makeEvent('ItemScaled', { itemId, scale: [sx, sy, sz] })
}

// ─── Event Classification ────────────────────────────────────────────────────

describe('classifyEvent', () => {
  it('classifies item events correctly', () => {
    expect(classifyEvent(placeEvent('a'))).toBe('add')
    expect(classifyEvent(removeEvent('a'))).toBe('remove')
    expect(classifyEvent(moveEvent('a', 1, 2, 3))).toBe('move')
    expect(classifyEvent(rotateEvent('a', 0, 1, 0))).toBe('rotate')
    expect(classifyEvent(scaleEvent('a', 2, 2, 2))).toBe('scale')
  })

  it('classifies group events', () => {
    const groupEvt = makeEvent('GroupCreated', { groupId: 'g1', itemIds: ['a'] })
    expect(classifyEvent(groupEvt)).toBe('group')
  })

  it('classifies other events', () => {
    const venueEvt = makeEvent('VenueCreated', { name: 'Test' })
    expect(classifyEvent(venueEvt)).toBe('other')
  })
})

// ─── Timeline ────────────────────────────────────────────────────────────────

describe('Timeline', () => {
  it('creates empty timeline', () => {
    const tl = createTimeline(VENUE_ID)
    expect(tl.rootBranchId).toBeDefined()
    expect(tl.activeBranchId).toBe(tl.rootBranchId)
    expect(tl.cursor).toBe(-1)
    expect(tl.branches.size).toBe(1)
  })

  it('appends events and advances cursor', () => {
    let tl = createTimeline(VENUE_ID)
    tl = appendEvent(tl, placeEvent('a'))
    expect(tl.cursor).toBe(0)
    expect(getActiveBranchLength(tl)).toBe(1)

    tl = appendEvent(tl, moveEvent('a', 5, 0, 5))
    expect(tl.cursor).toBe(1)
    expect(getActiveBranchLength(tl)).toBe(2)
  })

  it('reconstructs state at cursor', () => {
    let tl = createTimeline(VENUE_ID)
    tl = appendEvent(tl, placeEvent('a', 1, 0, 1))
    tl = appendEvent(tl, moveEvent('a', 5, 0, 5))

    const state = reconstructCurrent(tl)
    expect(state.items.size).toBe(1)
    expect(state.items.get('a')?.position).toEqual([5, 0, 5])
  })

  it('seeks to past state', () => {
    let tl = createTimeline(VENUE_ID)
    tl = appendEvent(tl, placeEvent('a', 1, 0, 1))
    tl = appendEvent(tl, moveEvent('a', 5, 0, 5))
    tl = appendEvent(tl, moveEvent('a', 10, 0, 10))

    tl = seekTo(tl, 0)
    const state = reconstructCurrent(tl)
    expect(state.items.get('a')?.position).toEqual([1, 0, 1])
  })

  it('seeks to -1 gives empty state', () => {
    let tl = createTimeline(VENUE_ID)
    tl = appendEvent(tl, placeEvent('a'))

    tl = seekTo(tl, -1)
    const state = reconstructCurrent(tl)
    expect(state.items.size).toBe(0)
  })

  it('clamps seek to valid range', () => {
    let tl = createTimeline(VENUE_ID)
    tl = appendEvent(tl, placeEvent('a'))

    tl = seekTo(tl, 100)
    expect(tl.cursor).toBe(0) // clamped to max

    tl = seekTo(tl, -10)
    expect(tl.cursor).toBe(-1) // clamped to min
  })

  it('appendEvents adds multiple at once', () => {
    let tl = createTimeline(VENUE_ID)
    tl = appendEvents(tl, [
      placeEvent('a', 1, 0, 1),
      placeEvent('b', 2, 0, 2),
      placeEvent('c', 3, 0, 3),
    ])
    expect(tl.cursor).toBe(2)
    expect(getActiveBranchLength(tl)).toBe(3)

    const state = reconstructCurrent(tl)
    expect(state.items.size).toBe(3)
  })

  it('truncates future when appending from past cursor', () => {
    let tl = createTimeline(VENUE_ID)
    tl = appendEvents(tl, [
      placeEvent('a'),
      placeEvent('b'),
      placeEvent('c'),
    ])

    // Seek back and append new event
    tl = seekTo(tl, 0)
    tl = appendEvent(tl, placeEvent('d'))

    // Should have 2 events: place 'a' and place 'd' (b, c truncated)
    expect(getActiveBranchLength(tl)).toBe(2)
    const state = reconstructCurrent(tl)
    expect(state.items.has('a')).toBe(true)
    expect(state.items.has('d')).toBe(true)
    expect(state.items.has('b')).toBe(false)
    expect(state.items.has('c')).toBe(false)
  })
})

// ─── Snapshots ───────────────────────────────────────────────────────────────

describe('Snapshots', () => {
  it('creates snapshots at SNAPSHOT_INTERVAL', () => {
    let tl = createTimeline(VENUE_ID)
    const events: DomainEvent[] = []
    for (let i = 0; i < SNAPSHOT_INTERVAL + 5; i++) {
      events.push(placeEvent(`item-${i}`, i, 0, 0))
    }
    tl = appendEvents(tl, events)

    const branch = tl.branches.get(tl.activeBranchId)!
    // Should have: initial (-1) + one at index SNAPSHOT_INTERVAL - 1
    expect(branch.snapshots.length).toBeGreaterThanOrEqual(2)
    const snap = branch.snapshots[1]!
    expect(snap.atIndex).toBe(SNAPSHOT_INTERVAL - 1)
  })

  it('findNearestSnapshot returns closest', () => {
    const snaps: TimelineSnapshot[] = [
      { atIndex: -1, version: 0, state: emptyVenueState(VENUE_ID) },
      { atIndex: 49, version: 50, state: emptyVenueState(VENUE_ID) },
      { atIndex: 99, version: 100, state: emptyVenueState(VENUE_ID) },
    ]

    expect(findNearestSnapshot(snaps, 30).atIndex).toBe(-1)
    expect(findNearestSnapshot(snaps, 49).atIndex).toBe(49)
    expect(findNearestSnapshot(snaps, 75).atIndex).toBe(49)
    expect(findNearestSnapshot(snaps, 99).atIndex).toBe(99)
    expect(findNearestSnapshot(snaps, 120).atIndex).toBe(99)
  })

  it('reconstruction via snapshot matches full replay', () => {
    let tl = createTimeline(VENUE_ID)
    const events: DomainEvent[] = []
    for (let i = 0; i < SNAPSHOT_INTERVAL * 2 + 10; i++) {
      events.push(placeEvent(`item-${i}`, i * 0.5, 0, i * 0.3))
    }
    tl = appendEvents(tl, events)

    // Reconstruct at various points and compare with full replay
    const fullState = events.reduce(
      (s, e) => applyEvent(s, e),
      emptyVenueState(VENUE_ID),
    )
    const tlState = reconstructCurrent(tl)
    expect(tlState.items.size).toBe(fullState.items.size)

    // Check at a point mid-stream
    const midIndex = SNAPSHOT_INTERVAL + 15
    const midStateFullReplay = events.slice(0, midIndex + 1).reduce(
      (s, e) => applyEvent(s, e),
      emptyVenueState(VENUE_ID),
    )
    const midStateTl = reconstructAt(tl, tl.activeBranchId, midIndex)
    expect(midStateTl.items.size).toBe(midStateFullReplay.items.size)
  })
})

// ─── Branches ────────────────────────────────────────────────────────────────

describe('Branches', () => {
  it('creates a branch at current cursor', () => {
    let tl = createTimeline(VENUE_ID)
    tl = appendEvents(tl, [
      placeEvent('a', 1, 0, 1),
      placeEvent('b', 2, 0, 2),
    ])

    const rootId = tl.activeBranchId
    tl = createBranch(tl, 'experiment')

    expect(tl.activeBranchId).not.toBe(rootId)
    expect(listBranches(tl).length).toBe(2)
    expect(tl.cursor).toBe(-1) // fresh branch

    // The branch fork state should include both items
    const state = reconstructCurrent(tl)
    expect(state.items.size).toBe(2)
  })

  it('branches diverge independently', () => {
    let tl = createTimeline(VENUE_ID)
    tl = appendEvents(tl, [
      placeEvent('a', 1, 0, 1),
    ])

    const rootId = tl.activeBranchId
    tl = createBranch(tl, 'alt')
    const altId = tl.activeBranchId

    // Add to alt branch
    tl = appendEvent(tl, placeEvent('b', 5, 0, 5))

    // Switch back to root
    tl = switchBranch(tl, rootId)
    tl = appendEvent(tl, placeEvent('c', 10, 0, 10))

    // Root has a + c
    let state = reconstructCurrent(tl)
    expect(state.items.has('a')).toBe(true)
    expect(state.items.has('c')).toBe(true)
    expect(state.items.has('b')).toBe(false)

    // Alt has a + b
    tl = switchBranch(tl, altId)
    state = reconstructCurrent(tl)
    expect(state.items.has('a')).toBe(true)
    expect(state.items.has('b')).toBe(true)
    expect(state.items.has('c')).toBe(false)
  })

  it('switchBranch sets cursor to end', () => {
    let tl = createTimeline(VENUE_ID)
    tl = appendEvents(tl, [placeEvent('a'), placeEvent('b')])

    const rootId = tl.activeBranchId
    tl = createBranch(tl, 'alt')
    tl = appendEvent(tl, placeEvent('c'))

    tl = switchBranch(tl, rootId)
    expect(tl.cursor).toBe(1) // root has 2 events → cursor at index 1
  })

  it('throws on unknown branch', () => {
    const tl = createTimeline(VENUE_ID)
    expect(() => switchBranch(tl, 'nonexistent')).toThrow()
  })
})

// ─── Event Markers ───────────────────────────────────────────────────────────

describe('Event Markers', () => {
  it('returns markers for active branch', () => {
    let tl = createTimeline(VENUE_ID)
    tl = appendEvents(tl, [
      placeEvent('a'),
      moveEvent('a', 5, 0, 5),
      removeEvent('a'),
    ])

    const markers = getEventMarkers(tl)
    expect(markers.length).toBe(3)
    expect(markers[0]!.category).toBe('add')
    expect(markers[1]!.category).toBe('move')
    expect(markers[2]!.category).toBe('remove')
  })
})

// ─── Visual Diff ─────────────────────────────────────────────────────────────

describe('Visual Diff', () => {
  function stateWith(
    items: Array<{ id: string; x: number; z: number; type?: string; ry?: number }>,
  ): ProjectedVenueState {
    const state = emptyVenueState(VENUE_ID)
    for (const item of items) {
      state.items.set(item.id, {
        id: item.id,
        furnitureType: (item.type ?? 'chair') as 'chair',
        position: [item.x, 0, item.z],
        rotation: [0, item.ry ?? 0, 0],
        scale: [1, 1, 1],
      })
    }
    return state
  }

  it('detects added items', () => {
    const before = stateWith([])
    const after = stateWith([{ id: 'a', x: 1, z: 1 }])

    const diff = computeDiff(before, after)
    expect(diff.added).toBe(1)
    expect(diff.diffs[0]!.status).toBe('added')
    expect(diff.diffs[0]!.after?.id).toBe('a')
  })

  it('detects removed items', () => {
    const before = stateWith([{ id: 'a', x: 1, z: 1 }])
    const after = stateWith([])

    const diff = computeDiff(before, after)
    expect(diff.removed).toBe(1)
    expect(diff.diffs[0]!.status).toBe('removed')
    expect(diff.diffs[0]!.before?.id).toBe('a')
  })

  it('detects moved items with displacement', () => {
    const before = stateWith([{ id: 'a', x: 1, z: 1 }])
    const after = stateWith([{ id: 'a', x: 5, z: 3 }])

    const diff = computeDiff(before, after)
    expect(diff.moved).toBe(1)
    const d = diff.diffs[0]!
    expect(d.status).toBe('moved')
    expect(d.displacement).toEqual([4, 0, 2])
  })

  it('detects modified items (non-position change)', () => {
    const before = stateWith([{ id: 'a', x: 1, z: 1, ry: 0 }])
    const after = stateWith([{ id: 'a', x: 1, z: 1, ry: Math.PI / 2 }])

    const diff = computeDiff(before, after)
    expect(diff.modified).toBe(1)
    expect(diff.diffs[0]!.status).toBe('modified')
  })

  it('detects unchanged items', () => {
    const before = stateWith([{ id: 'a', x: 1, z: 1 }])
    const after = stateWith([{ id: 'a', x: 1, z: 1 }])

    const diff = computeDiff(before, after)
    expect(diff.unchanged).toBe(1)
    expect(diff.diffs[0]!.status).toBe('unchanged')
  })

  it('handles mixed changes', () => {
    const before = stateWith([
      { id: 'a', x: 1, z: 1 },
      { id: 'b', x: 2, z: 2 },
      { id: 'c', x: 3, z: 3 },
    ])
    const after = stateWith([
      { id: 'a', x: 1, z: 1 },  // unchanged
      { id: 'b', x: 5, z: 5 },  // moved
      { id: 'd', x: 4, z: 4 },  // added (c removed)
    ])

    const diff = computeDiff(before, after)
    expect(diff.unchanged).toBe(1)
    expect(diff.moved).toBe(1)
    expect(diff.added).toBe(1)
    expect(diff.removed).toBe(1)
  })

  it('changedOnly filters unchanged', () => {
    const before = stateWith([{ id: 'a', x: 1, z: 1 }, { id: 'b', x: 2, z: 2 }])
    const after = stateWith([{ id: 'a', x: 1, z: 1 }, { id: 'b', x: 5, z: 5 }])

    const diff = computeDiff(before, after)
    const changed = changedOnly(diff)
    expect(changed.length).toBe(1)
    expect(changed[0]!.itemId).toBe('b')
  })

  it('filterByStatus works', () => {
    const before = stateWith([{ id: 'a', x: 1, z: 1 }])
    const after = stateWith([{ id: 'a', x: 5, z: 5 }, { id: 'b', x: 2, z: 2 }])

    const diff = computeDiff(before, after)
    expect(filterByStatus(diff, 'moved').length).toBe(1)
    expect(filterByStatus(diff, 'added').length).toBe(1)
  })
})

// ─── Three-Way Merge ─────────────────────────────────────────────────────────

describe('Three-Way Merge', () => {
  function stateWith(
    items: Array<{ id: string; x: number; z: number; type?: string }>,
  ): ProjectedVenueState {
    const s = emptyVenueState(VENUE_ID)
    for (const item of items) {
      s.items.set(item.id, {
        id: item.id,
        furnitureType: (item.type ?? 'chair') as 'chair',
        position: [item.x, 0, item.z],
        rotation: [0, 0, 0],
        scale: [1, 1, 1],
      })
    }
    return s
  }

  it('auto-merges non-conflicting changes', () => {
    const base = stateWith([{ id: 'a', x: 1, z: 1 }])
    const branchA = stateWith([
      { id: 'a', x: 1, z: 1 },
      { id: 'b', x: 5, z: 5 },  // A added b
    ])
    const branchB = stateWith([
      { id: 'a', x: 1, z: 1 },
      { id: 'c', x: 10, z: 10 }, // B added c
    ])

    const result = threeWayMerge(base, branchA, branchB)
    expect(result.conflicts.length).toBe(0)
    expect(result.mergedState.items.size).toBe(3)
    expect(result.mergedState.items.has('a')).toBe(true)
    expect(result.mergedState.items.has('b')).toBe(true)
    expect(result.mergedState.items.has('c')).toBe(true)
  })

  it('auto-merges concurrent moves with displacement sum', () => {
    const base = stateWith([{ id: 'a', x: 0, z: 0 }])
    const branchA = stateWith([{ id: 'a', x: 3, z: 0 }])  // A moved +3 on x
    const branchB = stateWith([{ id: 'a', x: 0, z: 4 }])  // B moved +4 on z

    const result = threeWayMerge(base, branchA, branchB)
    expect(result.conflicts.length).toBe(0)
    expect(result.autoMerged).toContain('a')

    const merged = result.mergedState.items.get('a')!
    expect(merged.position[0]).toBe(3) // base + A displacement + B displacement
    expect(merged.position[2]).toBe(4)
  })

  it('one branch changes, other unchanged → use changed version', () => {
    const base = stateWith([{ id: 'a', x: 1, z: 1 }])
    const branchA = stateWith([{ id: 'a', x: 5, z: 5 }])
    const branchB = stateWith([{ id: 'a', x: 1, z: 1 }])  // unchanged

    const result = threeWayMerge(base, branchA, branchB)
    expect(result.conflicts.length).toBe(0)
    expect(result.mergedState.items.get('a')?.position).toEqual([5, 0, 5])
  })

  it('one branch removes, other unchanged → remove', () => {
    const base = stateWith([{ id: 'a', x: 1, z: 1 }])
    const branchA = stateWith([])  // A removed a
    const branchB = stateWith([{ id: 'a', x: 1, z: 1 }])

    const result = threeWayMerge(base, branchA, branchB)
    expect(result.conflicts.length).toBe(0)
    expect(result.mergedState.items.has('a')).toBe(false)
  })

  it('detects move-remove conflict', () => {
    const base = stateWith([{ id: 'a', x: 1, z: 1 }])
    const branchA = stateWith([{ id: 'a', x: 5, z: 5 }])  // A moved
    const branchB = stateWith([])                            // B removed

    const result = threeWayMerge(base, branchA, branchB)
    expect(result.conflicts.length).toBe(1)
    expect(result.conflicts[0]!.kind).toBe('move-remove')
    expect(result.conflicts[0]!.itemId).toBe('a')
  })

  it('resolveConflict use-a keeps A value', () => {
    const base = stateWith([{ id: 'a', x: 1, z: 1 }])
    const branchA = stateWith([{ id: 'a', x: 5, z: 5 }])
    const branchB = stateWith([])

    let result = threeWayMerge(base, branchA, branchB)
    result = resolveConflict(result, 'a', 'use-a')

    expect(result.conflicts.length).toBe(0)
    expect(result.mergedState.items.has('a')).toBe(true)
    expect(result.mergedState.items.get('a')?.position).toEqual([5, 0, 5])
  })

  it('resolveConflict use-b removes when B removed', () => {
    const base = stateWith([{ id: 'a', x: 1, z: 1 }])
    const branchA = stateWith([{ id: 'a', x: 5, z: 5 }])
    const branchB = stateWith([])

    let result = threeWayMerge(base, branchA, branchB)
    result = resolveConflict(result, 'a', 'use-b')

    expect(result.conflicts.length).toBe(0)
    expect(result.mergedState.items.has('a')).toBe(false)
  })

  it('resolveConflict use-base restores original', () => {
    const base = stateWith([{ id: 'a', x: 1, z: 1 }])
    const branchA = stateWith([{ id: 'a', x: 5, z: 5 }])
    const branchB = stateWith([])

    let result = threeWayMerge(base, branchA, branchB)
    result = resolveConflict(result, 'a', 'use-base')

    expect(result.conflicts.length).toBe(0)
    expect(result.mergedState.items.get('a')?.position).toEqual([1, 0, 1])
  })

  it('resolveConflict merge-displacements sums vectors', () => {
    const base = stateWith([{ id: 'a', x: 0, z: 0 }])
    // Both modified (rotation + position → both-modified, not both-moved)
    const branchA: ProjectedVenueState = {
      ...emptyVenueState(VENUE_ID),
      items: new Map([['a', {
        id: 'a',
        furnitureType: 'chair' as const,
        position: [3, 0, 0] as [number, number, number],
        rotation: [0, 1, 0] as [number, number, number],
        scale: [1, 1, 1] as [number, number, number],
      }]]),
    }
    const branchB: ProjectedVenueState = {
      ...emptyVenueState(VENUE_ID),
      items: new Map([['a', {
        id: 'a',
        furnitureType: 'chair' as const,
        position: [0, 0, 4] as [number, number, number],
        rotation: [0, 2, 0] as [number, number, number],
        scale: [1, 1, 1] as [number, number, number],
      }]]),
    }

    let result = threeWayMerge(base, branchA, branchB)
    expect(result.conflicts.length).toBe(1)

    result = resolveConflict(result, 'a', 'merge-displacements')
    expect(result.conflicts.length).toBe(0)
    const merged = result.mergedState.items.get('a')!
    expect(merged.position[0]).toBe(3)
    expect(merged.position[2]).toBe(4)
  })
})

// ─── Integration: Timeline + Diff ────────────────────────────────────────────

describe('Timeline + Diff Integration', () => {
  it('diff between two timeline points', () => {
    let tl = createTimeline(VENUE_ID)
    tl = appendEvents(tl, [
      placeEvent('a', 1, 0, 1),
      placeEvent('b', 2, 0, 2),
      moveEvent('a', 5, 0, 5),
      removeEvent('b'),
    ])

    const stateBefore = reconstructAt(tl, tl.activeBranchId, 1)
    const stateAfter = reconstructAt(tl, tl.activeBranchId, 3)

    const diff = computeDiff(stateBefore, stateAfter)
    expect(diff.moved).toBe(1)
    expect(diff.removed).toBe(1)
  })

  it('diff between two branches', () => {
    let tl = createTimeline(VENUE_ID)
    tl = appendEvents(tl, [placeEvent('a', 1, 0, 1)])

    const rootId = tl.activeBranchId
    tl = createBranch(tl, 'alt')
    tl = appendEvent(tl, moveEvent('a', 5, 0, 5))

    tl = switchBranch(tl, rootId)
    tl = appendEvent(tl, placeEvent('b', 10, 0, 10))

    const rootState = reconstructCurrent(tl)
    tl = switchBranch(tl, tl.branches.get(tl.rootBranchId)!.id)

    // Get alt branch state
    const branches = listBranches(tl)
    const altBranch = branches.find(b => b.name === 'alt')!
    const altState = reconstructAt(tl, altBranch.id, altBranch.events.length - 1)

    const diff = computeDiff(rootState, altState)
    expect(diff.moved).toBe(1)    // 'a' moved
    expect(diff.removed).toBe(1)  // 'b' only in root
  })
})

// ─── Integration: Timeline + Merge ───────────────────────────────────────────

describe('Timeline + Merge Integration', () => {
  it('merges two branches via common ancestor', () => {
    let tl = createTimeline(VENUE_ID)
    tl = appendEvents(tl, [
      placeEvent('a', 0, 0, 0),
      placeEvent('b', 5, 0, 5),
    ])

    // Fork point state
    const baseState = reconstructCurrent(tl)

    const rootId = tl.activeBranchId
    tl = createBranch(tl, 'alt')
    const altId = tl.activeBranchId
    tl = appendEvent(tl, placeEvent('c', 10, 0, 10))  // alt adds c

    tl = switchBranch(tl, rootId)
    tl = appendEvent(tl, moveEvent('a', 3, 0, 3))     // root moves a

    const rootState = reconstructCurrent(tl)
    tl = switchBranch(tl, altId)
    const altState = reconstructCurrent(tl)

    const result = threeWayMerge(baseState, rootState, altState)
    expect(result.conflicts.length).toBe(0)
    expect(result.mergedState.items.size).toBe(3)
    expect(result.mergedState.items.get('a')?.position).toEqual([3, 0, 3])
    expect(result.mergedState.items.has('c')).toBe(true)
  })
})

// ─── Property-Based Tests ────────────────────────────────────────────────────

describe('Property-based tests', () => {
  const arbPlaceEvent = fc.record({
    itemId: fc.string({ minLength: 1, maxLength: 5 }),
    x: fc.float({ noNaN: true, noDefaultInfinity: true, min: -50, max: 50 }),
    z: fc.float({ noNaN: true, noDefaultInfinity: true, min: -50, max: 50 }),
  })

  it('reconstructAt(index) matches full replay up to index', () => {
    fc.assert(fc.property(
      fc.array(arbPlaceEvent, { minLength: 1, maxLength: 20 }),
      (placements) => {
        eventVersion = 0
        resetBranchCounter()

        let tl = createTimeline('v1')
        const events = placements.map(p => placeEvent(p.itemId, p.x, 0, p.z))
        tl = appendEvents(tl, events)

        // Pick a random index
        const idx = Math.floor(Math.random() * events.length)

        // Full replay
        let fullState = emptyVenueState('v1')
        for (let i = 0; i <= idx; i++) {
          fullState = applyEvent(fullState, events[i]!)
        }

        // Timeline reconstruction
        const tlState = reconstructAt(tl, tl.activeBranchId, idx)
        expect(tlState.items.size).toBe(fullState.items.size)
      },
    ), { numRuns: 50 })
  })

  it('diff is symmetric: added↔removed when swapped', () => {
    fc.assert(fc.property(
      fc.array(arbPlaceEvent, { minLength: 0, maxLength: 10 }),
      fc.array(arbPlaceEvent, { minLength: 0, maxLength: 10 }),
      (items1, items2) => {
        const state1 = emptyVenueState('v1')
        for (const p of items1) {
          state1.items.set(p.itemId, {
            id: p.itemId,
            furnitureType: 'chair',
            position: [p.x, 0, p.z],
            rotation: [0, 0, 0],
            scale: [1, 1, 1],
          })
        }
        const state2 = emptyVenueState('v1')
        for (const p of items2) {
          state2.items.set(p.itemId, {
            id: p.itemId,
            furnitureType: 'chair',
            position: [p.x, 0, p.z],
            rotation: [0, 0, 0],
            scale: [1, 1, 1],
          })
        }

        const d1 = computeDiff(state1, state2)
        const d2 = computeDiff(state2, state1)
        expect(d1.added).toBe(d2.removed)
        expect(d1.removed).toBe(d2.added)
      },
    ), { numRuns: 50 })
  })

  it('three-way merge: non-overlapping changes always auto-resolve', () => {
    fc.assert(fc.property(
      fc.array(arbPlaceEvent, { minLength: 1, maxLength: 5 }),
      fc.array(arbPlaceEvent, { minLength: 1, maxLength: 5 }),
      (aItems, bItems) => {
        // Ensure no overlap between A and B additions
        const aIds = new Set(aItems.map(i => `a_${i.itemId}`))
        const bIds = new Set(bItems.map(i => `b_${i.itemId}`))

        const base = emptyVenueState('v1')
        const branchA = emptyVenueState('v1')
        const branchB = emptyVenueState('v1')

        for (const p of aItems) {
          const id = `a_${p.itemId}`
          branchA.items.set(id, {
            id, furnitureType: 'chair',
            position: [p.x, 0, p.z], rotation: [0, 0, 0], scale: [1, 1, 1],
          })
        }
        for (const p of bItems) {
          const id = `b_${p.itemId}`
          branchB.items.set(id, {
            id, furnitureType: 'chair',
            position: [p.x, 0, p.z], rotation: [0, 0, 0], scale: [1, 1, 1],
          })
        }

        const result = threeWayMerge(base, branchA, branchB)
        expect(result.conflicts.length).toBe(0)
        // All A and B items should be in merged
        for (const id of aIds) expect(result.mergedState.items.has(id)).toBe(true)
        for (const id of bIds) expect(result.mergedState.items.has(id)).toBe(true)
      },
    ), { numRuns: 50 })
  })

  it('diff unchanged count equals items present in both states', () => {
    fc.assert(fc.property(
      fc.array(arbPlaceEvent, { minLength: 0, maxLength: 10 }),
      (items) => {
        const state = emptyVenueState('v1')
        for (const p of items) {
          state.items.set(p.itemId, {
            id: p.itemId, furnitureType: 'chair',
            position: [p.x, 0, p.z], rotation: [0, 0, 0], scale: [1, 1, 1],
          })
        }

        // Diff with itself should be all unchanged
        const diff = computeDiff(state, state)
        expect(diff.unchanged).toBe(state.items.size)
        expect(diff.added).toBe(0)
        expect(diff.removed).toBe(0)
        expect(diff.moved).toBe(0)
        expect(diff.modified).toBe(0)
      },
    ), { numRuns: 50 })
  })
})
