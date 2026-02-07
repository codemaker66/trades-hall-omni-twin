import { describe, it, expect } from 'vitest'
import { UndoManager } from '../undo-manager'
import { applyEvent, emptyVenueState, type ProjectedVenueState } from '../projector'
import type { DomainEvent } from '@omni-twin/types'

// ─── Helpers ─────────────────────────────────────────────────────────────────

const VENUE = 'venue-1'
const USER = 'user-1'

function makeEvent<T extends DomainEvent['type']>(
  type: T,
  version: number,
  payload: Extract<DomainEvent, { type: T }>['payload'],
): Extract<DomainEvent, { type: T }> {
  return {
    id: `evt-${version}`,
    type,
    timestamp: '2025-01-01T00:00:00Z',
    userId: USER,
    venueId: VENUE,
    version,
    payload,
  } as Extract<DomainEvent, { type: T }>
}

function applyEvents(state: ProjectedVenueState, events: DomainEvent[]): ProjectedVenueState {
  for (const event of events) {
    state = applyEvent(state, event)
  }
  return state
}

// ─── Basic Undo/Redo ─────────────────────────────────────────────────────────

describe('UndoManager', () => {
  it('starts with empty stacks', () => {
    const mgr = new UndoManager()
    expect(mgr.canUndo).toBe(false)
    expect(mgr.canRedo).toBe(false)
    expect(mgr.undoCount).toBe(0)
    expect(mgr.redoCount).toBe(0)
  })

  it('can undo after recording', () => {
    const mgr = new UndoManager()
    const state0 = emptyVenueState(VENUE)
    const placeEvent = makeEvent('ItemPlaced', 1, {
      itemId: 'a', furnitureType: 'chair', position: [5, 0, 5], rotation: [0, 0, 0],
    })

    mgr.record([placeEvent], state0)
    expect(mgr.canUndo).toBe(true)
    expect(mgr.canRedo).toBe(false)
  })

  it('undo produces compensating events for a placed item (remove it)', () => {
    const mgr = new UndoManager()
    const state0 = emptyVenueState(VENUE)
    const placeEvent = makeEvent('ItemPlaced', 1, {
      itemId: 'a', furnitureType: 'chair', position: [5, 0, 5], rotation: [0, 0, 0],
    })

    // Record + apply
    mgr.record([placeEvent], state0)
    const state1 = applyEvent(state0, placeEvent)

    // Undo
    const compensating = mgr.undo(state1)
    expect(compensating).not.toBeNull()
    expect(compensating!.length).toBe(1)
    expect(compensating![0]!.type).toBe('ItemRemoved')
  })

  it('undo produces compensating events for a moved item (move back)', () => {
    const mgr = new UndoManager()
    // Start with an item at [0, 0, 0]
    const placeEvent = makeEvent('ItemPlaced', 1, {
      itemId: 'a', furnitureType: 'chair', position: [0, 0, 0], rotation: [0, 0, 0],
    })
    const state0 = applyEvent(emptyVenueState(VENUE), placeEvent)

    const moveEvent = makeEvent('ItemMoved', 2, { itemId: 'a', position: [10, 0, 10] })
    mgr.record([moveEvent], state0)
    const state1 = applyEvent(state0, moveEvent)

    // Undo should move back to [0, 0, 0]
    const compensating = mgr.undo(state1)
    expect(compensating).not.toBeNull()
    expect(compensating!.some((e) => e.type === 'ItemMoved')).toBe(true)
  })

  it('undo for removed item re-places it', () => {
    const mgr = new UndoManager()
    const placeEvent = makeEvent('ItemPlaced', 1, {
      itemId: 'a', furnitureType: 'chair', position: [5, 0, 5], rotation: [0, 0, 0],
    })
    const state0 = applyEvent(emptyVenueState(VENUE), placeEvent)

    const removeEvent = makeEvent('ItemRemoved', 2, { itemId: 'a' })
    mgr.record([removeEvent], state0)
    const state1 = applyEvent(state0, removeEvent)

    const compensating = mgr.undo(state1)
    expect(compensating).not.toBeNull()
    expect(compensating!.some((e) => e.type === 'ItemPlaced')).toBe(true)
  })

  it('redo returns the original events', () => {
    const mgr = new UndoManager()
    const state0 = emptyVenueState(VENUE)
    const placeEvent = makeEvent('ItemPlaced', 1, {
      itemId: 'a', furnitureType: 'chair', position: [5, 0, 5], rotation: [0, 0, 0],
    })
    mgr.record([placeEvent], state0)
    const state1 = applyEvent(state0, placeEvent)

    mgr.undo(state1)
    expect(mgr.canRedo).toBe(true)

    const redoEvents = mgr.redo()
    expect(redoEvents).not.toBeNull()
    expect(redoEvents!).toEqual([placeEvent])
  })

  it('new action clears redo stack', () => {
    const mgr = new UndoManager()
    const state0 = emptyVenueState(VENUE)
    const placeA = makeEvent('ItemPlaced', 1, { itemId: 'a', furnitureType: 'chair', position: [0, 0, 0], rotation: [0, 0, 0] })
    const placeB = makeEvent('ItemPlaced', 2, { itemId: 'b', furnitureType: 'chair', position: [1, 0, 0], rotation: [0, 0, 0] })

    mgr.record([placeA], state0)
    const state1 = applyEvent(state0, placeA)
    mgr.undo(state1)
    expect(mgr.canRedo).toBe(true)

    // New action should clear redo
    mgr.record([placeB], state0)
    expect(mgr.canRedo).toBe(false)
  })

  it('returns null when nothing to undo/redo', () => {
    const mgr = new UndoManager()
    expect(mgr.undo(emptyVenueState(VENUE))).toBeNull()
    expect(mgr.redo()).toBeNull()
  })
})

// ─── Batch Operations ────────────────────────────────────────────────────────

describe('UndoManager batching', () => {
  it('batched events become a single undo entry', () => {
    const mgr = new UndoManager()
    const placeEvent = makeEvent('ItemPlaced', 1, { itemId: 'a', furnitureType: 'chair', position: [0, 0, 0], rotation: [0, 0, 0] })
    const state0 = applyEvent(emptyVenueState(VENUE), placeEvent)

    mgr.beginBatch()

    const move1 = makeEvent('ItemMoved', 2, { itemId: 'a', position: [1, 0, 0] })
    mgr.record([move1], state0)
    const state1 = applyEvent(state0, move1)

    const move2 = makeEvent('ItemMoved', 3, { itemId: 'a', position: [2, 0, 0] })
    mgr.record([move2], state1)

    mgr.endBatch()

    // Should be a single undo entry
    expect(mgr.undoCount).toBe(1)
  })

  it('empty batch does not create an undo entry', () => {
    const mgr = new UndoManager()
    mgr.beginBatch()
    mgr.endBatch()
    expect(mgr.undoCount).toBe(0)
  })

  it('undo a batch undoes all events in the batch', () => {
    const mgr = new UndoManager()
    const placeEvent = makeEvent('ItemPlaced', 1, { itemId: 'a', furnitureType: 'chair', position: [0, 0, 0], rotation: [0, 0, 0] })
    const state0 = applyEvent(emptyVenueState(VENUE), placeEvent)

    mgr.beginBatch()
    const move1 = makeEvent('ItemMoved', 2, { itemId: 'a', position: [5, 0, 0] })
    mgr.record([move1], state0)
    const state1 = applyEvent(state0, move1)
    const move2 = makeEvent('ItemMoved', 3, { itemId: 'a', position: [10, 0, 0] })
    mgr.record([move2], state1)
    const state2 = applyEvent(state1, move2)
    mgr.endBatch()

    // Undo the whole batch
    const compensating = mgr.undo(state2)
    expect(compensating).not.toBeNull()
    // Should produce a move back to original position [0, 0, 0]
    const moveBack = compensating!.find((e) => e.type === 'ItemMoved')
    expect(moveBack).toBeDefined()
  })
})

// ─── History Limit ───────────────────────────────────────────────────────────

describe('UndoManager history limit', () => {
  it('trims history beyond maxHistory', () => {
    const mgr = new UndoManager(3)
    const state = emptyVenueState(VENUE)

    for (let i = 0; i < 5; i++) {
      const event = makeEvent('ItemPlaced', i + 1, {
        itemId: `item-${i}`, furnitureType: 'chair', position: [i, 0, 0], rotation: [0, 0, 0],
      })
      mgr.record([event], state)
    }

    expect(mgr.undoCount).toBe(3)
  })
})

// ─── Clear ───────────────────────────────────────────────────────────────────

describe('UndoManager clear', () => {
  it('clears all history', () => {
    const mgr = new UndoManager()
    const state = emptyVenueState(VENUE)
    const event = makeEvent('ItemPlaced', 1, { itemId: 'a', furnitureType: 'chair', position: [0, 0, 0], rotation: [0, 0, 0] })
    mgr.record([event], state)

    mgr.clear()
    expect(mgr.canUndo).toBe(false)
    expect(mgr.canRedo).toBe(false)
  })
})
