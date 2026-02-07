import { describe, it, expect } from 'vitest'
import { applyEvent, projectState, emptyVenueState, type ProjectedVenueState } from '../projector'
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

// ─── VenueCreated ────────────────────────────────────────────────────────────

describe('VenueCreated', () => {
  it('sets the venue name', () => {
    const state = applyEvent(
      emptyVenueState(VENUE),
      makeEvent('VenueCreated', 1, { name: 'Grand Hall' }),
    )
    expect(state.name).toBe('Grand Hall')
    expect(state.version).toBe(1)
  })
})

// ─── VenueRenamed ────────────────────────────────────────────────────────────

describe('VenueRenamed', () => {
  it('updates the venue name', () => {
    let state = applyEvent(emptyVenueState(VENUE), makeEvent('VenueCreated', 1, { name: 'Old' }))
    state = applyEvent(state, makeEvent('VenueRenamed', 2, { name: 'New' }))
    expect(state.name).toBe('New')
    expect(state.version).toBe(2)
  })
})

// ─── VenueArchived ───────────────────────────────────────────────────────────

describe('VenueArchived', () => {
  it('marks the venue as archived', () => {
    let state = applyEvent(emptyVenueState(VENUE), makeEvent('VenueCreated', 1, { name: 'Hall' }))
    state = applyEvent(state, makeEvent('VenueArchived', 2, {}))
    expect(state.archived).toBe(true)
  })
})

// ─── ItemPlaced ──────────────────────────────────────────────────────────────

describe('ItemPlaced', () => {
  it('adds an item to the state', () => {
    const state = applyEvent(
      emptyVenueState(VENUE),
      makeEvent('ItemPlaced', 1, {
        itemId: 'item-1',
        furnitureType: 'chair',
        position: [5, 0, 3],
        rotation: [0, 0, 0],
      }),
    )
    expect(state.items.size).toBe(1)
    const item = state.items.get('item-1')!
    expect(item.furnitureType).toBe('chair')
    expect(item.position).toEqual([5, 0, 3])
    expect(item.scale).toEqual([1, 1, 1])
  })

  it('preserves groupId when set', () => {
    const state = applyEvent(
      emptyVenueState(VENUE),
      makeEvent('ItemPlaced', 1, {
        itemId: 'item-1',
        furnitureType: 'chair',
        position: [0, 0, 0],
        rotation: [0, 0, 0],
        groupId: 'g-1',
      }),
    )
    expect(state.items.get('item-1')!.groupId).toBe('g-1')
  })
})

// ─── ItemMoved ───────────────────────────────────────────────────────────────

describe('ItemMoved', () => {
  it('updates item position', () => {
    let state = applyEvent(
      emptyVenueState(VENUE),
      makeEvent('ItemPlaced', 1, { itemId: 'item-1', furnitureType: 'chair', position: [0, 0, 0], rotation: [0, 0, 0] }),
    )
    state = applyEvent(state, makeEvent('ItemMoved', 2, { itemId: 'item-1', position: [10, 0, 10] }))
    expect(state.items.get('item-1')!.position).toEqual([10, 0, 10])
  })

  it('is a no-op for non-existent item', () => {
    const state = applyEvent(emptyVenueState(VENUE), makeEvent('ItemMoved', 1, { itemId: 'no-item', position: [10, 0, 10] }))
    expect(state.items.size).toBe(0)
  })
})

// ─── ItemRotated ─────────────────────────────────────────────────────────────

describe('ItemRotated', () => {
  it('updates item rotation', () => {
    let state = applyEvent(
      emptyVenueState(VENUE),
      makeEvent('ItemPlaced', 1, { itemId: 'item-1', furnitureType: 'chair', position: [0, 0, 0], rotation: [0, 0, 0] }),
    )
    state = applyEvent(state, makeEvent('ItemRotated', 2, { itemId: 'item-1', rotation: [0, 1.57, 0] }))
    expect(state.items.get('item-1')!.rotation).toEqual([0, 1.57, 0])
  })
})

// ─── ItemScaled ──────────────────────────────────────────────────────────────

describe('ItemScaled', () => {
  it('updates item scale', () => {
    let state = applyEvent(
      emptyVenueState(VENUE),
      makeEvent('ItemPlaced', 1, { itemId: 'item-1', furnitureType: 'chair', position: [0, 0, 0], rotation: [0, 0, 0] }),
    )
    state = applyEvent(state, makeEvent('ItemScaled', 2, { itemId: 'item-1', scale: [2, 2, 2] }))
    expect(state.items.get('item-1')!.scale).toEqual([2, 2, 2])
  })
})

// ─── ItemRemoved ─────────────────────────────────────────────────────────────

describe('ItemRemoved', () => {
  it('removes an item from the state', () => {
    let state = applyEvent(
      emptyVenueState(VENUE),
      makeEvent('ItemPlaced', 1, { itemId: 'item-1', furnitureType: 'chair', position: [0, 0, 0], rotation: [0, 0, 0] }),
    )
    expect(state.items.size).toBe(1)
    state = applyEvent(state, makeEvent('ItemRemoved', 2, { itemId: 'item-1' }))
    expect(state.items.size).toBe(0)
  })
})

// ─── ItemsBatchMoved ─────────────────────────────────────────────────────────

describe('ItemsBatchMoved', () => {
  it('moves multiple items at once', () => {
    let state = emptyVenueState(VENUE)
    state = applyEvent(state, makeEvent('ItemPlaced', 1, { itemId: 'a', furnitureType: 'chair', position: [0, 0, 0], rotation: [0, 0, 0] }))
    state = applyEvent(state, makeEvent('ItemPlaced', 2, { itemId: 'b', furnitureType: 'chair', position: [0, 0, 0], rotation: [0, 0, 0] }))
    state = applyEvent(state, makeEvent('ItemsBatchMoved', 3, {
      moves: [
        { itemId: 'a', position: [5, 0, 5] },
        { itemId: 'b', position: [10, 0, 10] },
      ],
    }))
    expect(state.items.get('a')!.position).toEqual([5, 0, 5])
    expect(state.items.get('b')!.position).toEqual([10, 0, 10])
  })
})

// ─── GroupCreated ────────────────────────────────────────────────────────────

describe('GroupCreated', () => {
  it('creates a group and assigns items', () => {
    let state = emptyVenueState(VENUE)
    state = applyEvent(state, makeEvent('ItemPlaced', 1, { itemId: 'a', furnitureType: 'chair', position: [0, 0, 0], rotation: [0, 0, 0] }))
    state = applyEvent(state, makeEvent('ItemPlaced', 2, { itemId: 'b', furnitureType: 'chair', position: [1, 0, 0], rotation: [0, 0, 0] }))
    state = applyEvent(state, makeEvent('GroupCreated', 3, { groupId: 'g-1', itemIds: ['a', 'b'] }))

    expect(state.groups.has('g-1')).toBe(true)
    expect(state.items.get('a')!.groupId).toBe('g-1')
    expect(state.items.get('b')!.groupId).toBe('g-1')
  })
})

// ─── GroupDissolved ──────────────────────────────────────────────────────────

describe('GroupDissolved', () => {
  it('removes the group and clears groupId from items', () => {
    let state = emptyVenueState(VENUE)
    state = applyEvent(state, makeEvent('ItemPlaced', 1, { itemId: 'a', furnitureType: 'chair', position: [0, 0, 0], rotation: [0, 0, 0] }))
    state = applyEvent(state, makeEvent('ItemPlaced', 2, { itemId: 'b', furnitureType: 'chair', position: [1, 0, 0], rotation: [0, 0, 0] }))
    state = applyEvent(state, makeEvent('GroupCreated', 3, { groupId: 'g-1', itemIds: ['a', 'b'] }))
    state = applyEvent(state, makeEvent('GroupDissolved', 4, { groupId: 'g-1' }))

    expect(state.groups.has('g-1')).toBe(false)
    expect(state.items.get('a')!.groupId).toBeUndefined()
    expect(state.items.get('b')!.groupId).toBeUndefined()
  })
})

// ─── ItemsUngrouped ──────────────────────────────────────────────────────────

describe('ItemsUngrouped', () => {
  it('removes groupId from specified items', () => {
    let state = emptyVenueState(VENUE)
    state = applyEvent(state, makeEvent('ItemPlaced', 1, { itemId: 'a', furnitureType: 'chair', position: [0, 0, 0], rotation: [0, 0, 0] }))
    state = applyEvent(state, makeEvent('ItemPlaced', 2, { itemId: 'b', furnitureType: 'chair', position: [1, 0, 0], rotation: [0, 0, 0] }))
    state = applyEvent(state, makeEvent('GroupCreated', 3, { groupId: 'g-1', itemIds: ['a', 'b'] }))
    state = applyEvent(state, makeEvent('ItemsUngrouped', 4, { groupId: 'g-1', itemIds: ['a'] }))

    expect(state.items.get('a')!.groupId).toBeUndefined()
    expect(state.items.get('b')!.groupId).toBe('g-1')
    expect(state.groups.has('g-1')).toBe(true) // b still in group
  })

  it('dissolves group when last item removed', () => {
    let state = emptyVenueState(VENUE)
    state = applyEvent(state, makeEvent('ItemPlaced', 1, { itemId: 'a', furnitureType: 'chair', position: [0, 0, 0], rotation: [0, 0, 0] }))
    state = applyEvent(state, makeEvent('GroupCreated', 2, { groupId: 'g-1', itemIds: ['a'] }))
    state = applyEvent(state, makeEvent('ItemsUngrouped', 3, { groupId: 'g-1', itemIds: ['a'] }))

    expect(state.groups.has('g-1')).toBe(false)
  })
})

// ─── ScenarioCreated ─────────────────────────────────────────────────────────

describe('ScenarioCreated', () => {
  it('adds scenario to the state', () => {
    const state = applyEvent(
      emptyVenueState(VENUE),
      makeEvent('ScenarioCreated', 1, { scenarioId: 's-1', name: 'Wedding' }),
    )
    expect(state.scenarios.has('s-1')).toBe(true)
    expect(state.scenarios.get('s-1')!.name).toBe('Wedding')
  })
})

// ─── ScenarioDeleted ─────────────────────────────────────────────────────────

describe('ScenarioDeleted', () => {
  it('removes scenario from the state', () => {
    let state = applyEvent(emptyVenueState(VENUE), makeEvent('ScenarioCreated', 1, { scenarioId: 's-1', name: 'Wedding' }))
    state = applyEvent(state, makeEvent('ScenarioDeleted', 2, { scenarioId: 's-1' }))
    expect(state.scenarios.has('s-1')).toBe(false)
  })
})

// ─── projectState ────────────────────────────────────────────────────────────

describe('projectState', () => {
  it('projects a full event stream into state', () => {
    const events: DomainEvent[] = [
      makeEvent('VenueCreated', 1, { name: 'Grand Hall' }),
      makeEvent('ItemPlaced', 2, { itemId: 'chair-1', furnitureType: 'chair', position: [5, 0, 3], rotation: [0, 0, 0] }),
      makeEvent('ItemPlaced', 3, { itemId: 'chair-2', furnitureType: 'chair', position: [6, 0, 3], rotation: [0, 0, 0] }),
      makeEvent('GroupCreated', 4, { groupId: 'row-1', itemIds: ['chair-1', 'chair-2'] }),
      makeEvent('ItemMoved', 5, { itemId: 'chair-1', position: [5, 0, 5] }),
      makeEvent('ScenarioCreated', 6, { scenarioId: 's-1', name: 'Concert' }),
    ]

    const state = projectState(VENUE, events)

    expect(state.name).toBe('Grand Hall')
    expect(state.items.size).toBe(2)
    expect(state.items.get('chair-1')!.position).toEqual([5, 0, 5])
    expect(state.items.get('chair-2')!.position).toEqual([6, 0, 3])
    expect(state.groups.has('row-1')).toBe(true)
    expect(state.scenarios.has('s-1')).toBe(true)
    expect(state.version).toBe(6)
  })

  it('handles empty event stream', () => {
    const state = projectState(VENUE, [])
    expect(state.items.size).toBe(0)
    expect(state.version).toBe(0)
    expect(state.name).toBe('')
  })

  it('correctly handles item removal mid-stream', () => {
    const events: DomainEvent[] = [
      makeEvent('ItemPlaced', 1, { itemId: 'a', furnitureType: 'chair', position: [0, 0, 0], rotation: [0, 0, 0] }),
      makeEvent('ItemPlaced', 2, { itemId: 'b', furnitureType: 'round-table', position: [5, 0, 5], rotation: [0, 0, 0] }),
      makeEvent('ItemRemoved', 3, { itemId: 'a' }),
    ]

    const state = projectState(VENUE, events)
    expect(state.items.size).toBe(1)
    expect(state.items.has('a')).toBe(false)
    expect(state.items.has('b')).toBe(true)
  })
})

// ─── Immutability ────────────────────────────────────────────────────────────

describe('immutability', () => {
  it('does not mutate the previous state', () => {
    const state1 = applyEvent(
      emptyVenueState(VENUE),
      makeEvent('ItemPlaced', 1, { itemId: 'a', furnitureType: 'chair', position: [0, 0, 0], rotation: [0, 0, 0] }),
    )
    const state2 = applyEvent(state1, makeEvent('ItemMoved', 2, { itemId: 'a', position: [10, 0, 10] }))

    // state1 should be unchanged
    expect(state1.items.get('a')!.position).toEqual([0, 0, 0])
    expect(state2.items.get('a')!.position).toEqual([10, 0, 10])
    expect(state1.version).toBe(1)
    expect(state2.version).toBe(2)
  })
})
