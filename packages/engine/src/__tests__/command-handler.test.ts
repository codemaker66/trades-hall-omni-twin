import { describe, it, expect } from 'vitest'
import { handleCommand } from '../command-handler'
import { emptyVenueState, applyEvent, type ProjectedVenueState } from '../projector'
import type { Command, DomainEvent } from '@omni-twin/types'

// ─── Helpers ─────────────────────────────────────────────────────────────────

const VENUE = 'venue-1'
const USER = 'user-1'

function base() {
  return { userId: USER, venueId: VENUE }
}

function stateAfterEvents(events: DomainEvent[]): ProjectedVenueState {
  let state = emptyVenueState(VENUE)
  for (const event of events) {
    state = applyEvent(state, event)
  }
  return state
}

function placeItemEvents(itemId: string, version: number): DomainEvent {
  return {
    id: `evt-${version}`,
    type: 'ItemPlaced',
    timestamp: '2025-01-01T00:00:00Z',
    userId: USER,
    venueId: VENUE,
    version,
    payload: {
      itemId,
      furnitureType: 'chair',
      position: [0, 0, 0] as [number, number, number],
      rotation: [0, 0, 0] as [number, number, number],
    },
  } as DomainEvent
}

function venueCreatedEvent(): DomainEvent {
  return {
    id: 'evt-1',
    type: 'VenueCreated',
    timestamp: '2025-01-01T00:00:00Z',
    userId: USER,
    venueId: VENUE,
    version: 1,
    payload: { name: 'Test Hall' },
  } as DomainEvent
}

// ─── CreateVenue ─────────────────────────────────────────────────────────────

describe('handleCommand: CreateVenue', () => {
  it('produces VenueCreated event', () => {
    const cmd: Command = { ...base(), type: 'CreateVenue', name: 'Grand Hall' }
    const result = handleCommand(emptyVenueState(VENUE), cmd)

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.value).toHaveLength(1)
    expect(result.value[0]!.type).toBe('VenueCreated')
    expect(result.value[0]!.version).toBe(1)
  })
})

// ─── PlaceItem ───────────────────────────────────────────────────────────────

describe('handleCommand: PlaceItem', () => {
  it('produces ItemPlaced event', () => {
    const state = stateAfterEvents([venueCreatedEvent()])
    const cmd: Command = {
      ...base(), type: 'PlaceItem',
      itemId: 'chair-1', furnitureType: 'chair', position: [5, 0, 3], rotation: [0, 0, 0],
    }
    const result = handleCommand(state, cmd)

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.value).toHaveLength(1)
    expect(result.value[0]!.type).toBe('ItemPlaced')
    expect(result.value[0]!.version).toBe(2)
  })

  it('rejects duplicate item', () => {
    const state = stateAfterEvents([venueCreatedEvent(), placeItemEvents('chair-1', 2)])
    const cmd: Command = {
      ...base(), type: 'PlaceItem',
      itemId: 'chair-1', furnitureType: 'chair', position: [5, 0, 3], rotation: [0, 0, 0],
    }
    const result = handleCommand(state, cmd)

    expect(result.ok).toBe(false)
    if (result.ok) return
    expect(result.error.code).toBe('ITEM_ALREADY_EXISTS')
  })
})

// ─── MoveItem ────────────────────────────────────────────────────────────────

describe('handleCommand: MoveItem', () => {
  it('produces ItemMoved event', () => {
    const state = stateAfterEvents([venueCreatedEvent(), placeItemEvents('chair-1', 2)])
    const cmd: Command = { ...base(), type: 'MoveItem', itemId: 'chair-1', position: [10, 0, 10] }
    const result = handleCommand(state, cmd)

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.value[0]!.type).toBe('ItemMoved')
  })
})

// ─── RemoveItem ──────────────────────────────────────────────────────────────

describe('handleCommand: RemoveItem', () => {
  it('produces ItemRemoved event', () => {
    const state = stateAfterEvents([venueCreatedEvent(), placeItemEvents('chair-1', 2)])
    const cmd: Command = { ...base(), type: 'RemoveItem', itemId: 'chair-1' }
    const result = handleCommand(state, cmd)

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.value[0]!.type).toBe('ItemRemoved')
  })
})

// ─── MoveItemsBatch ──────────────────────────────────────────────────────────

describe('handleCommand: MoveItemsBatch', () => {
  it('produces ItemsBatchMoved event', () => {
    const state = stateAfterEvents([
      venueCreatedEvent(),
      placeItemEvents('a', 2),
      placeItemEvents('b', 3),
    ])
    const cmd: Command = {
      ...base(), type: 'MoveItemsBatch',
      moves: [
        { itemId: 'a', position: [5, 0, 5] },
        { itemId: 'b', position: [10, 0, 10] },
      ],
    }
    const result = handleCommand(state, cmd)

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.value).toHaveLength(1)
    expect(result.value[0]!.type).toBe('ItemsBatchMoved')
  })
})

// ─── GroupItems ──────────────────────────────────────────────────────────────

describe('handleCommand: GroupItems', () => {
  it('produces GroupCreated event', () => {
    const state = stateAfterEvents([
      venueCreatedEvent(),
      placeItemEvents('a', 2),
      placeItemEvents('b', 3),
    ])
    const cmd: Command = { ...base(), type: 'GroupItems', groupId: 'g-1', itemIds: ['a', 'b'] }
    const result = handleCommand(state, cmd)

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.value[0]!.type).toBe('GroupCreated')
  })
})

// ─── ImportLayout ────────────────────────────────────────────────────────────

describe('handleCommand: ImportLayout', () => {
  it('produces LayoutImported + ItemPlaced events', () => {
    const state = stateAfterEvents([venueCreatedEvent()])
    const cmd: Command = {
      ...base(), type: 'ImportLayout', sourceFormat: 'json',
      items: [
        { itemId: 'a', furnitureType: 'chair', position: [0, 0, 0], rotation: [0, 0, 0] },
        { itemId: 'b', furnitureType: 'round-table', position: [5, 0, 5], rotation: [0, 0, 0] },
      ],
    }
    const result = handleCommand(state, cmd)

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.value).toHaveLength(3)
    expect(result.value[0]!.type).toBe('LayoutImported')
    expect(result.value[1]!.type).toBe('ItemPlaced')
    expect(result.value[2]!.type).toBe('ItemPlaced')
    // Versions should be sequential
    expect(result.value[0]!.version).toBe(2)
    expect(result.value[1]!.version).toBe(3)
    expect(result.value[2]!.version).toBe(4)
  })
})

// ─── Version Sequencing ──────────────────────────────────────────────────────

describe('version sequencing', () => {
  it('assigns versions starting from state.version + 1', () => {
    const state = stateAfterEvents([
      venueCreatedEvent(),
      placeItemEvents('chair-1', 2),
      placeItemEvents('chair-2', 3),
    ])
    // State version is now 3
    const cmd: Command = { ...base(), type: 'PlaceItem', itemId: 'chair-3', furnitureType: 'chair', position: [1, 0, 1], rotation: [0, 0, 0] }
    const result = handleCommand(state, cmd)

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.value[0]!.version).toBe(4)
  })
})

// ─── Event IDs ───────────────────────────────────────────────────────────────

describe('event IDs', () => {
  it('assigns unique IDs to each event', () => {
    const state = stateAfterEvents([venueCreatedEvent()])
    const cmd: Command = {
      ...base(), type: 'ImportLayout', sourceFormat: 'json',
      items: [
        { itemId: 'a', furnitureType: 'chair', position: [0, 0, 0], rotation: [0, 0, 0] },
        { itemId: 'b', furnitureType: 'chair', position: [1, 0, 0], rotation: [0, 0, 0] },
      ],
    }
    const result = handleCommand(state, cmd)
    expect(result.ok).toBe(true)
    if (!result.ok) return

    const ids = result.value.map((e) => e.id)
    expect(new Set(ids).size).toBe(ids.length)
  })
})
