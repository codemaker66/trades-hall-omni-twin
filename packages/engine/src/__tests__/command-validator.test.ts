import { describe, it, expect } from 'vitest'
import { validateCommand, emptyProjectedState, type ValidatorVenueState, type ValidatorItem } from '../command-validator'
import type { Command, Position3D } from '@omni-twin/types'

// ─── Helpers ─────────────────────────────────────────────────────────────────

const USER = 'user-1'
const VENUE = 'venue-1'

function base() {
  return { userId: USER, venueId: VENUE }
}

function stateWith(overrides?: Partial<ValidatorVenueState>): ValidatorVenueState {
  return { ...emptyProjectedState(VENUE), name: 'Test Venue', ...overrides }
}

function itemAt(id: string, pos: Position3D = [0, 0, 0], groupId?: string): ValidatorItem {
  return { id, furnitureType: 'chair', position: pos, rotation: [0, 0, 0], groupId }
}

function stateWithItems(...items: ValidatorItem[]): ValidatorVenueState {
  const map = new Map(items.map((i) => [i.id, i]))
  const groups = new Set<string>()
  for (const item of items) {
    if (item.groupId) groups.add(item.groupId)
  }
  return stateWith({ items: map, groups })
}

// ─── CreateVenue ─────────────────────────────────────────────────────────────

describe('CreateVenue', () => {
  it('accepts a valid name', () => {
    const cmd: Command = { ...base(), type: 'CreateVenue', name: 'My Venue' }
    const result = validateCommand(cmd, stateWith())
    expect(result.ok).toBe(true)
  })

  it('rejects empty name', () => {
    const cmd: Command = { ...base(), type: 'CreateVenue', name: '' }
    const result = validateCommand(cmd, stateWith())
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('VENUE_NAME_EMPTY')
  })

  it('rejects whitespace-only name', () => {
    const cmd: Command = { ...base(), type: 'CreateVenue', name: '   ' }
    const result = validateCommand(cmd, stateWith())
    expect(result.ok).toBe(false)
  })

  it('rejects name over 200 chars', () => {
    const cmd: Command = { ...base(), type: 'CreateVenue', name: 'x'.repeat(201) }
    const result = validateCommand(cmd, stateWith())
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('VENUE_NAME_TOO_LONG')
  })
})

// ─── RenameVenue ─────────────────────────────────────────────────────────────

describe('RenameVenue', () => {
  it('accepts valid rename', () => {
    const cmd: Command = { ...base(), type: 'RenameVenue', name: 'New Name' }
    expect(validateCommand(cmd, stateWith()).ok).toBe(true)
  })

  it('rejects rename on archived venue', () => {
    const cmd: Command = { ...base(), type: 'RenameVenue', name: 'New Name' }
    const result = validateCommand(cmd, stateWith({ archived: true }))
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('VENUE_ARCHIVED')
  })
})

// ─── ArchiveVenue ────────────────────────────────────────────────────────────

describe('ArchiveVenue', () => {
  it('accepts archiving a non-archived venue', () => {
    const cmd: Command = { ...base(), type: 'ArchiveVenue' }
    expect(validateCommand(cmd, stateWith()).ok).toBe(true)
  })

  it('rejects archiving already-archived venue', () => {
    const cmd: Command = { ...base(), type: 'ArchiveVenue' }
    const result = validateCommand(cmd, stateWith({ archived: true }))
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('VENUE_ALREADY_ARCHIVED')
  })
})

// ─── PlaceItem ───────────────────────────────────────────────────────────────

describe('PlaceItem', () => {
  it('accepts valid placement', () => {
    const cmd: Command = {
      ...base(), type: 'PlaceItem',
      itemId: 'item-1', furnitureType: 'chair', position: [5, 0, 5], rotation: [0, 0, 0],
    }
    expect(validateCommand(cmd, stateWith()).ok).toBe(true)
  })

  it('rejects duplicate item ID', () => {
    const cmd: Command = {
      ...base(), type: 'PlaceItem',
      itemId: 'item-1', furnitureType: 'chair', position: [5, 0, 5], rotation: [0, 0, 0],
    }
    const result = validateCommand(cmd, stateWithItems(itemAt('item-1')))
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('ITEM_ALREADY_EXISTS')
  })

  it('rejects position out of bounds', () => {
    const cmd: Command = {
      ...base(), type: 'PlaceItem',
      itemId: 'item-1', furnitureType: 'chair', position: [999, 0, 0], rotation: [0, 0, 0],
    }
    const result = validateCommand(cmd, stateWith())
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('POSITION_OUT_OF_BOUNDS')
  })

  it('rejects NaN position', () => {
    const cmd: Command = {
      ...base(), type: 'PlaceItem',
      itemId: 'item-1', furnitureType: 'chair', position: [NaN, 0, 0], rotation: [0, 0, 0],
    }
    const result = validateCommand(cmd, stateWith())
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('INVALID_POSITION')
  })

  it('rejects Infinity position', () => {
    const cmd: Command = {
      ...base(), type: 'PlaceItem',
      itemId: 'item-1', furnitureType: 'chair', position: [Infinity, 0, 0], rotation: [0, 0, 0],
    }
    const result = validateCommand(cmd, stateWith())
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('INVALID_POSITION')
  })

  it('rejects archived venue', () => {
    const cmd: Command = {
      ...base(), type: 'PlaceItem',
      itemId: 'item-1', furnitureType: 'chair', position: [5, 0, 5], rotation: [0, 0, 0],
    }
    const result = validateCommand(cmd, stateWith({ archived: true }))
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('VENUE_ARCHIVED')
  })

  it('rejects non-existent group reference', () => {
    const cmd: Command = {
      ...base(), type: 'PlaceItem',
      itemId: 'item-1', furnitureType: 'chair', position: [5, 0, 5], rotation: [0, 0, 0], groupId: 'no-such-group',
    }
    const result = validateCommand(cmd, stateWith())
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('GROUP_NOT_FOUND')
  })
})

// ─── MoveItem ────────────────────────────────────────────────────────────────

describe('MoveItem', () => {
  it('accepts valid move', () => {
    const cmd: Command = { ...base(), type: 'MoveItem', itemId: 'item-1', position: [10, 0, 10] }
    expect(validateCommand(cmd, stateWithItems(itemAt('item-1'))).ok).toBe(true)
  })

  it('rejects non-existent item', () => {
    const cmd: Command = { ...base(), type: 'MoveItem', itemId: 'no-item', position: [10, 0, 10] }
    const result = validateCommand(cmd, stateWith())
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('ITEM_NOT_FOUND')
  })
})

// ─── RemoveItem ──────────────────────────────────────────────────────────────

describe('RemoveItem', () => {
  it('accepts removing existing item', () => {
    const cmd: Command = { ...base(), type: 'RemoveItem', itemId: 'item-1' }
    expect(validateCommand(cmd, stateWithItems(itemAt('item-1'))).ok).toBe(true)
  })

  it('rejects removing non-existent item', () => {
    const cmd: Command = { ...base(), type: 'RemoveItem', itemId: 'no-item' }
    const result = validateCommand(cmd, stateWith())
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('ITEM_NOT_FOUND')
  })
})

// ─── ScaleItem ───────────────────────────────────────────────────────────────

describe('ScaleItem', () => {
  it('accepts valid scale', () => {
    const cmd: Command = { ...base(), type: 'ScaleItem', itemId: 'item-1', scale: [1, 2, 1] }
    expect(validateCommand(cmd, stateWithItems(itemAt('item-1'))).ok).toBe(true)
  })

  it('rejects zero scale', () => {
    const cmd: Command = { ...base(), type: 'ScaleItem', itemId: 'item-1', scale: [1, 0, 1] }
    const result = validateCommand(cmd, stateWithItems(itemAt('item-1')))
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('INVALID_SCALE')
  })

  it('rejects negative scale', () => {
    const cmd: Command = { ...base(), type: 'ScaleItem', itemId: 'item-1', scale: [-1, 1, 1] }
    const result = validateCommand(cmd, stateWithItems(itemAt('item-1')))
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('INVALID_SCALE')
  })
})

// ─── MoveItemsBatch ──────────────────────────────────────────────────────────

describe('MoveItemsBatch', () => {
  it('accepts valid batch move', () => {
    const cmd: Command = {
      ...base(), type: 'MoveItemsBatch',
      moves: [
        { itemId: 'item-1', position: [1, 0, 1] },
        { itemId: 'item-2', position: [2, 0, 2] },
      ],
    }
    const state = stateWithItems(itemAt('item-1'), itemAt('item-2'))
    expect(validateCommand(cmd, state).ok).toBe(true)
  })

  it('rejects empty batch', () => {
    const cmd: Command = { ...base(), type: 'MoveItemsBatch', moves: [] }
    const result = validateCommand(cmd, stateWith())
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('EMPTY_BATCH')
  })

  it('rejects if any item missing', () => {
    const cmd: Command = {
      ...base(), type: 'MoveItemsBatch',
      moves: [{ itemId: 'item-1', position: [1, 0, 1] }, { itemId: 'no-item', position: [2, 0, 2] }],
    }
    const result = validateCommand(cmd, stateWithItems(itemAt('item-1')))
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('ITEM_NOT_FOUND')
  })
})

// ─── GroupItems ──────────────────────────────────────────────────────────────

describe('GroupItems', () => {
  it('accepts valid grouping', () => {
    const cmd: Command = { ...base(), type: 'GroupItems', groupId: 'g-1', itemIds: ['item-1', 'item-2'] }
    const state = stateWithItems(itemAt('item-1'), itemAt('item-2'))
    expect(validateCommand(cmd, state).ok).toBe(true)
  })

  it('rejects grouping fewer than 2 items', () => {
    const cmd: Command = { ...base(), type: 'GroupItems', groupId: 'g-1', itemIds: ['item-1'] }
    const state = stateWithItems(itemAt('item-1'))
    const result = validateCommand(cmd, state)
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('GROUP_TOO_SMALL')
  })

  it('rejects if group ID already exists', () => {
    const cmd: Command = { ...base(), type: 'GroupItems', groupId: 'g-existing', itemIds: ['item-1', 'item-2'] }
    const state = stateWithItems(itemAt('item-1'), itemAt('item-2', [1, 0, 0], 'g-existing'))
    // g-existing is already in groups because item-2 has it
    const result = validateCommand(cmd, state)
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('GROUP_ALREADY_EXISTS')
  })

  it('rejects if item already in a group', () => {
    const cmd: Command = { ...base(), type: 'GroupItems', groupId: 'g-new', itemIds: ['item-1', 'item-2'] }
    const state = stateWithItems(itemAt('item-1'), itemAt('item-2', [1, 0, 0], 'g-other'))
    // item-2 is already grouped
    const result = validateCommand(cmd, state)
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('ITEM_ALREADY_GROUPED')
  })
})

// ─── UngroupItems ────────────────────────────────────────────────────────────

describe('UngroupItems', () => {
  it('accepts valid ungrouping', () => {
    const cmd: Command = { ...base(), type: 'UngroupItems', groupId: 'g-1', itemIds: ['item-1', 'item-2'] }
    const state = stateWithItems(itemAt('item-1', [0, 0, 0], 'g-1'), itemAt('item-2', [1, 0, 0], 'g-1'))
    expect(validateCommand(cmd, state).ok).toBe(true)
  })

  it('rejects non-existent group', () => {
    const cmd: Command = { ...base(), type: 'UngroupItems', groupId: 'no-group', itemIds: ['item-1'] }
    const result = validateCommand(cmd, stateWith())
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('GROUP_NOT_FOUND')
  })
})

// ─── Scenario Commands ───────────────────────────────────────────────────────

describe('ActivateScenario', () => {
  it('accepts existing scenario', () => {
    const cmd: Command = { ...base(), type: 'ActivateScenario', scenarioId: 's-1' }
    const state = stateWith({ scenarios: new Set(['s-1']) })
    expect(validateCommand(cmd, state).ok).toBe(true)
  })

  it('rejects non-existent scenario', () => {
    const cmd: Command = { ...base(), type: 'ActivateScenario', scenarioId: 'no-scenario' }
    const result = validateCommand(cmd, stateWith())
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('SCENARIO_NOT_FOUND')
  })
})

describe('DeleteScenario', () => {
  it('accepts existing scenario', () => {
    const cmd: Command = { ...base(), type: 'DeleteScenario', scenarioId: 's-1' }
    const state = stateWith({ scenarios: new Set(['s-1']) })
    expect(validateCommand(cmd, state).ok).toBe(true)
  })

  it('rejects non-existent scenario', () => {
    const cmd: Command = { ...base(), type: 'DeleteScenario', scenarioId: 'no-scenario' }
    const result = validateCommand(cmd, stateWith())
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error.code).toBe('SCENARIO_NOT_FOUND')
  })
})

// ─── Always-valid commands ───────────────────────────────────────────────────

describe('CreateScenario', () => {
  it('always valid', () => {
    const cmd: Command = { ...base(), type: 'CreateScenario', scenarioId: 's-1', name: 'Test' }
    expect(validateCommand(cmd, stateWith()).ok).toBe(true)
  })
})

describe('CreateLayoutSnapshot', () => {
  it('always valid', () => {
    const cmd: Command = { ...base(), type: 'CreateLayoutSnapshot', snapshotId: 'snap-1' }
    expect(validateCommand(cmd, stateWith()).ok).toBe(true)
  })
})
