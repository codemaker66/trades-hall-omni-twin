/**
 * Pure state projector: applies domain events to produce venue state.
 * No side effects — takes state + event, returns new state.
 */

import type {
  DomainEvent,
  ItemId,
  FurnitureType,
  Position3D,
  Rotation3D,
} from '@omni-twin/types'

// ─── Projected State Types ───────────────────────────────────────────────────

export interface ProjectedItem {
  id: ItemId
  furnitureType: FurnitureType
  position: Position3D
  rotation: Rotation3D
  scale: Position3D
  groupId?: string
}

export interface ProjectedScenario {
  id: string
  name: string
}

export interface ProjectedVenueState {
  venueId: string
  name: string
  archived: boolean
  items: Map<ItemId, ProjectedItem>
  groups: Set<string>
  scenarios: Map<string, ProjectedScenario>
  version: number
}

// ─── Initial State ───────────────────────────────────────────────────────────

export function emptyVenueState(venueId: string): ProjectedVenueState {
  return {
    venueId,
    name: '',
    archived: false,
    items: new Map(),
    groups: new Set(),
    scenarios: new Map(),
    version: 0,
  }
}

// ─── Apply Single Event ──────────────────────────────────────────────────────

/**
 * Apply a single domain event to the current state, returning a new state.
 * Pure function — no mutations.
 */
export function applyEvent(state: ProjectedVenueState, event: DomainEvent): ProjectedVenueState {
  // Clone maps/sets for immutability
  const items = new Map(state.items)
  const groups = new Set(state.groups)
  const scenarios = new Map(state.scenarios)

  switch (event.type) {
    // ── Venue ──
    case 'VenueCreated':
      return {
        ...state,
        name: event.payload.name,
        version: event.version,
      }

    case 'VenueRenamed':
      return {
        ...state,
        name: event.payload.name,
        version: event.version,
      }

    case 'VenueArchived':
      return {
        ...state,
        archived: true,
        version: event.version,
      }

    // ── Items ──
    case 'ItemPlaced': {
      const item: ProjectedItem = {
        id: event.payload.itemId,
        furnitureType: event.payload.furnitureType,
        position: event.payload.position,
        rotation: event.payload.rotation,
        scale: [1, 1, 1],
        groupId: event.payload.groupId,
      }
      items.set(item.id, item)
      return { ...state, items, version: event.version }
    }

    case 'ItemMoved': {
      const existing = items.get(event.payload.itemId)
      if (!existing) return { ...state, version: event.version }
      items.set(existing.id, { ...existing, position: event.payload.position })
      return { ...state, items, version: event.version }
    }

    case 'ItemRotated': {
      const existing = items.get(event.payload.itemId)
      if (!existing) return { ...state, version: event.version }
      items.set(existing.id, { ...existing, rotation: event.payload.rotation })
      return { ...state, items, version: event.version }
    }

    case 'ItemScaled': {
      const existing = items.get(event.payload.itemId)
      if (!existing) return { ...state, version: event.version }
      items.set(existing.id, { ...existing, scale: event.payload.scale })
      return { ...state, items, version: event.version }
    }

    case 'ItemRemoved': {
      items.delete(event.payload.itemId)
      return { ...state, items, version: event.version }
    }

    case 'ItemsBatchMoved': {
      for (const move of event.payload.moves) {
        const existing = items.get(move.itemId)
        if (existing) {
          items.set(existing.id, { ...existing, position: move.position })
        }
      }
      return { ...state, items, version: event.version }
    }

    case 'ItemsBatchRotated': {
      for (const rot of event.payload.rotations) {
        const existing = items.get(rot.itemId)
        if (existing) {
          items.set(existing.id, { ...existing, rotation: rot.rotation })
        }
      }
      return { ...state, items, version: event.version }
    }

    // ── Groups ──
    case 'GroupCreated': {
      groups.add(event.payload.groupId)
      for (const itemId of event.payload.itemIds) {
        const existing = items.get(itemId)
        if (existing) {
          items.set(itemId, { ...existing, groupId: event.payload.groupId })
        }
      }
      return { ...state, items, groups, version: event.version }
    }

    case 'GroupDissolved': {
      groups.delete(event.payload.groupId)
      for (const [id, item] of items) {
        if (item.groupId === event.payload.groupId) {
          items.set(id, { ...item, groupId: undefined })
        }
      }
      return { ...state, items, groups, version: event.version }
    }

    case 'ItemsGrouped': {
      groups.add(event.payload.groupId)
      for (const itemId of event.payload.itemIds) {
        const existing = items.get(itemId)
        if (existing) {
          items.set(itemId, { ...existing, groupId: event.payload.groupId })
        }
      }
      return { ...state, items, groups, version: event.version }
    }

    case 'ItemsUngrouped': {
      for (const itemId of event.payload.itemIds) {
        const existing = items.get(itemId)
        if (existing) {
          items.set(itemId, { ...existing, groupId: undefined })
        }
      }
      // Check if group is now empty
      let groupStillHasMembers = false
      for (const [, item] of items) {
        if (item.groupId === event.payload.groupId) {
          groupStillHasMembers = true
          break
        }
      }
      if (!groupStillHasMembers) {
        groups.delete(event.payload.groupId)
      }
      return { ...state, items, groups, version: event.version }
    }

    // ── Scenarios ──
    case 'ScenarioCreated': {
      scenarios.set(event.payload.scenarioId, {
        id: event.payload.scenarioId,
        name: event.payload.name,
      })
      return { ...state, scenarios, version: event.version }
    }

    case 'ScenarioActivated':
      return { ...state, version: event.version }

    case 'ScenarioDeleted': {
      scenarios.delete(event.payload.scenarioId)
      return { ...state, scenarios, version: event.version }
    }

    // ── Layout ──
    case 'LayoutSnapshotCreated':
      return { ...state, version: event.version }

    case 'LayoutImported':
      return { ...state, version: event.version }
  }
}

// ─── Project Full State ──────────────────────────────────────────────────────

/**
 * Project a list of events into a complete venue state.
 * Events must be in version order.
 */
export function projectState(venueId: string, events: DomainEvent[]): ProjectedVenueState {
  let state = emptyVenueState(venueId)
  for (const event of events) {
    state = applyEvent(state, event)
  }
  return state
}
