/**
 * Store adapter: bridges the event-sourcing engine with a Zustand-style store.
 * When ENABLE_EVENT_SOURCING is on, store mutations dispatch commands internally.
 * When off, existing direct mutation works unchanged.
 */

import type {
  Command,
  DomainEvent,
  ItemId,
  FurnitureType,
  Position3D,
  Rotation3D,
} from '@omni-twin/types'
import { handleCommand } from './command-handler'
import { applyEvent, emptyVenueState, type ProjectedVenueState } from './projector'
import { UndoManager } from './undo-manager'

// ─── Types ───────────────────────────────────────────────────────────────────

export interface EventSourcedStoreState {
  /** The projected venue state from events. */
  projected: ProjectedVenueState
  /** Event log (in-memory for now, can be synced to event store). */
  eventLog: DomainEvent[]
  /** Undo manager. */
  undoManager: UndoManager
}

export interface EventSourcedStoreActions {
  /** Dispatch a command. Returns the produced events or null on validation error. */
  dispatch: (command: Command) => DomainEvent[] | null
  /** Undo the last action. Returns compensating events or null. */
  undo: () => DomainEvent[] | null
  /** Redo the last undone action. Returns events or null. */
  redo: () => DomainEvent[] | null
  /** Begin a batch (for drag operations). */
  beginBatch: () => void
  /** End the current batch. */
  endBatch: () => void
  /** Reset the store to initial state. */
  reset: (venueId: string) => void
  /** Get the current projected state. */
  getState: () => ProjectedVenueState
}

// ─── Adapter ─────────────────────────────────────────────────────────────────

/**
 * Create an event-sourced store adapter.
 * Manages projected state + event log + undo/redo.
 */
export function createEventSourcedStore(venueId: string, userId: string): EventSourcedStoreState & EventSourcedStoreActions {
  let state: EventSourcedStoreState = {
    projected: emptyVenueState(venueId),
    eventLog: [],
    undoManager: new UndoManager(),
  }

  function applyEvents(events: DomainEvent[]): void {
    for (const event of events) {
      state.projected = applyEvent(state.projected, event)
    }
    state.eventLog.push(...events)
  }

  const actions: EventSourcedStoreActions = {
    dispatch(command: Command): DomainEvent[] | null {
      const stateBefore = state.projected
      const result = handleCommand(state.projected, command)
      if (!result.ok) return null

      applyEvents(result.value)
      state.undoManager.record(result.value, stateBefore)
      return result.value
    },

    undo(): DomainEvent[] | null {
      const compensating = state.undoManager.undo(state.projected)
      if (!compensating) return null
      applyEvents(compensating)
      return compensating
    },

    redo(): DomainEvent[] | null {
      const events = state.undoManager.redo()
      if (!events) return null
      applyEvents(events)
      return events
    },

    beginBatch(): void {
      state.undoManager.beginBatch()
    },

    endBatch(): void {
      state.undoManager.endBatch()
    },

    reset(newVenueId: string): void {
      state.projected = emptyVenueState(newVenueId)
      state.eventLog = []
      state.undoManager.clear()
    },

    getState(): ProjectedVenueState {
      return state.projected
    },
  }

  return { ...state, ...actions }
}

// ─── Command Helpers ─────────────────────────────────────────────────────────

/** Create a PlaceItem command from common parameters. */
export function placeItemCommand(
  userId: string,
  venueId: string,
  itemId: ItemId,
  furnitureType: FurnitureType,
  position: Position3D,
  rotation: Rotation3D,
  groupId?: string,
): Command {
  return { type: 'PlaceItem', userId, venueId, itemId, furnitureType, position, rotation, groupId }
}

/** Create a MoveItem command. */
export function moveItemCommand(userId: string, venueId: string, itemId: ItemId, position: Position3D): Command {
  return { type: 'MoveItem', userId, venueId, itemId, position }
}

/** Create a RemoveItem command. */
export function removeItemCommand(userId: string, venueId: string, itemId: ItemId): Command {
  return { type: 'RemoveItem', userId, venueId, itemId }
}

/** Create a MoveItemsBatch command. */
export function moveItemsBatchCommand(
  userId: string,
  venueId: string,
  moves: Array<{ itemId: ItemId; position: Position3D }>,
): Command {
  return { type: 'MoveItemsBatch', userId, venueId, moves }
}

/** Create a GroupItems command. */
export function groupItemsCommand(userId: string, venueId: string, groupId: string, itemIds: ItemId[]): Command {
  return { type: 'GroupItems', userId, venueId, groupId, itemIds }
}
