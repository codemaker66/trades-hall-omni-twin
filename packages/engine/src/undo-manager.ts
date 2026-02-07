/**
 * Event-sourced undo/redo manager.
 * Tracks command history and produces compensating events for undo.
 * Supports batch operations for single undo entries during drag.
 */

import type { DomainEvent } from '@omni-twin/types'
import { applyEvent, type ProjectedVenueState } from './projector'

// ─── Types ───────────────────────────────────────────────────────────────────

/** A recorded command entry with its original events and the state before execution. */
export interface UndoEntry {
  /** The events produced by this command (or batch). */
  events: DomainEvent[]
  /** Snapshot of items affected, for producing compensating events. */
  previousState: ItemSnapshot[]
}

/** Minimal snapshot of an item before a change. */
export interface ItemSnapshot {
  id: string
  existed: boolean
  position?: [number, number, number]
  rotation?: [number, number, number]
  scale?: [number, number, number]
  groupId?: string
  furnitureType?: string
}

// ─── Undo Manager ────────────────────────────────────────────────────────────

export class UndoManager {
  private undoStack: UndoEntry[] = []
  private redoStack: UndoEntry[] = []
  private batchEntry: UndoEntry | null = null
  private maxHistory: number

  constructor(maxHistory = 100) {
    this.maxHistory = maxHistory
  }

  get canUndo(): boolean {
    return this.undoStack.length > 0
  }

  get canRedo(): boolean {
    return this.redoStack.length > 0
  }

  get undoCount(): number {
    return this.undoStack.length
  }

  get redoCount(): number {
    return this.redoStack.length
  }

  /**
   * Record events from a command execution.
   * Call this after handleCommand succeeds and events are applied.
   */
  record(events: DomainEvent[], stateBefore: ProjectedVenueState): void {
    const snapshots = captureSnapshots(events, stateBefore)
    const entry: UndoEntry = { events, previousState: snapshots }

    if (this.batchEntry) {
      // Merge into current batch
      this.batchEntry.events.push(...events)
      this.batchEntry.previousState.push(...snapshots)
      return
    }

    this.undoStack.push(entry)
    if (this.undoStack.length > this.maxHistory) {
      this.undoStack.shift()
    }
    // New action clears redo stack
    this.redoStack = []
  }

  /**
   * Begin a batch — all recorded events until endBatch() become a single undo entry.
   */
  beginBatch(): void {
    this.batchEntry = { events: [], previousState: [] }
  }

  /**
   * End the current batch. If any events were recorded, push as single undo entry.
   */
  endBatch(): void {
    if (this.batchEntry && this.batchEntry.events.length > 0) {
      this.undoStack.push(this.batchEntry)
      if (this.undoStack.length > this.maxHistory) {
        this.undoStack.shift()
      }
      this.redoStack = []
    }
    this.batchEntry = null
  }

  /**
   * Generate compensating events to undo the last action.
   * Returns the compensating events and moves the entry to redo stack.
   */
  undo(currentState: ProjectedVenueState): DomainEvent[] | null {
    const entry = this.undoStack.pop()
    if (!entry) return null

    const compensating = generateCompensatingEvents(entry, currentState)
    this.redoStack.push(entry)
    return compensating
  }

  /**
   * Re-apply the last undone action.
   * Returns the original events to replay.
   */
  redo(): DomainEvent[] | null {
    const entry = this.redoStack.pop()
    if (!entry) return null

    this.undoStack.push(entry)
    return entry.events
  }

  /** Clear all history. */
  clear(): void {
    this.undoStack = []
    this.redoStack = []
    this.batchEntry = null
  }
}

// ─── Snapshot Capture ────────────────────────────────────────────────────────

/**
 * Capture the state of items affected by events BEFORE they are applied.
 */
function captureSnapshots(events: DomainEvent[], state: ProjectedVenueState): ItemSnapshot[] {
  const snapshots: ItemSnapshot[] = []
  const seen = new Set<string>()

  for (const event of events) {
    const itemIds = getAffectedItemIds(event)
    for (const id of itemIds) {
      if (seen.has(id)) continue
      seen.add(id)

      const item = state.items.get(id)
      if (item) {
        snapshots.push({
          id,
          existed: true,
          position: [...item.position],
          rotation: [...item.rotation],
          scale: [...item.scale],
          groupId: item.groupId,
          furnitureType: item.furnitureType,
        })
      } else {
        snapshots.push({ id, existed: false })
      }
    }
  }

  return snapshots
}

/**
 * Extract item IDs affected by an event.
 */
function getAffectedItemIds(event: DomainEvent): string[] {
  switch (event.type) {
    case 'ItemPlaced':
    case 'ItemMoved':
    case 'ItemRotated':
    case 'ItemScaled':
    case 'ItemRemoved':
      return [event.payload.itemId]
    case 'ItemsBatchMoved':
      return event.payload.moves.map((m) => m.itemId)
    case 'ItemsBatchRotated':
      return event.payload.rotations.map((r) => r.itemId)
    case 'GroupCreated':
    case 'ItemsGrouped':
      return event.payload.itemIds
    case 'GroupDissolved':
      return [] // Items are handled by iterating in the projector
    case 'ItemsUngrouped':
      return event.payload.itemIds
    default:
      return []
  }
}

// ─── Compensating Events ─────────────────────────────────────────────────────

/**
 * Generate compensating events to reverse an undo entry.
 * These restore items to their pre-command state.
 */
function generateCompensatingEvents(
  entry: UndoEntry,
  currentState: ProjectedVenueState,
): DomainEvent[] {
  const events: DomainEvent[] = []
  const baseVersion = currentState.version

  let versionOffset = 0
  for (const snapshot of entry.previousState) {
    versionOffset++
    const base = {
      id: `undo-${Date.now()}-${versionOffset}`,
      timestamp: new Date().toISOString(),
      userId: 'system',
      venueId: currentState.venueId,
      version: baseVersion + versionOffset,
    }

    const currentItem = currentState.items.get(snapshot.id)

    if (!snapshot.existed) {
      // Item didn't exist before — remove it to undo
      if (currentItem) {
        events.push({ ...base, type: 'ItemRemoved', payload: { itemId: snapshot.id } } as DomainEvent)
      }
    } else if (!currentItem) {
      // Item existed before but is now gone — re-place it
      events.push({
        ...base,
        type: 'ItemPlaced',
        payload: {
          itemId: snapshot.id,
          furnitureType: snapshot.furnitureType!,
          position: snapshot.position!,
          rotation: snapshot.rotation!,
          ...(snapshot.groupId ? { groupId: snapshot.groupId } : {}),
        },
      } as DomainEvent)
    } else {
      // Item exists in both — restore position/rotation/scale
      if (
        snapshot.position &&
        (currentItem.position[0] !== snapshot.position[0] ||
          currentItem.position[1] !== snapshot.position[1] ||
          currentItem.position[2] !== snapshot.position[2])
      ) {
        events.push({
          ...base,
          type: 'ItemMoved',
          payload: { itemId: snapshot.id, position: snapshot.position },
        } as DomainEvent)
        versionOffset++
      }
      if (
        snapshot.rotation &&
        (currentItem.rotation[0] !== snapshot.rotation[0] ||
          currentItem.rotation[1] !== snapshot.rotation[1] ||
          currentItem.rotation[2] !== snapshot.rotation[2])
      ) {
        events.push({
          ...base,
          type: 'ItemRotated',
          version: baseVersion + versionOffset,
          payload: { itemId: snapshot.id, rotation: snapshot.rotation },
        } as DomainEvent)
        versionOffset++
      }
    }
  }

  return events
}
