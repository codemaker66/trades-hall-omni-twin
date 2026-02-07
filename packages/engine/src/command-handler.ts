/**
 * Command handler: validates commands against state and produces domain events.
 * Pure function — no side effects.
 */

import type {
  Command,
  DomainEvent,
  ValidationError,
  Result,
} from '@omni-twin/types'
import { validateCommand, type ValidatorVenueState } from './command-validator'
import type { ProjectedVenueState } from './projector'

// ─── Types ───────────────────────────────────────────────────────────────────

type NewEvent = Omit<DomainEvent, 'id' | 'version'>

// ─── UUID stub ───────────────────────────────────────────────────────────────

function uuid(): string {
  // Prefer crypto.randomUUID when available (Node 19+, modern browsers).
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  // Fallback for older environments.
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16)
  })
}

// ─── Event Factories ─────────────────────────────────────────────────────────

function makeBase(cmd: Command): { timestamp: string; userId: string; venueId: string } {
  return {
    timestamp: new Date().toISOString(),
    userId: cmd.userId,
    venueId: cmd.venueId,
  }
}

// ─── Command → Events Mapping ────────────────────────────────────────────────

function commandToEvents(cmd: Command): NewEvent[] {
  const base = makeBase(cmd)

  switch (cmd.type) {
    case 'CreateVenue':
      return [{ ...base, type: 'VenueCreated', payload: { name: cmd.name } }]

    case 'RenameVenue':
      return [{ ...base, type: 'VenueRenamed', payload: { name: cmd.name } }]

    case 'ArchiveVenue':
      return [{ ...base, type: 'VenueArchived', payload: {} }]

    case 'PlaceItem':
      return [{
        ...base, type: 'ItemPlaced',
        payload: {
          itemId: cmd.itemId,
          furnitureType: cmd.furnitureType,
          position: cmd.position,
          rotation: cmd.rotation,
          ...(cmd.groupId ? { groupId: cmd.groupId } : {}),
        },
      }]

    case 'MoveItem':
      return [{ ...base, type: 'ItemMoved', payload: { itemId: cmd.itemId, position: cmd.position } }]

    case 'RotateItem':
      return [{ ...base, type: 'ItemRotated', payload: { itemId: cmd.itemId, rotation: cmd.rotation } }]

    case 'ScaleItem':
      return [{ ...base, type: 'ItemScaled', payload: { itemId: cmd.itemId, scale: cmd.scale } }]

    case 'RemoveItem':
      return [{ ...base, type: 'ItemRemoved', payload: { itemId: cmd.itemId } }]

    case 'MoveItemsBatch':
      return [{ ...base, type: 'ItemsBatchMoved', payload: { moves: cmd.moves } }]

    case 'RotateItemsBatch':
      return [{ ...base, type: 'ItemsBatchRotated', payload: { rotations: cmd.rotations } }]

    case 'GroupItems':
      return [{ ...base, type: 'GroupCreated', payload: { groupId: cmd.groupId, itemIds: cmd.itemIds } }]

    case 'UngroupItems':
      return [{ ...base, type: 'ItemsUngrouped', payload: { groupId: cmd.groupId, itemIds: cmd.itemIds } }]

    case 'CreateScenario':
      return [{ ...base, type: 'ScenarioCreated', payload: { scenarioId: cmd.scenarioId, name: cmd.name } }]

    case 'ActivateScenario':
      return [{ ...base, type: 'ScenarioActivated', payload: { scenarioId: cmd.scenarioId } }]

    case 'DeleteScenario':
      return [{ ...base, type: 'ScenarioDeleted', payload: { scenarioId: cmd.scenarioId } }]

    case 'CreateLayoutSnapshot':
      return [{ ...base, type: 'LayoutSnapshotCreated', payload: { snapshotId: cmd.snapshotId, itemCount: 0 } }]

    case 'ImportLayout':
      return [
        // First emit the LayoutImported event
        { ...base, type: 'LayoutImported', payload: { sourceFormat: cmd.sourceFormat, itemCount: cmd.items.length } },
        // Then emit ItemPlaced for each item
        ...cmd.items.map((item) => ({
          ...base,
          type: 'ItemPlaced' as const,
          payload: {
            itemId: item.itemId,
            furnitureType: item.furnitureType,
            position: item.position,
            rotation: item.rotation,
            ...(item.groupId ? { groupId: item.groupId } : {}),
          },
        })),
      ]
  }
}

// ─── Adapt ProjectedVenueState → ValidatorVenueState ─────────────────────────

function toValidatorState(state: ProjectedVenueState): ValidatorVenueState {
  return {
    venueId: state.venueId,
    name: state.name,
    archived: state.archived,
    items: state.items,
    groups: state.groups,
    scenarios: state.scenarios, // Map.has() satisfies the { has } interface
  }
}

// ─── Main Handler ────────────────────────────────────────────────────────────

/**
 * Handle a command: validate against current state, produce events if valid.
 *
 * @param state - Current projected venue state
 * @param command - Command to handle
 * @returns Result with array of new events (without id/version assigned) or validation error
 */
export function handleCommand(
  state: ProjectedVenueState,
  command: Command,
): Result<DomainEvent[], ValidationError> {
  // Validate
  const validation = validateCommand(command, toValidatorState(state))
  if (!validation.ok) return validation

  // Map to events, assigning IDs and versions
  const newEvents = commandToEvents(command)
  const nextVersion = state.version

  const events: DomainEvent[] = newEvents.map((evt, i) => ({
    ...evt,
    id: uuid(),
    version: nextVersion + i + 1,
  } as DomainEvent))

  return { ok: true, value: events }
}
