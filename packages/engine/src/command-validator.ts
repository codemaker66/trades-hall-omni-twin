/**
 * Pure command validation functions.
 * Validates commands against the current projected state.
 * Returns Result<ValidatedCommand, ValidationError>.
 */

import type {
  Command,
  PlaceItemCommand,
  MoveItemCommand,
  RotateItemCommand,
  ScaleItemCommand,
  RemoveItemCommand,
  MoveItemsBatchCommand,
  RotateItemsBatchCommand,
  GroupItemsCommand,
  UngroupItemsCommand,
  CreateVenueCommand,
  RenameVenueCommand,
  ActivateScenarioCommand,
  DeleteScenarioCommand,
  CreateLayoutSnapshotCommand,
  ImportLayoutCommand,
  ValidationError,
  Result,
  Position3D,
  ItemId,
  FurnitureType,
} from '@omni-twin/types'

// ─── Projected State Shape ───────────────────────────────────────────────────

/** Minimal item shape the validator needs. */
export interface ValidatorItem {
  id: ItemId
  furnitureType: FurnitureType
  position: Position3D
  rotation: [number, number, number]
  groupId?: string
}

/** Minimal venue state shape for validation.
 *  Accepts both Set<string> and Map<string, T> for scenarios (via `has` method).
 */
export interface ValidatorVenueState {
  venueId: string
  name: string
  archived: boolean
  items: Map<ItemId, ValidatorItem>
  groups: Set<string>
  scenarios: { has(key: string): boolean }
}

// ─── Venue Bounds ────────────────────────────────────────────────────────────

/** Default venue bounds (can be expanded later per-venue). */
const VENUE_BOUNDS = {
  minX: -100,
  maxX: 100,
  minZ: -100,
  maxZ: 100,
  minY: 0,
  maxY: 20,
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function err(code: string, message: string, field?: string): Result<never, ValidationError> {
  return { ok: false, error: { code, message, field } }
}

function ok<T>(value: T): Result<T, ValidationError> {
  return { ok: true, value }
}

function isPositionInBounds(pos: Position3D): boolean {
  const [x, y, z] = pos
  return (
    x >= VENUE_BOUNDS.minX && x <= VENUE_BOUNDS.maxX &&
    y >= VENUE_BOUNDS.minY && y <= VENUE_BOUNDS.maxY &&
    z >= VENUE_BOUNDS.minZ && z <= VENUE_BOUNDS.maxZ
  )
}

function isFinitePosition(pos: Position3D): boolean {
  return pos.every((v) => Number.isFinite(v))
}

// ─── Validators ──────────────────────────────────────────────────────────────

function validateCreateVenue(cmd: CreateVenueCommand): Result<CreateVenueCommand, ValidationError> {
  if (!cmd.name.trim()) return err('VENUE_NAME_EMPTY', 'Venue name cannot be empty', 'name')
  if (cmd.name.length > 200) return err('VENUE_NAME_TOO_LONG', 'Venue name must be 200 characters or less', 'name')
  return ok(cmd)
}

function validateRenameVenue(cmd: RenameVenueCommand, state: ValidatorVenueState): Result<RenameVenueCommand, ValidationError> {
  if (state.archived) return err('VENUE_ARCHIVED', 'Cannot rename an archived venue')
  if (!cmd.name.trim()) return err('VENUE_NAME_EMPTY', 'Venue name cannot be empty', 'name')
  if (cmd.name.length > 200) return err('VENUE_NAME_TOO_LONG', 'Venue name must be 200 characters or less', 'name')
  return ok(cmd)
}

function validateArchiveVenue(state: ValidatorVenueState): Result<null, ValidationError> {
  if (state.archived) return err('VENUE_ALREADY_ARCHIVED', 'Venue is already archived')
  return ok(null)
}

function validatePlaceItem(cmd: PlaceItemCommand, state: ValidatorVenueState): Result<PlaceItemCommand, ValidationError> {
  if (state.archived) return err('VENUE_ARCHIVED', 'Cannot place items in an archived venue')
  if (state.items.has(cmd.itemId)) return err('ITEM_ALREADY_EXISTS', `Item ${cmd.itemId} already exists`, 'itemId')
  if (!isFinitePosition(cmd.position)) return err('INVALID_POSITION', 'Position must contain finite numbers', 'position')
  if (!isPositionInBounds(cmd.position)) return err('POSITION_OUT_OF_BOUNDS', 'Position is outside venue bounds', 'position')
  if (!isFinitePosition(cmd.rotation)) return err('INVALID_ROTATION', 'Rotation must contain finite numbers', 'rotation')
  if (cmd.groupId && !state.groups.has(cmd.groupId)) return err('GROUP_NOT_FOUND', `Group ${cmd.groupId} does not exist`, 'groupId')
  return ok(cmd)
}

function validateMoveItem(cmd: MoveItemCommand, state: ValidatorVenueState): Result<MoveItemCommand, ValidationError> {
  if (state.archived) return err('VENUE_ARCHIVED', 'Cannot move items in an archived venue')
  if (!state.items.has(cmd.itemId)) return err('ITEM_NOT_FOUND', `Item ${cmd.itemId} not found`, 'itemId')
  if (!isFinitePosition(cmd.position)) return err('INVALID_POSITION', 'Position must contain finite numbers', 'position')
  if (!isPositionInBounds(cmd.position)) return err('POSITION_OUT_OF_BOUNDS', 'Position is outside venue bounds', 'position')
  return ok(cmd)
}

function validateRotateItem(cmd: RotateItemCommand, state: ValidatorVenueState): Result<RotateItemCommand, ValidationError> {
  if (state.archived) return err('VENUE_ARCHIVED', 'Cannot rotate items in an archived venue')
  if (!state.items.has(cmd.itemId)) return err('ITEM_NOT_FOUND', `Item ${cmd.itemId} not found`, 'itemId')
  if (!isFinitePosition(cmd.rotation)) return err('INVALID_ROTATION', 'Rotation must contain finite numbers', 'rotation')
  return ok(cmd)
}

function validateScaleItem(cmd: ScaleItemCommand, state: ValidatorVenueState): Result<ScaleItemCommand, ValidationError> {
  if (state.archived) return err('VENUE_ARCHIVED', 'Cannot scale items in an archived venue')
  if (!state.items.has(cmd.itemId)) return err('ITEM_NOT_FOUND', `Item ${cmd.itemId} not found`, 'itemId')
  if (!isFinitePosition(cmd.scale)) return err('INVALID_SCALE', 'Scale must contain finite numbers', 'scale')
  if (cmd.scale.some((v) => v <= 0)) return err('INVALID_SCALE', 'Scale values must be positive', 'scale')
  return ok(cmd)
}

function validateRemoveItem(cmd: RemoveItemCommand, state: ValidatorVenueState): Result<RemoveItemCommand, ValidationError> {
  if (state.archived) return err('VENUE_ARCHIVED', 'Cannot remove items from an archived venue')
  if (!state.items.has(cmd.itemId)) return err('ITEM_NOT_FOUND', `Item ${cmd.itemId} not found`, 'itemId')
  return ok(cmd)
}

function validateMoveItemsBatch(cmd: MoveItemsBatchCommand, state: ValidatorVenueState): Result<MoveItemsBatchCommand, ValidationError> {
  if (state.archived) return err('VENUE_ARCHIVED', 'Cannot move items in an archived venue')
  if (cmd.moves.length === 0) return err('EMPTY_BATCH', 'Batch move must contain at least one item')
  for (const move of cmd.moves) {
    if (!state.items.has(move.itemId)) return err('ITEM_NOT_FOUND', `Item ${move.itemId} not found`, 'itemId')
    if (!isFinitePosition(move.position)) return err('INVALID_POSITION', `Invalid position for item ${move.itemId}`, 'position')
    if (!isPositionInBounds(move.position)) return err('POSITION_OUT_OF_BOUNDS', `Position out of bounds for item ${move.itemId}`, 'position')
  }
  return ok(cmd)
}

function validateRotateItemsBatch(cmd: RotateItemsBatchCommand, state: ValidatorVenueState): Result<RotateItemsBatchCommand, ValidationError> {
  if (state.archived) return err('VENUE_ARCHIVED', 'Cannot rotate items in an archived venue')
  if (cmd.rotations.length === 0) return err('EMPTY_BATCH', 'Batch rotate must contain at least one item')
  for (const rot of cmd.rotations) {
    if (!state.items.has(rot.itemId)) return err('ITEM_NOT_FOUND', `Item ${rot.itemId} not found`, 'itemId')
    if (!isFinitePosition(rot.rotation)) return err('INVALID_ROTATION', `Invalid rotation for item ${rot.itemId}`, 'rotation')
  }
  return ok(cmd)
}

function validateGroupItems(cmd: GroupItemsCommand, state: ValidatorVenueState): Result<GroupItemsCommand, ValidationError> {
  if (state.archived) return err('VENUE_ARCHIVED', 'Cannot group items in an archived venue')
  if (cmd.itemIds.length < 2) return err('GROUP_TOO_SMALL', 'A group must contain at least 2 items', 'itemIds')
  if (state.groups.has(cmd.groupId)) return err('GROUP_ALREADY_EXISTS', `Group ${cmd.groupId} already exists`, 'groupId')
  for (const id of cmd.itemIds) {
    if (!state.items.has(id)) return err('ITEM_NOT_FOUND', `Item ${id} not found`, 'itemIds')
    const item = state.items.get(id)!
    if (item.groupId) return err('ITEM_ALREADY_GROUPED', `Item ${id} is already in a group`, 'itemIds')
  }
  return ok(cmd)
}

function validateUngroupItems(cmd: UngroupItemsCommand, state: ValidatorVenueState): Result<UngroupItemsCommand, ValidationError> {
  if (state.archived) return err('VENUE_ARCHIVED', 'Cannot ungroup items in an archived venue')
  if (!state.groups.has(cmd.groupId)) return err('GROUP_NOT_FOUND', `Group ${cmd.groupId} not found`, 'groupId')
  for (const id of cmd.itemIds) {
    if (!state.items.has(id)) return err('ITEM_NOT_FOUND', `Item ${id} not found`, 'itemIds')
  }
  return ok(cmd)
}

function validateActivateScenario(cmd: ActivateScenarioCommand, state: ValidatorVenueState): Result<ActivateScenarioCommand, ValidationError> {
  if (!state.scenarios.has(cmd.scenarioId)) return err('SCENARIO_NOT_FOUND', `Scenario ${cmd.scenarioId} not found`, 'scenarioId')
  return ok(cmd)
}

function validateDeleteScenario(cmd: DeleteScenarioCommand, state: ValidatorVenueState): Result<DeleteScenarioCommand, ValidationError> {
  if (!state.scenarios.has(cmd.scenarioId)) return err('SCENARIO_NOT_FOUND', `Scenario ${cmd.scenarioId} not found`, 'scenarioId')
  return ok(cmd)
}

function validateImportLayout(cmd: ImportLayoutCommand, state: ValidatorVenueState): Result<ImportLayoutCommand, ValidationError> {
  if (state.archived) return err('VENUE_ARCHIVED', 'Cannot import layout into an archived venue')
  for (const item of cmd.items) {
    if (!isFinitePosition(item.position)) return err('INVALID_POSITION', `Invalid position for item ${item.itemId}`, 'position')
    if (!isPositionInBounds(item.position)) return err('POSITION_OUT_OF_BOUNDS', `Position out of bounds for item ${item.itemId}`, 'position')
  }
  return ok(cmd)
}

// ─── Main Validator ──────────────────────────────────────────────────────────

/**
 * Validate a command against the current projected state.
 * Returns the validated command if valid, or a ValidationError if not.
 */
export function validateCommand(cmd: Command, state: ValidatorVenueState): Result<Command, ValidationError> {
  switch (cmd.type) {
    case 'CreateVenue':
      return validateCreateVenue(cmd)
    case 'RenameVenue':
      return validateRenameVenue(cmd, state)
    case 'ArchiveVenue':
      return validateArchiveVenue(state) as Result<Command, ValidationError>
    case 'PlaceItem':
      return validatePlaceItem(cmd, state)
    case 'MoveItem':
      return validateMoveItem(cmd, state)
    case 'RotateItem':
      return validateRotateItem(cmd, state)
    case 'ScaleItem':
      return validateScaleItem(cmd, state)
    case 'RemoveItem':
      return validateRemoveItem(cmd, state)
    case 'MoveItemsBatch':
      return validateMoveItemsBatch(cmd, state)
    case 'RotateItemsBatch':
      return validateRotateItemsBatch(cmd, state)
    case 'GroupItems':
      return validateGroupItems(cmd, state)
    case 'UngroupItems':
      return validateUngroupItems(cmd, state)
    case 'CreateScenario':
      return ok(cmd) // No state-dependent validation needed
    case 'ActivateScenario':
      return validateActivateScenario(cmd, state)
    case 'DeleteScenario':
      return validateDeleteScenario(cmd, state)
    case 'CreateLayoutSnapshot':
      return ok(cmd as CreateLayoutSnapshotCommand) // Always valid
    case 'ImportLayout':
      return validateImportLayout(cmd, state)
  }
}

/** Create an empty projected state for a new venue. */
export function emptyProjectedState(venueId: string): ValidatorVenueState {
  return {
    venueId,
    name: '',
    archived: false,
    items: new Map(),
    groups: new Set(),
    scenarios: new Set(),
  }
}
