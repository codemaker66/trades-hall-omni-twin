import type { VenueId, ItemId, UserId, Position3D, Rotation3D, FurnitureType } from './index'

// ─── Base ────────────────────────────────────────────────────────────────────

/** Base shape shared by all commands. */
export interface CommandBase {
  /** Command type discriminator */
  type: string
  /** User issuing the command */
  userId: UserId
  /** Target venue */
  venueId: VenueId
}

// ─── Venue Commands ──────────────────────────────────────────────────────────

export interface CreateVenueCommand extends CommandBase {
  type: 'CreateVenue'
  name: string
}

export interface RenameVenueCommand extends CommandBase {
  type: 'RenameVenue'
  name: string
}

export interface ArchiveVenueCommand extends CommandBase {
  type: 'ArchiveVenue'
}

// ─── Item Commands ───────────────────────────────────────────────────────────

export interface PlaceItemCommand extends CommandBase {
  type: 'PlaceItem'
  itemId: ItemId
  furnitureType: FurnitureType
  position: Position3D
  rotation: Rotation3D
  groupId?: string
}

export interface MoveItemCommand extends CommandBase {
  type: 'MoveItem'
  itemId: ItemId
  position: Position3D
}

export interface RotateItemCommand extends CommandBase {
  type: 'RotateItem'
  itemId: ItemId
  rotation: Rotation3D
}

export interface ScaleItemCommand extends CommandBase {
  type: 'ScaleItem'
  itemId: ItemId
  scale: Position3D
}

export interface RemoveItemCommand extends CommandBase {
  type: 'RemoveItem'
  itemId: ItemId
}

/** Batch move for multi-select drag. */
export interface MoveItemsBatchCommand extends CommandBase {
  type: 'MoveItemsBatch'
  moves: Array<{ itemId: ItemId; position: Position3D }>
}

/** Batch rotate for multi-select rotation. */
export interface RotateItemsBatchCommand extends CommandBase {
  type: 'RotateItemsBatch'
  rotations: Array<{ itemId: ItemId; rotation: Rotation3D }>
}

// ─── Group Commands ──────────────────────────────────────────────────────────

export interface GroupItemsCommand extends CommandBase {
  type: 'GroupItems'
  groupId: string
  itemIds: ItemId[]
}

export interface UngroupItemsCommand extends CommandBase {
  type: 'UngroupItems'
  groupId: string
  itemIds: ItemId[]
}

// ─── Scenario Commands ───────────────────────────────────────────────────────

export interface CreateScenarioCommand extends CommandBase {
  type: 'CreateScenario'
  scenarioId: string
  name: string
}

export interface ActivateScenarioCommand extends CommandBase {
  type: 'ActivateScenario'
  scenarioId: string
}

export interface DeleteScenarioCommand extends CommandBase {
  type: 'DeleteScenario'
  scenarioId: string
}

// ─── Layout Commands ─────────────────────────────────────────────────────────

export interface CreateLayoutSnapshotCommand extends CommandBase {
  type: 'CreateLayoutSnapshot'
  snapshotId: string
}

export interface ImportLayoutCommand extends CommandBase {
  type: 'ImportLayout'
  sourceFormat: string
  items: Array<{
    itemId: ItemId
    furnitureType: FurnitureType
    position: Position3D
    rotation: Rotation3D
    groupId?: string
  }>
}

// ─── Discriminated Union ─────────────────────────────────────────────────────

/** All commands as a discriminated union on `type`. */
export type Command =
  | CreateVenueCommand
  | RenameVenueCommand
  | ArchiveVenueCommand
  | PlaceItemCommand
  | MoveItemCommand
  | RotateItemCommand
  | ScaleItemCommand
  | RemoveItemCommand
  | MoveItemsBatchCommand
  | RotateItemsBatchCommand
  | GroupItemsCommand
  | UngroupItemsCommand
  | CreateScenarioCommand
  | ActivateScenarioCommand
  | DeleteScenarioCommand
  | CreateLayoutSnapshotCommand
  | ImportLayoutCommand

/** All possible command type discriminators. */
export type CommandType = Command['type']

// ─── Result Type ─────────────────────────────────────────────────────────────

/** Validation error for a rejected command. */
export interface ValidationError {
  code: string
  message: string
  field?: string
}

/** Result of validating or handling a command. */
export type Result<T, E = ValidationError> =
  | { ok: true; value: T }
  | { ok: false; error: E }
