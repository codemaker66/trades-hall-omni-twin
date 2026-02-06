// Domain event types, command types, and shared entity types.
// These will be populated in Layer 1 (Tranches 1.1–1.2).

/** Unique identifier for a venue */
export type VenueId = string

/** Unique identifier for a furniture item */
export type ItemId = string

/** Unique identifier for a user */
export type UserId = string

/** 3D position tuple */
export type Position3D = [x: number, y: number, z: number]

/** 3D rotation tuple (Euler angles) */
export type Rotation3D = [x: number, y: number, z: number]

// Auth types
export type { AuthUser, Session, RegisterInput, LoginInput, AuthResponse, AuthError, UserRole, VenueRole } from './auth'

// Domain event types
export {
  assertNever,
  type DomainEventBase,
  type DomainEvent,
  type DomainEventType,
  type EventByType,
  type VenueCreated,
  type VenueRenamed,
  type VenueArchived,
  type ItemPlaced,
  type ItemMoved,
  type ItemRotated,
  type ItemScaled,
  type ItemRemoved,
  type ItemsBatchMoved,
  type ItemsBatchRotated,
  type GroupCreated,
  type GroupDissolved,
  type ItemsGrouped,
  type ItemsUngrouped,
  type ScenarioCreated,
  type ScenarioActivated,
  type ScenarioDeleted,
  type LayoutSnapshotCreated,
  type LayoutImported,
} from './events'

// Command types
export {
  type CommandBase,
  type Command,
  type CommandType,
  type CreateVenueCommand,
  type RenameVenueCommand,
  type ArchiveVenueCommand,
  type PlaceItemCommand,
  type MoveItemCommand,
  type RotateItemCommand,
  type ScaleItemCommand,
  type RemoveItemCommand,
  type MoveItemsBatchCommand,
  type RotateItemsBatchCommand,
  type GroupItemsCommand,
  type UngroupItemsCommand,
  type CreateScenarioCommand,
  type ActivateScenarioCommand,
  type DeleteScenarioCommand,
  type CreateLayoutSnapshotCommand,
  type ImportLayoutCommand,
  type ValidationError,
  type Result,
} from './commands'

/** Furniture type enum */
export type FurnitureType =
  | 'chair'
  | 'round-table'
  | 'rect-table'
  | 'trestle-table'
  | 'podium'
  | 'stage'
  | 'bar'
