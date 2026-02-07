import type { VenueId, ItemId, UserId, Position3D, Rotation3D, FurnitureType } from './index'

// ─── Base ────────────────────────────────────────────────────────────────────

/** Base shape shared by all domain events. */
export interface DomainEventBase {
  /** Unique event ID (UUID v4) */
  id: string
  /** Event type discriminator */
  type: string
  /** ISO-8601 timestamp */
  timestamp: string
  /** User who caused the event */
  userId: UserId
  /** Venue this event belongs to */
  venueId: VenueId
  /** Monotonically increasing version per venue stream */
  version: number
}

// ─── Venue Events ────────────────────────────────────────────────────────────

export interface VenueCreated extends DomainEventBase {
  type: 'VenueCreated'
  payload: {
    name: string
  }
}

export interface VenueRenamed extends DomainEventBase {
  type: 'VenueRenamed'
  payload: {
    name: string
  }
}

export interface VenueArchived extends DomainEventBase {
  type: 'VenueArchived'
  payload: Record<string, never>
}

// ─── Item Events ─────────────────────────────────────────────────────────────

export interface ItemPlaced extends DomainEventBase {
  type: 'ItemPlaced'
  payload: {
    itemId: ItemId
    furnitureType: FurnitureType
    position: Position3D
    rotation: Rotation3D
    groupId?: string
  }
}

export interface ItemMoved extends DomainEventBase {
  type: 'ItemMoved'
  payload: {
    itemId: ItemId
    position: Position3D
  }
}

export interface ItemRotated extends DomainEventBase {
  type: 'ItemRotated'
  payload: {
    itemId: ItemId
    rotation: Rotation3D
  }
}

export interface ItemScaled extends DomainEventBase {
  type: 'ItemScaled'
  payload: {
    itemId: ItemId
    scale: Position3D
  }
}

export interface ItemRemoved extends DomainEventBase {
  type: 'ItemRemoved'
  payload: {
    itemId: ItemId
  }
}

/** Batch move for multi-item drag operations. */
export interface ItemsBatchMoved extends DomainEventBase {
  type: 'ItemsBatchMoved'
  payload: {
    moves: Array<{ itemId: ItemId; position: Position3D }>
  }
}

/** Batch rotate for multi-item rotation. */
export interface ItemsBatchRotated extends DomainEventBase {
  type: 'ItemsBatchRotated'
  payload: {
    rotations: Array<{ itemId: ItemId; rotation: Rotation3D }>
  }
}

// ─── Group Events ────────────────────────────────────────────────────────────

export interface GroupCreated extends DomainEventBase {
  type: 'GroupCreated'
  payload: {
    groupId: string
    itemIds: ItemId[]
  }
}

export interface GroupDissolved extends DomainEventBase {
  type: 'GroupDissolved'
  payload: {
    groupId: string
  }
}

export interface ItemsGrouped extends DomainEventBase {
  type: 'ItemsGrouped'
  payload: {
    groupId: string
    itemIds: ItemId[]
  }
}

export interface ItemsUngrouped extends DomainEventBase {
  type: 'ItemsUngrouped'
  payload: {
    groupId: string
    itemIds: ItemId[]
  }
}

// ─── Scenario Events ─────────────────────────────────────────────────────────

export interface ScenarioCreated extends DomainEventBase {
  type: 'ScenarioCreated'
  payload: {
    scenarioId: string
    name: string
  }
}

export interface ScenarioActivated extends DomainEventBase {
  type: 'ScenarioActivated'
  payload: {
    scenarioId: string
  }
}

export interface ScenarioDeleted extends DomainEventBase {
  type: 'ScenarioDeleted'
  payload: {
    scenarioId: string
  }
}

// ─── Layout Events ───────────────────────────────────────────────────────────

export interface LayoutSnapshotCreated extends DomainEventBase {
  type: 'LayoutSnapshotCreated'
  payload: {
    snapshotId: string
    itemCount: number
  }
}

export interface LayoutImported extends DomainEventBase {
  type: 'LayoutImported'
  payload: {
    sourceFormat: string
    itemCount: number
  }
}

// ─── Discriminated Union ─────────────────────────────────────────────────────

/** All domain events as a discriminated union on `type`. */
export type DomainEvent =
  | VenueCreated
  | VenueRenamed
  | VenueArchived
  | ItemPlaced
  | ItemMoved
  | ItemRotated
  | ItemScaled
  | ItemRemoved
  | ItemsBatchMoved
  | ItemsBatchRotated
  | GroupCreated
  | GroupDissolved
  | ItemsGrouped
  | ItemsUngrouped
  | ScenarioCreated
  | ScenarioActivated
  | ScenarioDeleted
  | LayoutSnapshotCreated
  | LayoutImported

/** All possible event type discriminators. */
export type DomainEventType = DomainEvent['type']

/** Helper: extract event shape by type. */
export type EventByType<T extends DomainEventType> = Extract<DomainEvent, { type: T }>

// ─── Exhaustive Check ────────────────────────────────────────────────────────

/** Compile-time exhaustive switch helper. Ensures all event types are handled. */
export function assertNever(event: never): never {
  throw new Error(`Unhandled event type: ${(event as DomainEventBase).type}`)
}
