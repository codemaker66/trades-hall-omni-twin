// Spatial engine: event sourcing, command handling, ECS, and WASM bridge.

export const ENGINE_VERSION = '0.1.0'

export {
  validateCommand,
  emptyProjectedState,
  type ValidatorItem,
  type ValidatorVenueState,
} from './command-validator'

export {
  applyEvent,
  projectState,
  emptyVenueState,
  type ProjectedItem,
  type ProjectedScenario,
  type ProjectedVenueState,
} from './projector'

export { handleCommand } from './command-handler'

export {
  UndoManager,
  type UndoEntry,
  type ItemSnapshot,
} from './undo-manager'

export {
  createEventSourcedStore,
  placeItemCommand,
  moveItemCommand,
  removeItemCommand,
  moveItemsBatchCommand,
  groupItemsCommand,
  type EventSourcedStoreState,
  type EventSourcedStoreActions,
} from './store-adapter'

export {
  registerMigration,
  clearMigrations,
  migrateEvent,
  migrateEvents,
  needsMigration,
  toVersionedEvent,
  type EventMigration,
  type VersionedEvent,
} from './event-migrator'

// ECS
export {
  Position,
  Rotation,
  Scale,
  BoundingBox,
  FurnitureTag,
  GroupMember,
  Selectable,
  Draggable,
  furnitureTypeToIndex,
  indexToFurnitureType,
  createEcsWorld,
  resetEcsWorld,
  createFurnitureEntity,
  removeFurnitureEntity,
  isFurnitureEntity,
  getAllEntityIds,
  EntityIdMap,
  type EcsWorld,
  type FurnitureEntityInput,
  // Systems
  SpatialHash,
  type AABB,
  getEntityAABB,
  aabbOverlap,
  detectCollisions,
  checkCollision,
  checkPlacementCollision,
  type CollisionPair,
  rectSelect,
  pointSelect,
  snapToGrid,
  snapToHeightGrid,
  findNearest,
  findKNearest,
  snapToNearest,
  type NearestResult,
  // Bridge
  EcsBridge,
  type BridgeItem,
} from './ecs'

// Incremental computation
export {
  IncrementalGraph,
  type InputNode,
  type DerivedNode,
  type ObserverNode,
} from './incremental'
