// ECS module barrel export

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
} from './components'

export {
  createEcsWorld,
  resetEcsWorld,
  createFurnitureEntity,
  removeFurnitureEntity,
  isFurnitureEntity,
  getAllEntityIds,
  EntityIdMap,
  type EcsWorld,
  type FurnitureEntityInput,
} from './world'

// Systems
export {
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
} from './systems'

// Bridge
export { EcsBridge, type BridgeItem } from './bridge'
