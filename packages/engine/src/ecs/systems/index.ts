// ECS systems barrel export

export { SpatialHash, type AABB } from './spatial-index'

export {
  getEntityAABB,
  aabbOverlap,
  detectCollisions,
  checkCollision,
  checkPlacementCollision,
  type CollisionPair,
} from './collision'

export { rectSelect, pointSelect } from './selection'

export {
  snapToGrid,
  snapToHeightGrid,
  findNearest,
  findKNearest,
  snapToNearest,
  type NearestResult,
} from './snapping'
