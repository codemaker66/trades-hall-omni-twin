// Types
export type {
  Vec3, SpatialOp, SpatialOpType, AddOp, RemoveOp, MoveOp, RotateOp, ScaleOp,
  ObjectSnapshot, StateVector,
} from './types'
export { VEC3_ZERO, VEC3_ONE, vec3Add, vec3Eq } from './types'

// Operations
export {
  generateOpId, resetOpIdCounter, parseOpId,
  createAddOp, createRemoveOp, createMoveOp, createRotateOp, createScaleOp,
  opCompare,
} from './operation'

// State reconstruction
export { reconstructObject, reconstructAll } from './state'

// Merge
export { mergeOpSets, opSetDifference } from './merge'

// Document
export { SpatialDocument } from './document'

// Sync protocol
export {
  encodeStateVector, decodeStateVector,
  computeSyncMessage, applySyncMessage, fullSync,
} from './sync'
