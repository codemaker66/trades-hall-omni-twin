// Types
export type {
  AABB2D, Point2D, RoomGeometry, AnalysisItem,
  CollisionResult, SightlineResult, CrowdFlowResult, GpuCapabilities,
} from './types'

// Collision detection
export { detectCollisions, detectCollisionsCPU, COLLISION_SHADER } from './collision'

// Sightline analysis
export { analyzeSightlines, analyzeSightlinesCPU, SIGHTLINE_SHADER } from './sightlines'

// Crowd flow simulation
export {
  simulateCrowdFlow, simulateCrowdFlowCPU, CROWD_FLOW_SHADER,
  type CrowdFlowParams,
} from './crowd-flow'
