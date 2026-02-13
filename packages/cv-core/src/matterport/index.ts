// ---------------------------------------------------------------------------
// CV-1: Matterport — barrel export
// ---------------------------------------------------------------------------

export {
  createSweep,
  createFloorPlan,
  sweepDistance,
} from './api-types.js';

export type {
  Sweep,
  FloorPlan,
  Room,
  MatterportModel,
  E57Header,
  CoordinateTransform,
} from './api-types.js';

export {
  computeRoomArea,
  extractRoomBoundaries,
  isPointInRoom,
  computeFloorPlanBounds,
} from './floor-plan.js';

export {
  parseE57Header,
  transformPoint,
  computeTransformFromPoints,
} from './e57-types.js';
