// ---------------------------------------------------------------------------
// CV-8: Point Cloud Processing — barrel export
// ---------------------------------------------------------------------------

export {
  statisticalOutlierRemoval,
  voxelDownsample,
} from './cleaning.js';
export {
  icpPointToPoint,
  ransacGlobalRegistration,
  findClosestPoint,
} from './registration.js';
export {
  ransacPlaneDetection,
  douglasPeucker,
  extractFloorBoundary,
} from './room-boundary.js';
export { estimateNormals, orientNormals } from './normals.js';
