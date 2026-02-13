// ---------------------------------------------------------------------------
// CV-11: XR — barrel export
// ---------------------------------------------------------------------------

export {
  rayPlaneIntersection,
  snapToSurface,
  hitTestFloor,
} from './hit-testing.js';

export {
  createPose6DoF,
  interpolatePose,
  poseToMatrix,
  matrixToPose,
  composePoses,
  inversePose,
} from './pose.js';

export {
  createUSDZMeta,
  validateUSDZConstraints,
  estimateUSDZFileSize,
} from './usdz-types.js';
