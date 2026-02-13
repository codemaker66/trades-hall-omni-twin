// ---------------------------------------------------------------------------
// CV-10: Segmentation — barrel export
// ---------------------------------------------------------------------------

export {
  createMask,
  maskUnion,
  maskIntersection,
  computeIoU,
  maskToPolygon,
  connectedComponents,
} from './mask-ops.js';

export {
  nonMaxSuppression,
  softNMS,
  bboxIoU,
} from './nms.js';

export {
  estimateRealDimensions,
  measureDistance,
  estimateVolume,
} from './dimension.js';

export {
  createBBox2D,
  createBBox3D,
  bboxArea,
  bboxContains,
  fitOrientedBBox,
  mergeBBoxes,
} from './bbox.js';
