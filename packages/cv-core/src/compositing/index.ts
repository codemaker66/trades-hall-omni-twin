// ---------------------------------------------------------------------------
// CV-3: Compositing — barrel export
// ---------------------------------------------------------------------------

export {
  createDepthBuffer,
  unprojectPixel,
  compositeDepthOrder,
  linearizeDepth,
} from './depth-compositing.js';

export {
  rayEllipsoidIntersection,
  rayTriangleIntersection,
  rayAABBIntersection,
} from './raycasting.js';
