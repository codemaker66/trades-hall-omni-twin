// ---------------------------------------------------------------------------
// CV-9: Depth Estimation — barrel export
// ---------------------------------------------------------------------------

export {
  createDepthMap,
  fillHoles,
  medianFilter,
  computeDepthGradient,
} from './depth-processing.js';

export {
  computeDisparity,
  disparityToDepth,
  estimateBaseline,
} from './stereo.js';

export {
  createTSDFVolume,
  integrateTSDFFrame,
  extractMeshFromTSDF,
} from './fusion.js';
