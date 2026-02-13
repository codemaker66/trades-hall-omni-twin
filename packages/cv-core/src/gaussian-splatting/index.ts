// ---------------------------------------------------------------------------
// CV-2: Gaussian Splatting — barrel export
// ---------------------------------------------------------------------------

export {
  createGaussian3D,
  covarianceFromScaleRotation,
  projectGaussian2D,
  evaluateGaussian2D,
} from './gaussian.js';

export {
  radixSortByDepth,
  assignTiles,
} from './sorting.js';

export {
  quantizePositions,
  truncateSH,
  estimateCompressedSize,
} from './compression.js';

export {
  createTSDFGrid,
  integrateTSDF,
  marchingCubes,
} from './mesh-extraction.js';

export type {
  TSDFGrid,
  DepthIntrinsics,
} from './mesh-extraction.js';
