// ---------------------------------------------------------------------------
// CV-6: Mesh Processing — Barrel Export
// ---------------------------------------------------------------------------

export {
  computeQuadricError,
  buildQuadricMatrices,
  decimateMesh,
} from './decimation.js';

export {
  computeAngleBasedFlattening,
  packAtlas,
  chartBoundaryLength,
} from './uv-unwrap.js';

export {
  bakeNormalMap,
  computeAmbientOcclusion,
} from './baking.js';

export {
  createGLTFAccessor,
  estimateGLTFSize,
  validateMeshManifold,
} from './gltf-types.js';
